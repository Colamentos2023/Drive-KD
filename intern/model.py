from __future__ import annotations

import glob
import importlib
import logging
import math
import os
import re
import sys
from contextlib import contextmanager
from typing import Any, Final, Literal, Protocol, Sequence, TypeAlias, cast, overload, runtime_checkable

import torch

logger = logging.getLogger(__name__)

default_model_dir: str = "/root/keith/colamentos/models"
default_checkpoint_dir: str = "checkpoints"

Tensor: TypeAlias = torch.Tensor
ImageTensor: TypeAlias = Tensor
ContentBlock: TypeAlias = str | ImageTensor
ContentBlocks: TypeAlias = Sequence[ContentBlock]


def set_default_model_dir(dir: str) -> None:
    global default_model_dir
    default_model_dir = dir


def set_default_checkpoint_dir(dir: str) -> None:
    global default_checkpoint_dir
    default_checkpoint_dir = dir


def get_default_model_dir() -> str:
    return default_model_dir


def get_default_checkpoint_dir() -> str:
    return default_checkpoint_dir


@runtime_checkable
class LLMConfig(Protocol):
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    bos_token_id: int
    eos_token_id: int


@runtime_checkable
class VisionConfig(Protocol):
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    image_size: int
    patch_size: int


@runtime_checkable
class VLMConfig(Protocol):
    llm_config: LLMConfig
    vision_config: VisionConfig
    downsample_ratio: float
    force_image_size: int
    hidden_size: int
    output_attentions: bool
    output_hidden_states: bool


@runtime_checkable
class VisionLanguageModel(Protocol):
    vision_model: Any
    language_model: Any

    def extract_feature(self, image_tensor: ImageTensor) -> Tensor: ...


class MaskedContentBlocks(Sequence[ContentBlock]):
    content_blocks: list[ContentBlock]
    loss_mask: list[bool]

    def __init__(self, content_blocks: ContentBlocks, loss_mask: Sequence[bool] | bool = True) -> None:
        self.content_blocks = list(content_blocks)
        if isinstance(loss_mask, bool):
            self.loss_mask = [bool(loss_mask)] * len(self.content_blocks)
        else:
            lm = list(bool(x) for x in loss_mask)
            if len(lm) != len(self.content_blocks):
                raise ValueError("loss_mask length must match content_blocks length")
            self.loss_mask = lm

    def __len__(self) -> int:
        return len(self.content_blocks)

    @overload
    def __getitem__(self, index: int) -> ContentBlock: ...

    @overload
    def __getitem__(self, index: slice) -> MaskedContentBlocks: ...

    def __getitem__(self, index: int | slice) -> ContentBlock | MaskedContentBlocks:
        if isinstance(index, int):
            return self.content_blocks[index]
        return MaskedContentBlocks(self.content_blocks[index], self.loss_mask[index])

    def __add__(self, other: MaskedContentBlocks) -> MaskedContentBlocks:
        return MaskedContentBlocks(self.content_blocks + other.content_blocks, self.loss_mask + other.loss_mask)


ContentType: TypeAlias = ContentBlock | ContentBlocks | MaskedContentBlocks
DeviceMap: TypeAlias = dict[str, int]


class ModelManager:
    model_name: str
    model_path: str
    cache_dir: str
    device_map: DeviceMap
    config: VLMConfig
    model: VisionLanguageModel
    tokenizer: Any
    torch_dtype: torch.dtype
    train_steps: int

    def __init__(
        self,
        model_name: str | None = None,
        model_path: str | None = None,
        cache_dir: str = ".cache",
        torch_dtype: torch.dtype | str | None = None,
        resume_from_checkpoint: str | bool = False,
        **kwargs: Any,
    ) -> None:
        self.cache_dir = str(cache_dir)

        if torch_dtype is None:
            resolved_dtype = torch.bfloat16
        elif isinstance(torch_dtype, torch.dtype):
            resolved_dtype = torch_dtype
        elif isinstance(torch_dtype, str):
            s = torch_dtype.strip().lower()
            if s in ("bf16", "bfloat16", "torch.bfloat16"):
                resolved_dtype = torch.bfloat16
            elif s in ("fp16", "float16", "half", "torch.float16"):
                resolved_dtype = torch.float16
            elif s in ("fp32", "float32", "torch.float32"):
                resolved_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported torch_dtype string: {torch_dtype}")
        else:
            raise TypeError(f"torch_dtype must be torch.dtype | str | None, got {type(torch_dtype)}")

        self.torch_dtype = resolved_dtype
        self.train_steps = 0

        self.load_model_metadata(model_name, model_path, resume_from_checkpoint, **kwargs)
        self.build_device_map()
        self.load_model()

    @staticmethod
    def find_last_checkpoint(model_name: str, resume_from_checkpoint: str | bool = True) -> str | None:
        if resume_from_checkpoint is False:
            return None
        checkpoint_dir = default_checkpoint_dir if resume_from_checkpoint is True else str(resume_from_checkpoint)
        available_checkpoints = glob.glob(f"{checkpoint_dir}/{model_name}/step_*")
        if not available_checkpoints:
            return None
        model_path: str | None = None
        current_step = 0
        for path in available_checkpoints:
            match_result = re.match(r".*/step_(\d+)$", path)
            if not match_result:
                continue
            step = int(match_result.group(1))
            if step > current_step:
                current_step = step
                model_path = path
        return model_path

    def load_model_metadata(
        self,
        model_name: str | None = None,
        model_path: str | None = None,
        resume_from_checkpoint: str | bool = True,
        **kwargs: Any,
    ) -> None:
        self.train_steps = 0

        if model_name is None:
            if model_path is None:
                raise ValueError("Either model_name or model_path must be provided.")
            model_name = os.path.basename(model_path)
        elif model_path is None:
            last_checkpoint = self.find_last_checkpoint(model_name, resume_from_checkpoint)
            if last_checkpoint:
                model_path = last_checkpoint
                train_step_match = re.search(r".*/step_(\d+)$", model_path)
                if train_step_match is not None:
                    self.train_steps = int(train_step_match.group(1))
            else:
                model_path = os.path.join(default_model_dir, model_name)

        self.model_name = str(model_name)
        self.model_path = str(model_path)

        transformers = importlib.import_module("transformers")
        AutoConfig = cast(Any, getattr(transformers, "AutoConfig"))

        cfg_any = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, cache_dir=self.cache_dir)
        self.config = cast(VLMConfig, cfg_any)

        kwargs.setdefault("return_dict_in_generate", True)
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("output_attentions", True)

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def build_device_map(self) -> None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available.")

        num_llm_layers = int(self.config.llm_config.num_hidden_layers)
        num_vision_layers = int(self.config.vision_config.num_hidden_layers)

        device_map: DeviceMap = {}

        if num_gpus == 1:
            device_map["vision_model.embeddings"] = 0
            for i in range(num_vision_layers):
                device_map[f"vision_model.encoder.layers.{i}"] = 0

            for i in range(num_llm_layers):
                device_map[f"language_model.model.layers.{i}"] = 0

            device_map["language_model.model.tok_embeddings"] = 0
            device_map["language_model.model.embed_tokens"] = 0
            device_map["language_model.model.rotary_emb"] = 0
            device_map["language_model.model.norm"] = 0
            device_map["language_model.lm_head"] = 0
            device_map["language_model.output"] = 0
            device_map["mlp1"] = 0

            self.device_map = device_map
            return

        gpus = list(range(num_gpus))
        gpu_load: list[int] = [0 for _ in range(num_gpus)]

        def bump(g: int, u: int = 1) -> None:
            gpu_load[g] += int(u)

        def pick_least_loaded(exclude: set[int] | None = None) -> int:
            ex = exclude or set()
            cand = [g for g in gpus if g not in ex]
            if not cand:
                return 0
            return min(cand, key=lambda gg: (gpu_load[gg], gg))

        def assign_layers_contiguous(prefix: str, n_layers: int) -> None:
            if n_layers <= 0:
                return
            base = n_layers // num_gpus
            rem = n_layers % num_gpus
            idx = 0
            for gi, g in enumerate(gpus):
                take = base + (1 if gi < rem else 0)
                for _ in range(take):
                    if idx >= n_layers:
                        break
                    device_map[f"{prefix}.{idx}"] = g
                    bump(g, 2)
                    idx += 1

        vision_embed_gpu = pick_least_loaded()
        device_map["vision_model.embeddings"] = vision_embed_gpu
        bump(vision_embed_gpu, 2)
        assign_layers_contiguous("vision_model.encoder.layers", num_vision_layers)

        embed_gpu = pick_least_loaded()

        if num_llm_layers > 0:
            assign_layers_contiguous("language_model.model.layers", num_llm_layers)
            device_map["language_model.model.layers.0"] = embed_gpu
            bump(embed_gpu, 2)
            last_layer_gpu = int(device_map.get(f"language_model.model.layers.{num_llm_layers - 1}", embed_gpu))
        else:
            last_layer_gpu = embed_gpu

        head_gpu = last_layer_gpu

        device_map["language_model.model.tok_embeddings"] = embed_gpu
        device_map["language_model.model.embed_tokens"] = embed_gpu
        device_map["language_model.model.rotary_emb"] = embed_gpu
        bump(embed_gpu, 3)

        device_map["language_model.model.norm"] = head_gpu
        device_map["language_model.lm_head"] = head_gpu
        bump(head_gpu, 2)

        out_gpu = pick_least_loaded()
        device_map["language_model.output"] = out_gpu
        bump(out_gpu, 1)

        mlp1_gpu = pick_least_loaded()
        device_map["mlp1"] = mlp1_gpu
        bump(mlp1_gpu, 1)

        self.device_map = device_map

    @staticmethod
    def _patch_loaded_internvl_vit_sources() -> list[str]:
        """
        Patch known InternVL remote source pattern that is incompatible with meta-device init:
            [x.item() for x in torch.linspace(...)]
        """
        patched_paths: list[str] = []
        replacement = "dpr = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device='cpu').tolist()"
        pattern = re.compile(
            r"dpr\s*=\s*\[x\.item\(\)\s*for\s*x\s*in\s*torch\.linspace\(\s*0\s*,\s*config\.drop_path_rate\s*,\s*config\.num_hidden_layers\s*\)\s*\]"
        )

        for module in list(sys.modules.values()):
            mod_file = getattr(module, "__file__", None)
            if not isinstance(mod_file, str):
                continue
            if os.path.basename(mod_file) != "modeling_intern_vit.py":
                continue

            try:
                with open(mod_file, "r", encoding="utf-8") as f:
                    src = f.read()
            except Exception:
                continue

            if replacement in src:
                continue

            new_src, n = pattern.subn(replacement, src, count=1)
            if n <= 0:
                legacy = "dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]"
                if legacy in src:
                    new_src = src.replace(legacy, replacement, 1)
                    n = 1

            if n <= 0:
                continue

            try:
                with open(mod_file, "w", encoding="utf-8") as f:
                    f.write(new_src)
                patched_paths.append(mod_file)
            except Exception:
                continue

        return patched_paths

    @staticmethod
    def _clear_remote_transformer_modules() -> None:
        names = [n for n in list(sys.modules.keys()) if n.startswith("transformers_modules.")]
        for n in names:
            sys.modules.pop(n, None)
        importlib.invalidate_caches()

    @staticmethod
    def _is_missing_all_tied_keys_error(exc: BaseException) -> bool:
        msg = str(exc)
        return ("all_tied_weights_keys" in msg) and ("InternVLChatModel" in msg)

    @staticmethod
    def _patch_loaded_internvl_chat_classes() -> list[str]:
        """
        Patch compatibility for newer transformers that access
        `model.all_tied_weights_keys`.
        """
        patched: list[str] = []
        for module in list(sys.modules.values()):
            mod_file = getattr(module, "__file__", None)
            if not isinstance(mod_file, str):
                continue
            if os.path.basename(mod_file) != "modeling_internvl_chat.py":
                continue

            cls = getattr(module, "InternVLChatModel", None)
            if cls is None:
                continue
            if hasattr(cls, "all_tied_weights_keys"):
                continue

            def _all_tied_weights_keys(self: Any) -> dict[str, str]:
                keys = getattr(self, "_tied_weights_keys", None)
                if keys is None:
                    return {}
                if isinstance(keys, dict):
                    return {str(k): str(v) for k, v in keys.items()}
                try:
                    return {str(k): str(k) for k in list(keys)}
                except Exception:
                    return {}

            setattr(cls, "all_tied_weights_keys", property(_all_tied_weights_keys))
            patched.append(mod_file)

        return patched

    @staticmethod
    @contextmanager
    def _force_cpu_linspace_context():
        """
        Guard against remote model code calling `torch.linspace(...).item()` under meta default device.
        """
        orig_linspace = torch.linspace

        def _safe_linspace(start: Any, end: Any, steps: Any, *args: Any, **kwargs: Any) -> Tensor:
            device = kwargs.get("device", None)
            if device is None or str(device) == "meta":
                kwargs["device"] = "cpu"
            return cast(Tensor, orig_linspace(start, end, steps, *args, **kwargs))

        cast(Any, torch).linspace = _safe_linspace
        try:
            yield
        finally:
            cast(Any, torch).linspace = orig_linspace

    def load_model(self) -> None:
        transformers = importlib.import_module("transformers")
        AutoModel = cast(Any, getattr(transformers, "AutoModel"))
        AutoTokenizer = cast(Any, getattr(transformers, "AutoTokenizer"))

        try:
            model_any = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                config=self.config,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                device_map=self.device_map,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            meta_item_err = "tensor.item() cannot be called on meta tensors"
            if meta_item_err not in msg:
                raise

            logger.warning(
                "Meta-device init failed for model '%s' with device_map. "
                "Falling back to CPU load + dispatch_model. Original error: %s",
                self.model_path,
                str(e),
            )

            model_any: Any | None = None

            patched_paths = self._patch_loaded_internvl_vit_sources()
            if patched_paths:
                logger.warning(
                    "Patched InternVL remote source files to avoid meta-tensor item(): %s",
                    patched_paths,
                )
                self._clear_remote_transformer_modules()
                try:
                    with self._force_cpu_linspace_context():
                        model_any = AutoModel.from_pretrained(
                            self.model_path,
                            torch_dtype=self.torch_dtype,
                            config=self.config,
                            trust_remote_code=True,
                            cache_dir=self.cache_dir,
                            device_map=self.device_map,
                        )
                except Exception as retry_e:
                    if self._is_missing_all_tied_keys_error(retry_e):
                        patched_chat = self._patch_loaded_internvl_chat_classes()
                        logger.warning(
                            "Patched InternVL chat class for all_tied_weights_keys compatibility: %s",
                            patched_chat,
                        )
                        model_any = None
                    else:
                        retry_msg = str(retry_e).lower()
                        if meta_item_err not in retry_msg:
                            raise
                        logger.warning(
                            "Retry after source patch still hit meta init. "
                            "Continue with CPU load + dispatch_model. Error: %s",
                            str(retry_e),
                        )

            if model_any is None:
                # Some HF/accelerate failure paths may leave default device as "meta".
                # Force CPU for fallback model construction so custom __init__ code
                # (e.g., torch.linspace(...).item()) does not run on meta tensors.
                if hasattr(torch, "set_default_device"):
                    try:
                        cast(Any, torch).set_default_device("cpu")
                    except Exception:
                        pass

                with torch.device("cpu"):
                    with self._force_cpu_linspace_context():
                        try:
                            model_any = AutoModel.from_pretrained(
                                self.model_path,
                                torch_dtype=self.torch_dtype,
                                config=self.config,
                                trust_remote_code=True,
                                cache_dir=self.cache_dir,
                                low_cpu_mem_usage=False,
                                device_map=None,
                            )
                        except Exception as cpu_e:
                            if not self._is_missing_all_tied_keys_error(cpu_e):
                                raise
                            patched_chat = self._patch_loaded_internvl_chat_classes()
                            logger.warning(
                                "Patched InternVL chat class for all_tied_weights_keys compatibility: %s",
                                patched_chat,
                            )
                            model_any = AutoModel.from_pretrained(
                                self.model_path,
                                torch_dtype=self.torch_dtype,
                                config=self.config,
                                trust_remote_code=True,
                                cache_dir=self.cache_dir,
                                low_cpu_mem_usage=False,
                                device_map=None,
                            )

                dispatched = False
                try:
                    accelerate = importlib.import_module("accelerate")
                    dispatch_model = cast(Any, getattr(accelerate, "dispatch_model", None))
                    if callable(dispatch_model):
                        model_any = dispatch_model(model_any, device_map=self.device_map)
                        dispatched = True
                except Exception as dispatch_err:
                    logger.warning("dispatch_model fallback failed: %s", str(dispatch_err))

                if not dispatched and torch.cuda.is_available():
                    # Last-resort fallback: keep training runnable on a single GPU.
                    target_gpu = 0
                    if isinstance(self.device_map, dict):
                        gpu_ids = [int(v) for v in self.device_map.values() if isinstance(v, int)]
                        if len(gpu_ids) > 0:
                            target_gpu = min(gpu_ids)
                    model_any = model_any.to(torch.device(f"cuda:{target_gpu}"))

        self.model = cast(VisionLanguageModel, model_any)

        tokenizer_any = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = tokenizer_any

    def vision_input_device(self) -> torch.device:
        vm = getattr(self.model, "vision_model", None)
        if vm is not None:
            emb = getattr(vm, "embeddings", None)
            if emb is not None:
                try:
                    return next(emb.parameters()).device
                except StopIteration:
                    pass
            try:
                return next(vm.parameters()).device
            except StopIteration:
                pass
        try:
            return next(cast(Any, self.model).parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def embed_image(self, image_tensor: ImageTensor) -> Tensor:
        features_sum = self.model.extract_feature(image_tensor).sum(dim=0, keepdim=True)
        return features_sum / math.sqrt(float(image_tensor.shape[0]))

    @overload
    def embed_text(self, text: str, output_ids: Literal[False] = False) -> Tensor: ...

    @overload
    def embed_text(self, text: str, output_ids: Literal[True]) -> tuple[Tensor, torch.IntTensor]: ...

    def embed_text(self, text: str, output_ids: bool = False) -> Tensor | tuple[Tensor, torch.IntTensor]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        token_ids_cpu = cast(torch.IntTensor, enc.input_ids[0].to(torch.int64))

        emb_layer = self.model.language_model.get_input_embeddings()
        emb_device = next(emb_layer.parameters()).device

        token_ids_dev = token_ids_cpu.to(emb_device, non_blocking=True)
        input_embeds = emb_layer(token_ids_dev).unsqueeze(0)

        if output_ids:
            return input_embeds, token_ids_cpu
        return input_embeds

    def run_with_content(self, content: ContentType = "", **kwargs: Any) -> Any:
        if isinstance(content, (str, torch.Tensor)):
            content_seq: list[ContentBlock] = [cast(ContentBlock, content)]
            loss_mask_seq = [True]
        elif isinstance(content, MaskedContentBlocks):
            content_seq = list(content.content_blocks)
            loss_mask_seq = list(content.loss_mask)
        else:
            content_seq = list(cast(Sequence[ContentBlock], content))
            loss_mask_seq = [True] * len(content_seq)

        input_embeds: list[Tensor] = []
        input_token_ids: list[int] = []

        ignore_id: Final[int] = -100

        emb_layer = self.model.language_model.get_input_embeddings()
        concat_device = next(emb_layer.parameters()).device

        for item, mask in zip(content_seq, loss_mask_seq):
            if isinstance(item, str):
                text_embedding, text_ids = self.embed_text(item, output_ids=True)

                if text_embedding.device != concat_device:
                    text_embedding = text_embedding.to(concat_device, non_blocking=True)

                input_embeds.append(text_embedding)

                ids_cpu = cast(torch.Tensor, text_ids)
                if mask:
                    input_token_ids.extend(ids_cpu.tolist())
                else:
                    input_token_ids.extend([ignore_id] * int(ids_cpu.numel()))
                continue

            img = item.to(self.vision_input_device(), self.torch_dtype)
            image_embedding = self.embed_image(img)
            if image_embedding.ndim != 3:
                raise RuntimeError("Image embedding should be 3D tensor")
            batch_size, num_image_tokens, _ = image_embedding.shape
            if int(batch_size) != 1:
                raise RuntimeError("Image embedding batch size should be 1")

            if image_embedding.device != concat_device:
                image_embedding = image_embedding.to(concat_device, non_blocking=True)

            input_embeds.append(image_embedding)
            input_token_ids.extend([ignore_id] * int(num_image_tokens))

        output_hidden_states = kwargs.pop("output_hidden_states", bool(self.config.output_hidden_states))
        output_attentions = kwargs.pop("output_attentions", bool(self.config.output_attentions))

        if len(input_embeds) == 0:
            raise RuntimeError("No input embeddings produced (empty content).")

        inputs_embeds = torch.concat(input_embeds, dim=-2)

        labels = torch.tensor(
            input_token_ids,
            dtype=torch.int64,
            device=inputs_embeds.device,
        ).unsqueeze(0)

        language_output = self.model.language_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            labels=labels,
            **kwargs,
        )

        return language_output

    def __call__(self, content: ContentType, **kwargs: Any) -> Any:
        return self.run_with_content(content, **kwargs)

    def freeze_vision_model(self) -> None:
        vm = getattr(self.model, "vision_model", None)
        if vm is None:
            return
        for param in cast(Any, vm).parameters():
            param.requires_grad = False

    def unfreeze_vision_model(self) -> None:
        vm = getattr(self.model, "vision_model", None)
        if vm is None:
            return
        for param in cast(Any, vm).parameters():
            param.requires_grad = True

    def freeze_language_model(self) -> None:
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            return
        for param in cast(Any, lm).parameters():
            param.requires_grad = False

    def unfreeze_language_model(self) -> None:
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            return
        for param in cast(Any, lm).parameters():
            param.requires_grad = True

    def freeze(self) -> None:
        self.freeze_vision_model()
        self.freeze_language_model()

    def unfreeze(self) -> None:
        self.unfreeze_vision_model()
        self.unfreeze_language_model()

    def eval(self) -> None:
        cast(Any, self.model).eval()

    def train(self) -> None:
        cast(Any, self.model).train()
