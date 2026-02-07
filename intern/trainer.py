from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol, Sequence, TypeAlias, TypedDict, cast

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .image_loader import ImageLoader
from .markers import image_pattern, img_end, img_start
from .model import (
    ContentBlock,
    ContentBlocks,
    ContentType,
    ImageTensor,
    MaskedContentBlocks,
    ModelManager,
    default_checkpoint_dir,
)

logger = logging.getLogger(__name__)

Tensor: TypeAlias = torch.Tensor


class CausalLMOutputLike(Protocol):
    logits: Tensor
    loss: Tensor
    attentions: Sequence[Tensor] | None
    past_key_values: Any


class DistillationRecord(TypedDict):
    first_attention_Atv_loss: float
    intermediate_attention_loss: float
    penult_attention_Atv_loss: float
    total_distill_loss: float
    total_distill_loss_static: float


class TeacherStepRecord(TypedDict):
    teacher_index: int
    teacher_role: str
    dataset_index: int
    teacher_weight: float
    ground_truth_loss: float
    distillation_record: DistillationRecord
    loss: float
    loss_static: float
    loss_weighted: float
    loss_static_weighted: float
    hard_loss: float
    hard_loss_static: float
    hard_loss_weighted: float
    hard_loss_static_weighted: float
    soft_loss: float
    soft_loss_static: float
    soft_loss_weighted: float
    soft_loss_static_weighted: float
    used_weights: dict[str, float]
    static_weights: dict[str, float]
    dynamic_enabled: bool
    term_losses_static: dict[str, float]
    term_losses_static_weighted: dict[str, float]
    term_losses_raw: dict[str, float]
    hard_loss_raw: float
    soft_terms_raw: dict[str, float]


class TrainStepRecord(TypedDict):
    dataset_index: int
    ground_truth_loss: float
    loss: float
    loss_static: float
    hard_loss: float
    hard_loss_static: float
    soft_loss: float
    soft_loss_static: float
    per_teacher: list[TeacherStepRecord]
    dyn_weights: dict[str, float]
    step: int
    optim_step: int


DistillationHook: TypeAlias = Callable[[TrainStepRecord], Any]


class DistillationTeacher(nn.Module):
    loaded_teachers: dict[str, ModelManager] = {}

    model_manager: ModelManager
    intermediate_layer_start: int
    intermediate_layer_end: int

    def __init__(
        self,
        model_manager: ModelManager,
        *,
        intermediate_layer_start: int = 0,
        intermediate_layer_end: int = -2,
    ) -> None:
        super().__init__()
        self.model_manager = model_manager
        self.intermediate_layer_start = int(intermediate_layer_start)
        self.intermediate_layer_end = int(intermediate_layer_end)

    @classmethod
    def load_teacher(
        cls,
        teacher_model: ModelManager | str,
        *,
        freeze: bool = True,
        torch_dtype: str | None = None,
        device_map: Any = "auto",
    ) -> ModelManager:
        if isinstance(teacher_model, ModelManager):
            manager = teacher_model
        else:
            key = str(teacher_model)
            if key in cls.loaded_teachers:
                manager = cls.loaded_teachers[key]
            else:
                manager = ModelManager(
                    model_name=teacher_model,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
                cls.loaded_teachers[key] = manager

        if freeze:
            manager.freeze()
            manager.eval()
        return manager

    @staticmethod
    def _mse_sum_last_mean(student_map: Tensor, teacher_map: Tensor) -> Tensor:
        mse = (student_map - teacher_map).pow(2)
        return mse.mean(dim=-1).mean()

    @staticmethod
    def _atv_mask_from_is_image(is_image_mask: Tensor) -> Tensor:
        m = is_image_mask.bool()
        return (~m).unsqueeze(-1) & m.unsqueeze(1)

    @classmethod
    def _atv_loss_masked_fill_sum_last_mean(
        cls,
        student_map: Tensor,
        teacher_map: Tensor,
        atv_mask: Tensor,
    ) -> Tensor:
        mse = (student_map - teacher_map).pow(2)

        mask_f = atv_mask.to(dtype=mse.dtype)
        mse_masked = mse * mask_f
        
        denom = mask_f.sum(dim=(-2, -1)).clamp_min(1.0)
        numer = mse_masked.sum(dim=(-2, -1))
        return (numer / denom).mean()

    @staticmethod
    def _resolve_index(spec: int, total_layers: int) -> int:
        idx = spec if spec >= 0 else total_layers + spec
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Layer index out of range: spec={spec}, resolved={idx}, total_layers={total_layers}")
        return idx

    @classmethod
    def _resolve_indices_index_only(cls, start_spec: int, end_spec: int, total_layers: int) -> list[int]:
        if total_layers <= 0:
            return []
        s_idx = cls._resolve_index(int(start_spec), total_layers)
        e_idx = cls._resolve_index(int(end_spec), total_layers)
        if s_idx > e_idx:
            raise ValueError(f"Invalid index range: start_idx={s_idx} > end_idx={e_idx}")
        return list(range(s_idx, e_idx + 1))

    @classmethod
    def _build_compo_layer_mapping(
        cls,
        student_indices: list[int],
        teacher_indices: list[int],
    ) -> dict[int, list[int]]:
        k = len(student_indices)
        m = len(teacher_indices)
        if k == 0 or m == 0:
            return {}
        group_size = max(1, m - k + 1)
        mapping: dict[int, list[int]] = {}
        for i, s_layer in enumerate(student_indices):
            t_start_idx = min(i, m - group_size)
            mapping[s_layer] = teacher_indices[t_start_idx : t_start_idx + group_size]
        return mapping

    @staticmethod
    def _zero_like_from_map(map_tensor: Tensor, *, base_device: torch.device) -> Tensor:
        z = map_tensor.sum() * 0.0
        return z.to(base_device)

    def _compute_intermediate_attention_loss_full(
        self,
        student_attentions: Sequence[Tensor],
        teacher_attentions: Sequence[Tensor],
        *,
        base_device: torch.device,
    ) -> Tensor:
        n_student = len(student_attentions)
        n_teacher = len(teacher_attentions)
        if n_student == 0 or n_teacher == 0:
            return torch.tensor(0.0, device=base_device)

        student_indices = self._resolve_indices_index_only(self.intermediate_layer_start, self.intermediate_layer_end, n_student)
        teacher_indices = self._resolve_indices_index_only(self.intermediate_layer_start, self.intermediate_layer_end, n_teacher)
        mapping = self._build_compo_layer_mapping(student_indices, teacher_indices)
        if not mapping:
            return self._zero_like_from_map(student_attentions[0].mean(dim=1), base_device=base_device)

        total = torch.tensor(0.0, device=base_device)
        count = 0

        for s_layer, t_layers in mapping.items():
            if not t_layers:
                continue

            s_att = student_attentions[s_layer]
            s_dev = s_att.device
            s_map = s_att.mean(dim=1)

            t_sum: Tensor | None = None
            for t_layer in t_layers:
                t_map = teacher_attentions[t_layer].detach().mean(dim=1)
                if t_map.device != s_dev:
                    t_map = t_map.to(s_dev)
                t_sum = t_map if t_sum is None else (t_sum + t_map)

            if t_sum is None:
                continue

            t_target = t_sum / float(len(t_layers))
            layer_loss = self._mse_sum_last_mean(s_map, t_target)
            total = total + layer_loss.to(base_device)
            count += 1

        if count == 0:
            return self._zero_like_from_map(student_attentions[0].mean(dim=1), base_device=base_device)
        return total / float(count)

    def get_distillation_loss(
        self,
        *,
        role: str,
        student_output: CausalLMOutputLike,
        teacher_output: CausalLMOutputLike,
        is_image_mask: Tensor | None,
    ) -> tuple[dict[str, Tensor], DistillationRecord]:
        base_device = student_output.logits.device

        first_attention_Atv_loss = torch.tensor(0.0, device=base_device)
        intermediate_attention_loss = torch.tensor(0.0, device=base_device)
        penult_attention_Atv_loss = torch.tensor(0.0, device=base_device)

        student_attentions = student_output.attentions
        teacher_attentions = teacher_output.attentions

        loss_terms: dict[str, Tensor] = {}

        if not isinstance(student_attentions, (list, tuple)) or not isinstance(teacher_attentions, (list, tuple)):
            record: DistillationRecord = {
                "first_attention_Atv_loss": 0.0,
                "intermediate_attention_loss": 0.0,
                "penult_attention_Atv_loss": 0.0,
                "total_distill_loss": 0.0,
                "total_distill_loss_static": 0.0,
            }
            return loss_terms, record

        if role == "perception":
            if len(student_attentions) == 0 or len(teacher_attentions) == 0:
                record = {
                    "first_attention_Atv_loss": 0.0,
                    "intermediate_attention_loss": 0.0,
                    "penult_attention_Atv_loss": 0.0,
                    "total_distill_loss": 0.0,
                    "total_distill_loss_static": 0.0,
                }
                return loss_terms, record

            s_first_att = student_attentions[0]
            s_dev = s_first_att.device
            s_first = s_first_att.mean(dim=1)

            t_first = teacher_attentions[0].detach().mean(dim=1)
            if t_first.device != s_dev:
                t_first = t_first.to(s_dev)

            if is_image_mask is None:
                first_attention_Atv_loss = self._zero_like_from_map(s_first, base_device=base_device)
            else:
                atv_mask = self._atv_mask_from_is_image(is_image_mask.to(s_dev))
                if bool(atv_mask.any().item()):
                    first_attention_Atv_loss = self._atv_loss_masked_fill_sum_last_mean(s_first, t_first, atv_mask).to(base_device)
                else:
                    first_attention_Atv_loss = self._zero_like_from_map(s_first, base_device=base_device)

            loss_terms["first_att_Atv"] = first_attention_Atv_loss

        elif role == "reasoning":
            if len(student_attentions) == 0 or len(teacher_attentions) == 0:
                record = {
                    "first_attention_Atv_loss": 0.0,
                    "intermediate_attention_loss": 0.0,
                    "penult_attention_Atv_loss": 0.0,
                    "total_distill_loss": 0.0,
                    "total_distill_loss_static": 0.0,
                }
                return loss_terms, record

            intermediate_attention_loss = self._compute_intermediate_attention_loss_full(
                student_attentions,
                teacher_attentions,
                base_device=base_device,
            )
            loss_terms["intermediate"] = intermediate_attention_loss

        elif role == "planning":
            if len(student_attentions) < 2 or len(teacher_attentions) < 2:
                record = {
                    "first_attention_Atv_loss": 0.0,
                    "intermediate_attention_loss": 0.0,
                    "penult_attention_Atv_loss": 0.0,
                    "total_distill_loss": 0.0,
                    "total_distill_loss_static": 0.0,
                }
                return loss_terms, record

            s_pen_att = student_attentions[-2]
            s_dev = s_pen_att.device
            s_penult = s_pen_att.mean(dim=1)

            t_penult = teacher_attentions[-2].detach().mean(dim=1)
            if t_penult.device != s_dev:
                t_penult = t_penult.to(s_dev)

            if is_image_mask is None:
                penult_attention_Atv_loss = self._zero_like_from_map(s_penult, base_device=base_device)
            else:
                atv_mask = self._atv_mask_from_is_image(is_image_mask.to(s_dev))
                if bool(atv_mask.any().item()):
                    penult_attention_Atv_loss = self._atv_loss_masked_fill_sum_last_mean(s_penult, t_penult, atv_mask).to(base_device)
                else:
                    penult_attention_Atv_loss = self._zero_like_from_map(s_penult, base_device=base_device)

            loss_terms["penult_att_Atv"] = penult_attention_Atv_loss

        raw_total = torch.tensor(0.0, device=base_device)
        for v in loss_terms.values():
            raw_total = raw_total + v

        record_out: DistillationRecord = {
            "first_attention_Atv_loss": float(first_attention_Atv_loss.detach().item()),
            "intermediate_attention_loss": float(intermediate_attention_loss.detach().item()),
            "penult_attention_Atv_loss": float(penult_attention_Atv_loss.detach().item()),
            "total_distill_loss": float(raw_total.detach().item()),
            "total_distill_loss_static": float(raw_total.detach().item()),
        }
        return loss_terms, record_out


def _softplus_inv(y: Tensor) -> Tensor:
    return torch.log(torch.expm1(y).clamp_min(1e-20))


@dataclass
class _AccumStats:
    loss_sums: dict[str, Tensor]
    reweight_norm_sums: dict[str, Tensor]
    count: int


class _OnlineReweightGroup:
    teacher_role: str
    active_losses: list[str]
    static_weight_map: dict[str, float]
    device: torch.device

    use_dynamic_loss_weights: bool
    dynamic_weight_lr: float
    reweight_alpha: float
    dynamic_weight_eps: float
    dynamic_weight_min: float
    dynamic_weight_max: float
    min_dynamic_weight_ce: float

    loss_weight_params: nn.ParameterDict | None
    weight_optimizer: torch.optim.Optimizer | None
    loss_init_values: dict[str, float]
    _accum_stats: _AccumStats | None

    def __init__(
        self,
        *,
        teacher_role: str,
        active_losses: list[str],
        static_weight_map: dict[str, float],
        device: torch.device,
        use_dynamic_loss_weights: bool,
        dynamic_weight_lr: float,
        reweight_alpha: float,
        dynamic_weight_eps: float,
        dynamic_weight_min: float,
        dynamic_weight_max: float,
        min_dynamic_weight_ce: float,
    ) -> None:
        self.teacher_role = str(teacher_role)
        self.active_losses = list(active_losses)
        self.static_weight_map = {k: float(v) for k, v in static_weight_map.items()}
        self.device = device

        self.use_dynamic_loss_weights = bool(use_dynamic_loss_weights)
        self.dynamic_weight_lr = float(dynamic_weight_lr)
        self.reweight_alpha = float(reweight_alpha)
        self.dynamic_weight_eps = float(dynamic_weight_eps)
        self.dynamic_weight_min = float(dynamic_weight_min)
        self.dynamic_weight_max = float(dynamic_weight_max)
        self.min_dynamic_weight_ce = float(min_dynamic_weight_ce)

        self.loss_weight_params = None
        self.weight_optimizer = None
        self.loss_init_values = {}
        self._accum_stats = None

        if self.use_dynamic_loss_weights and len(self.active_losses) >= 2:
            init_params: dict[str, nn.Parameter] = {}
            for n in self.active_losses:
                w0 = float(self.static_weight_map.get(n, 1.0))
                w0 = max(self.dynamic_weight_min, min(self.dynamic_weight_max, w0))
                wi = torch.tensor(w0, dtype=torch.float32, device=self.device)
                ai = _softplus_inv(wi)
                init_params[n] = nn.Parameter(ai)
            self.loss_weight_params = nn.ParameterDict(init_params)
            self.weight_optimizer = torch.optim.AdamW(
                self.loss_weight_params.parameters(),
                lr=self.dynamic_weight_lr,
                betas=(0.9, 0.99),
                weight_decay=0.0,
            )

    def current_loss_weights(self) -> dict[str, Tensor]:
        weights: dict[str, Tensor] = {}
        if not (self.use_dynamic_loss_weights and self.loss_weight_params is not None):
            for n in self.active_losses:
                weights[n] = torch.tensor(
                    float(self.static_weight_map.get(n, 1.0)),
                    dtype=torch.float32,
                    device=self.device,
                )
            return weights

        for n in self.active_losses:
            ai = self.loss_weight_params[n]
            wi = F.softplus(ai) + float(self.dynamic_weight_eps)
            wi = torch.clamp(wi, min=self.dynamic_weight_min, max=self.dynamic_weight_max)
            weights[n] = wi

        if "ce" in weights:
            weights["ce"] = torch.clamp(weights["ce"], min=self.min_dynamic_weight_ce)

        return weights

    @staticmethod
    def _dot_grads(ga: list[Tensor | None], gb: list[Tensor | None]) -> float:
        total: float = 0.0
        for a, b in zip(ga, gb):
            if a is None or b is None:
                continue
            total += float((a.detach().float() * b.detach().float()).sum().cpu().item())
        return total

    @staticmethod
    def _norm_sq_grads(g: list[Tensor | None]) -> float:
        total: float = 0.0
        for t in g:
            if t is None:
                continue
            total += float((t.detach().float() * t.detach().float()).sum().cpu().item())
        return total

    def accum_add_stats(self, losses: dict[str, Tensor], base_norms: dict[str, Tensor]) -> None:
        if not (self.use_dynamic_loss_weights and self.loss_weight_params is not None):
            return
        if len(self.active_losses) <= 1:
            return

        if self._accum_stats is None:
            self._accum_stats = _AccumStats(loss_sums={}, reweight_norm_sums={}, count=0)

        for n in self.active_losses:
            if n not in losses or n not in base_norms:
                continue
            prev_l = self._accum_stats.loss_sums.get(n)
            prev_g = self._accum_stats.reweight_norm_sums.get(n)

            l_val = losses[n].detach()
            g_val = base_norms[n].detach()

            self._accum_stats.loss_sums[n] = l_val if prev_l is None else (prev_l + l_val)
            self._accum_stats.reweight_norm_sums[n] = g_val if prev_g is None else (prev_g + g_val)

        self._accum_stats.count += 1

    def accum_pop_mean_stats(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if self._accum_stats is None or self._accum_stats.count <= 0:
            self._accum_stats = None
            return {}, {}
        c = float(self._accum_stats.count)
        mean_losses: dict[str, Tensor] = {}
        mean_norms: dict[str, Tensor] = {}
        for n, s in self._accum_stats.loss_sums.items():
            mean_losses[n] = s / c
        for n, s in self._accum_stats.reweight_norm_sums.items():
            mean_norms[n] = s / c
        self._accum_stats = None
        return mean_losses, mean_norms

    def update_dynamic_weights_online_reweighting(
        self,
        mean_losses: dict[str, Tensor],
        mean_base_norms: dict[str, Tensor],
    ) -> dict[str, float]:
        if not (self.use_dynamic_loss_weights and self.loss_weight_params is not None and self.weight_optimizer is not None):
            return {}
        if len(self.active_losses) <= 1:
            return {}

        avail = [n for n in self.active_losses if (n in mean_losses and n in mean_base_norms)]
        if len(avail) <= 1:
            return {}

        for n in avail:
            if n not in self.loss_init_values:
                self.loss_init_values[n] = float(mean_losses[n].detach().item())

        w_now = self.current_loss_weights()

        G: dict[str, Tensor] = {}
        for n in avail:
            G[n] = w_now[n].to(mean_base_norms[n].device) * mean_base_norms[n]

        G_sum = torch.stack([G[n] for n in avail]).sum()
        if float(G_sum.detach().item()) <= 0.0:
            return {}

        ratios: list[float] = []
        for n in avail:
            ratios.append(max(0.0, float(self.static_weight_map.get(n, 0.0))))
        s_ratio = sum(ratios)
        if s_ratio <= 0.0:
            ratios = [1.0 for _ in avail]
            s_ratio = float(len(avail))
        ratios = [r / s_ratio for r in ratios]

        r_list: list[Tensor] = []
        for n in avail:
            Li0 = max(1e-12, float(self.loss_init_values.get(n, 1.0)))
            r_list.append((mean_losses[n] / Li0).detach())
        r_bar = torch.stack(r_list).mean()

        targets: dict[str, Tensor] = {}
        for n, p_i, r_i in zip(avail, ratios, r_list):
            if self.reweight_alpha == 0.0:
                targets[n] = torch.tensor(float(p_i), device=G_sum.device) * G_sum
            else:
                targets[n] = torch.tensor(float(p_i), device=G_sum.device) * G_sum * ((r_i / r_bar).pow(self.reweight_alpha))

        obj = torch.tensor(0.0, device=G_sum.device)
        for n in avail:
            obj = obj + (G[n] - targets[n]).abs()

        self.weight_optimizer.zero_grad()
        obj.backward()
        self.weight_optimizer.step()

        with torch.no_grad():
            w_new = self.current_loss_weights()
            subset_sum = torch.stack([w_new[n] for n in avail]).sum().clamp_min(1e-20)

            target_subset_sum = sum(max(0.0, float(self.static_weight_map.get(n, 0.0))) for n in avail)
            if target_subset_sum <= 0.0:
                target_subset_sum = float(len(avail))
            scale = float(target_subset_sum) / float(subset_sum.detach().item())

            for n in avail:
                w_new[n] = w_new[n] * scale
                w_new[n] = torch.clamp(w_new[n], min=self.dynamic_weight_min, max=self.dynamic_weight_max)

            for n in avail:
                if n == "ce":
                    w_new[n] = torch.clamp(w_new[n], min=self.min_dynamic_weight_ce)

            for n in avail:
                wi = w_new[n].detach().to(self.loss_weight_params[n].device)
                ai = _softplus_inv(wi)
                self.loss_weight_params[n].data.copy_(ai)

        return {n: float(w_new[n].detach().item()) for n in avail}


class DistillationTrainer:
    teacher_managers: list[ModelManager]
    teachers: list[DistillationTeacher]
    student_manager: ModelManager

    teacher_weights_by_dataset: list[list[float]]
    teacher_weights_by_dataset_norm: list[list[float]]
    teacher_roles: list[str]

    grad_accum_steps: int
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None
    optimizer_step: int

    reweight_group: _OnlineReweightGroup
    reweight_groups: list[_OnlineReweightGroup]

    image_loader: ImageLoader

    use_lr_scheduler: bool
    lr_scheduler_type: str
    total_optim_steps: int
    warmup_optim_steps: int
    lr_min_ratio: float

    static_ce_weight: float
    teacher_soft_loss_weights: list[dict[str, float]]

    def __init__(
        self,
        *,
        teacher_models: Sequence[ModelManager | str],
        student_model: ModelManager | str,
        teacher_weights_by_dataset: Sequence[Sequence[float]] | None = None,
        static_ce_weight: float = 1.0,
        teacher_soft_loss_weights: Sequence[dict[str, float]] | None = None,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        grad_accum_steps: int = 4,
        use_dynamic_loss_weights: bool = True,
        use_agp: bool | None = None,
        dynamic_weight_lr: float = 3e-4,
        reweight_alpha: float = 0.0,
        dynamic_weight_eps: float = 1e-8,
        dynamic_weight_min: float = 1e-6,
        dynamic_weight_max: float = 1e6,
        min_dynamic_weight_ce: float = 0.6,
        intermediate_layer_start: int = 0,
        intermediate_layer_end: int = -2,
        use_lr_scheduler: bool = True,
        lr_scheduler_type: str = "cosine",
        total_optim_steps: int = 0,
        warmup_optim_steps: int = 0,
        lr_min_ratio: float = 0.01,
        torch_dtype: str | None = None,
        device_map: Any = "auto",
    ) -> None:
        if len(teacher_models) <= 0:
            raise ValueError("teacher_models is empty.")

        self.grad_accum_steps = int(grad_accum_steps)

        self.use_lr_scheduler = bool(use_lr_scheduler)
        self.lr_scheduler_type = str(lr_scheduler_type)
        self.total_optim_steps = int(total_optim_steps)
        self.warmup_optim_steps = int(warmup_optim_steps)
        self.lr_min_ratio = float(lr_min_ratio)

        self.optimizer_step = 0
        self.lr_scheduler = None

        if len(teacher_models) == 3:
            self.teacher_roles = ["perception", "reasoning", "planning"]
        else:
            self.teacher_roles = [f"teacher{i}" for i in range(len(teacher_models))]

        self.teacher_managers = []
        for tm in teacher_models:
            manager = DistillationTeacher.load_teacher(
                tm,
                freeze=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            self.teacher_managers.append(manager)

        num_teachers = len(self.teacher_managers)
        
        if use_agp is None:
            self.use_agp = (num_teachers == 3)
        else:
            self.use_agp = bool(use_agp)

        if isinstance(student_model, ModelManager):
            self.student_manager = student_model
        else:
            self.student_manager = ModelManager(
                model_name=student_model,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        self.student_manager.train()

        if not hasattr(self.student_manager, "train_steps"):
            self.student_manager.train_steps = 0

        try:
            student_device = next(self.student_manager.model.parameters()).device
        except StopIteration:
            student_device = torch.device("cpu")

        force_img_size = int(getattr(self.student_manager.config, "force_image_size", 448))
        self.image_loader = ImageLoader(chunk_size=force_img_size)

        self.teacher_weights_by_dataset = self._parse_teacher_weights_by_dataset(
            teacher_weights_by_dataset, num_teachers=num_teachers
        )
        self.teacher_weights_by_dataset_norm = self._normalize_teacher_weights(self.teacher_weights_by_dataset)

        self.teachers = []
        for _t_idx in range(num_teachers):
            self.teachers.append(
                DistillationTeacher(
                    model_manager=self.teacher_managers[_t_idx],
                    intermediate_layer_start=int(intermediate_layer_start),
                    intermediate_layer_end=int(intermediate_layer_end),
                )
            )

        self.static_ce_weight = float(static_ce_weight)

        if teacher_soft_loss_weights is None:
            tmp: list[dict[str, float]] = []
            for t_idx in range(num_teachers):
                if t_idx == 0:
                    tmp.append({"first_att_Atv": 1.0})
                elif t_idx == 1:
                    tmp.append({"intermediate": 1.0})
                elif t_idx == 2:
                    tmp.append({"penult_att_Atv": 1.0})
                else:
                    tmp.append({"intermediate": 1.0})
            teacher_soft_loss_weights = tmp

        if len(teacher_soft_loss_weights) != num_teachers:
            raise ValueError(
                f"teacher_soft_loss_weights length {len(teacher_soft_loss_weights)} must match num_teachers {num_teachers}."
            )

        normalized_soft: list[dict[str, float]] = []
        for d in teacher_soft_loss_weights:
            out: dict[str, float] = {}
            for k, v in d.items():
                if k in ("first_att_Atv", "intermediate", "penult_att_Atv"):
                    out[k] = float(v)
            normalized_soft.append(out)
        self.teacher_soft_loss_weights = normalized_soft

        static_map_global: dict[str, float] = {
            "ce": float(self.static_ce_weight),
            "p:first_att_Atv": 0.0,
            "r:intermediate": 0.0,
            "d:penult_att_Atv": 0.0,
        }

        if num_teachers >= 1:
            static_map_global["p:first_att_Atv"] = float(self.teacher_soft_loss_weights[0].get("first_att_Atv", 0.0))
        if num_teachers >= 2:
            static_map_global["r:intermediate"] = float(self.teacher_soft_loss_weights[1].get("intermediate", 0.0))
        if num_teachers >= 3:
            static_map_global["d:penult_att_Atv"] = float(self.teacher_soft_loss_weights[2].get("penult_att_Atv", 0.0))

        active_losses_global: list[str] = ["ce"]
        for k, v in static_map_global.items():
            if k == "ce":
                continue
            if float(v) != 0.0:
                active_losses_global.append(k)

        params = self._get_trainable_params(self.student_manager.model)
        if len(params) == 0:
            raise ValueError("Student model has no trainable parameters (requires_grad=True).")

        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=float(weight_decay),
            betas=(0.9, 0.999),
        )

        self.reweight_group = _OnlineReweightGroup(
            teacher_role="global",
            active_losses=active_losses_global,
            static_weight_map=static_map_global,
            device=student_device,
            use_dynamic_loss_weights=bool(use_dynamic_loss_weights),
            dynamic_weight_lr=float(dynamic_weight_lr),
            reweight_alpha=float(reweight_alpha),
            dynamic_weight_eps=float(dynamic_weight_eps),
            dynamic_weight_min=float(dynamic_weight_min),
            dynamic_weight_max=float(dynamic_weight_max),
            min_dynamic_weight_ce=float(min_dynamic_weight_ce),
        )

        self.reweight_groups = [self.reweight_group]

        if self.use_lr_scheduler:
            self.lr_scheduler = self._build_lr_scheduler()
        else:
            self.lr_scheduler = None

        logger.info(
            f"Initialized DistillationTrainer: num_teachers={num_teachers}, grad_accum_steps={self.grad_accum_steps}, use_lr_scheduler={self.use_lr_scheduler}"
        )

    @staticmethod
    def _parse_teacher_weights_by_dataset(
        teacher_weights_by_dataset: Sequence[Sequence[float]] | None,
        *,
        num_teachers: int,
    ) -> list[list[float]]:
        if teacher_weights_by_dataset is None:
            if num_teachers == 3:
                return [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            if num_teachers <= 0:
                raise ValueError("num_teachers must be > 0")
            u = 1.0 / float(num_teachers)
            return [[u for _ in range(num_teachers)]]

        rows: list[list[float]] = []
        for row in teacher_weights_by_dataset:
            r = [float(x) for x in row]
            if len(r) == 0:
                continue
            rows.append(r)

        if len(rows) == 0:
            if num_teachers == 3:
                return [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            u = 1.0 / float(num_teachers)
            return [[u for _ in range(num_teachers)]]

        for i, r in enumerate(rows):
            if len(r) != num_teachers:
                raise ValueError(f"teacher_weights_by_dataset row {i} length {len(r)} != num_teachers {num_teachers}")

        if len(rows) == 1 and num_teachers == 3:
            return [list(rows[0]), list(rows[0]), list(rows[0])]

        return rows

    @staticmethod
    def _normalize_teacher_weights(weights: list[list[float]]) -> list[list[float]]:
        normed: list[list[float]] = []
        for row in weights:
            s = sum(float(max(0.0, w)) for w in row)
            if s <= 0:
                normed.append([0.0 for _ in row])
            else:
                normed.append([float(max(0.0, w)) / s for w in row])
        return normed

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        base_lr = float(self.optimizer.param_groups[0]["lr"])
        total_steps = int(self.total_optim_steps)
        warmup_steps = int(self.warmup_optim_steps)
        min_ratio = float(self.lr_min_ratio)

        def lr_lambda(step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            remain = max(1, total_steps - warmup_steps)
            prog = float(step - warmup_steps) / float(remain)
            prog = max(0.0, min(1.0, prog))
            if self.lr_scheduler_type.lower() == "linear":
                return max(min_ratio, 1.0 - prog)
            cos_val = 0.5 * (1.0 + math.cos(math.pi * prog))
            return max(min_ratio, cos_val)

        logger.info(
            f"LR scheduler: type={self.lr_scheduler_type}, base_lr={base_lr:.2e}, total_steps={total_steps}, warmup_steps={warmup_steps}, min_ratio={min_ratio}"
        )
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def resolve_content_list(self, content_list: ContentBlocks) -> ContentBlocks:
        resolved_content: list[ContentBlock] = []
        for item in content_list:
            if isinstance(item, str):
                resolved_content.extend(self.resolve_content_text(item))
            else:
                resolved_content.append(item)
        return resolved_content

    def resolve_masked_content_list(self, masked_content_list: MaskedContentBlocks) -> MaskedContentBlocks:
        resolved_content: list[ContentBlock] = []
        resolved_loss_mask: list[bool] = []

        content_blocks = list(masked_content_list.content_blocks)
        loss_mask = list(masked_content_list.loss_mask)
        if len(content_blocks) != len(loss_mask):
            raise ValueError("MaskedContentBlocks.content_blocks and loss_mask must have same length.")

        for item, mask in zip(content_blocks, loss_mask):
            if isinstance(item, str):
                resolved_text_list = self.resolve_content_text(item)
                resolved_content.extend(resolved_text_list)
                resolved_loss_mask.extend([mask] * len(resolved_text_list))
            else:
                resolved_content.append(item)
                resolved_loss_mask.append(mask)

        return MaskedContentBlocks(content_blocks=resolved_content, loss_mask=resolved_loss_mask)

    def resolve_content_text(self, content: str) -> ContentBlocks:
        resolved_content: list[ContentBlock] = []
        image_path_tag_pattern = r"<image_path>(.*?)</image_path>"
        if re.search(image_path_tag_pattern, content):
            image_path_list = re.findall(image_path_tag_pattern, content)
            content_split = re.split(image_path_tag_pattern, content)
        else:
            image_path_list = re.findall(image_pattern, content)
            content_split = re.split(image_pattern, content)

        for i in range(len(image_path_list)):
            resolved_content.append(content_split[i])
            resolved_content.append(img_start)
            resolved_content.append(cast(ImageTensor, self.image_loader.load_image_tensor(image_path_list[i])))
            resolved_content.append(img_end)

        resolved_content.append(content_split[-1])
        return resolved_content

    def resolve_content(self, content: str | ContentBlocks | MaskedContentBlocks) -> ContentType:
        if isinstance(content, str):
            return self.resolve_content_text(content)
        if isinstance(content, MaskedContentBlocks):
            return self.resolve_masked_content_list(content)
        if isinstance(content, list):
            return self.resolve_content_list(content)
        raise TypeError(f"Unsupported content type: {type(content)}")

    def _forward_student_with_mask(self, content: ContentType) -> tuple[CausalLMOutputLike, Tensor | None]:
        if isinstance(content, (str, torch.Tensor)):
            content_seq: list[ContentBlock] = [cast(ContentBlock, content)]
            loss_mask_seq = [True]
        elif isinstance(content, MaskedContentBlocks):
            loss_mask_seq = list(content.loss_mask)
            content_seq = list(content.content_blocks)
        else:
            content_seq = list(cast(Sequence[ContentBlock], content))
            loss_mask_seq = [True] * len(content_seq)

        input_embeds: list[Tensor] = []
        input_token_ids: list[int] = []
        is_image_mask_list: list[bool] = []

        ignore_id = -100

        emb_layer = self.student_manager.model.language_model.get_input_embeddings()
        concat_device = next(emb_layer.parameters()).device

        for item, keep_loss in zip(content_seq, loss_mask_seq):
            if isinstance(item, str):
                token_embeds, token_ids = self.student_manager.embed_text(item, output_ids=True)

                if token_embeds.device != concat_device:
                    token_embeds = token_embeds.to(concat_device, non_blocking=True)

                token_ids_list = cast(torch.Tensor, token_ids).tolist()

                input_embeds.append(token_embeds)
                if keep_loss:
                    input_token_ids.extend(token_ids_list)
                else:
                    input_token_ids.extend([ignore_id] * len(token_ids_list))
                is_image_mask_list.extend([False] * len(token_ids_list))
            else:
                img = item.to(self.student_manager.vision_input_device(), self.student_manager.torch_dtype)
                image_embedding = self.student_manager.embed_image(img)
                if image_embedding.ndim != 3:
                    raise RuntimeError("Image embedding should be a 3D tensor")
                batch_size, num_image_tokens, _ = image_embedding.shape
                if batch_size != 1:
                    raise RuntimeError("Image embedding batch size should be 1")

                if image_embedding.device != concat_device:
                    image_embedding = image_embedding.to(concat_device, non_blocking=True)

                input_embeds.append(image_embedding)
                input_token_ids.extend([ignore_id] * int(num_image_tokens))
                is_image_mask_list.extend([True] * int(num_image_tokens))

        if len(input_embeds) == 0:
            raise RuntimeError("No input embeddings produced for this sample.")

        inputs_embeds = torch.concat(input_embeds, dim=-2)
        labels = torch.tensor(input_token_ids, dtype=torch.int64, device=inputs_embeds.device).unsqueeze(0)

        language_output = self.student_manager.model.language_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            output_attentions=True,
            labels=labels,
            return_dict=True,
        )

        out = cast(CausalLMOutputLike, language_output)

        if len(is_image_mask_list) != int(inputs_embeds.shape[1]):
            return out, None

        mask_tensor = torch.tensor(is_image_mask_list, dtype=torch.bool, device=inputs_embeds.device).unsqueeze(0)
        return out, mask_tensor

    @staticmethod
    def _get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _set_grads_none(params: list[nn.Parameter]) -> None:
        for p in params:
            p.grad = None

    @staticmethod
    def _copy_grads(params: list[nn.Parameter]) -> list[Tensor | None]:
        out: list[Tensor | None] = []
        for p in params:
            if p.grad is None:
                out.append(None)
            else:
                out.append(p.grad.detach().clone())
        return out

    @staticmethod
    def _grads_norm(grads: list[Tensor | None]) -> Tensor:
        total_sq: float = 0.0
        for g in grads:
            if g is None:
                continue
            total_sq += float((g.detach().float().pow(2).sum()).cpu().item())
        return torch.tensor(math.sqrt(total_sq + 1e-20), dtype=torch.float32, device="cpu")

    def train_step(
        self,
        content: ContentType,
        *,
        dataset_index: int,
        step_optimizer: bool = True,
        loss_scale: float = 1.0,
    ) -> TrainStepRecord:
        try:
            if not (0 <= dataset_index < len(self.teacher_weights_by_dataset_norm)):
                return {
                    "dataset_index": int(dataset_index),
                    "ground_truth_loss": 0.0,
                    "loss": 0.0,
                    "loss_static": 0.0,
                    "hard_loss": 0.0,
                    "hard_loss_static": 0.0,
                    "soft_loss": 0.0,
                    "soft_loss_static": 0.0,
                    "per_teacher": [],
                    "dyn_weights": {},
                    "step": int(self.student_manager.train_steps),
                    "optim_step": int(self.optimizer_step),
                }

            teacher_mix = self.teacher_weights_by_dataset_norm[dataset_index]
            active_teacher_indices = [i for i, w in enumerate(teacher_mix) if float(w) > 0.0]
            if not active_teacher_indices:
                return {
                    "dataset_index": int(dataset_index),
                    "ground_truth_loss": 0.0,
                    "loss": 0.0,
                    "loss_static": 0.0,
                    "hard_loss": 0.0,
                    "hard_loss_static": 0.0,
                    "soft_loss": 0.0,
                    "soft_loss_static": 0.0,
                    "per_teacher": [],
                    "dyn_weights": {},
                    "step": int(self.student_manager.train_steps),
                    "optim_step": int(self.optimizer_step),
                }

            group = self.reweight_group
            static_w: dict[str, float] = dict(group.static_weight_map)

            def role_prefix(role: str, idx: int) -> str:
                if role == "perception":
                    return "p"
                if role == "reasoning":
                    return "r"
                if role == "planning":
                    return "d"
                return f"t{idx}"

            def teacher_att_name(role: str) -> str:
                if role == "perception":
                    return "first_att_Atv"
                if role == "reasoning":
                    return "intermediate"
                if role == "planning":
                    return "penult_att_Atv"
                return "intermediate"

            filtered: list[int] = []
            for t_idx in active_teacher_indices:
                role = self.teacher_roles[t_idx] if t_idx < len(self.teacher_roles) else f"teacher{t_idx}"
                pref = role_prefix(role, t_idx)
                att_name = teacher_att_name(role)
                att_key = f"{pref}:{att_name}"
                need_att = (att_key in group.active_losses) and (float(static_w.get(att_key, 0.0)) != 0.0)
                if need_att:
                    filtered.append(t_idx)

            active_teacher_indices = filtered
            if not active_teacher_indices:
                student_output, _ = self._forward_student_with_mask(content)
                ce_raw = float(student_output.loss.detach().item())
                return {
                    "dataset_index": int(dataset_index),
                    "ground_truth_loss": float(ce_raw),
                    "loss": float(ce_raw),
                    "loss_static": float(ce_raw),
                    "hard_loss": float(ce_raw),
                    "hard_loss_static": float(ce_raw),
                    "soft_loss": 0.0,
                    "soft_loss_static": 0.0,
                    "per_teacher": [],
                    "dyn_weights": {k: 0.0 for k in ["ce", "p:first_att_Atv", "r:intermediate", "d:penult_att_Atv"]},
                    "step": int(self.student_manager.train_steps),
                    "optim_step": int(self.optimizer_step),
                }

            student_output, is_image_mask = self._forward_student_with_mask(content)
            ground_truth_loss_t = student_output.loss
            base_device = ground_truth_loss_t.device

            teacher_loss_terms: dict[int, dict[str, Tensor]] = {}
            teacher_records_raw: dict[int, DistillationRecord] = {}

            for t_idx in active_teacher_indices:
                role = self.teacher_roles[t_idx] if t_idx < len(self.teacher_roles) else f"teacher{t_idx}"
                with torch.no_grad():
                    t_out_any = self.teacher_managers[t_idx](
                        content,
                        output_attentions=True,
                        output_hidden_states=False,
                    )
                t_out = cast(CausalLMOutputLike, t_out_any)

                loss_terms, dist_record = self.teachers[t_idx].get_distillation_loss(
                    role=str(role),
                    student_output=student_output,
                    teacher_output=t_out,
                    is_image_mask=is_image_mask,
                )

                expected_att = teacher_att_name(role)
                if expected_att not in loss_terms:
                    raise RuntimeError(f"Teacher[{t_idx}:{role}] required '{expected_att}' but it was not produced.")

                teacher_loss_terms[t_idx] = loss_terms
                teacher_records_raw[t_idx] = dist_record

            all_task_keys_ordered: list[str] = ["ce", "p:first_att_Atv", "r:intermediate", "d:penult_att_Atv"]

            task_losses: dict[str, Tensor] = {"ce": ground_truth_loss_t}
            per_teacher_records: list[TeacherStepRecord] = []

            for t_idx in active_teacher_indices:
                role = self.teacher_roles[t_idx] if t_idx < len(self.teacher_roles) else f"teacher{t_idx}"
                pref = role_prefix(role, t_idx)
                att_name = teacher_att_name(role)
                att_key = f"{pref}:{att_name}"

                att_loss = teacher_loss_terms[t_idx][att_name]
                task_losses[att_key] = att_loss

                soft_terms_raw: dict[str, float] = {att_name: float(att_loss.detach().item())}

                per_teacher_records.append(
                    {
                        "teacher_index": int(t_idx),
                        "teacher_role": str(role),
                        "dataset_index": int(dataset_index),
                        "teacher_weight": float(teacher_mix[t_idx]),
                        "ground_truth_loss": float(ground_truth_loss_t.detach().item()),
                        "distillation_record": teacher_records_raw[t_idx],
                        "loss": 0.0,
                        "loss_static": 0.0,
                        "loss_weighted": 0.0,
                        "loss_static_weighted": 0.0,
                        "hard_loss": 0.0,
                        "hard_loss_static": 0.0,
                        "hard_loss_weighted": 0.0,
                        "hard_loss_static_weighted": 0.0,
                        "soft_loss": 0.0,
                        "soft_loss_static": 0.0,
                        "soft_loss_weighted": 0.0,
                        "soft_loss_static_weighted": 0.0,
                        "used_weights": {},
                        "static_weights": {},
                        "dynamic_enabled": False,
                        "term_losses_static": {},
                        "term_losses_static_weighted": {},
                        "term_losses_raw": {},
                        "hard_loss_raw": float(ground_truth_loss_t.detach().item()),
                        "soft_terms_raw": soft_terms_raw,
                    }
                )

            params = self._get_trainable_params(self.student_manager.model)
            prev_accum_grads = self._copy_grads(params)

            base_grads: dict[str, list[Tensor | None]] = {}
            base_norms: dict[str, Tensor] = {}

            def backward_capture(loss: Tensor, *, retain: bool) -> list[Tensor | None]:
                self._set_grads_none(params)
                (loss * float(loss_scale)).backward(retain_graph=retain)
                return self._copy_grads(params)

            task_keys_for_backward = list(task_losses.keys())
            for k in task_keys_for_backward:
                lt = task_losses[k]
                if not torch.is_tensor(lt):
                    raise RuntimeError(f"task_losses['{k}'] is not a tensor.")
                if not lt.requires_grad:
                    raise RuntimeError(f"task_losses['{k}'] does NOT require grad.")

            total_backwards = len(task_keys_for_backward)
            bw_count = 0
            for k in task_keys_for_backward:
                bw_count += 1
                retain = bw_count < total_backwards
                g = backward_capture(task_losses[k], retain=retain)
                base_grads[k] = g
                base_norms[k] = self._grads_norm(g)

            w_now = group.current_loss_weights()

            dyn_weights_out: dict[str, float] = {}
            for key in all_task_keys_ordered:
                if key in w_now:
                    dyn_weights_out[key] = float(w_now[key].detach().item())
                else:
                    dyn_weights_out[key] = 0.0

            for key in all_task_keys_ordered:
                if key not in static_w:
                    static_w[key] = 0.0

            task_grads_weighted: dict[str, list[Tensor | None]] = {}

            for k in task_keys_for_backward:
                g_base = base_grads[k]
                if k == "ce":
                    w_total = w_now.get("ce", torch.tensor(float(static_w["ce"]), device=group.device)).detach()
                else:
                    mix = 1.0
                    if k.startswith("p:"):
                        mix = float(teacher_mix[0]) if len(teacher_mix) > 0 else 1.0
                    elif k.startswith("r:"):
                        mix = float(teacher_mix[1]) if len(teacher_mix) > 1 else 1.0
                    elif k.startswith("d:"):
                        mix = float(teacher_mix[2]) if len(teacher_mix) > 2 else 1.0
                    else:
                        mix = 1.0

                    w_internal = w_now.get(k, torch.tensor(float(static_w.get(k, 0.0)), device=group.device)).detach()
                    w_total = w_internal * float(mix)

                g_weighted: list[Tensor | None] = []
                for gi in g_base:
                    if gi is None:
                        g_weighted.append(None)
                    else:
                        g_weighted.append(gi * w_total.to(gi.device))
                task_grads_weighted[k] = g_weighted

            def sum_grads(grads_dict: dict[str, list[Tensor | None]]) -> list[Tensor | None]:
                out: list[Tensor | None] = []
                for p_idx in range(len(params)):
                    acc: Tensor | None = None
                    for gg in grads_dict.values():
                        gk = gg[p_idx]
                        if gk is None:
                            continue
                        acc = gk if acc is None else (acc + gk)
                    out.append(acc)
                return out

            def _dot_list_grads(ga: list[Tensor | None], gb: list[Tensor | None]) -> float:
                return _OnlineReweightGroup._dot_grads(ga, gb)

            def _norm_sq_list_grads(g: list[Tensor | None]) -> float:
                return _OnlineReweightGroup._norm_sq_grads(g)

            def _project_follower_against_anchor(
                follower: list[Tensor | None],
                anchor: list[Tensor | None],
            ) -> list[Tensor | None]:
                dot = _dot_list_grads(follower, anchor)
                if dot >= 0.0:
                    return follower
                denom = max(_norm_sq_list_grads(anchor), 1e-20)
                coef = float(dot / denom)
                out: list[Tensor | None] = []
                for gf, ga in zip(follower, anchor):
                    if gf is None:
                        out.append(None)
                        continue
                    if ga is None:
                        out.append(gf)
                        continue
                    aa = ga
                    if aa.device != gf.device:
                        aa = aa.to(gf.device)
                    out.append(gf - coef * aa)
                return out

            if not bool(getattr(self, "use_agp", False)):
                total_grad = sum_grads({k: task_grads_weighted[k] for k in task_keys_for_backward})
            else:
                # --- AGP Stage 1: asymmetric anchor (ce) / follower (each capability distill) ---
                g_sup = task_grads_weighted["ce"]

                cap_keys_order = ["p:first_att_Atv", "r:intermediate", "d:penult_att_Atv"]
                present_caps = [ck for ck in cap_keys_order if ck in task_grads_weighted]

                merged: dict[str, list[Tensor | None]] = {}
                for ck in present_caps:
                    g_f = task_grads_weighted[ck]
                    g_f_proj = _project_follower_against_anchor(g_f, g_sup)
                    # anchor unchanged; merge
                    merged[ck] = sum_grads({"sup": g_sup, "f": g_f_proj})

                if not merged:
                    total_grad = g_sup
                else:
                    # --- AGP Stage 2: shuffled symmetric pairwise projections among merged capability grads ---
                    perm = list(present_caps)
                    random.shuffle(perm)

                    tilde: dict[str, list[Tensor | None]] = {k: merged[k] for k in present_caps}

                    for ci in perm:
                        for cj in perm:
                            if ci == cj:
                                continue
                            tilde[ci] = _project_follower_against_anchor(tilde[ci], merged[cj])

                    total_grad = sum_grads(tilde)

            for p_idx, p in enumerate(params):
                g_prev = prev_accum_grads[p_idx]
                g_new = total_grad[p_idx]
                if g_prev is None and g_new is None:
                    p.grad = None
                elif g_prev is None:
                    p.grad = g_new
                elif g_new is None:
                    p.grad = g_prev
                else:
                    p.grad = g_prev + g_new

            self.student_manager.train_steps += 1

            ce_raw = float(ground_truth_loss_t.detach().item())
            ce_static_total = float(static_w["ce"]) * ce_raw

            soft_static_total_weighted = 0.0

            for rec in per_teacher_records:
                role = rec["teacher_role"]
                t_idx = int(rec["teacher_index"])
                if role == "perception":
                    att_key = "p:first_att_Atv"
                    mix_w = float(teacher_mix[0]) if len(teacher_mix) > 0 else 1.0
                elif role == "reasoning":
                    att_key = "r:intermediate"
                    mix_w = float(teacher_mix[1]) if len(teacher_mix) > 1 else 1.0
                elif role == "planning":
                    att_key = "d:penult_att_Atv"
                    mix_w = float(teacher_mix[2]) if len(teacher_mix) > 2 else 1.0
                else:
                    att_key = f"t{t_idx}:intermediate"
                    mix_w = float(teacher_mix[t_idx]) if t_idx < len(teacher_mix) else 1.0

                att_raw = float(task_losses.get(att_key, torch.tensor(0.0, device=base_device)).detach().item())
                per_teacher_soft_static = float(static_w.get(att_key, 0.0)) * att_raw
                rec["soft_loss_static"] = float(per_teacher_soft_static)

                soft_static_total_weighted += float(mix_w) * per_teacher_soft_static

            total_loss_static = ce_static_total + soft_static_total_weighted

            losses_for_stats: dict[str, Tensor] = {}
            norms_for_stats: dict[str, Tensor] = {}
            for k in task_keys_for_backward:
                losses_for_stats[k] = task_losses[k].detach()
                norms_for_stats[k] = base_norms[k].to(group.device).detach()
            group.accum_add_stats(losses_for_stats, norms_for_stats)

            if step_optimizer:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer_step += 1

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                mean_losses, mean_norms = group.accum_pop_mean_stats()
                if group.use_dynamic_loss_weights and group.loss_weight_params is not None:
                    _ = group.update_dynamic_weights_online_reweighting(mean_losses, mean_norms)

            record: TrainStepRecord = {
                "dataset_index": int(dataset_index),
                "ground_truth_loss": float(ce_raw),
                "loss": float(total_loss_static),
                "loss_static": float(total_loss_static),
                "hard_loss": float(ce_static_total),
                "hard_loss_static": float(ce_static_total),
                "soft_loss": float(soft_static_total_weighted),
                "soft_loss_static": float(soft_static_total_weighted),
                "per_teacher": per_teacher_records,
                "dyn_weights": dyn_weights_out,
                "step": int(self.student_manager.train_steps),
                "optim_step": int(self.optimizer_step),
            }
            return record

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                try:
                    self.optimizer.zero_grad()
                except Exception:
                    pass
                try:
                    for p in self._get_trainable_params(self.student_manager.model):
                        p.grad = None
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return {
                    "dataset_index": int(dataset_index),
                    "ground_truth_loss": 0.0,
                    "loss": 0.0,
                    "loss_static": 0.0,
                    "hard_loss": 0.0,
                    "hard_loss_static": 0.0,
                    "soft_loss": 0.0,
                    "soft_loss_static": 0.0,
                    "per_teacher": [],
                    "dyn_weights": {},
                    "step": int(self.student_manager.train_steps),
                    "optim_step": int(self.optimizer_step),
                }
            raise

    def distill(
        self,
        dataloader: Sequence[tuple[ContentType, int]],
        *,
        temperature: float = 1.0,
        hook: DistillationHook | None = None,
        progress_info: dict[str, Any] | None = None,
    ) -> None:
        if len(dataloader) == 0:
            return

        if not hasattr(self, "_eta_global_epoch_sec_sum"):
            self._eta_global_epoch_sec_sum = 0.0
            self._eta_global_epoch_done = 0
            self._eta_global_avg_epoch_sec = 0.0

        if not hasattr(self, "_eta_exp_stats"):
            self._eta_exp_stats: dict[str, dict[str, float]] = {}

        if not hasattr(self, "_eta_last_exp_uid"):
            self._eta_last_exp_uid: str | None = None

        self.student_manager.train()
        self.optimizer.zero_grad()
        accum_counter = 0

        epoch_start_time = datetime.now()

        exp_uid = "default_exp"
        if progress_info is not None and isinstance(progress_info.get("exp_uid", None), str):
            exp_uid = str(progress_info["exp_uid"])

        if self._eta_last_exp_uid is None:
            self._eta_last_exp_uid = exp_uid
        elif self._eta_last_exp_uid != exp_uid:
            self._eta_last_exp_uid = exp_uid

        if exp_uid not in self._eta_exp_stats:
            self._eta_exp_stats[exp_uid] = {"sec_sum": 0.0, "done": 0.0, "avg": 0.0}

        base_desc = f"[{exp_uid}]"
        if progress_info is not None and isinstance(progress_info.get("prefix", None), str):
            base_desc = str(progress_info["prefix"])

        pbar = tqdm(dataloader, desc=base_desc)
        for idx, (content, dataset_index) in enumerate(pbar):
            accum_counter += 1
            do_step = (accum_counter % self.grad_accum_steps == 0) or (idx == len(dataloader) - 1)

            record = self.train_step(
                content,
                dataset_index=int(dataset_index),
                step_optimizer=do_step,
                loss_scale=1.0 / float(self.grad_accum_steps),
            )

            pbar.set_postfix({"loss": f"{record['loss']:.3f}"})

            if progress_info is not None:
                total_epochs_all = int(progress_info.get("total_epochs_all", 0))
                done_epochs_all = int(progress_info.get("done_epochs", 0))

                exp_total_epochs = int(progress_info.get("exp_total_epochs", total_epochs_all))
                exp_done_epochs = int(progress_info.get("exp_done_epochs", done_epochs_all))

                total_in_epoch = float(pbar.total or 1.0)
                frac_epoch = float(pbar.n) / total_in_epoch if total_in_epoch > 0 else 0.0

                global_start_time = progress_info.get("global_start_time", None)
                elapsed_sec = 0.0
                if isinstance(global_start_time, datetime):
                    elapsed_td = datetime.now() - global_start_time
                    elapsed_sec = float(elapsed_td.total_seconds())
                    elapsed_str = str(elapsed_td).split(".")[0]
                else:
                    elapsed_str = "0:00:00"

                pi_g_avg = float(progress_info.get("global_avg_epoch_sec", 0.0))
                pi_e_avg = float(progress_info.get("exp_avg_epoch_sec", 0.0))

                g_avg = float(getattr(self, "_eta_global_avg_epoch_sec", 0.0))
                exp_st = self._eta_exp_stats.get(exp_uid, {"sec_sum": 0.0, "done": 0.0, "avg": 0.0})
                e_done = int(exp_st.get("done", 0.0))
                e_avg = float(exp_st.get("avg", 0.0))

                exp_epoch_est = 0.0
                if pi_e_avg > 0.0:
                    exp_epoch_est = pi_e_avg
                elif e_done > 0 and e_avg > 0.0:
                    exp_epoch_est = e_avg

                if exp_epoch_est <= 0.0:
                    exp_epoch_est = pi_g_avg if pi_g_avg > 0.0 else g_avg

                curr_epoch_est = float(exp_epoch_est)

                rep_remaining_str = "unknown"
                total_remaining_str = "unknown"
                rep_remaining_sec: float | None = None
                total_remaining_sec: float | None = None

                if curr_epoch_est > 0.0 and total_epochs_all > 0:
                    rem_curr = max(0.0, 1.0 - frac_epoch) * curr_epoch_est

                    exp_remaining = max(0, exp_total_epochs - exp_done_epochs - 1)
                    global_remaining = max(0, total_epochs_all - done_epochs_all - 1)
                    future_remaining = max(0, global_remaining - exp_remaining)

                    future_epoch_est = pi_g_avg if pi_g_avg > 0.0 else g_avg

                    if future_epoch_est > 0.0:
                        rep_remaining_sec_val = float(rem_curr) + float(exp_remaining) * float(curr_epoch_est)
                        total_remaining_sec_val = float(rep_remaining_sec_val) + float(future_remaining) * float(future_epoch_est)

                        rep_remaining_sec_val = max(0.0, rep_remaining_sec_val)
                        total_remaining_sec_val = max(0.0, total_remaining_sec_val)

                        rep_remaining_sec = rep_remaining_sec_val
                        total_remaining_sec = total_remaining_sec_val

                        rep_remaining_str = str(timedelta(seconds=int(rep_remaining_sec_val)))
                        total_remaining_str = str(timedelta(seconds=int(total_remaining_sec_val)))

                if total_remaining_sec is not None and (elapsed_sec + total_remaining_sec) > 0.0:
                    global_progress = elapsed_sec / (elapsed_sec + total_remaining_sec)
                else:
                    if total_epochs_all > 0:
                        global_progress = (float(done_epochs_all) + float(frac_epoch)) / float(total_epochs_all)
                    else:
                        global_progress = 0.0

                desc_prefix = progress_info.get("prefix", base_desc)
                desc = (
                    f"{desc_prefix}"
                    f"[{global_progress * 100.0:.2f}% - {elapsed_str} - "
                    f"rep_rem {rep_remaining_str} - total_rem {total_remaining_str}]"
                )
                pbar.set_description(desc)

            if hook is not None:
                hook(record)

        epoch_sec = float((datetime.now() - epoch_start_time).total_seconds())
        if epoch_sec < 0.0:
            epoch_sec = 0.0

        self._eta_global_epoch_done = int(self._eta_global_epoch_done) + 1
        self._eta_global_epoch_sec_sum = float(self._eta_global_epoch_sec_sum) + epoch_sec
        self._eta_global_avg_epoch_sec = (
            float(self._eta_global_epoch_sec_sum) / float(self._eta_global_epoch_done)
            if self._eta_global_epoch_done > 0
            else 0.0
        )

        st = self._eta_exp_stats[exp_uid]
        st["done"] = float(int(st.get("done", 0.0)) + 1)
        st["sec_sum"] = float(st.get("sec_sum", 0.0)) + epoch_sec
        st["avg"] = float(st["sec_sum"]) / float(st["done"]) if st["done"] > 0 else 0.0
        self._eta_exp_stats[exp_uid] = st

        try:
            ckpt_dir: str | None = None
            if progress_info is not None and isinstance(progress_info.get("checkpoint_dir", None), str):
                ckpt_dir = str(progress_info["checkpoint_dir"])

            if progress_info is not None:
                epoch_num = int(progress_info.get("done_epochs", 0)) + 1
            else:
                epoch_num = int(getattr(self, "_eta_global_epoch_done", 0))

            remark = f"{exp_uid}_epoch{epoch_num:04d}_{epoch_num % 100:02d}"
            self.save_checkpoint(checkpoint_dir=ckpt_dir, remark=remark)
        except Exception:
            pass

        logger.info(f"Epoch finished in {str(timedelta(seconds=int(epoch_sec)))}")

    def save_checkpoint(self, checkpoint_dir: str | None = None, remark: str | None = "") -> None:
        if checkpoint_dir is None:
            checkpoint_dir = default_checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        raw_student_name = getattr(self.student_manager, "model_name", "student")
        safe_student_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(raw_student_name)).strip("_")

        fname = safe_student_name
        if remark:
            safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(remark)).strip("_")
            fname = f"{fname}_{safe}"

        ckpt_dir = os.path.join(checkpoint_dir, fname)
        os.makedirs(ckpt_dir, exist_ok=True)

        student_suffix = ""
        if remark:
            m = re.search(r"_(\d{2})$", str(remark))
            if m:
                student_suffix = m.group(1)

        student_model_name_out = self.student_manager.model_name
        if student_suffix:
            student_model_name_out = f"{student_model_name_out}_{student_suffix}"

        src_dir = getattr(self.student_manager, "model_path", None)
        if isinstance(src_dir, str) and os.path.isdir(src_dir):
            skip_names = {
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            }
            skip_exts = {".safetensors", ".bin"}

            try:
                for name in os.listdir(src_dir):
                    src = os.path.join(src_dir, name)
                    dst = os.path.join(ckpt_dir, name)

                    if os.path.isdir(src):
                        continue

                    if name in skip_names:
                        continue
                    _, ext = os.path.splitext(name)
                    if ext in skip_exts:
                        continue

                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            self.student_manager.model.save_pretrained(ckpt_dir, safe_serialization=True)
        except TypeError:
            self.student_manager.model.save_pretrained(ckpt_dir)
        except Exception:
            torch.save(self.student_manager.model.state_dict(), os.path.join(ckpt_dir, "student_state.pt"))

        try:
            tok = getattr(self.student_manager, "tokenizer", None)
            if tok is not None:
                tok.save_pretrained(ckpt_dir)
        except Exception:
            pass

        torch.save(self.student_manager.model.state_dict(), os.path.join(ckpt_dir, "student_state.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer_state.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "lr_scheduler_state.pt"))

        try:
            torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))
        except Exception:
            pass

        reweight_payload: list[dict[str, Any]] = []
        for gi, g in enumerate(self.reweight_groups):
            entry: dict[str, Any] = {
                "teacher_role": g.teacher_role,
                "active_losses": list(g.active_losses),
                "static_weight_map": dict(g.static_weight_map),
                "use_dynamic_loss_weights": bool(g.use_dynamic_loss_weights),
                "dynamic_weight_lr": float(g.dynamic_weight_lr),
                "reweight_alpha": float(g.reweight_alpha),
                "dynamic_weight_eps": float(g.dynamic_weight_eps),
                "dynamic_weight_min": float(g.dynamic_weight_min),
                "dynamic_weight_max": float(g.dynamic_weight_max),
                "min_dynamic_weight_ce": float(g.min_dynamic_weight_ce),
                "loss_init_values": dict(g.loss_init_values),
            }

            if g.loss_weight_params is not None:
                fn = f"reweight_{gi}_{g.teacher_role}_loss_weight_params.pt"
                torch.save(g.loss_weight_params.state_dict(), os.path.join(ckpt_dir, fn))
                entry["loss_weight_params_state_file"] = fn

            if g.weight_optimizer is not None:
                fn = f"reweight_{gi}_{g.teacher_role}_weight_optimizer.pt"
                torch.save(g.weight_optimizer.state_dict(), os.path.join(ckpt_dir, fn))
                entry["weight_optimizer_state_file"] = fn

            reweight_payload.append(entry)

        meta: dict[str, Any] = {
            "version": 2,
            "teacher_model_names": [m.model_name for m in self.teacher_managers],
            "student_model_name": student_model_name_out,
            "teacher_roles": list(self.teacher_roles),
            "teacher_weights_by_dataset": list(self.teacher_weights_by_dataset),
            "grad_accum_steps": int(self.grad_accum_steps),
            "optimizer_step": int(self.optimizer_step),
            "student_train_steps": int(self.student_manager.train_steps),
            "use_lr_scheduler": bool(self.use_lr_scheduler),
            "lr_scheduler_type": str(self.lr_scheduler_type),
            "total_optim_steps": int(self.total_optim_steps),
            "warmup_optim_steps": int(self.warmup_optim_steps),
            "lr_min_ratio": float(self.lr_min_ratio),
            "reweight_groups": reweight_payload,
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        try:
            with open(os.path.join(ckpt_dir, "trainer_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        try:
            g = self.reweight_group
            dyn_payload = {
                "use_dynamic_loss_weights": bool(g.use_dynamic_loss_weights),
                "active_losses": list(g.active_losses),
                "static_weight_map": dict(g.static_weight_map),
                "loss_init_values": dict(g.loss_init_values),
                "loss_weight_params": (g.loss_weight_params.state_dict() if g.loss_weight_params is not None else None),
                "weight_optimizer": (g.weight_optimizer.state_dict() if g.weight_optimizer is not None else None),
            }
            torch.save(dyn_payload, os.path.join(ckpt_dir, "dynamic_weights.pth"))
        except Exception:
            pass

        logger.info(f"Checkpoint saved: {ckpt_dir}")

