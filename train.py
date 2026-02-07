from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from intern.model import ModelManager, set_default_checkpoint_dir
from intern.qa_loader import QADialogLoader
from intern.trainer import DistillationTrainer


DEFAULT_TEACHER_MODEL = "models/InternVL3-8B"
DEFAULT_STUDENT_MODEL = "models/InternVL3-1B"
DEFAULT_DATA_JSON = "data/demo.json"


def _setup_logger(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _parse_float_triplet(value: str, field_name: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 comma-separated values, got: {value}")
    row = [float(p) for p in parts]
    if all(v <= 0.0 for v in row):
        raise ValueError(f"{field_name} must contain at least one positive value, got: {row}")
    return row


def _normalize_question_type(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    aliases: dict[str, str] = {
        "perception": "perception",
        "perceptual": "perception",
        "perceive": "perception",
        "per": "perception",
        "p": "perception",
        "reasoning": "reasoning",
        "reason": "reasoning",
        "logical_reasoning": "reasoning",
        "infer": "reasoning",
        "r": "reasoning",
        "planning": "planning",
        "plan": "planning",
        "pl": "planning",
        "d": "planning",
        "decision": "planning",
    }
    if s in aliases:
        return aliases[s]

    if "percep" in s:
        return "perception"
    if "reason" in s:
        return "reasoning"
    if "plan" in s:
        return "planning"
    return None


def _extract_raw_items(blob: Any) -> list[dict[str, Any]]:
    if isinstance(blob, list):
        return [x for x in blob if isinstance(x, dict)]

    if isinstance(blob, dict):
        for key in ("data", "items", "samples", "dataset", "records"):
            v = blob.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    raise ValueError("JSON root must be a list or contain one of keys: data/items/samples/dataset/records")


def _canonicalize_dataset_items(
    items: list[dict[str, Any]],
    *,
    on_unknown_type: str,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    counts = {"perception": 0, "reasoning": 0, "planning": 0}
    unknown_count = 0
    out: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        qtype_raw = item.get("question_type", item.get("type", item.get("capability")))
        qtype = _normalize_question_type(qtype_raw)

        if qtype is None:
            unknown_count += 1
            if on_unknown_type == "error":
                raise ValueError(
                    f"Unknown question_type at item {idx}: {qtype_raw!r}. "
                    "Use --on-unknown-type to set a fallback capability."
                )
            qtype = on_unknown_type

        new_item = dict(item)
        new_item["question_type"] = qtype
        out.append(new_item)
        counts[qtype] += 1

    return out, counts, unknown_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Main training script for multi-teacher InternVL distillation "
            "(perception/reasoning/planning, with AGP and teacher mixing)."
        )
    )

    parser.add_argument("--data-json", type=str, default=DEFAULT_DATA_JSON)

    parser.add_argument("--teacher-model-path", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-perception-path", type=str, default=None)
    parser.add_argument("--teacher-reasoning-path", type=str, default=None)
    parser.add_argument("--teacher-planning-path", type=str, default=None)
    parser.add_argument("--student-model-path", type=str, default=DEFAULT_STUDENT_MODEL)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--use-agp", action="store_true", default=True)
    parser.add_argument("--no-agp", dest="use_agp", action="store_false")

    parser.add_argument("--use-dynamic-loss-weights", action="store_true", default=True)
    parser.add_argument("--no-dynamic-loss-weights", dest="use_dynamic_loss_weights", action="store_false")
    parser.add_argument("--dynamic-weight-lr", type=float, default=3e-4)
    parser.add_argument("--reweight-alpha", type=float, default=0.0)
    parser.add_argument("--dynamic-weight-min", type=float, default=1e-6)
    parser.add_argument("--dynamic-weight-max", type=float, default=1e6)
    parser.add_argument("--min-dynamic-weight-ce", type=float, default=0.6)

    parser.add_argument("--w-ce", type=float, default=1.0)
    parser.add_argument("--w-perception", type=float, default=1.0, help="Weight for first-layer t2v loss.")
    parser.add_argument("--w-reasoning", type=float, default=1.0, help="Weight for intermediate attention loss.")
    parser.add_argument("--w-planning", type=float, default=1.0, help="Weight for penultimate t2v loss.")

    parser.add_argument(
        "--mix-perception",
        type=str,
        default="0.8,0.1,0.1",
        help="Teacher mix row for perception samples: p_teacher,r_teacher,pl_teacher",
    )
    parser.add_argument(
        "--mix-reasoning",
        type=str,
        default="0.1,0.8,0.1",
        help="Teacher mix row for reasoning samples: p_teacher,r_teacher,pl_teacher",
    )
    parser.add_argument(
        "--mix-planning",
        type=str,
        default="0.1,0.1,0.8",
        help="Teacher mix row for planning samples: p_teacher,r_teacher,pl_teacher",
    )

    parser.add_argument(
        "--on-unknown-type",
        type=str,
        choices=["error", "perception", "reasoning", "planning"],
        default="error",
        help="Fallback capability if question_type cannot be normalized.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--intermediate-layer-start", type=int, default=0)
    parser.add_argument("--intermediate-layer-end", type=int, default=-2)

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional override for CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'.",
    )

    parser.add_argument("--use-lr-scheduler", action="store_true", default=True)
    parser.add_argument("--no-lr-scheduler", dest="use_lr_scheduler", action="store_false")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--total-optim-steps", type=int, default=0)
    parser.add_argument("--warmup-optim-steps", type=int, default=0)
    parser.add_argument("--lr-min-ratio", type=float, default=0.01)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _setup_logger(args.log_level)
    logger = logging.getLogger("train")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()
        logger.info("Set CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("No CUDA devices found. This training script requires at least one GPU.")
    logger.info("CUDA device count=%d", gpu_count)
    for i in range(gpu_count):
        logger.info("GPU[%d]: %s", i, torch.cuda.get_device_name(i))

    set_default_checkpoint_dir(args.checkpoint_dir)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    data_json = Path(args.data_json)
    if not data_json.exists():
        raise FileNotFoundError(f"Dataset json not found: {data_json}")

    with data_json.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    raw_items = _extract_raw_items(blob)
    if not raw_items:
        raise ValueError(f"No valid items loaded from {data_json}")

    dataset_items, counts, unknown_count = _canonicalize_dataset_items(
        raw_items,
        on_unknown_type=args.on_unknown_type,
    )

    if args.max_samples > 0:
        dataset_items = dataset_items[: args.max_samples]
        logger.info("Using first %d samples due to --max-samples", len(dataset_items))

    logger.info(
        "Loaded %d samples (perception=%d, reasoning=%d, planning=%d, unknown_handled=%d)",
        len(dataset_items),
        counts["perception"],
        counts["reasoning"],
        counts["planning"],
        unknown_count,
    )

    teacher_p = args.teacher_perception_path or args.teacher_model_path
    teacher_r = args.teacher_reasoning_path or args.teacher_model_path
    teacher_pl = args.teacher_planning_path or args.teacher_model_path

    teacher_manager_cache: dict[str, ModelManager] = {}

    def get_teacher_manager(path: str) -> ModelManager:
        if path not in teacher_manager_cache:
            teacher_manager_cache[path] = ModelManager(
                model_path=path,
                torch_dtype=args.torch_dtype,
                resume_from_checkpoint=False,
            )
        return teacher_manager_cache[path]

    teacher_models = [
        get_teacher_manager(teacher_p),
        get_teacher_manager(teacher_r),
        get_teacher_manager(teacher_pl),
    ]

    student_model = ModelManager(
        model_path=args.student_model_path,
        torch_dtype=args.torch_dtype,
        resume_from_checkpoint=False,
    )

    teacher_weights_by_dataset = [
        _parse_float_triplet(args.mix_perception, "mix_perception"),
        _parse_float_triplet(args.mix_reasoning, "mix_reasoning"),
        _parse_float_triplet(args.mix_planning, "mix_planning"),
    ]
    logger.info("teacher_weights_by_dataset=%s", teacher_weights_by_dataset)

    loss_weight_config = {
        "ce": float(args.w_ce),
        "perception": float(args.w_perception),
        "reasoning": float(args.w_reasoning),
        "planning": float(args.w_planning),
    }

    trainer = DistillationTrainer(
        teacher_models=teacher_models,
        student_model=student_model,
        teacher_weights_by_dataset=teacher_weights_by_dataset,
        static_ce_weight=loss_weight_config["ce"],
        teacher_soft_loss_weights=[
            {"first_att_Atv": loss_weight_config["perception"]},
            {"intermediate": loss_weight_config["reasoning"]},
            {"penult_att_Atv": loss_weight_config["planning"]},
        ],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_accum_steps=int(args.grad_accum_steps),
        use_dynamic_loss_weights=bool(args.use_dynamic_loss_weights),
        use_agp=bool(args.use_agp),
        dynamic_weight_lr=float(args.dynamic_weight_lr),
        reweight_alpha=float(args.reweight_alpha),
        dynamic_weight_min=float(args.dynamic_weight_min),
        dynamic_weight_max=float(args.dynamic_weight_max),
        min_dynamic_weight_ce=float(args.min_dynamic_weight_ce),
        intermediate_layer_start=int(args.intermediate_layer_start),
        intermediate_layer_end=int(args.intermediate_layer_end),
        use_lr_scheduler=bool(args.use_lr_scheduler),
        lr_scheduler_type=str(args.lr_scheduler_type),
        total_optim_steps=int(args.total_optim_steps),
        warmup_optim_steps=int(args.warmup_optim_steps),
        lr_min_ratio=float(args.lr_min_ratio),
        torch_dtype=args.torch_dtype,
    )

    question_type_to_model = {
        "perception": 0,
        "reasoning": 1,
        "planning": 2,
    }
    loader = QADialogLoader(
        data=dataset_items,
        resolver=trainer.resolve_content_text,
        question_type_to_model=question_type_to_model,
        is_open_ending=False,
    )

    logger.info(
        "Training setup: epochs=%d, lr=%.2e, grad_accum_steps=%d, use_agp=%s, dynamic_weights=%s",
        args.epochs,
        args.lr,
        args.grad_accum_steps,
        args.use_agp,
        args.use_dynamic_loss_weights,
    )
    logger.info(
        "Models: teacher_p=%s | teacher_r=%s | teacher_pl=%s | student=%s",
        teacher_p,
        teacher_r,
        teacher_pl,
        args.student_model_path,
    )
    logger.info("Loss weights: %s", loss_weight_config)
    logger.info("Data: %s", args.data_json)

    state = {
        "last_log_step": -1,
    }

    def hook(rec: dict[str, Any]) -> None:
        step = int(rec.get("step", 0))
        if args.log_every > 0 and step > 0 and step % args.log_every == 0 and step != state["last_log_step"]:
            state["last_log_step"] = step
            logging.info(
                "step=%d optim_step=%d loss=%.6f hard=%.6f soft=%.6f",
                step,
                int(rec.get("optim_step", 0)),
                float(rec.get("loss", 0.0)),
                float(rec.get("hard_loss", 0.0)),
                float(rec.get("soft_loss", 0.0)),
            )

    global_start = datetime.now()

    for epoch_idx in range(args.epochs):
        if args.shuffle:
            order = list(range(len(loader)))
            random.shuffle(order)
            epoch_loader = loader[order]
        else:
            epoch_loader = loader

        progress_info = {
            "exp_uid": "drive_kd",
            "prefix": f"[epoch {epoch_idx + 1}/{args.epochs}]",
            "total_epochs_all": int(args.epochs),
            "done_epochs": int(epoch_idx),
            "exp_total_epochs": int(args.epochs),
            "exp_done_epochs": int(epoch_idx),
            "global_start_time": global_start,
            "checkpoint_dir": args.checkpoint_dir,
        }
        trainer.distill(epoch_loader, hook=hook, progress_info=progress_info)

    logger.info("Training completed. Final student_train_steps=%d", int(trainer.student_manager.train_steps))


if __name__ == "__main__":
    main()

