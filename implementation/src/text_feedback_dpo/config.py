from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_KEYS = {"run_id", "student_model", "teacher_model", "student_revision", "teacher_revision", "dataset", "training", "slurm"}
ALLOWED_KEYS = REQUIRED_KEYS | {"generation", "teacher_generation", "evaluation", "storage", "seeds"}


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml is required to load config files") from exc
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    unknown = sorted(set(data) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {', '.join(unknown)}")
    missing = sorted(REQUIRED_KEYS - set(data))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
    dataset = data["dataset"]
    training = data["training"]
    slurm = data["slurm"]
    if not isinstance(dataset, dict) or dataset.get("name") != "searchqa" or not dataset.get("source") or not dataset.get("revision"):
        raise ValueError("dataset.name, dataset.source, and dataset.revision are required")
    if dataset.get("max_length") != 4096:
        raise ValueError("dataset.max_length must be exactly 4096")
    if not isinstance(training, dict) or training.get("full_finetuning") is not True:
        raise ValueError("training.full_finetuning must be true")
    if training.get("method") not in {"sft", "dpo", "grpo", "dapo"}:
        raise ValueError("training.method must be sft, dpo, grpo, or dapo")
    if not isinstance(slurm, dict) or not slurm.get("partition") or not isinstance(slurm.get("gpus"), int) or slurm["gpus"] < 1:
        raise ValueError("slurm.partition and positive slurm.gpus are required")
    return data
