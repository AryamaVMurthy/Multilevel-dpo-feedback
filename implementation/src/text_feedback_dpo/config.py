from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {"run_id", "student_model", "teacher_model", "student_revision", "teacher_revision", "dataset", "training", "slurm", "retrieval"}
ALLOWED_KEYS = REQUIRED_KEYS | {"generation", "teacher_generation", "evaluation", "storage", "seeds"}
RETRIEVAL_KEYS = {"backend", "top_k", "k1", "b", "schema_version"}


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
    retrieval = data["retrieval"]
    if not isinstance(retrieval, dict):
        raise ValueError("retrieval must be a mapping")
    unknown_retrieval = sorted(set(retrieval) - RETRIEVAL_KEYS)
    if unknown_retrieval:
        raise ValueError(f"unknown retrieval config keys: {', '.join(unknown_retrieval)}")
    missing_retrieval = sorted(RETRIEVAL_KEYS - set(retrieval))
    if missing_retrieval:
        raise ValueError(f"missing retrieval config keys: {', '.join(missing_retrieval)}")
    if retrieval["backend"] != "fixed_bm25":
        raise ValueError(f"unknown retrieval backend: {retrieval['backend']!r}")
    if isinstance(retrieval["top_k"], bool) or not isinstance(retrieval["top_k"], int) or retrieval["top_k"] != 8:
        raise ValueError("retrieval.top_k must be exactly 8")
    if (
        isinstance(retrieval["k1"], bool)
        or not isinstance(retrieval["k1"], Real)
        or not math.isfinite(retrieval["k1"])
        or retrieval["k1"] <= 0
    ):
        raise ValueError("retrieval.k1 must be a finite number greater than zero")
    if (
        isinstance(retrieval["b"], bool)
        or not isinstance(retrieval["b"], Real)
        or not math.isfinite(retrieval["b"])
        or not 0 <= retrieval["b"] <= 1
    ):
        raise ValueError("retrieval.b must be a finite number between zero and one")
    if retrieval["schema_version"] != 1:
        raise ValueError("retrieval.schema_version must be 1")
    return data
