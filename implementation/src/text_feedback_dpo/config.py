from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "run_id",
    "student_model",
    "teacher_model",
    "teacher_mode",
    "max_examples",
    "output_dir",
    "generation",
    "teacher_generation",
    "slurm",
}

OPTIONAL_KEYS = {
    "allow_missing_slurm_account_for_local",
}

ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml is required to load config files") from exc

    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def load_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)

    unknown = sorted(set(config) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {', '.join(unknown)}")

    missing = sorted(REQUIRED_KEYS - set(config))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")

    if config["teacher_mode"] not in {"stronger_model", "same_model_privileged"}:
        raise ValueError("teacher_mode must be stronger_model or same_model_privileged")

    if not isinstance(config["max_examples"], int) or config["max_examples"] <= 0:
        raise ValueError("max_examples must be a positive integer")

    slurm = config["slurm"]
    if not isinstance(slurm, dict):
        raise ValueError("slurm must be a mapping")
    if not config.get("allow_missing_slurm_account_for_local") and not slurm.get("account"):
        raise ValueError("slurm.account is required unless allow_missing_slurm_account_for_local is true")

    return config
