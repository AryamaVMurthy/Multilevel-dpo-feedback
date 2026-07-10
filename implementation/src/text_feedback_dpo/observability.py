from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from text_feedback_dpo.io import write_json_atomic


RUN_MANIFEST_REQUIRED_FIELDS = (
    "git_commit",
    "config_hash",
    "dataset_manifest_hash",
    "seed",
    "source_revision",
    "model_revisions",
    "package_versions",
    "slurm",
    "gpu_telemetry",
    "token_counts",
    "latency_ms",
    "throughput",
    "peak_memory_bytes",
    "pair_metrics",
    "evaluator_metrics",
    "training_metrics",
    "architecture",
    "optimizer",
    "candidate_id",
    "promotion_stage",
    "selection_evidence",
    "search_ledger_hash",
    "failure_ledger",
)


class JsonlLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.start_ns = time.monotonic_ns()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_name: str, **fields: Any) -> None:
        elapsed_ms = (time.monotonic_ns() - self.start_ns) // 1_000_000
        payload = {
            "event_name": event_name,
            "run_id": self.run_id,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": fields.pop("status", "ok"),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def failure(self, *, stage: str, error_code: str, message: str, **fields: Any) -> None:
        self.event(
            "failure",
            status="error",
            stage=stage,
            error_code=error_code,
            message=message,
            **fields,
        )


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    missing = [field for field in RUN_MANIFEST_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"run manifest is missing required field: {missing[0]}")
    write_json_atomic(path, payload)


def validate_run_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"run manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    missing = [field for field in RUN_MANIFEST_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"run manifest is missing required field: {missing[0]}")
    return payload


class TensorBoardScalarLogger:
    """Optional secondary view; canonical JSONL logging remains mandatory."""

    def __init__(self, log_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("TensorBoard logging requested but torch.utils.tensorboard is unavailable") from exc
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def scalar(self, name: str, value: float, step: int) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("TensorBoard scalar name must be non-empty")
        self._writer.add_scalar(name, float(value), int(step))
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
