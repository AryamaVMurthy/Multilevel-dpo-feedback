from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from collections.abc import Mapping
from pathlib import Path

from text_feedback_dpo.training import retained_training_checkpoints


REQUIRED_ENVIRONMENT = (
    "EXPECTED_COMMIT",
    "CONFIG_SHA256",
    "TRAIN_SHA256",
    "EVAL_SHA256",
    "SPLIT_REPORT_SHA256",
    "START_MODEL",
    "START_MODEL_SHA256",
    "START_REVISION",
    "FINAL_MODEL_SHA256",
    "DATASET_SOURCE",
    "DATASET_REVISION",
    "PROMPT_HASH",
    "RETRIEVAL_HASH",
    "SOURCE_SCHEMA_HASH",
    "EPOCHS",
    "LEARNING_RATE",
    "SAVE_STEPS",
    "EVAL_STEPS",
    "GRADIENT_ACCUMULATION_STEPS",
    "EXPECTED_TRAIN_PAIRS",
    "EXPECTED_EVAL_PAIRS",
    "GPU_TELEMETRY",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_full_sft_manifest(output: Path, environment: Mapping[str, str]) -> dict[str, object]:
    missing = [name for name in REQUIRED_ENVIRONMENT if not environment.get(name)]
    if missing:
        raise ValueError(f"full SFT manifest environment is missing {missing[0]}")
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"full SFT manifest already exists: {manifest_path}")
    final_model = output / "final" / "model.safetensors"
    if not final_model.is_file():
        raise FileNotFoundError(f"full SFT final model is missing: {final_model}")
    if file_sha256(final_model) != environment["FINAL_MODEL_SHA256"]:
        raise ValueError("full SFT final model hash mismatch")
    telemetry = Path(environment["GPU_TELEMETRY"])
    if not telemetry.is_file() or telemetry.stat().st_size == 0:
        raise FileNotFoundError(f"full SFT GPU telemetry is missing or empty: {telemetry}")

    checkpoints = []
    for step, checkpoint in retained_training_checkpoints(output):
        model = checkpoint / "model.safetensors"
        state = checkpoint / "trainer_state.json"
        if not model.is_file() or not state.is_file():
            raise ValueError(f"checkpoint is incomplete: {checkpoint}")
        state_value = json.loads(state.read_text(encoding="utf-8"))
        checkpoints.append({
            "path": str(checkpoint),
            "step": step,
            "model_sha256": file_sha256(model),
            "trainer_state_sha256": file_sha256(state),
            "log_history": state_value.get("log_history", []),
        })
    if not checkpoints:
        raise ValueError("full SFT produced no retained numeric checkpoints")
    eval_losses = [
        entry["eval_loss"]
        for checkpoint in checkpoints
        for entry in checkpoint["log_history"]
        if isinstance(entry, dict) and isinstance(entry.get("eval_loss"), (int, float))
    ]
    manifest: dict[str, object] = {
        "status": "complete",
        "stage": "verified_student_no_hint_full_sft",
        "commit_hash": environment["EXPECTED_COMMIT"],
        "config_sha256": environment["CONFIG_SHA256"],
        "train_sha256": environment["TRAIN_SHA256"],
        "eval_sha256": environment["EVAL_SHA256"],
        "split_report_sha256": environment["SPLIT_REPORT_SHA256"],
        "model": {
            "initial_path": environment["START_MODEL"],
            "initial_model_sha256": environment["START_MODEL_SHA256"],
            "revision": environment["START_REVISION"],
            "final_model_sha256": environment["FINAL_MODEL_SHA256"],
            "full_finetuning": True,
            "optimizer_state": "fresh",
        },
        "dataset": {"source": environment["DATASET_SOURCE"], "revision": environment["DATASET_REVISION"]},
        "prompt_hash": environment["PROMPT_HASH"],
        "retrieval_hash": environment["RETRIEVAL_HASH"],
        "source_schema_hash": environment["SOURCE_SCHEMA_HASH"],
        "training": {
            "bf16": True,
            "tf32": True,
            "deepspeed": "zero3",
            "max_length": 4096,
            "max_steps": -1,
            "epochs": float(environment["EPOCHS"]),
            "learning_rate": float(environment["LEARNING_RATE"]),
            "save_steps": int(environment["SAVE_STEPS"]),
            "eval_steps": int(environment["EVAL_STEPS"]),
            "gradient_accumulation_steps": int(environment["GRADIENT_ACCUMULATION_STEPS"]),
            "effective_batch_size": 4 * int(environment["GRADIENT_ACCUMULATION_STEPS"]),
            "train_pairs": int(environment["EXPECTED_TRAIN_PAIRS"]),
            "eval_pairs": int(environment["EXPECTED_EVAL_PAIRS"]),
        },
        "retained_checkpoints": checkpoints,
        "best_retained_eval_loss": min(eval_losses) if eval_losses else None,
        "gpu_telemetry": str(telemetry),
        "gpu_telemetry_sha256": file_sha256(telemetry),
        "node": socket.gethostname(),
        "platform": platform.platform(),
        "fallback_reason": "none",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a completed full-SFT run without rerunning training")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_full_sft_manifest(args.output, os.environ)
    print(
        "event=full_sft_manifest_complete "
        f"manifest={args.output / 'run-manifest.json'} "
        f"retained_checkpoints={len(manifest['retained_checkpoints'])} fallback_reason=none"
    )


if __name__ == "__main__":
    main()
