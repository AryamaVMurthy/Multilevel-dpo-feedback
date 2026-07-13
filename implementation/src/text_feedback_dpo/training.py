from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


PRIMARY_STUDENT_MODEL = "Qwen/Qwen3-4B-Base"
PRIMARY_STUDENT_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
FALLBACK_STUDENT_MODEL = "Qwen/Qwen3-1.7B-Base"
MAX_SEQUENCE_LENGTH = 4096


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dataset_identity_hash(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash the exact DPO input rows, excluding only computed ref-log columns."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("reference-log-probability input rows must be a sequence")
    stripped = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("reference-log-probability input rows must be mappings")
        stripped.append({key: value for key, value in row.items() if key not in {"ref_chosen_logps", "ref_rejected_logps"}})
    return _sha256((_canonical_json(stripped) + "\n").encode("utf-8"))


def _installed_trl_version() -> str:
    try:
        return importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("TRL is required to create a reference-log-probability manifest") from exc


def build_reference_manifest(
    *,
    model: str,
    model_revision: str,
    reference_checkpoint_hash: str,
    tokenizer: str,
    tokenizer_revision: str,
    data_hash: str,
    prompt_context_schema: Mapping[str, object],
    max_length: int,
    trl_version: str | None = None,
) -> dict[str, object]:
    if max_length != MAX_SEQUENCE_LENGTH:
        raise ValueError("reference-log-probability max_length must be exactly 4096")
    values = {
        "model": {"id": model, "revision": model_revision},
        "reference_checkpoint_hash": reference_checkpoint_hash,
        "tokenizer": {"id": tokenizer, "revision": tokenizer_revision},
        "data_hash": data_hash,
        "prompt_context_schema": json.loads(_canonical_json(prompt_context_schema)),
        "max_length": max_length,
        "trl_version": trl_version or _installed_trl_version(),
    }
    if not isinstance(model, str) or not model.strip() or not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("reference manifest requires non-empty model identity and revision")
    if not isinstance(tokenizer, str) or not tokenizer.strip() or not isinstance(tokenizer_revision, str) or not tokenizer_revision.strip():
        raise ValueError("reference manifest requires non-empty tokenizer identity and revision")
    for field in ("reference_checkpoint_hash", "data_hash"):
        if not isinstance(values[field], str) or len(values[field]) != 64 or any(character not in "0123456789abcdef" for character in values[field]):
            raise ValueError(f"reference manifest {field} must be a lowercase SHA-256")
    if not isinstance(prompt_context_schema, Mapping) or not prompt_context_schema:
        raise ValueError("reference manifest prompt_context_schema must be a non-empty mapping")
    return values


_IDENTITY_MANIFEST_KEYS = (
    "model", "reference_checkpoint_hash", "tokenizer", "data_hash", "prompt_context_schema", "max_length", "trl_version",
)


def validate_precomputed_reference_manifest(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise TypeError("reference manifests must be mappings")
    for key in _IDENTITY_MANIFEST_KEYS:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"reference manifest mismatch at {key}")


def _validate_ref_row(row: Mapping[str, object], index: int) -> None:
    required = ("id", "prompt", "chosen", "rejected", "ref_chosen_logps", "ref_rejected_logps")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"reference-log-probability row {index} is missing {missing[0]}")
    for key in ("id", "prompt", "chosen", "rejected"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"reference-log-probability row {index} {key} must be non-empty text")
    for key in ("ref_chosen_logps", "ref_rejected_logps"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"reference-log-probability row {index} {key} must be finite")


def write_precomputed_reference_log_probs(path: Path, rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]) -> dict[str, object]:
    rows = list(rows)
    if not rows:
        raise ValueError("reference-log-probability artifact cannot be empty")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"reference-log-probability row {index} must be a mapping")
        _validate_ref_row(row, index)
    if manifest.get("data_hash") != dataset_identity_hash(rows):
        raise ValueError("reference manifest data_hash does not match persisted input rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")
    result = dict(manifest)
    result["rows"] = len(rows)
    result["artifact_sha256"] = _sha256(path.read_bytes())
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
    return result


def load_precomputed_reference_log_probs(path: Path, expected_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"precomputed reference-log-probability artifact does not exist: {path}")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("reference-log-probability artifact manifest is missing")
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_precomputed_reference_manifest(actual_manifest, expected_manifest)
    actual_hash = _sha256(path.read_bytes())
    if actual_manifest.get("artifact_sha256") != actual_hash:
        raise ValueError("reference-log-probability artifact hash mismatch")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"reference-log-probability row {index} must be a mapping")
        _validate_ref_row(row, index)
        rows.append(row)
    if not rows or actual_manifest.get("rows") != len(rows):
        raise ValueError("reference-log-probability artifact row count mismatch")
    if actual_manifest.get("data_hash") != dataset_identity_hash(rows):
        raise ValueError("reference-log-probability data hash mismatch")
    return rows


def _validate_steps(max_steps: int) -> int:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or (max_steps == 0 or max_steps < -1):
        raise ValueError("max_steps must be -1 for an explicitly selected full run or a positive smoke bound")
    return max_steps


def build_method_config(method: str, *, max_length: int, max_steps: int | None = None) -> dict:
    if method not in {"sft", "dpo", "grpo", "dapo"}:
        raise ValueError(f"unknown method: {method}")
    if max_length != MAX_SEQUENCE_LENGTH:
        raise ValueError("max_length must remain exactly 4096")
    if max_steps is None:
        raise ValueError("max_steps must be explicitly selected by launcher or config")
    return {
        "method": method,
        "full_finetuning": True,
        "adapter": None,
        "peft_config": None,
        "max_length": max_length,
        "max_steps": _validate_steps(max_steps),
        "bf16": True,
        "gradient_checkpointing": True,
        "deepspeed_stage": 3,
    }


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"authorized fallback evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("authorized fallback evidence must be a JSON object")
    return value


def validate_student_model_selection(
    config: Mapping[str, object],
    *,
    requested_model: str | None = None,
    requested_revision: str | None = None,
) -> tuple[str, str]:
    configured_model = config.get("student_model")
    configured_revision = config.get("student_revision")
    if configured_model != PRIMARY_STUDENT_MODEL or configured_revision != PRIMARY_STUDENT_REVISION:
        raise ValueError("student model configuration must pin Qwen/Qwen3-4B-Base at the approved revision")
    if (requested_model is None) != (requested_revision is None):
        raise ValueError("model override requires both --model and --model-revision")
    if requested_model is None:
        return PRIMARY_STUDENT_MODEL, PRIMARY_STUDENT_REVISION
    if requested_model == PRIMARY_STUDENT_MODEL and requested_revision == PRIMARY_STUDENT_REVISION:
        return requested_model, requested_revision
    training = config.get("training")
    if not isinstance(training, Mapping) or requested_model != training.get("student_fallback_model") or requested_revision != training.get("student_fallback_revision"):
        raise ValueError("model override is not the pinned primary or configured fallback")
    evidence_path = training.get("student_fallback_oom_artifact")
    intended_hash = training.get("intended_config_hash")
    if not isinstance(evidence_path, (str, Path)) or not isinstance(intended_hash, str) or len(intended_hash) != 64:
        raise ValueError("fallback requires persisted intended-config CUDA OOM evidence and config identity")
    evidence = _read_json_object(Path(evidence_path))
    required = {
        "status": "failed", "error_type": "cuda_oom", "authorized_fallback": True,
        "intended_model": PRIMARY_STUDENT_MODEL, "intended_revision": PRIMARY_STUDENT_REVISION,
        "intended_config_hash": intended_hash,
    }
    if any(evidence.get(key) != value for key, value in required.items()):
        raise ValueError("fallback requires matching persisted intended-config CUDA OOM evidence")
    return str(requested_model), str(requested_revision)
