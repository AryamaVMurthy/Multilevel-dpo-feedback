from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable


def _require_manifest_value(key: str, value: object) -> None:
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"cache manifest field must be explicit and non-empty: {key}")


def student_policy_identity(*, student_model: str, student_revision: str, policy_version: str) -> dict:
    fields = {
        "student_model": student_model,
        "student_revision": student_revision,
        "policy_version": policy_version,
    }
    for field, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"student policy identity requires pinned non-empty {field}")
    identity = {"identity": "student-policy-v1", **fields}
    return {**identity, "sha256": _identity_hash(identity)}


def build_cache_manifest(
    *,
    student_model: str,
    student_revision: str,
    teacher_model: str,
    teacher_revision: str,
    teacher_identity: str,
    teacher_quantization: str,
    teacher_fallback_reason: str | None,
    dataset_revision: str,
    dataset_hash: str,
    dataset_schema: str,
    source_schema_version: int,
    source_schema_hash: str,
    retrieval_config: dict,
    retrieval_hash: str,
    prompt_version: str,
    prompt_hash: str,
    response_schema_version: int,
    response_schema_hash: str,
    evaluator_version: str,
    evaluator_hash: str,
    policy_version: str,
    student_thinking_mode: str,
    teacher_thinking: bool,
    decoding: dict,
    intervention_policy: dict,
    sibling_count: int,
    sibling_seeds: list[int],
    seed: int,
    policy_hash: str,
) -> dict:
    from text_feedback_dpo.batch_generation import (
        EVALUATOR_VERSION, FIXED_B, FIXED_K1, FIXED_TOP_K, PROMPT_VERSION, RESPONSE_SCHEMA_VERSION,
    )
    from text_feedback_dpo.runtime import validate_teacher_identity
    from text_feedback_dpo.prompts import prompt_builder_identity
    from text_feedback_dpo.searchqa import SOURCE_SCHEMA, SOURCE_SCHEMA_VERSION

    actual_teacher_identity = validate_teacher_identity(
        teacher_model, revision=teacher_revision, quantization=teacher_quantization,
        fallback_reason=teacher_fallback_reason,
    )
    if teacher_identity != actual_teacher_identity:
        raise ValueError("teacher model/identity/4bit mapping mismatch")
    for field, value in (("student_model", student_model), ("student_revision", student_revision),
                         ("teacher_revision", teacher_revision), ("dataset_revision", dataset_revision),
                         ("policy_version", policy_version)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"cache manifest requires pinned non-empty {field}")
    policy_identity = student_policy_identity(
        student_model=student_model,
        student_revision=student_revision,
        policy_version=policy_version,
    )
    hashes = {
        "dataset_hash": dataset_hash, "source_schema_hash": source_schema_hash,
        "retrieval_hash": retrieval_hash, "prompt_hash": prompt_hash,
        "response_schema_hash": response_schema_hash, "evaluator_hash": evaluator_hash,
        "policy_hash": policy_hash,
    }
    for field, value in hashes.items():
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"cache manifest {field} must be a lowercase SHA-256")
    if policy_hash != policy_identity["sha256"]:
        raise ValueError("cache manifest policy_hash identity mismatch")
    expected_retrieval = {
        "identity": "fixed_bm25", "schema_version": 1,
        "requested_top_k": FIXED_TOP_K, "k1": FIXED_K1, "b": FIXED_B,
    }
    prompt_builders = prompt_builder_identity()
    expected_identities = {
        "source_schema_hash": _identity_hash({"identity": SOURCE_SCHEMA, "version": SOURCE_SCHEMA_VERSION}),
        "retrieval_hash": _identity_hash(expected_retrieval),
        "prompt_hash": _identity_hash({"identity": PROMPT_VERSION, "builders": prompt_builders}),
        "response_schema_hash": _identity_hash({"identity": "cited-response", "schema_version": RESPONSE_SCHEMA_VERSION}),
        "evaluator_hash": _identity_hash({"identity": EVALUATOR_VERSION}),
    }
    if dataset_schema != SOURCE_SCHEMA or source_schema_version != SOURCE_SCHEMA_VERSION:
        raise ValueError("cache manifest dataset/source schema identity mismatch")
    if retrieval_config != expected_retrieval:
        raise ValueError("cache manifest retrieval configuration mismatch")
    if prompt_version != PROMPT_VERSION or response_schema_version != RESPONSE_SCHEMA_VERSION or evaluator_version != EVALUATOR_VERSION:
        raise ValueError("cache manifest prompt/response/evaluator identity mismatch")
    for field, expected in expected_identities.items():
        if hashes[field] != expected:
            raise ValueError(f"cache manifest {field} identity mismatch")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("cache manifest seed must be a nonnegative integer")
    manifest = {
        "manifest_version": 5,
        "schema_version": 5,
        "student_model": student_model,
        "student_revision": student_revision,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "teacher_identity": teacher_identity,
        "teacher_quantization": teacher_quantization,
        "teacher_fallback_reason": teacher_fallback_reason,
        "dataset_revision": dataset_revision,
        "dataset_hash": dataset_hash,
        "dataset_schema": dataset_schema,
        "source_schema_version": source_schema_version,
        "source_schema_hash": source_schema_hash,
        "retrieval_config": retrieval_config,
        "retrieval_hash": retrieval_hash,
        "prompt_version": prompt_version,
        "prompt_builders": prompt_builders,
        "prompt_hash": prompt_hash,
        "response_schema_version": response_schema_version,
        "response_schema_hash": response_schema_hash,
        "evaluator_version": evaluator_version,
        "evaluator_hash": evaluator_hash,
        "policy_version": policy_version,
        "policy_identity": policy_identity,
        "student_thinking_mode": student_thinking_mode,
        "teacher_thinking": teacher_thinking,
        "decoding": decoding,
        "intervention_policy": intervention_policy,
        "sibling_count": sibling_count,
        "sibling_seeds": sibling_seeds,
        "seed": seed,
        "policy_hash": policy_hash,
    }
    for key, value in manifest.items():
        if key != "teacher_fallback_reason":
            _require_manifest_value(key, value)
    if isinstance(source_schema_version, bool) or not isinstance(source_schema_version, int):
        raise ValueError("source_schema_version must be an integer")
    if isinstance(response_schema_version, bool) or not isinstance(response_schema_version, int):
        raise ValueError("response_schema_version must be an integer")
    if not isinstance(teacher_thinking, bool):
        raise ValueError("teacher_thinking must be boolean")
    if teacher_identity == "primary_qwen3_32b_4bit" and teacher_fallback_reason is not None:
        raise ValueError("primary teacher cache identity cannot contain a fallback reason")
    if teacher_identity == "fallback_qwen3_14b_4bit" and (not isinstance(teacher_fallback_reason, str) or not teacher_fallback_reason.strip()):
        raise ValueError("fallback teacher cache identity requires a documented fallback reason")
    if isinstance(sibling_count, bool) or not isinstance(sibling_count, int) or sibling_count <= 0:
        raise ValueError("sibling_count must be a positive integer")
    if not isinstance(sibling_seeds, list) or any(isinstance(seed_value, bool) or not isinstance(seed_value, int) for seed_value in sibling_seeds):
        raise ValueError("sibling_seeds must be an explicit list of integer seeds")
    if len(sibling_seeds) != sibling_count:
        raise ValueError("sibling_count must exactly match sibling_seeds")
    if any(seed_value < 0 for seed_value in sibling_seeds):
        raise ValueError("sibling_seeds must be nonnegative")
    if len(set(sibling_seeds)) != len(sibling_seeds):
        raise ValueError("sibling_seeds must be unique")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {**manifest, "cache_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".manifest.json")


def _identity_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_existing_manifest(cache_path: Path, expected: dict) -> None:
    path = _manifest_path(cache_path)
    if not path.exists():
        raise ValueError(f"cache exists without required manifest: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise ValueError(f"cache manifest mismatch in fields: {', '.join(differing)}")


def _row_identity(example: dict, cache_manifest: dict) -> str:
    explicit = example.get("row_identity")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError(f"row {example.get('id')} has invalid explicit row_identity")
        return explicit
    payload = json.dumps(
        {"example": example, "source_schema_hash": cache_manifest["source_schema_hash"], "policy_hash": cache_manifest["policy_hash"]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_cache(cache_path: Path, rows: list[dict], manifest: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    _manifest_path(cache_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_cache_objects(cache_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(cache_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"cache JSONL contains a blank record: {cache_path.name}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cache JSONL contains invalid JSON: {cache_path.name}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"cache JSONL record must be an object: {cache_path.name}:{line_number}")
        rows.append(row)
    return rows


def load_or_build_rollouts(*, examples: list[dict], cache_path: Path, cache_manifest: dict, generate: Callable[[dict], dict]) -> list[dict]:
    expected_identity = {str(example["id"]): _row_identity(example, cache_manifest) for example in examples}
    cached: dict[str, dict] = {}
    if cache_path.exists():
        _verify_existing_manifest(cache_path, cache_manifest)
        for row in _read_cache_objects(cache_path):
            example_id = row.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("cached rollout missing example_id")
            if example_id not in expected_identity:
                raise ValueError(f"cached rollout has unexpected example_id: {example_id}")
            if row.get("cache_hash") != cache_manifest["cache_hash"]:
                raise ValueError(f"cached rollout cache_hash mismatch for {example_id}")
            if row.get("row_identity") != expected_identity[example_id]:
                raise ValueError(f"cached rollout row_identity mismatch for {example_id}")
            if example_id in cached:
                raise ValueError(f"duplicate cached rollout: {example_id}")
            cached[example_id] = row
    rows = []
    for example in examples:
        example_id = str(example["id"])
        if example_id in cached:
            rows.append(cached[example_id])
            continue
        generated = generate(example)
        if not isinstance(generated, dict):
            raise ValueError(f"generator returned a non-mapping rollout for {example_id}")
        rows.append({"example_id": example_id, "row_identity": expected_identity[example_id], "cache_hash": cache_manifest["cache_hash"], **generated})
    _write_cache(cache_path, rows, cache_manifest)
    return rows


def load_or_build_trajectories(*, examples: list[dict], cache_path: Path, cache_manifest: dict, generate: Callable[[list[dict]], list[dict]]) -> list[dict]:
    from text_feedback_dpo.retrieval import validate_source_records
    from text_feedback_dpo.trajectories import revalidate_cached_trajectory

    for example in examples:
        try:
            validate_source_records(example.get("sources"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"trajectory example {example.get('id')} has invalid canonical sources: {exc}") from exc
    expected_ids = [str(example["id"]) for example in examples]
    example_by_id = {str(example["id"]): example for example in examples}
    expected_identity = {str(example["id"]): _row_identity(example, cache_manifest) for example in examples}
    cached: dict[str, dict] = {}
    if cache_path.exists():
        _verify_existing_manifest(cache_path, cache_manifest)
        for row in _read_cache_objects(cache_path):
            row_id = row.get("id")
            if str(row_id) not in expected_identity:
                raise ValueError(f"cached trajectory has unexpected id: {row_id}")
            if row.get("cache_hash") != cache_manifest["cache_hash"]:
                raise ValueError(f"cached trajectory cache_hash mismatch for {row_id}")
            if row.get("row_identity") != expected_identity[str(row_id)]:
                raise ValueError(f"cached trajectory row_identity mismatch for {row_id}")
            if str(row_id) in cached:
                raise ValueError(f"duplicate cached trajectory: {row_id}")
            cached[str(row_id)] = revalidate_cached_trajectory(
                row, example=example_by_id[str(row_id)],
                expected_sibling_seeds=cache_manifest["sibling_seeds"],
            )
    missing = [example for example in examples if str(example["id"]) not in cached]
    if missing:
        generated = generate(missing)
        if not isinstance(generated, list) or len(generated) != len(missing):
            raise ValueError(f"trajectory generator cardinality mismatch: expected {len(missing)}, got {len(generated) if isinstance(generated, list) else type(generated).__name__}")
        missing_ids = {str(example["id"]) for example in missing}
        for row in generated:
            if not isinstance(row, dict) or str(row.get("id")) not in missing_ids:
                raise ValueError(f"trajectory generator returned unexpected id: {row.get('id') if isinstance(row, dict) else type(row).__name__}")
            row_id = str(row["id"])
            if row_id in cached:
                raise ValueError(f"duplicate generated trajectory: {row_id}")
            forbidden_wrapper_fields = [field for field in ("cache_hash", "row_identity") if field in row]
            if forbidden_wrapper_fields:
                raise ValueError(
                    f"generated trajectory {row_id} must not provide {forbidden_wrapper_fields[0]}"
                )
            wrapped = {
                **row,
                "cache_hash": cache_manifest["cache_hash"],
                "row_identity": expected_identity[row_id],
            }
            cached[row_id] = revalidate_cached_trajectory(
                wrapped, example=example_by_id[row_id],
                expected_sibling_seeds=cache_manifest["sibling_seeds"],
            )
    rows = [cached[example_id] for example_id in expected_ids]
    _write_cache(cache_path, rows, cache_manifest)
    return rows
