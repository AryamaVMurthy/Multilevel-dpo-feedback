from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable


def _require_manifest_value(key: str, value: object) -> None:
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"cache manifest field must be explicit and non-empty: {key}")


def build_cache_manifest(
    *,
    student_model: str,
    student_revision: str,
    teacher_model: str,
    teacher_revision: str,
    dataset_revision: str,
    dataset_schema: str,
    source_schema_version: int,
    source_schema_hash: str,
    retrieval_config: dict,
    retrieval_hash: str,
    prompt_version: str,
    response_schema_version: int,
    evaluator_version: str,
    policy_version: str,
    student_thinking_mode: str,
    teacher_thinking: bool,
    decoding: dict,
    intervention_policy: dict,
    sibling_seeds: list[int],
    seed: int,
    policy_hash: str,
) -> dict:
    manifest = {
        "manifest_version": 2,
        "schema_version": 2,
        "student_model": student_model,
        "student_revision": student_revision,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "dataset_revision": dataset_revision,
        "dataset_schema": dataset_schema,
        "source_schema_version": source_schema_version,
        "source_schema_hash": source_schema_hash,
        "retrieval_config": retrieval_config,
        "retrieval_hash": retrieval_hash,
        "prompt_version": prompt_version,
        "response_schema_version": response_schema_version,
        "evaluator_version": evaluator_version,
        "policy_version": policy_version,
        "student_thinking_mode": student_thinking_mode,
        "teacher_thinking": teacher_thinking,
        "decoding": decoding,
        "intervention_policy": intervention_policy,
        "sibling_seeds": sibling_seeds,
        "seed": seed,
        "policy_hash": policy_hash,
    }
    for key, value in manifest.items():
        _require_manifest_value(key, value)
    if isinstance(source_schema_version, bool) or not isinstance(source_schema_version, int):
        raise ValueError("source_schema_version must be an integer")
    if isinstance(response_schema_version, bool) or not isinstance(response_schema_version, int):
        raise ValueError("response_schema_version must be an integer")
    if not isinstance(teacher_thinking, bool):
        raise ValueError("teacher_thinking must be boolean")
    if not isinstance(sibling_seeds, list) or any(isinstance(seed_value, bool) or not isinstance(seed_value, int) for seed_value in sibling_seeds):
        raise ValueError("sibling_seeds must be an explicit list of integer seeds")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {**manifest, "cache_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".manifest.json")


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


def load_or_build_rollouts(*, examples: list[dict], cache_path: Path, cache_manifest: dict, generate: Callable[[dict], dict]) -> list[dict]:
    expected_identity = {str(example["id"]): _row_identity(example, cache_manifest) for example in examples}
    cached: dict[str, dict] = {}
    if cache_path.exists():
        _verify_existing_manifest(cache_path, cache_manifest)
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
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
    expected_ids = [str(example["id"]) for example in examples]
    expected_identity = {str(example["id"]): _row_identity(example, cache_manifest) for example in examples}
    cached: dict[str, dict] = {}
    if cache_path.exists():
        _verify_existing_manifest(cache_path, cache_manifest)
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if str(row_id) not in expected_identity:
                raise ValueError(f"cached trajectory has unexpected id: {row_id}")
            if row.get("cache_hash") != cache_manifest["cache_hash"]:
                raise ValueError(f"cached trajectory cache_hash mismatch for {row_id}")
            if row.get("row_identity") != expected_identity[str(row_id)]:
                raise ValueError(f"cached trajectory row_identity mismatch for {row_id}")
            if str(row_id) in cached:
                raise ValueError(f"duplicate cached trajectory: {row_id}")
            cached[str(row_id)] = row
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
            cached[row_id] = {"cache_hash": cache_manifest["cache_hash"], "row_identity": expected_identity[row_id], **row}
    rows = [cached[example_id] for example_id in expected_ids]
    _write_cache(cache_path, rows, cache_manifest)
    return rows
