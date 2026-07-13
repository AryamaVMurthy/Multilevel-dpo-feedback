from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Callable


def build_cache_manifest(
    *, student_model: str, student_revision: str, teacher_model: str,
    teacher_revision: str, dataset_revision: str, prompt_version: str,
    student_thinking_mode: str, teacher_thinking: bool, decoding: dict,
    intervention_policy: dict, seed: int, policy_hash: str,
) -> dict:
    manifest = {
        "schema_version": 1,
        "student_model": student_model,
        "student_revision": student_revision,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "dataset_revision": dataset_revision,
        "prompt_version": prompt_version,
        "student_thinking_mode": student_thinking_mode,
        "teacher_thinking": teacher_thinking,
        "decoding": decoding,
        "intervention_policy": intervention_policy,
        "seed": seed,
        "policy_hash": policy_hash,
    }
    for key, value in manifest.items():
        if value is None or value == "" or value == {}:
            raise ValueError(f"cache manifest field must be explicit and non-empty: {key}")
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


def _write_cache(cache_path: Path, rows: list[dict], manifest: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    _manifest_path(cache_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_or_build_rollouts(*, examples: list[dict], cache_path: Path, cache_manifest: dict, generate: Callable[[dict], dict]) -> list[dict]:
    cached: dict[str, dict] = {}
    if cache_path.exists():
        _verify_existing_manifest(cache_path, cache_manifest)
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = row.get("example_id")
            if not example_id:
                raise ValueError("cached rollout missing example_id")
            if row.get("cache_hash") != cache_manifest["cache_hash"]:
                raise ValueError(f"cached rollout cache_hash mismatch for {example_id}")
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
        if not isinstance(generated, dict) or not generated.get("response"):
            raise ValueError(f"generator returned no response for {example_id}")
        rows.append({"example_id": example_id, "cache_hash": cache_manifest["cache_hash"], **generated})
    _write_cache(cache_path, rows, cache_manifest)
    return rows


def load_or_build_trajectories(*, examples: list[dict], cache_path: Path, cache_manifest: dict, generate: Callable[[list[dict]], list[dict]]) -> list[dict]:
    cached: dict[str, dict] = {}
    if cache_path.exists():
        _verify_existing_manifest(cache_path, cache_manifest)
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cache_hash") != cache_manifest["cache_hash"]:
                raise ValueError(f"cached trajectory cache_hash mismatch for {row.get('id')}")
            if row.get("id") in cached:
                raise ValueError(f"duplicate cached trajectory: {row.get('id')}")
            cached[str(row["id"])] = row
    missing = [example for example in examples if str(example["id"]) not in cached]
    if missing:
        generated = generate(missing)
        if len(generated) != len(missing):
            raise ValueError(f"trajectory generator cardinality mismatch: expected {len(missing)}, got {len(generated)}")
        for row in generated:
            if str(row.get("id")) not in {str(example["id"]) for example in missing}:
                raise ValueError(f"trajectory generator returned unexpected id: {row.get('id')}")
            cached[str(row["id"])] = {"cache_hash": cache_manifest["cache_hash"], **row}
    rows = [cached[str(example["id"])] for example in examples]
    _write_cache(cache_path, rows, cache_manifest)
    return rows
