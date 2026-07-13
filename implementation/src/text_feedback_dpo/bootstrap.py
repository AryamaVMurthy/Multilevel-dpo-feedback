from __future__ import annotations

import hashlib
import heapq
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from text_feedback_dpo.trajectories import _structured_hash, validate_active_artifact


class _ReverseSelectionKey:
    __slots__ = ("value",)

    def __init__(self, value: tuple[int, str]) -> None:
        self.value = value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseSelectionKey):
            return NotImplemented
        return self.value > other.value


def select_bootstrap_pool(
    examples: Iterable[Mapping[str, object]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    """Select the lowest stable ID hashes while retaining only ``count`` large rows."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("bootstrap pool count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap pool seed must be a nonnegative integer")
    heap: list[tuple[_ReverseSelectionKey, dict[str, object]]] = []
    seen: set[str] = set()
    total = 0
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError(f"bootstrap pool row {total} must be a mapping")
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id or example_id in seen:
            raise ValueError(f"bootstrap pool has invalid or duplicate id at row {total}: {example_id!r}")
        seen.add(example_id)
        total += 1
        score = int.from_bytes(hashlib.sha256(f"{seed}|{example_id}".encode()).digest(), "big")
        entry = (_ReverseSelectionKey((score, example_id)), dict(row))
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[0].value < heap[0][0].value:
            heapq.heapreplace(heap, entry)
    if total < count:
        raise ValueError(f"requested {count} bootstrap examples but input contains only {total}")
    return [row for _key, row in sorted(heap, key=lambda item: item[0].value)]


def _validate_inputs(examples: Sequence[Mapping[str, object]], seeds: Sequence[int]) -> list[str]:
    if not examples:
        raise ValueError("bootstrap collection requires at least one example")
    ids: list[str] = []
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("bootstrap examples require non-empty string ids")
        if example_id in ids:
            raise ValueError(f"duplicate example id: {example_id}")
        ids.append(example_id)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("bootstrap seeds must be a non-empty sequence of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("bootstrap seeds must be unique")
    return ids


def collect_bootstrap_rollouts(
    examples: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int],
    generate_seed_batch: Callable[..., list[object]],
) -> list[dict[str, object]]:
    """Collect canonical no-hint student candidates with one batched call per seed."""
    ids = _validate_inputs(examples, seeds)
    candidates_by_id: dict[str, list[dict[str, object]]] = {example_id: [] for example_id in ids}
    for seed in seeds:
        outputs = generate_seed_batch(list(examples), seed=seed)
        if not isinstance(outputs, list) or len(outputs) != len(examples):
            actual = len(outputs) if isinstance(outputs, list) else "non-list"
            raise ValueError(
                f"bootstrap generation cardinality mismatch for seed {seed}: "
                f"expected {len(examples)}, got {actual}"
            )
        for example, output in zip(examples, outputs, strict=True):
            artifact = validate_active_artifact(output, example=example, hints=[])
            example_id = str(example["id"])
            candidates_by_id[example_id].append({
                "seed": seed,
                "provenance": "student",
                "no_hint": True,
                "artifact": artifact,
                "artifact_hash": _structured_hash(artifact),
            })
    rows = [
        {
            "id": example_id,
            "example_identity": _structured_hash(example),
            "seeds": list(seeds),
            "candidate_count": len(seeds),
            "candidates": candidates_by_id[example_id],
        }
        for example_id, example in zip(ids, examples, strict=True)
    ]
    return validate_bootstrap_rows(rows, examples=examples, expected_seeds=seeds)


def validate_bootstrap_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    examples: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    ids = _validate_inputs(examples, expected_seeds)
    if len(rows) != len(examples):
        raise ValueError(f"bootstrap row cardinality mismatch: expected {len(examples)}, got {len(rows)}")
    validated: list[dict[str, Any]] = []
    for expected_id, example, row in zip(ids, examples, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("id") != expected_id:
            raise ValueError(f"bootstrap row order/ID mismatch for {expected_id}")
        if row.get("example_identity") != _structured_hash(example):
            raise ValueError(f"bootstrap example identity mismatch for {expected_id}")
        if row.get("seeds") != list(expected_seeds) or row.get("candidate_count") != len(expected_seeds):
            raise ValueError(f"bootstrap seed identity mismatch for {expected_id}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(expected_seeds):
            raise ValueError(f"bootstrap candidate cardinality mismatch for {expected_id}")
        normalized_candidates: list[dict[str, object]] = []
        for seed, candidate in zip(expected_seeds, candidates, strict=True):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"bootstrap candidate for {expected_id} must be a mapping")
            if candidate.get("seed") != seed:
                raise ValueError(f"bootstrap candidate seed mismatch for {expected_id}")
            if candidate.get("provenance") != "student" or candidate.get("no_hint") is not True:
                raise ValueError(f"bootstrap candidate provenance mismatch for {expected_id} seed {seed}")
            if any("teacher" in str(key).lower() for key in candidate):
                raise ValueError(f"bootstrap candidate contains forbidden teacher field for {expected_id} seed {seed}")
            artifact = validate_active_artifact(candidate.get("artifact"), example=example, hints=[])
            if candidate.get("artifact_hash") != _structured_hash(artifact):
                raise ValueError(f"bootstrap artifact hash mismatch for {expected_id} seed {seed}")
            normalized_candidates.append({**dict(candidate), "artifact": artifact})
        validated.append({**dict(row), "candidates": normalized_candidates})
    return validated
