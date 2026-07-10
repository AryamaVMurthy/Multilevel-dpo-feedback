from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from text_feedback_dpo.io import read_jsonl_zst, write_json_atomic


def shard_rows(rows: Sequence[Mapping[str, Any]], *, shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    """Partition rows contiguously with deterministic balanced boundaries."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    copied = [dict(row) for row in rows]
    ids = [row.get("id") for row in copied]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise ValueError("every row must have a non-empty string id before sharding")
    if len(set(ids)) != len(ids):
        raise ValueError("row ids must be unique before sharding")
    base, remainder = divmod(len(copied), num_shards)
    start = shard_index * base + min(shard_index, remainder)
    size = base + (1 if shard_index < remainder else 0)
    return copied[start : start + size]


def _read_metadata(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required {label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _context_matches(
    payload: Mapping[str, Any],
    *,
    config_hash: str,
    dataset_manifest_hash: str,
    protocol_hash: str,
    shard_index: int,
    num_shards: int,
) -> None:
    expected = {
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "protocol_hash": protocol_hash,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        labels = [
            "config hash" if key == "config_hash" else "protocol hash" if key == "protocol_hash" else key
            for key in mismatches
        ]
        raise ValueError(f"shard metadata context mismatch in {', '.join(labels)}")


def write_progress(
    path: Path,
    *,
    config_hash: str,
    dataset_manifest_hash: str,
    protocol_hash: str,
    shard_index: int,
    num_shards: int,
    last_completed_local_index: int,
    records_written: int,
) -> None:
    """Advance a shard marker only after its compressed record has been fsynced."""

    if last_completed_local_index < -1:
        raise ValueError("last_completed_local_index must be at least -1")
    if records_written < 0:
        raise ValueError("records_written must be non-negative")
    complete_path = path.parent / "complete.json"
    if complete_path.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {complete_path}")
    if path.exists():
        previous = _read_metadata(path, label="progress marker")
        try:
            _context_matches(
                previous,
                config_hash=config_hash,
                dataset_manifest_hash=dataset_manifest_hash,
                protocol_hash=protocol_hash,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        except ValueError as exc:
            raise ValueError(f"cannot resume shard: {exc}") from exc
        if int(previous.get("last_completed_local_index", -1)) > last_completed_local_index:
            raise ValueError("progress marker cannot move backwards")
        if int(previous.get("records_written", 0)) > records_written:
            raise ValueError("progress record count cannot move backwards")
    write_json_atomic(
        path,
        {
            "config_hash": config_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "protocol_hash": protocol_hash,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "last_completed_local_index": last_completed_local_index,
            "records_written": records_written,
        },
    )


def next_local_index(
    path: Path,
    *,
    config_hash: str,
    dataset_manifest_hash: str,
    protocol_hash: str,
    shard_index: int,
    num_shards: int,
) -> int:
    if not path.exists():
        return 0
    payload = _read_metadata(path, label="progress marker")
    _context_matches(
        payload,
        config_hash=config_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        protocol_hash=protocol_hash,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    return int(payload.get("last_completed_local_index", -1)) + 1


def complete_shard(
    shard_dir: Path,
    *,
    config_hash: str,
    dataset_manifest_hash: str,
    protocol_hash: str,
    shard_index: int,
    num_shards: int,
    expected_records: int,
) -> None:
    if expected_records < 0:
        raise ValueError("expected_records must be non-negative")
    complete_path = shard_dir / "complete.json"
    if complete_path.exists():
        raise FileExistsError(f"shard is already complete: {complete_path}")
    progress = _read_metadata(shard_dir / "progress.json", label="progress marker")
    _context_matches(
        progress,
        config_hash=config_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        protocol_hash=protocol_hash,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if int(progress.get("records_written", -1)) != expected_records:
        raise ValueError("progress marker record count does not match expected shard size")
    expected_last = expected_records - 1
    if int(progress.get("last_completed_local_index", -2)) != expected_last:
        raise ValueError("progress marker does not cover every expected local example")
    records_path = shard_dir / "records.jsonl.zst"
    records = read_jsonl_zst(records_path)
    if len(records) != expected_records:
        raise ValueError("compressed record count does not match expected shard size")
    write_json_atomic(
        complete_path,
        {
            "config_hash": config_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "protocol_hash": protocol_hash,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "expected_records": expected_records,
            "records_sha256": __import__("hashlib").sha256(records_path.read_bytes()).hexdigest(),
        },
    )


def merge_completed_shards(
    root: Path,
    *,
    expected_shards: int,
    config_hash: str,
    dataset_manifest_hash: str,
    protocol_hash: str,
) -> list[dict[str, Any]]:
    if expected_shards <= 0:
        raise ValueError("expected_shards must be positive")
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for shard_index in range(expected_shards):
        shard_dir = root / f"shard-{shard_index:04d}"
        marker = _read_metadata(shard_dir / "complete.json", label="completion marker")
        _context_matches(
            marker,
            config_hash=config_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            protocol_hash=protocol_hash,
            shard_index=shard_index,
            num_shards=expected_shards,
        )
        records = read_jsonl_zst(shard_dir / "records.jsonl.zst")
        expected_records = int(marker.get("expected_records", -1))
        if expected_records < 0 or len(records) != expected_records:
            raise ValueError(f"completed shard {shard_index} record count is invalid")
        for row in records:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"completed shard {shard_index} has a row without id")
            if row_id in seen_ids:
                raise ValueError(f"merged collection contains duplicate id: {row_id}")
            seen_ids.add(row_id)
            merged.append(row)
    return merged
