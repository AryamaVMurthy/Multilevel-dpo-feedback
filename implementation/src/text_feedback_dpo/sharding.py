from __future__ import annotations

import hashlib
import json
import os
from itertools import zip_longest
from pathlib import Path


def shard_jsonl(source: Path, output_dir: Path, *, shard_count: int) -> dict:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"shard input does not exist: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"shard-{index}.jsonl" for index in range(shard_count)]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise ValueError(f"refusing to overwrite existing shard: {existing[0]}")
    handles = [path.open("w", encoding="utf-8") for path in paths]
    counts = [0] * shard_count
    seen = set()
    try:
        with source.open(encoding="utf-8") as input_handle:
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                example_id = str(row.get("id", ""))
                if not example_id or example_id in seen:
                    raise ValueError(f"input requires unique non-empty id at line {line_number}: {example_id!r}")
                seen.add(example_id)
                shard_index = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest(), 16) % shard_count
                handles[shard_index].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[shard_index] += 1
    finally:
        for handle in handles:
            handle.close()
    manifest = {"source": str(source), "shards": shard_count, "rows": len(seen), "shard_rows": counts, "assignment": "sha256(id) modulo shard_count"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def merge_prediction_shards(shard_dir: Path, output: Path, *, shard_count: int) -> dict:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary merge output already exists: {temporary}")
    seen = set()
    row_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for shard_index in range(shard_count):
                source_path = shard_dir / f"shard-{shard_index}.jsonl"
                prediction_path = shard_dir / f"predictions-{shard_index}.jsonl"
                if not source_path.is_file() or not prediction_path.is_file():
                    raise FileNotFoundError(f"missing shard pair for index {shard_index}")
                with source_path.open(encoding="utf-8") as source, prediction_path.open(encoding="utf-8") as predictions:
                    for line_number, pair in enumerate(zip_longest(source, predictions), start=1):
                        source_line, prediction_line = pair
                        if source_line is None or prediction_line is None:
                            raise ValueError(f"cardinality mismatch in shard {shard_index} at line {line_number}")
                        source_row = json.loads(source_line)
                        prediction_row = json.loads(prediction_line)
                        source_id = str(source_row.get("id", ""))
                        prediction_id = str(prediction_row.get("id", ""))
                        if source_id != prediction_id:
                            raise ValueError(f"ID mismatch in shard {shard_index} at line {line_number}: {source_id} != {prediction_id}")
                        if prediction_id in seen:
                            raise ValueError(f"duplicate prediction ID across shards: {prediction_id}")
                        seen.add(prediction_id)
                        destination.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")
                        row_count += 1
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"rows": row_count, "shards": shard_count, "output": str(output)}
