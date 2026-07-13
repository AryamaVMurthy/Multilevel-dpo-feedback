#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from text_feedback_dpo.bootstrap import select_bootstrap_pool
from text_feedback_dpo.dataset import write_jsonl


def stream_and_hash(path: Path, digest):
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"blank source row at {path}:{line_number}")
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid source JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"source row must be an object at {path}:{line_number}")
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a deterministic train-only bootstrap pool")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-input-sha256", required=True)
    args = parser.parse_args()
    if len(args.expected_input_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.expected_input_sha256
    ):
        raise ValueError("expected input identity must be a lowercase SHA-256")

    digest = hashlib.sha256()
    selected = select_bootstrap_pool(
        stream_and_hash(args.input, digest), count=args.count, seed=args.seed
    )
    actual_hash = digest.hexdigest()
    if actual_hash != args.expected_input_sha256:
        raise ValueError(
            f"input SHA-256 mismatch: expected {args.expected_input_sha256}, got {actual_hash}"
        )
    write_jsonl(selected, args.output)
    selected_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    ids_hash = hashlib.sha256(
        json.dumps([row["id"] for row in selected], separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": str(args.input),
        "source_sha256": actual_hash,
        "selection": "lowest_sha256_seed_pipe_id_v1",
        "seed": args.seed,
        "rows": len(selected),
        "selected_ids_sha256": ids_hash,
        "output_sha256": selected_hash,
        "required_files": [args.output.name],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
