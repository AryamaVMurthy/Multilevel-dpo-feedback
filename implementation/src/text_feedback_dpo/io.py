from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def parse_jsonl_record(line: str, *, path: Path, line_number: int) -> dict:
    """Parse one required JSON-object record with exact file/line diagnostics."""
    if not line.strip():
        raise ValueError(f"blank JSONL record at {path}:{line_number}")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL record at {path}:{line_number} must be a JSON object")
    return value


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"required JSONL file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            yield parse_jsonl_record(line, path=path, line_number=line_number)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
