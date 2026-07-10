from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"required JSONL file does not exist: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl_zst(path: Path, row: dict[str, Any]) -> None:
    """Append one independently compressed JSONL frame and fsync it."""

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError("zstandard is required for compressed artifacts") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(zstd.ZstdCompressor().compress(payload))
        handle.flush()
        import os

        os.fsync(handle.fileno())


def read_jsonl_zst(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required compressed JSONL file does not exist: {path}")
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError("zstandard is required for compressed artifacts") from exc
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        with zstd.ZstdDecompressor().stream_reader(handle) as reader:
            buffer = b""
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines.pop()
                for line in lines:
                    if line.strip():
                        rows.append(json.loads(line.decode("utf-8")))
            if buffer.strip():
                rows.append(json.loads(buffer.decode("utf-8")))
    return rows


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a small metadata object atomically and fsync both file and directory."""

    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
