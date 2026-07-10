from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping


EXPECTED_OFFICIAL_COUNTS = {"train": 99820, "validation": 13393, "test": 27248}


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_from_name(name: str) -> str | None:
    normalized = name.lower().replace("-", "_")
    matches = []
    if re.search(r"(^|[_/.])train([_/.]|$)", normalized):
        matches.append("train")
    if re.search(r"(^|[_/.])(validation|valid|dev)([_/.]|$)", normalized):
        matches.append("validation")
    if re.search(r"(^|[_/.])test([_/.]|$)", normalized):
        matches.append("test")
    if len(set(matches)) > 1:
        return None
    return matches[0] if matches else None


def _decode_payload(raw: bytes, name: str) -> Any:
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in SearchQA artifact {name}:{line_number}: {exc}") from exc
        if not rows:
            raise ValueError(f"SearchQA artifact {name} is not a JSON object, array, or JSONL file")
        return rows


def _split_payload(payload: Any) -> dict[str, list[dict[str, Any]]] | None:
    if not isinstance(payload, dict):
        return None
    aliases = {"train": "train", "validation": "validation", "valid": "validation", "dev": "validation", "test": "test"}
    output: dict[str, list[dict[str, Any]]] = {}
    for key, role in aliases.items():
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"SearchQA combined payload field {key} must be a list of objects")
        output[role] = list(value)
    return output or None


def _rows_from_payload(payload: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return list(payload)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
        if all(isinstance(row, dict) for row in rows):
            return list(rows)
    raise ValueError(f"SearchQA artifact {name} does not contain a list of row objects")


def _flatten_evidence(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        for key in ("snippets", "snippet", "text", "content"):
            if key in value:
                return _flatten_evidence(value[key])
        return []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_evidence(item))
        return [item for item in output if item]
    return []


def convert_original_searchqa_row(row: Mapping[str, Any], *, split: str, index: int) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answer_value = row.get("answer", row.get("gold_answer", row.get("answers")))
    if isinstance(answer_value, list):
        answer_value = answer_value[0] if answer_value else ""
    answer = str(answer_value or "").strip()
    if not question or not answer:
        raise ValueError(f"SearchQA {split} row {index} is missing question or answer")
    evidence = _flatten_evidence(row.get("search_results", row.get("evidence")))
    if not evidence:
        raise ValueError(f"SearchQA {split} row {index} is missing controlled evidence")
    source_id = str(row.get("id", row.get("qid", index)))
    aliases_value = row.get("answer_aliases", row.get("answers", [answer]))
    if isinstance(aliases_value, str):
        aliases = [aliases_value]
    elif isinstance(aliases_value, list) and all(str(item).strip() for item in aliases_value):
        aliases = [str(item).strip() for item in aliases_value]
    else:
        raise ValueError(f"SearchQA {split} row {index} has invalid answer aliases")
    return {
        "id": f"searchqa-{split}-{source_id}",
        "domain": "search_qa",
        "problem": question,
        "gold_answer": answer,
        "answers": aliases,
        "answer_aliases": aliases,
        "answer_type": str(row.get("answer_type", "unknown")),
        "evidence": evidence,
        "context": "\n".join(evidence),
        "source": "nyu-dl/SearchQA",
        "source_key": f"{split}:{source_id}",
        "source_metadata": {
            "category": row.get("category"),
            "air_date": row.get("air_date"),
            "value": row.get("value"),
            "round": row.get("round"),
            "show_number": row.get("show_number"),
        },
        "source_split": split,
        "source_index": index,
    }


def _read_source(path: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if path.is_file() and zipfile.is_zipfile(path):
        artifact_hash = artifact_sha256(path)
        named_payloads: list[tuple[str, Any]] = []
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/") and name.lower().endswith((".json", ".jsonl"))]
            if not members:
                raise ValueError(f"SearchQA archive contains no JSON or JSONL files: {path}")
            for name in sorted(members):
                named_payloads.append((name, _decode_payload(archive.read(name), name)))
    elif path.is_dir():
        files = sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in {".json", ".jsonl"})
        if not files:
            raise ValueError(f"SearchQA directory contains no JSON or JSONL files: {path}")
        digest = hashlib.sha256()
        named_payloads = []
        for file in files:
            raw = file.read_bytes()
            digest.update(str(file.relative_to(path)).encode("utf-8"))
            digest.update(hashlib.sha256(raw).digest())
            named_payloads.append((str(file.relative_to(path)), _decode_payload(raw, str(file))))
        artifact_hash = digest.hexdigest()
    else:
        raise FileNotFoundError(f"SearchQA source artifact does not exist: {path}")

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for name, payload in named_payloads:
        combined = _split_payload(payload)
        if combined is not None:
            for split, rows in combined.items():
                split_rows[split].extend(rows)
            continue
        split = _split_from_name(name)
        if split is None:
            raise ValueError(f"cannot infer official SearchQA split from artifact name: {name}")
        split_rows[split].extend(_rows_from_payload(payload, name))
    missing = [split for split, rows in split_rows.items() if not rows]
    if missing:
        raise ValueError(f"SearchQA source is missing official splits: {', '.join(missing)}")
    return split_rows, artifact_hash


def load_original_searchqa(
    path: Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    raw_splits, artifact_hash = _read_source(path)
    expected = dict(expected_counts or {})
    for split, count in expected.items():
        if split not in raw_splits or len(raw_splits[split]) != int(count):
            observed = len(raw_splits.get(split, []))
            raise ValueError(f"SearchQA {split} count mismatch: expected {count}, observed {observed}")
    converted = {
        split: [convert_original_searchqa_row(row, split=split, index=index) for index, row in enumerate(rows)]
        for split, rows in raw_splits.items()
    }
    return {
        "source": "nyu-dl/SearchQA",
        "artifact_sha256": artifact_hash,
        "splits": converted,
    }
