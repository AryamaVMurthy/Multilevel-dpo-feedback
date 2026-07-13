from __future__ import annotations

import hashlib
import json
import io
import zipfile
from pathlib import Path
from text_feedback_dpo.searchqa import materialize_row, pack_evidence
from text_feedback_dpo.prompts import build_student_prompt


def dataset_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_searchqa_split(source: str, split: str, *, revision: str, limit: int | None = None) -> list[dict]:
    rows, _ = load_searchqa_split_with_stats(source, split, revision=revision, limit=limit)
    return rows


def load_searchqa_split_with_stats(source: str, split: str, *, revision: str, limit: int | None = None) -> tuple[list[dict], dict]:
    if not source:
        raise ValueError("dataset source is required")
    if source == "kyunghyuncho/search_qa":
        rows, stats = _load_official_searchqa_zip(split, revision, limit)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for SearchQA materialization") from exc
        dataset = load_dataset(source, split=split, revision=revision)
        rows = []
        dropped_rows = 0
        for index, raw in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            rows.append(materialize_row(raw, split=split, index=index))
        stats = {"source_rows": len(rows), "materialized_rows": len(rows), "dropped_rows": dropped_rows, "drop_reasons": {}}
    if not rows:
        raise ValueError(f"SearchQA split {split!r} produced zero rows")
    return rows, stats


def _load_official_searchqa_zip(split: str, revision: str, limit: int | None) -> tuple[list[dict], dict]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for the official SearchQA archive") from exc
    filename = {"train": "data/train_test_val/train.zip", "validation": "data/train_test_val/val.zip", "test": "data/train_test_val/test.zip"}.get(split)
    if filename is None:
        raise ValueError("official SearchQA split must be train, validation, or test")
    archive_path = hf_hub_download(repo_id="kyunghyuncho/search_qa", filename=filename, repo_type="dataset", revision=revision)
    rows = []
    source_rows = 0
    dropped_rows = 0
    drop_reasons: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for index, member in enumerate(sorted(name for name in archive.namelist() if name.endswith(".json"))):
            if limit is not None and len(rows) >= limit:
                break
            source_rows += 1
            with archive.open(member) as handle:
                raw = json.loads(io.TextIOWrapper(handle, encoding="utf-8").read())
            try:
                rows.append(materialize_row(raw, split=split, index=index))
            except ValueError as exc:
                if "no usable non-empty evidence snippets" not in str(exc):
                    raise
                dropped_rows += 1
                reason = "no_usable_evidence"
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    return rows, {"source_rows": source_rows, "materialized_rows": len(rows), "dropped_rows": dropped_rows, "drop_reasons": drop_reasons}


def attach_evidence(rows: list[dict], *, max_evidence_tokens: int, token_count) -> list[dict]:
    if max_evidence_tokens <= 0:
        raise ValueError("max_evidence_tokens must be positive")
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["packed_evidence"] = pack_evidence(row["snippets"], max_tokens=max_evidence_tokens, token_count=token_count)
        enriched["prompt"] = build_student_prompt(enriched, [])
        output.append(enriched)
    return output


def build_sft_rows(rows: list[dict]) -> list[dict]:
    return [build_sft_row(row) for row in rows]


def build_sft_row(row: dict) -> dict:
    completion = str(row.get("gold_answer", "")).strip()
    if not completion:
        raise ValueError("SFT completion cannot be empty")
    return {"id": row["id"], "prompt": build_student_prompt(row, []), "completion": completion}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
