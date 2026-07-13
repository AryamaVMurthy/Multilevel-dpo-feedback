from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_sft_reproduction_report(
    rows: Sequence[Mapping[str, object]],
    generated_by_id: Mapping[str, Any],
) -> tuple[list[dict], dict]:
    """Compare decoded checkpoint continuations with exact verified SFT targets."""
    row_ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError("SFT reproduction rows require non-empty string IDs")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("SFT reproduction rows contain a duplicate ID")
    if set(row_ids) != set(generated_by_id):
        missing = sorted(set(row_ids) - set(generated_by_id))
        unexpected = sorted(set(generated_by_id) - set(row_ids))
        raise ValueError(f"SFT reproduction generation ID parity mismatch: missing={missing} unexpected={unexpected}")

    records: list[dict] = []
    for row in rows:
        row_id = str(row["id"])
        task = row.get("task")
        completion = row.get("completion")
        if task not in {"query", "response"}:
            raise ValueError(f"SFT reproduction row {row_id} requires task=query or task=response")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"SFT reproduction row {row_id} requires a non-empty completion")
        generated = generated_by_id[row_id]
        text = getattr(generated, "text", None)
        truncated = getattr(generated, "truncated", None)
        if not isinstance(text, str) or not isinstance(truncated, bool):
            raise TypeError(f"SFT reproduction generation {row_id} must expose text and truncated")
        reference_text = completion.strip()
        generated_text = text.strip()
        records.append({
            "id": row_id,
            "task": task,
            "reference": reference_text,
            "generated": generated_text,
            "exact": generated_text == reference_text,
            "empty": not generated_text,
            "truncated": truncated,
        })

    summary: dict[str, object] = {
        "rows": len(records),
        "exact": sum(record["exact"] for record in records),
        "empty": sum(record["empty"] for record in records),
        "truncated": sum(record["truncated"] for record in records),
        "tasks": {},
        "comparison": "decoded_text_strip_boundary_whitespace_only_no_repair",
    }
    summary["exact_rate"] = summary["exact"] / max(1, len(records))
    for task in ("query", "response"):
        task_records = [record for record in records if record["task"] == task]
        exact = sum(record["exact"] for record in task_records)
        summary["tasks"][task] = {
            "rows": len(task_records),
            "exact": exact,
            "exact_rate": exact / max(1, len(task_records)),
            "empty": sum(record["empty"] for record in task_records),
            "truncated": sum(record["truncated"] for record in task_records),
        }
    return records, summary
