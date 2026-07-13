from __future__ import annotations

import csv
import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from text_feedback_dpo.trajectories import revalidate_cached_trajectory


def _require_unique_rows(rows: Sequence[Mapping[str, object]], *, label: str) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for index, row in enumerate(rows):
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id or example_id in indexed:
            raise ValueError(f"{label} requires unique non-empty ids; invalid row {index}: {example_id!r}")
        indexed[example_id] = row
    return indexed


def _source_view(source: Mapping[str, object]) -> dict[str, object]:
    return {
        "source_id": source.get("source_id"),
        "rank": source.get("rank"),
        "title": source.get("title"),
        "snippet": source.get("snippet"),
    }


def _sibling_view(sibling: Mapping[str, object]) -> dict[str, object]:
    return {
        "seed": sibling.get("seed"),
        "raw_query": sibling.get("raw_query"),
        "raw_response": sibling.get("raw_response"),
        "error_code": sibling.get("error_code"),
        "verified_no_hint_success": sibling.get("verified_no_hint_success"),
        "future_sibling_gain": sibling.get("future_sibling_gain"),
    }


def audit_trajectories(
    examples: Sequence[Mapping[str, object]],
    trajectories: Sequence[Mapping[str, object]],
    *,
    sibling_seeds: Sequence[int],
) -> dict[str, object]:
    """Canonically revalidate trajectories and expose exact human-audit evidence.

    Missing runtime telemetry remains explicit ``null`` and is summarized as an
    observability gap. It is never estimated from words, characters, or wall time.
    """
    example_by_id = _require_unique_rows(examples, label="canonical examples")
    trajectory_by_id = _require_unique_rows(trajectories, label="trajectories")
    if set(example_by_id) != set(trajectory_by_id):
        missing = sorted(set(example_by_id) - set(trajectory_by_id))
        extra = sorted(set(trajectory_by_id) - set(example_by_id))
        raise ValueError(f"trajectory/example ID parity mismatch; missing={missing[:3]} extra={extra[:3]}")

    rows: list[dict[str, object]] = []
    normalized_trajectories: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    unavailable: set[str] = set()

    for example_id in example_by_id:
        example = example_by_id[example_id]
        trajectory = revalidate_cached_trajectory(
            trajectory_by_id[example_id],
            example=example,
            expected_sibling_seeds=sibling_seeds,
        )
        normalized_trajectories.append(trajectory)
        interventions = {
            int(item["attempt_index"]): item for item in trajectory["interventions"]
        }
        attempts = trajectory["attempts"]
        siblings = [_sibling_view(item) for item in trajectory["no_hint_siblings"]]
        for attempt_index, attempt in enumerate(attempts):
            artifact = attempt["artifact"]
            intervention = interventions.get(attempt_index)
            retry = attempts[attempt_index + 1] if attempt_index + 1 < len(attempts) else None
            error_code = artifact.get("error_code")
            error_counts[str(error_code or "none")] += 1
            repair_region = attempt.get("responsible_region")
            repair_counts[str(repair_region or "success")] += 1
            latency_seconds = attempt.get("latency_seconds")
            teacher_prompt_token_count = (
                intervention.get("teacher_prompt_token_count") if intervention is not None else None
            )
            if latency_seconds is None:
                unavailable.add("latency_seconds")
            if intervention is not None and teacher_prompt_token_count is None:
                unavailable.add("teacher_prompt_token_count")
            rows.append({
                "id": example_id,
                "attempt_index": attempt_index,
                "question": example.get("question"),
                "gold_answer": example.get("gold_answer"),
                "raw_query": artifact.get("raw_query"),
                "top_sources": [_source_view(source) for source in artifact["ranked_search_results"]],
                "raw_response": artifact.get("raw_response"),
                "error_code": error_code,
                "correct": attempt.get("correct"),
                "responsible_region": repair_region,
                "diagnostics": attempt.get("diagnostics"),
                "teacher_hint": intervention.get("hint") if intervention is not None else None,
                "teacher_prompt_hash": intervention.get("teacher_prompt_hash") if intervention is not None else None,
                "teacher_prompt_token_count": teacher_prompt_token_count,
                "retry_raw_query": retry["artifact"].get("raw_query") if retry is not None else None,
                "retry_raw_response": retry.get("response") if retry is not None else None,
                "siblings": siblings,
                "teacher_leakage": False,
                "latency_seconds": latency_seconds,
                "sft_eligible": trajectory["sft_eligible"],
                "preference_eligible": trajectory["preference_eligible"],
                "resolved": trajectory["resolved"],
            })

    summary = {
        "trajectories": len(normalized_trajectories),
        "attempt_rows": len(rows),
        "resolved": sum(bool(item["resolved"]) for item in normalized_trajectories),
        "resolved_without_hint": sum(
            bool(item["resolved"]) and not item["interventions"] for item in normalized_trajectories
        ),
        "resolved_after_hint": sum(
            bool(item["resolved"]) and bool(item["interventions"]) for item in normalized_trajectories
        ),
        "unresolved": sum(not item["resolved"] for item in normalized_trajectories),
        "sft_eligible": sum(bool(item["sft_eligible"]) for item in normalized_trajectories),
        "preference_eligible": sum(bool(item["preference_eligible"]) for item in normalized_trajectories),
        "teacher_leakage": 0,
        "no_hint_sibling_successes": sum(
            int(sibling["verified_no_hint_success"])
            for item in normalized_trajectories
            for sibling in item["no_hint_siblings"]
        ),
        "no_hint_siblings": sum(len(item["no_hint_siblings"]) for item in normalized_trajectories),
        "error_counts": dict(sorted(error_counts.items())),
        "responsible_region_counts": dict(sorted(repair_counts.items())),
        "unavailable_observability": sorted(unavailable),
    }
    return {"schema_version": 1, "summary": summary, "rows": rows}


def _cell(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def write_trajectory_audit(result: Mapping[str, object], *, output_prefix: Path) -> dict[str, Path]:
    rows = result.get("rows")
    summary = result.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, Mapping):
        raise ValueError("trajectory audit result requires rows and summary")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_prefix.with_suffix(".json"),
        "jsonl": output_prefix.with_suffix(".jsonl"),
        "csv": output_prefix.with_suffix(".csv"),
        "html": output_prefix.with_suffix(".html"),
    }
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fieldnames = list(rows[0]) if rows else []
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)

    summary_html = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_cell(value))}</td></tr>"
        for key, value in summary.items()
    )
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in fieldnames)
    body = "".join(
        "<tr>" + "".join(f"<td><pre>{html.escape(_cell(row.get(name)))}</pre></td>" for name in fieldnames) + "</tr>"
        for row in rows
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Trajectory audit</title>"
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:6px;vertical-align:top}"
        "pre{white-space:pre-wrap;max-width:44rem}</style></head><body>"
        f"<h1>Trajectory audit</h1><h2>Summary</h2><table>{summary_html}</table>"
        f"<h2>Attempts</h2><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>"
    )
    paths["html"].write_text(document, encoding="utf-8")
    return paths
