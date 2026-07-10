from __future__ import annotations

from pathlib import Path
from typing import Any

from text_feedback_dpo.io import read_jsonl, write_json_atomic, write_jsonl
from text_feedback_dpo.report import write_html_report


def _index(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{label} contains a row without a non-empty id")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate id: {row_id}")
        indexed[row_id] = row
    return indexed


def audit_checkpoint_evaluation(
    *,
    predictions_path: Path,
    labels_path: Path,
    output_dir: Path,
    minimum_labels: int,
    minimum_agreement: float,
    max_truncation_rate: float,
) -> dict[str, Any]:
    if minimum_labels <= 0:
        raise ValueError("minimum_labels must be positive")
    if not 0 <= minimum_agreement <= 1:
        raise ValueError("minimum_agreement must be between 0 and 1")
    if not 0 <= max_truncation_rate <= 1:
        raise ValueError("max_truncation_rate must be between 0 and 1")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation audit: {output_dir}")
    predictions = read_jsonl(predictions_path)
    labels = read_jsonl(labels_path)
    if not predictions:
        raise ValueError("evaluation audit requires non-empty predictions")
    if len(labels) < minimum_labels:
        raise ValueError(
            f"evaluation audit requires at least {minimum_labels} manual labels; found {len(labels)}"
        )
    prediction_by_id = _index(predictions, label="predictions")
    label_by_id = _index(labels, label="manual labels")
    unknown_labels = sorted(set(label_by_id) - set(prediction_by_id))
    if unknown_labels:
        raise ValueError(f"manual label has no prediction: {unknown_labels[0]}")

    agreements = 0
    disagreements: list[dict[str, Any]] = []
    for row_id, label in label_by_id.items():
        manual_correct = label.get("manual_correct")
        if not isinstance(manual_correct, bool):
            raise ValueError(f"manual label {row_id} must contain boolean manual_correct")
        if not isinstance(label.get("notes"), str) or not str(label["notes"]).strip():
            raise ValueError(f"manual label {row_id} must contain non-empty notes")
        evaluator_result = prediction_by_id[row_id].get("evaluator_result")
        if not isinstance(evaluator_result, dict) or not isinstance(evaluator_result.get("correct"), bool):
            raise ValueError(f"prediction {row_id} has no boolean evaluator correctness")
        evaluator_correct = bool(evaluator_result["correct"])
        if evaluator_correct == manual_correct:
            agreements += 1
        else:
            disagreements.append(
                {
                    "id": row_id,
                    "evaluator_correct": evaluator_correct,
                    "manual_correct": manual_correct,
                    "notes": label["notes"],
                }
            )

    metadata_complete = all(
        isinstance(row.get("prompt_tokens"), int)
        and isinstance(row.get("generated_tokens"), int)
        and isinstance(row.get("terminated"), bool)
        and isinstance(row.get("truncated"), bool)
        and row.get("finish_reason") in {"eos", "length", "other"}
        for row in predictions
    )
    teacher_free = all(row.get("teacher_free") is True for row in predictions)
    truncation_rate = sum(bool(row.get("truncated")) for row in predictions) / len(predictions)
    agreement = agreements / len(labels)
    failed_gates: list[str] = []
    if agreement < minimum_agreement:
        failed_gates.append("manual_agreement")
    if not metadata_complete:
        failed_gates.append("generation_metadata")
    if not teacher_free:
        failed_gates.append("teacher_free")
    if truncation_rate > max_truncation_rate:
        failed_gates.append("truncation_rate")
    result = {
        "schema": "checkpoint-evaluation-audit-v1",
        "passed": not failed_gates,
        "predictions": len(predictions),
        "manual_labels": len(labels),
        "manual_agreement": agreement,
        "minimum_agreement": minimum_agreement,
        "disagreements": len(disagreements),
        "generation_metadata_complete": metadata_complete,
        "teacher_free": teacher_free,
        "truncation_rate": truncation_rate,
        "max_truncation_rate": max_truncation_rate,
        "failed_gates": failed_gates,
    }
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "audit.json", result)
    write_jsonl(output_dir / "disagreements.jsonl", disagreements)
    write_html_report(output_dir / "report.html", result)
    return result
