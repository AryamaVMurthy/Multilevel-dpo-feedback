from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from text_feedback_dpo.answer_evaluation import evaluate_domain_answer
from text_feedback_dpo.io import read_jsonl, read_jsonl_zst, write_json_atomic, write_jsonl
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rescore_checkpoint_evaluation(
    *,
    predictions_path: Path,
    examples_path: Path,
    output_dir: Path,
    source_commit: str,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("rescore source_commit must be a lowercase 40-character Git SHA")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation rescore: {output_dir}")
    predictions = read_jsonl(predictions_path)
    examples = (
        read_jsonl_zst(examples_path)
        if examples_path.name.endswith(".zst")
        else read_jsonl(examples_path)
    )
    if not predictions:
        raise ValueError("evaluation rescore requires non-empty predictions")
    prediction_by_id = _index(predictions, label="predictions")
    example_by_id = _index(examples, label="examples")
    if set(prediction_by_id) != set(example_by_id):
        raise ValueError("evaluation rescore prediction/example ids must match exactly")

    rescored: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    original_correct = 0
    rescored_correct = 0
    requires_model_judgment = 0
    for prediction in predictions:
        row_id = str(prediction["id"])
        example = example_by_id[row_id]
        evaluator_result = prediction.get("evaluator_result")
        if not isinstance(evaluator_result, dict):
            raise ValueError(f"prediction {row_id} has no evaluator_result")
        answer = evaluator_result.get("answer")
        model_correct = evaluator_result.get("model_correct")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"prediction {row_id} evaluator_result has no answer")
        if not isinstance(model_correct, bool):
            raise ValueError(f"prediction {row_id} evaluator_result has no model_correct")
        original = evaluator_result.get("correct")
        if not isinstance(original, bool):
            raise ValueError(f"prediction {row_id} evaluator_result has no boolean correct")
        deterministic = evaluate_domain_answer(
            domain=str(example.get("domain", "")),
            prediction=answer,
            example=example,
            actual_answer_type=str(evaluator_result.get("answer_type", "unknown")),
            evidence_supported=evaluator_result.get("evidence_supported"),
        )
        needs_model = bool(deterministic.get("requires_model_judgment"))
        corrected = model_correct if needs_model else model_correct and bool(deterministic["correct"])
        updated_result = {
            **evaluator_result,
            "correct": corrected,
            "deterministic": deterministic,
            "deterministic_correct": bool(deterministic["correct"]),
            "requires_model_judgment": needs_model,
        }
        rescored.append(
            {
                **prediction,
                "evaluator_result": updated_result,
                "rescore_source_commit": source_commit,
                "original_evaluator_result_sha256": _canonical_sha256(evaluator_result),
            }
        )
        original_correct += int(original)
        rescored_correct += int(corrected)
        requires_model_judgment += int(needs_model)
        if original != corrected:
            changed_ids.append(row_id)

    output_dir.mkdir(parents=True)
    output_predictions = output_dir / "predictions.jsonl"
    write_jsonl(output_predictions, rescored)
    result = {
        "schema": "checkpoint-evaluation-rescore-v1",
        "source_commit": source_commit,
        "input_predictions_sha256": _sha256(predictions_path),
        "input_examples_sha256": _sha256(examples_path),
        "output_predictions_sha256": _sha256(output_predictions),
        "examples": len(rescored),
        "original_correct": original_correct,
        "rescored_correct": rescored_correct,
        "changed_decisions": len(changed_ids),
        "changed_ids": changed_ids,
        "requires_model_judgment": requires_model_judgment,
    }
    write_json_atomic(output_dir / "rescore.json", result)
    return result


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
        and row.get("finish_reason") in {"eos", "final_answer", "length", "other"}
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
