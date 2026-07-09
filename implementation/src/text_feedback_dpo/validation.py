from __future__ import annotations

import json
from pathlib import Path

from text_feedback_dpo.io import read_jsonl
from text_feedback_dpo.observability import JsonlLogger


REQUIRED_ARTIFACTS = (
    "events.jsonl",
    "examples.jsonl",
    "rollouts.jsonl",
    "corrections.jsonl",
    "pairs.jsonl",
    "rejections.jsonl",
    "metrics.json",
)


def _read_metrics(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metrics.json is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("metrics.json must contain a JSON object")
    return value


def _require_same_ids(name: str, rows: list[dict], expected_ids: set[str]) -> None:
    actual_ids = {str(row.get("id")) for row in rows}
    if len(actual_ids) != len(rows) or "None" in actual_ids:
        raise ValueError(f"{name} has missing or duplicate ids")
    if actual_ids != expected_ids:
        raise ValueError(f"{name} ids do not match examples ids")


def _write_validation(path: Path, payload: dict) -> None:
    (path / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_run(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required run artifacts: {', '.join(missing)}")

    metrics = _read_metrics(output_dir / "metrics.json")
    run_id = metrics.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("metrics.json is missing run_id")
    logger = JsonlLogger(output_dir / "events.jsonl", run_id=run_id)

    try:
        for key in ("student_model", "teacher_model", "accepted_pairs"):
            if key not in metrics:
                raise ValueError(f"metrics.json is missing {key}")
        if not metrics["student_model"] or not metrics["teacher_model"]:
            raise ValueError("metrics.json has empty model metadata")
        if not isinstance(metrics["accepted_pairs"], int) or metrics["accepted_pairs"] <= 0:
            raise ValueError("accepted_pairs must be positive for real validation")

        events = read_jsonl(output_dir / "events.jsonl")
        event_names = {event.get("event_name") for event in events}
        if "run_start" not in event_names or "run_end" not in event_names:
            raise ValueError("events.jsonl must include run_start and run_end")

        examples = read_jsonl(output_dir / "examples.jsonl")
        if not examples:
            raise ValueError("examples.jsonl is empty")
        example_ids = {str(row.get("id")) for row in examples}
        if len(example_ids) != len(examples) or "None" in example_ids:
            raise ValueError("examples.jsonl has missing or duplicate ids")
        _require_same_ids("rollouts.jsonl", read_jsonl(output_dir / "rollouts.jsonl"), example_ids)
        _require_same_ids("corrections.jsonl", read_jsonl(output_dir / "corrections.jsonl"), example_ids)

        pairs = read_jsonl(output_dir / "pairs.jsonl")
        if len(pairs) != metrics["accepted_pairs"]:
            raise ValueError("pairs.jsonl count does not match accepted_pairs")
        for pair in pairs:
            prompt = str(pair.get("prompt", ""))
            feedback = str(pair.get("metadata", {}).get("feedback", ""))
            gold_answers = [str(row["gold_answer"]) for row in examples if row.get("id") == pair.get("id") and "gold_answer" in row]
            if (feedback and feedback in prompt) or "gold answer:" in prompt.lower() or any(
                answer and answer in prompt for answer in gold_answers
            ):
                raise ValueError("pair prompt leaks teacher feedback or gold answer")
            if not pair.get("chosen") or not pair.get("rejected"):
                raise ValueError("pair is missing chosen or rejected rollout")

        if not list(output_dir.glob("gpu-*.csv")):
            raise ValueError("missing GPU telemetry CSV")

        result = {
            "valid": True,
            "run_id": run_id,
            "accepted_pairs": metrics["accepted_pairs"],
            "student_model": metrics["student_model"],
            "teacher_model": metrics["teacher_model"],
        }
        _write_validation(output_dir, result)
        logger.event("validation_complete", stage="validate_run", **result)
        return result
    except Exception as exc:
        logger.failure(
            stage="validate_run",
            error_code="run_artifact_validation_failed",
            message=str(exc),
        )
        raise
