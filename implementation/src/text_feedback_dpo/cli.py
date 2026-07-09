from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_feedback_dpo.io import read_jsonl, write_jsonl
from text_feedback_dpo.observability import JsonlLogger
from text_feedback_dpo.report import write_html_report
from text_feedback_dpo.scoring import evaluate_rollout


def _index_by_id(rows: list[dict], source_name: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        row_id = row.get("id")
        if not row_id:
            raise ValueError(f"{source_name} row is missing required id")
        if row_id in indexed:
            raise ValueError(f"{source_name} contains duplicate id: {row_id}")
        indexed[row_id] = row
    return indexed


def run_basic_pipeline(
    *,
    examples_path: Path,
    rollouts_path: Path,
    corrections_path: Path,
    output_dir: Path,
    run_id: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "events.jsonl", run_id=run_id)
    logger.event(
        "run_start",
        stage="load_inputs",
        examples_path=str(examples_path),
        rollouts_path=str(rollouts_path),
        corrections_path=str(corrections_path),
        output_dir=str(output_dir),
    )

    examples = read_jsonl(examples_path)
    rollouts = _index_by_id(read_jsonl(rollouts_path), "rollouts")
    corrections = _index_by_id(read_jsonl(corrections_path), "corrections")

    pairs: list[dict] = []
    rejections: list[dict] = []
    verification_missing_rejections = 0

    for example in examples:
        example_id = example.get("id")
        if not example_id:
            raise ValueError("example row is missing required id")
        if example_id not in rollouts:
            raise ValueError(f"missing rollout for example id: {example_id}")
        if example_id not in corrections:
            raise ValueError(f"missing correction for example id: {example_id}")

        gold_answer = str(example["gold_answer"])
        original_rollout = str(rollouts[example_id]["rollout"])
        corrected_rollout = str(corrections[example_id]["corrected_rollout"])
        original_result = evaluate_rollout(original_rollout, gold_answer)
        corrected_result = evaluate_rollout(corrected_rollout, gold_answer)

        logger.event(
            "example_evaluated",
            stage="evaluate",
            example_id=example_id,
            domain=example.get("domain"),
            original_score=original_result["score"],
            corrected_score=corrected_result["score"],
            corrected_format_valid=corrected_result["format_valid"],
            corrected_verification_present=corrected_result["verification_present"],
            corrected_error_code=corrected_result["error_code"],
        )

        if not corrected_result["verification_present"]:
            verification_missing_rejections += 1

        accepted = (
            original_result["score"] < corrected_result["score"]
            and corrected_result["format_valid"]
            and corrected_result["verification_present"]
        )
        if accepted:
            pairs.append(
                {
                    "id": example_id,
                    "prompt": str(example["problem"]),
                    "chosen": corrected_rollout,
                    "rejected": original_rollout,
                    "metadata": {
                        "domain": example.get("domain"),
                        "original_score": original_result["score"],
                        "corrected_score": corrected_result["score"],
                        "feedback": corrections[example_id].get("feedback", ""),
                    },
                }
            )
        else:
            reason = corrected_result["error_code"] or "corrected_not_better"
            rejections.append(
                {
                    "id": example_id,
                    "reason": reason,
                    "original_result": original_result,
                    "corrected_result": corrected_result,
                }
            )

    metrics = {
        "run_id": run_id,
        "examples_total": len(examples),
        "accepted_pairs": len(pairs),
        "rejected_examples": len(rejections),
        "verification_missing_rejections": verification_missing_rejections,
    }
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    write_jsonl(output_dir / "rejections.jsonl", rejections)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_html_report(output_dir / "report.html", metrics)
    logger.event("run_end", stage="complete", **metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basic_pipeline", nargs="?")
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--corrections", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.basic_pipeline != "basic-pipeline":
        raise SystemExit("Only the basic-pipeline command is implemented.")
    result = run_basic_pipeline(
        examples_path=args.examples,
        rollouts_path=args.rollouts,
        corrections_path=args.corrections,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
