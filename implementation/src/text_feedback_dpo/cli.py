from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_feedback_dpo.config import load_config
from text_feedback_dpo.io import read_jsonl, write_jsonl
from text_feedback_dpo.models import ModelProvider, TransformersModelProvider
from text_feedback_dpo.observability import JsonlLogger
from text_feedback_dpo.prompts import build_student_prompt, build_teacher_prompt
from text_feedback_dpo.report import write_html_report
from text_feedback_dpo.scoring import evaluate_rollout


FAKE_STUDENT_ROLLOUT = """<plan>
Solve with one arithmetic branch.
</plan>
<think branch="A">
2 + 2 = 5.
</think>
<reflect>
Branch comparison: one branch.
Evidence / derivation check: the arithmetic should be checked.
Verification: recalculating 2 + 2 does not support 5.
Decision: answer
</reflect>
<final>
5
</final>"""


FAKE_TEACHER_OUTPUT = """<feedback>
The student should recompute the arithmetic and verify before final.
</feedback>

<corrected_rollout>
<plan>
Solve with one arithmetic branch and verify before final.
</plan>
<think branch="A">
2 + 2 = 4.
</think>
<reflect>
Branch comparison: one branch is sufficient.
Evidence / derivation check: direct addition gives 4.
Verification: recalculating 2 + 2 gives 4.
Decision: answer
</reflect>
<final>
4
</final>
</corrected_rollout>"""


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


def _extract_required_block(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    end = text.find(end_tag, start + len(start_tag))
    if start < 0 or end < 0:
        raise ValueError(f"teacher output missing <{tag}> block")
    return text[start + len(start_tag) : end].strip()


def _default_smoke_examples(max_examples: int) -> list[dict]:
    examples = [
        {
            "id": "math-1",
            "domain": "math",
            "problem": "What is 2 + 2?",
            "gold_answer": "4",
        },
        {
            "id": "math-2",
            "domain": "math",
            "problem": "What is 3 + 3?",
            "gold_answer": "6",
        },
        {
            "id": "math-3",
            "domain": "math",
            "problem": "What is 5 - 2?",
            "gold_answer": "3",
        },
        {
            "id": "math-4",
            "domain": "math",
            "problem": "What is 7 + 1?",
            "gold_answer": "8",
        },
        {
            "id": "math-5",
            "domain": "math",
            "problem": "What is 10 / 2?",
            "gold_answer": "5",
        },
    ]
    return examples[:max_examples]


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


def run_generate_pipeline(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    model_provider: ModelProvider | None = None,
    fake_smoke: bool = False,
) -> dict:
    config = load_config(config_path)
    run_id = str(config["run_id"])
    output = output_dir or Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "events.jsonl", run_id=run_id)
    logger.event("run_start", stage="generate", config_path=str(config_path), output_dir=str(output))

    if fake_smoke and model_provider is None:
        from text_feedback_dpo.models import FakeModelProvider

        logger.event(
            "fake_smoke_enabled",
            stage="generate",
            fallback_reason="explicit --fake-smoke test mode requested",
        )
        provider = FakeModelProvider({"student": FAKE_STUDENT_ROLLOUT, "teacher": FAKE_TEACHER_OUTPUT})
    else:
        provider = model_provider or TransformersModelProvider(
            model_ids={"student": str(config["student_model"]), "teacher": str(config["teacher_model"])},
        )

    examples = _default_smoke_examples(int(config["max_examples"]))
    rollouts: list[dict] = []
    corrections: list[dict] = []
    pairs: list[dict] = []
    rejections: list[dict] = []
    verification_missing_rejections = 0

    for example in examples:
        student_prompt = build_student_prompt(str(example["problem"]), str(example["domain"]))
        student_rollout = provider.generate("student", student_prompt, **config["generation"])
        rollouts.append({"id": example["id"], "prompt": student_prompt, "rollout": student_rollout})
        original_result = evaluate_rollout(student_rollout, str(example["gold_answer"]))
        logger.event(
            "student_generated",
            stage="student_generation",
            example_id=example["id"],
            domain=example["domain"],
            original_score=original_result["score"],
            original_error_code=original_result["error_code"],
        )

        teacher_prompt = build_teacher_prompt(
            problem=str(example["problem"]),
            gold_answer=str(example["gold_answer"]),
            student_rollout=student_rollout,
            result=original_result,
            domain=str(example["domain"]),
            teacher_mode=str(config["teacher_mode"]),
        )
        teacher_output = provider.generate("teacher", teacher_prompt, **config["teacher_generation"])
        feedback = _extract_required_block(teacher_output, "feedback")
        corrected_rollout = _extract_required_block(teacher_output, "corrected_rollout")
        corrected_result = evaluate_rollout(corrected_rollout, str(example["gold_answer"]))
        corrections.append(
            {
                "id": example["id"],
                "prompt": teacher_prompt,
                "feedback": feedback,
                "corrected_rollout": corrected_rollout,
                "corrected_result": corrected_result,
            }
        )
        logger.event(
            "teacher_corrected",
            stage="teacher_correction",
            example_id=example["id"],
            domain=example["domain"],
            corrected_score=corrected_result["score"],
            corrected_error_code=corrected_result["error_code"],
            corrected_verification_present=corrected_result["verification_present"],
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
                    "id": example["id"],
                    "prompt": str(example["problem"]),
                    "chosen": corrected_rollout,
                    "rejected": student_rollout,
                    "metadata": {
                        "domain": example["domain"],
                        "original_score": original_result["score"],
                        "corrected_score": corrected_result["score"],
                        "feedback": feedback,
                    },
                }
            )
        else:
            rejections.append(
                {
                    "id": example["id"],
                    "reason": corrected_result["error_code"] or "corrected_not_better",
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
        "student_model": config["student_model"],
        "teacher_model": config["teacher_model"],
        "teacher_mode": config["teacher_mode"],
    }
    write_jsonl(output / "examples.jsonl", examples)
    write_jsonl(output / "rollouts.jsonl", rollouts)
    write_jsonl(output / "corrections.jsonl", corrections)
    write_jsonl(output / "pairs.jsonl", pairs)
    write_jsonl(output / "rejections.jsonl", rejections)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_html_report(output / "report.html", metrics)
    logger.event("run_end", stage="complete", **metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    basic = subparsers.add_parser("basic-pipeline")
    basic.add_argument("--examples", required=True, type=Path)
    basic.add_argument("--rollouts", required=True, type=Path)
    basic.add_argument("--corrections", required=True, type=Path)
    basic.add_argument("--output-dir", required=True, type=Path)
    basic.add_argument("--run-id", required=True)
    generate = subparsers.add_parser("generate-pipeline")
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--output-dir", type=Path)
    generate.add_argument("--fake-smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "basic-pipeline":
        result = run_basic_pipeline(
            examples_path=args.examples,
            rollouts_path=args.rollouts,
            corrections_path=args.corrections,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    elif args.command == "generate-pipeline":
        result = run_generate_pipeline(
            config_path=args.config,
            output_dir=args.output_dir,
            fake_smoke=args.fake_smoke,
        )
    else:
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
