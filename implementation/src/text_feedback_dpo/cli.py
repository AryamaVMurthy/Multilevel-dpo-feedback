from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from text_feedback_dpo.config import load_config
from text_feedback_dpo.benchmarks import load_benchmark_examples
from text_feedback_dpo.io import read_jsonl, write_jsonl
from text_feedback_dpo.evaluators import (
    ModelOutputParseError,
    make_model_evaluator,
    make_model_guidance_guard,
)
from text_feedback_dpo.dataset_manifests import materialize_paper_dataset
from text_feedback_dpo.experiment_config import load_paper_experiment, validate_paper_experiment
from text_feedback_dpo.methods import build_native_iterative_guidance_pairs
from text_feedback_dpo.models import ModelProvider, TransformersModelProvider
from text_feedback_dpo.observability import JsonlLogger
from text_feedback_dpo.prompts import (
    build_native_student_prompt,
    build_privileged_guidance_prompt,
    build_student_prompt,
    build_teacher_prompt,
)
from text_feedback_dpo.report import write_html_report
from text_feedback_dpo.scoring import evaluate_rollout
from text_feedback_dpo.training import (
    load_training_rows,
    run_distillation_training,
    run_dpo_training,
    run_grpo_training,
)
from text_feedback_dpo.validation import validate_run


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
    content = text[start + len(start_tag) : end].strip()
    if not content or "..." in content:
        raise ValueError(f"teacher output contains a placeholder in <{tag}> block")
    return content


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
        try:
            feedback = _extract_required_block(teacher_output, "feedback")
            corrected_rollout = _extract_required_block(teacher_output, "corrected_rollout")
        except ValueError as exc:
            failure = {
                "id": example["id"],
                "domain": example["domain"],
                "error_code": "teacher_output_parse_failed",
                "message": str(exc),
                "raw_teacher_output": teacher_output,
            }
            write_jsonl(output / "teacher_failures.jsonl", [failure])
            logger.failure(
                stage="teacher_correction",
                error_code="teacher_output_parse_failed",
                message=str(exc),
                example_id=example["id"],
                domain=example["domain"],
                raw_output_path=str(output / "teacher_failures.jsonl"),
            )
            raise
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


def run_native_pipeline(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    model_provider: ModelProvider | None = None,
    evaluator=None,
    guidance_guard=None,
) -> dict:
    config = load_config(config_path)
    run_id = str(config["run_id"])
    output = output_dir or Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "events.jsonl", run_id=run_id)
    logger.event("run_start", stage="native_iterative_guidance", config_path=str(config_path))

    if "benchmarks" in config:
        examples = load_benchmark_examples(config["benchmarks"])
    else:
        if "examples_path" not in config:
            raise ValueError("native pipeline requires examples_path or benchmarks")
        examples_path = Path(str(config["examples_path"]))
        if not examples_path.is_absolute():
            examples_path = config_path.parent / examples_path
        examples = read_jsonl(examples_path)
    max_examples = int(config["max_examples"])
    if len(examples) < max_examples:
        raise ValueError(
            f"examples file has {len(examples)} rows but max_examples={max_examples} was requested"
        )
    examples = examples[:max_examples]

    provider = model_provider or TransformersModelProvider(
        model_ids={
            "student": str(config["student_model"]),
            "teacher": str(config["teacher_model"]),
            "evaluator": str(config.get("evaluator_model", config["teacher_model"])),
        },
    )
    evaluator_fn = evaluator or make_model_evaluator(
        generate=provider.generate,
        generation_kwargs=dict(config["evaluator_generation"]),
    )
    guidance_guard_fn = guidance_guard or make_model_guidance_guard(
        generate=provider.generate,
        generation_kwargs=dict(config["evaluator_generation"]),
    )
    guidance_events: list[dict] = []
    generation_events: list[dict] = []

    def base_prompt_builder(example: dict) -> str:
        return build_native_student_prompt(
            problem=str(example["problem"]),
            domain=str(example["domain"]),
            evidence=example.get("evidence"),
        )

    def retry_prompt_builder(base_prompt: str, guidance: str) -> str:
        return (
            f"{base_prompt}\n\nTeacher guidance for reconsideration:\n{guidance}\n"
            "Solve the original problem again and provide your best answer."
        )

    def student_generate(prompt: str) -> str:
        start = time.monotonic_ns()
        response = provider.generate("student", prompt, **dict(config["generation"]))
        generation_events.append(
            {
                "role": "student",
                "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                "generated_tokens_estimate": len(response.split()),
            }
        )
        return response

    def teacher_guidance(example: dict, rollout: str, result: dict, attempt: int) -> str:
        prompt = build_privileged_guidance_prompt(
            problem=str(example["problem"]),
            gold_answer=str(example["gold_answer"]),
            rollout=rollout,
            result=result,
            domain=str(example["domain"]),
        )
        start = time.monotonic_ns()
        guidance = provider.generate("teacher", prompt, **dict(config["teacher_generation"]))
        generation_events.append(
            {
                "role": "teacher",
                "example_id": str(example["id"]),
                "attempt": attempt,
                "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                "generated_tokens_estimate": len(guidance.split()),
            }
        )
        guidance_events.append(
            {
                "id": str(example["id"]),
                "attempt": attempt,
                "prompt": prompt,
                "guidance": guidance,
                "teacher_model": config["teacher_model"],
            }
        )
        return guidance

    try:
        result = build_native_iterative_guidance_pairs(
            examples=examples,
            base_prompt_builder=base_prompt_builder,
            retry_prompt_builder=retry_prompt_builder,
            student_generate=student_generate,
            evaluate=evaluator_fn,
            teacher_guidance=teacher_guidance,
            guidance_guard=guidance_guard_fn,
            max_guidance_steps=int(config["max_guidance_steps"]),
            max_guidance_regenerations=int(config["max_guidance_regenerations"]),
        )
    except ModelOutputParseError as exc:
        write_jsonl(
            output / "model_failures.jsonl",
            [
                {
                    "role": exc.role,
                    "error_code": "model_output_parse_failed",
                    "message": str(exc),
                    "raw_output": exc.raw,
                }
            ],
        )
        logger.failure(
            stage=exc.role,
            error_code="model_output_parse_failed",
            message=str(exc),
            raw_output_path=str(output / "model_failures.jsonl"),
        )
        raise
    metrics = {
        "run_id": run_id,
        "method": "native_iterative_guidance_dpo",
        "student_model": config["student_model"],
        "teacher_model": config["teacher_model"],
        "evaluator_model": config.get("evaluator_model", config["teacher_model"]),
        "teacher_mode": config["teacher_mode"],
        "generation_events": len(generation_events),
        "average_generated_tokens_estimate": (
            sum(row["generated_tokens_estimate"] for row in generation_events) / len(generation_events)
            if generation_events
            else 0.0
        ),
        "average_generation_latency_ms": (
            sum(row["latency_ms"] for row in generation_events) / len(generation_events)
            if generation_events
            else 0.0
        ),
        **result["metrics"],
    }
    write_jsonl(output / "examples.jsonl", examples)
    write_jsonl(output / "attempts.jsonl", result["attempts"])
    write_jsonl(output / "guidance.jsonl", guidance_events)
    write_jsonl(output / "generation_events.jsonl", generation_events)
    write_jsonl(output / "pairs.jsonl", result["pairs"])
    write_jsonl(output / "response_sft.jsonl", result["response_sft"])
    write_jsonl(output / "failures.jsonl", result["failures"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_html_report(output / "report.html", metrics)
    for row in result["attempts"]:
        logger.event(
            "attempt_evaluated",
            stage="native_iterative_guidance",
            example_id=row["id"],
            attempt=row["attempt"],
            correct=bool(row["result"].get("correct")),
            evaluator_confidence=row["result"].get("confidence"),
        )
    logger.event("run_end", stage="complete", **metrics)
    return metrics


def run_training(
    *,
    method: str,
    data_path: Path,
    model_id: str,
    output_dir: Path,
    max_steps: int,
) -> dict:
    rows = load_training_rows(data_path)
    if method == "dpo":
        result = run_dpo_training(
            model_id=model_id,
            pairs=rows,
            output_dir=output_dir,
            max_steps=max_steps,
            baseline=True,
        )
    elif method == "multilevel_dpo":
        result = run_dpo_training(
            model_id=model_id,
            pairs=rows,
            output_dir=output_dir,
            max_steps=max_steps,
            baseline=False,
        )
    elif method == "distill":
        result = run_distillation_training(
            model_id=model_id,
            rows=rows,
            output_dir=output_dir,
            max_steps=max_steps,
        )
    elif method == "grpo":
        examples = []
        for row in rows:
            if "problem" not in row or "gold_answer" not in row:
                raise ValueError("GRPO data rows require problem and gold_answer")
            examples.append(
                {
                    **row,
                    "prompt": build_native_student_prompt(
                        problem=str(row["problem"]),
                        domain=str(row["domain"]),
                        evidence=row.get("evidence"),
                    ),
                }
            )
        result = run_grpo_training(
            model_id=model_id,
            examples=examples,
            output_dir=output_dir,
            max_steps=max_steps,
        )
    else:
        raise ValueError("method must be dpo, multilevel_dpo, distill, or grpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {key: value for key, value in result.items() if key != "history"}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "history.jsonl", result.get("history", []))
    write_html_report(output_dir / "report.html", metrics, training_history=result.get("history", []))
    return metrics


def run_validate_paper_config(config_path: Path) -> dict[str, object]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    return {
        "valid": True,
        "experiment_id": config.experiment_id,
        "dataset": config.dataset.name,
        "student_model": config.models["student"]["id"],
        "teacher_model": config.models["teacher"]["id"],
        "require_freeze_manifest_for_test": config.require_freeze_manifest_for_test,
    }


def run_materialize_dataset(config_path: Path, source_path: Path, output_dir: Path) -> dict[str, object]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    return materialize_paper_dataset(config, source_path, output_dir)


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
    native = subparsers.add_parser("native-pipeline")
    native.add_argument("--config", required=True, type=Path)
    native.add_argument("--output-dir", type=Path)
    train = subparsers.add_parser("train")
    train.add_argument("--method", required=True, choices=["dpo", "multilevel_dpo", "distill", "grpo"])
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--model-id", required=True)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--max-steps", required=True, type=int)
    validate = subparsers.add_parser("validate-run")
    validate.add_argument("--output-dir", required=True, type=Path)
    paper_config = subparsers.add_parser("validate-paper-config")
    paper_config.add_argument("--config", required=True, type=Path)
    materialize = subparsers.add_parser("materialize-dataset")
    materialize.add_argument("--config", required=True, type=Path)
    materialize.add_argument("--source-path", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)
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
    elif args.command == "native-pipeline":
        result = run_native_pipeline(
            config_path=args.config,
            output_dir=args.output_dir,
        )
    elif args.command == "train":
        result = run_training(
            method=args.method,
            data_path=args.data,
            model_id=args.model_id,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
        )
    elif args.command == "validate-run":
        result = validate_run(args.output_dir)
    elif args.command == "validate-paper-config":
        result = run_validate_paper_config(args.config)
    elif args.command == "materialize-dataset":
        result = run_materialize_dataset(args.config, args.source_path, args.output_dir)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
