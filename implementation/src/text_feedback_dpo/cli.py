from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from text_feedback_dpo.config import load_config
from text_feedback_dpo.benchmarks import load_benchmark_examples
from text_feedback_dpo.collection import collect_paper_shard, merge_paper_collection, paper_generation_kwargs
from text_feedback_dpo.io import read_jsonl, read_jsonl_zst, write_json_atomic, write_jsonl
from text_feedback_dpo.evaluators import (
    ModelOutputParseError,
    make_model_evaluator,
    make_model_guidance_guard,
)
from text_feedback_dpo.evaluation_audit import (
    audit_checkpoint_evaluation,
    rescore_checkpoint_evaluation,
)
from text_feedback_dpo.dataset_manifests import (
    audit_paper_dataset,
    materialize_paper_dataset,
    materialize_preflight_subset,
)
from text_feedback_dpo.experiment_config import load_paper_experiment, validate_paper_experiment
from text_feedback_dpo.methods import build_native_iterative_guidance_pairs
from text_feedback_dpo.models import ModelProvider, TransformersModelProvider
from text_feedback_dpo.observability import JsonlLogger
from text_feedback_dpo.paper_training import train_paper_dpo, train_paper_grpo
from text_feedback_dpo.hyperparameter_search import (
    DpoCandidate,
    GrpoCandidate,
    build_dpo_candidates,
    build_grpo_candidates,
    create_search_ledger,
    freeze_selection,
    promote_stage,
    register_observation,
)
from text_feedback_dpo.heldout import (
    build_baseline_evaluation_freeze,
    build_transformers_checkpoint_generator,
    evaluate_checkpoint,
    merge_checkpoint_evaluations,
)
from text_feedback_dpo.preference_data import build_preference_datasets
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
from text_feedback_dpo.sharding import shard_rows


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

    def teacher_guidance(
        example: dict,
        rollout: str,
        result: dict,
        attempt: int,
        regeneration: int,
        prior_reviews: list[dict[str, Any]],
    ) -> str:
        prompt = build_privileged_guidance_prompt(
            problem=str(example["problem"]),
            gold_answer=str(example["gold_answer"]),
            rollout=rollout,
            result=result,
            domain=str(example["domain"]),
            prior_reviews=prior_reviews,
        )
        start = time.monotonic_ns()
        guidance = provider.generate("teacher", prompt, **dict(config["teacher_generation"]))
        generation_events.append(
            {
                "role": "teacher",
                "example_id": str(example["id"]),
                "attempt": attempt,
                "regeneration": regeneration,
                "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                "generated_tokens_estimate": len(guidance.split()),
            }
        )
        guidance_events.append(
            {
                "id": str(example["id"]),
                "attempt": attempt,
                "regeneration": regeneration,
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
    result = materialize_paper_dataset(config, source_path, output_dir)
    manifest = result["manifest"]
    return {
        "schema": "paper-dataset-materialization-summary-v1",
        "dataset": manifest["metadata"]["dataset"],
        "output_dir": result["output_dir"],
        "roles": manifest["roles"],
        "nested_roles": manifest["nested_roles"],
        "content_sha256": manifest["content_sha256"],
    }


def run_audit_dataset(config_path: Path, dataset_dir: Path, output_path: Path) -> dict[str, object]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    return audit_paper_dataset(config, dataset_dir, output_path=output_path)


def run_materialize_preflight_subset(
    *,
    source_path: Path,
    output_path: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    return materialize_preflight_subset(
        source_path=source_path,
        output_path=output_path,
        count=count,
        seed=seed,
    )


def run_collect_shard(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    shard_index: int,
    num_shards: int,
    source_commit: str,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    rows_path = dataset_dir / f"{split}.jsonl.zst"
    examples = read_jsonl_zst(rows_path)
    return collect_paper_shard(
        config=config,
        config_path=config_path,
        examples=examples,
        dataset_dir=dataset_dir,
        output_root=output_dir,
        split=split,
        shard_index=shard_index,
        num_shards=num_shards,
        source_commit=source_commit,
    )


def run_merge_collection(
    *,
    config_path: Path,
    dataset_dir: Path,
    collection_dir: Path,
    expected_shards: int,
    output_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    return merge_paper_collection(
        config_path=config_path,
        dataset_dir=dataset_dir,
        collection_dir=collection_dir,
        expected_shards=expected_shards,
        output_path=output_path,
        source_commit=source_commit,
    )


def run_build_preferences(
    *,
    collection_path: Path,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing preference output directory: {output_dir}")
    records = read_jsonl_zst(collection_path)
    attempts: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for record in records:
        example_id = record.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("merged collection record is missing id")
        if example_id in seen_groups:
            raise ValueError(f"merged collection contains duplicate group: {example_id}")
        seen_groups.add(example_id)
        group_attempts = record.get("attempts")
        if not isinstance(group_attempts, list):
            raise ValueError(f"merged collection record {example_id} is missing attempts")
        for attempt in group_attempts:
            if not isinstance(attempt, dict):
                raise ValueError(f"merged collection record {example_id} contains invalid attempt")
            attempts.append({"id": example_id, **attempt})
    examples = read_jsonl_zst(dataset_path)
    datasets = build_preference_datasets(
        attempts=attempts,
        examples=examples,
        seed=seed,
        base_prompt_builder=lambda example: build_native_student_prompt(
            problem=str(example["problem"]),
            domain=str(example["domain"]),
            evidence=example.get("evidence"),
        ),
    )
    output_dir.mkdir(parents=True)
    artifact_names = ("standard", "multilevel", "matched", "response_sft", "unresolved")
    artifacts: dict[str, dict[str, Any]] = {}
    for method in artifact_names:
        path = output_dir / f"{method}.jsonl"
        write_jsonl(path, datasets[method])
        artifacts[method] = {
            "path": path.name,
            "rows": len(datasets[method]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "schema": "paper-preference-v2",
            "seed": seed,
            "source_collection": str(collection_path),
            "source_collection_sha256": hashlib.sha256(collection_path.read_bytes()).hexdigest(),
            "source_dataset": str(dataset_path),
            "source_dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "metrics": datasets["metrics"],
            "artifacts": artifacts,
        },
    )
    return datasets["metrics"]


def _paper_candidates(config: Any, method: str) -> list[Any]:
    if method in {"standard_dpo", "multilevel_dpo", "matched_dpo"}:
        return build_dpo_candidates(
            learning_rates=config.dpo_search.learning_rates,
            betas=config.dpo_search.betas,
            weight_decay=config.optimizer.weight_decay,
            warmup_fraction=config.optimizer.warmup_fraction,
            scheduler=config.optimizer.scheduler,
            loss_type=config.dpo_search.loss_type,
        )
    if method == "ld_dpo":
        return [
            candidate
            for alpha in config.dpo_search.ld_alpha_values
            for candidate in build_dpo_candidates(
                learning_rates=config.dpo_search.learning_rates,
                betas=config.dpo_search.betas,
                weight_decay=config.optimizer.weight_decay,
                warmup_fraction=config.optimizer.warmup_fraction,
                scheduler=config.optimizer.scheduler,
                loss_type="sigmoid",
                ld_alpha=alpha,
            )
        ]
    if method in {"grpo", "dapo_sensitivity"}:
        return build_grpo_candidates(
            learning_rates=config.grpo_search.learning_rates,
            kl_betas=config.grpo_search.kl_betas,
            epsilon=config.grpo_search.epsilon,
            num_iterations=config.grpo_search.num_iterations,
            num_generations=config.grpo_search.num_generations,
            loss_type=config.grpo_search.loss_type,
        )
    raise ValueError(f"unsupported paper method: {method}")


def run_init_search_ledger(*, config_path: Path, method: str, output_path: Path, dataset_manifest_hash: str) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    candidates = _paper_candidates(config, method)
    promote_counts = config.dpo_search.promote_counts if method.endswith("dpo") else config.grpo_search.promote_counts
    ledger = create_search_ledger(
        method=method,
        candidates=candidates,
        promote_counts=promote_counts,
        dataset_manifest_hash=dataset_manifest_hash,
        seed=config.dataset.seed,
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite search ledger: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"method": method, "candidates": len(candidates), "ledger": str(output_path)}


def run_promote_search_stage(*, ledger_path: Path, stage: int) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    promoted = promote_stage(ledger, stage=stage)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"stage": stage, "promoted": promoted}


def run_freeze_search(*, ledger_path: Path, candidate_id: str, stage: int, output_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = freeze_selection(ledger, candidate_id=candidate_id, stage=stage)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite freeze manifest: {output_path}")
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _candidate_from_ledger(ledger: dict[str, Any], candidate_id: str) -> Any:
    payload = ledger.get("candidates", {}).get(candidate_id)
    if not isinstance(payload, dict):
        raise ValueError(f"candidate is not present in search ledger: {candidate_id}")
    if "beta" in payload:
        return DpoCandidate(
            learning_rate=float(payload["learning_rate"]),
            beta=float(payload["beta"]),
            weight_decay=float(payload["weight_decay"]),
            warmup_fraction=float(payload["warmup_fraction"]),
            scheduler=str(payload["scheduler"]),
            loss_type=str(payload.get("loss_type", "sigmoid_norm")),
            ld_alpha=float(payload["ld_alpha"]) if payload.get("ld_alpha") is not None else None,
        )
    return GrpoCandidate(
        learning_rate=float(payload["learning_rate"]),
        kl_beta=float(payload["kl_beta"]),
        epsilon=float(payload["epsilon"]),
        num_iterations=int(payload["num_iterations"]),
        num_generations=int(payload["num_generations"]),
        loss_type=str(payload["loss_type"]),
    )


def _read_rows_for_paper(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_zst(path) if path.name.endswith(".zst") else read_jsonl(path)


def _paper_evaluator(config: Any):
    provider = TransformersModelProvider(
        model_ids={"evaluator": config.models["evaluator"]["id"]},
        model_revisions={"evaluator": config.models["evaluator"]["revision"]},
    )
    return make_model_evaluator(
        generate=provider.generate_result,
        generation_kwargs=paper_generation_kwargs(config, role="evaluator"),
        max_regenerations=int(config.collection["max_guidance_regenerations"]),
    )


def _paper_examples_with_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "prompt": build_native_student_prompt(
                problem=str(row["problem"]),
                domain=str(row["domain"]),
                evidence=row.get("evidence"),
            ),
        }
        for row in rows
    ]


def _selection_metric(config: Any, metrics: dict[str, Any]) -> float:
    if config.dataset.name in {"gsm8k", "math"}:
        return float(metrics["math"]["exact_accuracy"])
    return float(metrics["search_qa"]["exact_match"])


def run_tune_paper(
    *,
    config_path: Path,
    method: str,
    candidate_id: str,
    stage: int,
    data_path: Path,
    validation_path: Path,
    output_dir: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("method") != method:
        raise ValueError("search ledger method does not match tune method")
    candidate = _candidate_from_ledger(ledger, candidate_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    if method.endswith("dpo"):
        train_metrics = train_paper_dpo(
            config=config,
            method=method,
            pairs=_read_rows_for_paper(data_path),
            output_dir=output_dir / "train",
            candidate=candidate,
            seed=config.dataset.seed,
        )
    else:
        evaluator = _paper_evaluator(config)
        train_metrics = train_paper_grpo(
            config=config,
            method=method,
            examples=_paper_examples_with_prompts(_read_rows_for_paper(data_path)),
            output_dir=output_dir / "train",
            candidate=candidate,
            seed=config.dataset.seed,
            evaluator=evaluator,
        )
    adapter_manifest = json.loads((output_dir / "train" / "adapter_manifest.json").read_text(encoding="utf-8"))
    generator = build_transformers_checkpoint_generator(
        model_id=config.models["student"]["id"],
        revision=config.models["student"]["revision"],
        generation_kwargs=paper_generation_kwargs(config, role="student"),
        adapter_dir=output_dir / "train",
        adapter_base_revision=config.models["student"]["revision"],
        adapter_lora_coverage_hash=adapter_manifest["lora"]["coverage_hash"],
    )
    validation_metrics = evaluate_checkpoint(
        examples=_read_rows_for_paper(validation_path),
        generate=generator,
        evaluator=_paper_evaluator(config),
        output_dir=output_dir / "validation",
        checkpoint_kind="adapter",
        base_model_revision=config.models["student"]["revision"],
        seed=config.dataset.seed,
        test=False,
        adapter_manifest=adapter_manifest,
    )
    artifact_hash = hashlib.sha256((output_dir / "train" / "train_metrics.json").read_bytes()).hexdigest()
    register_observation(
        ledger,
        candidate_id=candidate_id,
        stage=stage,
        status="valid",
        metrics={"selection_metric": _selection_metric(config, validation_metrics), "gpu_hours": 0.0},
        artifact_hash=artifact_hash,
    )
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"candidate_id": candidate_id, "stage": stage, "validation": validation_metrics, "train": train_metrics}


def run_train_paper(
    *,
    config_path: Path,
    method: str,
    seed: int,
    data_path: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
    evaluator: Any | None = None,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    if freeze.get("method") != method or not freeze.get("candidate"):
        raise ValueError("freeze manifest does not contain the requested method and candidate")
    payload = freeze["candidate"]
    candidate = _candidate_from_ledger({"candidates": {freeze["candidate_id"]: payload}}, freeze["candidate_id"])
    if method.endswith("dpo"):
        return train_paper_dpo(
            config=config,
            method=method,
            pairs=_read_rows_for_paper(data_path),
            output_dir=output_dir,
            candidate=candidate,
            seed=seed,
        )
    if evaluator is None:
        evaluator = _paper_evaluator(config)
    return train_paper_grpo(
        config=config,
        method=method,
        examples=_paper_examples_with_prompts(_read_rows_for_paper(data_path)),
        output_dir=output_dir,
        candidate=candidate,
        seed=seed,
        evaluator=evaluator,
    )


def run_evaluate_paper(
    *,
    config_path: Path,
    checkpoint: Path | None,
    checkpoint_kind: str,
    data_path: Path,
    split: str,
    output_dir: Path,
    freeze_manifest: Path | None,
    source_commit: str,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    test = split == "test"
    if freeze_manifest is None or not freeze_manifest.exists():
        raise FileNotFoundError("paper evaluation requires an existing --freeze-manifest")
    if checkpoint_kind not in {"base", "adapter"}:
        raise ValueError("checkpoint_kind must be base or adapter")
    adapter_manifest: dict[str, Any] | None = None
    if checkpoint_kind == "base":
        if checkpoint is not None:
            raise ValueError("base evaluation must not receive an adapter checkpoint path")
        expected_freeze = _build_baseline_freeze_from_paths(
            config_path=config_path,
            dataset_manifest_path=data_path.parent / "manifest.json",
            source_commit=source_commit,
        )
        actual_freeze = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        if actual_freeze != expected_freeze:
            raise ValueError(
                "baseline freeze manifest does not match the source, config, dataset, models, or evaluation protocol"
            )
    else:
        if checkpoint is None:
            raise ValueError("adapter evaluation requires --checkpoint")
        adapter_manifest_path = checkpoint / "adapter_manifest.json"
        if not adapter_manifest_path.exists():
            raise FileNotFoundError(f"adapter manifest does not exist: {adapter_manifest_path}")
        adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    generator = build_transformers_checkpoint_generator(
        model_id=config.models["student"]["id"],
        revision=config.models["student"]["revision"],
        generation_kwargs=paper_generation_kwargs(config, role="student"),
        adapter_dir=checkpoint if checkpoint_kind == "adapter" else None,
        adapter_base_revision=(
            config.models["student"]["revision"] if checkpoint_kind == "adapter" else None
        ),
        adapter_lora_coverage_hash=(
            adapter_manifest["lora"]["coverage_hash"] if adapter_manifest is not None else None
        ),
    )
    examples = _read_rows_for_paper(data_path)
    evaluation_rows = shard_rows(examples, shard_index=shard_index, num_shards=num_shards)
    return evaluate_checkpoint(
        examples=evaluation_rows,
        generate=generator,
        evaluator=_paper_evaluator(config),
        output_dir=output_dir,
        checkpoint_kind=checkpoint_kind,
        base_model_revision=config.models["student"]["revision"],
        seed=int(config.evaluation["generation_seed"]),
        test=test,
        freeze_manifest=freeze_manifest,
        adapter_manifest=adapter_manifest,
        require_generation_metadata=True,
        shard_index=shard_index,
        num_shards=num_shards,
    )


def _build_baseline_freeze_from_paths(
    *,
    config_path: Path,
    dataset_manifest_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest does not exist: {dataset_manifest_path}")
    return build_baseline_evaluation_freeze(
        experiment_id=config.experiment_id,
        source_commit=source_commit,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest(),
        student_model=config.models["student"],
        evaluator_model=config.models["evaluator"],
        prompt_protocol=str(config.collection["prompt_protocol"]),
        student_generation=paper_generation_kwargs(config, role="student"),
        evaluator_generation=paper_generation_kwargs(config, role="evaluator"),
        generation_seed=int(config.evaluation["generation_seed"]),
    )


def run_freeze_baseline(
    *,
    config_path: Path,
    dataset_manifest_path: Path,
    source_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite baseline freeze manifest: {output_path}")
    freeze = _build_baseline_freeze_from_paths(
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        source_commit=source_commit,
    )
    write_json_atomic(output_path, freeze)
    return freeze


def run_merge_evaluations(
    *,
    config_path: Path,
    data_path: Path,
    split: str,
    shard_root: Path,
    expected_shards: int,
    output_dir: Path,
    checkpoint_kind: str,
    freeze_manifest: Path,
    source_commit: str,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    if checkpoint_kind == "base":
        expected_freeze = _build_baseline_freeze_from_paths(
            config_path=config_path,
            dataset_manifest_path=data_path.parent / "manifest.json",
            source_commit=source_commit,
        )
        actual_freeze = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        if actual_freeze != expected_freeze:
            raise ValueError("baseline freeze manifest does not match the merge protocol")
    metrics = merge_checkpoint_evaluations(
        examples=_read_rows_for_paper(data_path),
        shard_root=shard_root,
        expected_shards=expected_shards,
        output_dir=output_dir,
        checkpoint_kind=checkpoint_kind,
        base_model_revision=config.models["student"]["revision"],
        seed=int(config.evaluation["generation_seed"]),
        test=split == "test",
        freeze_manifest=freeze_manifest,
    )
    report_metrics: dict[str, Any] = {
        "checkpoint_kind": checkpoint_kind,
        "split": split,
        "generation_seed": int(config.evaluation["generation_seed"]),
    }
    for section, values in metrics.items():
        if section == "per_example" or not isinstance(values, dict):
            continue
        for key, value in values.items():
            report_metrics[f"{section}.{key}"] = value
    write_html_report(output_dir / "report.html", report_metrics)
    return metrics


def run_audit_evaluation(
    *,
    config_path: Path,
    predictions_path: Path,
    labels_path: Path,
    output_dir: Path,
    minimum_labels: int,
) -> dict[str, Any]:
    config = load_paper_experiment(config_path)
    validate_paper_experiment(config)
    return audit_checkpoint_evaluation(
        predictions_path=predictions_path,
        labels_path=labels_path,
        output_dir=output_dir,
        minimum_labels=minimum_labels,
        minimum_agreement=float(config.evaluation["minimum_evaluator_audit_agreement"]),
        max_truncation_rate=float(config.evaluation["max_truncation_rate"]),
    )


def run_rescore_evaluation(
    *,
    predictions_path: Path,
    examples_path: Path,
    output_dir: Path,
    source_commit: str,
) -> dict[str, Any]:
    return rescore_checkpoint_evaluation(
        predictions_path=predictions_path,
        examples_path=examples_path,
        output_dir=output_dir,
        source_commit=source_commit,
    )


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
    audit_dataset = subparsers.add_parser("audit-dataset")
    audit_dataset.add_argument("--config", required=True, type=Path)
    audit_dataset.add_argument("--dataset-dir", required=True, type=Path)
    audit_dataset.add_argument("--output-path", required=True, type=Path)
    preflight_subset = subparsers.add_parser("materialize-preflight-subset")
    preflight_subset.add_argument("--source-path", required=True, type=Path)
    preflight_subset.add_argument("--output-path", required=True, type=Path)
    preflight_subset.add_argument("--count", required=True, type=int)
    preflight_subset.add_argument("--seed", required=True, type=int)
    collect = subparsers.add_parser("collect-shard")
    collect.add_argument("--config", required=True, type=Path)
    collect.add_argument("--dataset-dir", required=True, type=Path)
    collect.add_argument("--output-dir", required=True, type=Path)
    collect.add_argument("--split", required=True)
    collect.add_argument("--shard-index", required=True, type=int)
    collect.add_argument("--num-shards", required=True, type=int)
    collect.add_argument("--source-commit", required=True)
    merge = subparsers.add_parser("merge-collection")
    merge.add_argument("--config", required=True, type=Path)
    merge.add_argument("--dataset-dir", required=True, type=Path)
    merge.add_argument("--collection-dir", required=True, type=Path)
    merge.add_argument("--expected-shards", required=True, type=int)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--source-commit", required=True)
    preferences = subparsers.add_parser("build-preferences")
    preferences.add_argument("--collection", required=True, type=Path)
    preferences.add_argument("--dataset", required=True, type=Path)
    preferences.add_argument("--output-dir", required=True, type=Path)
    preferences.add_argument("--seed", required=True, type=int)
    init_ledger = subparsers.add_parser("init-search-ledger")
    init_ledger.add_argument("--config", required=True, type=Path)
    init_ledger.add_argument("--method", required=True, choices=["standard_dpo", "multilevel_dpo", "matched_dpo", "ld_dpo", "grpo"])
    init_ledger.add_argument("--dataset-manifest-hash", required=True)
    init_ledger.add_argument("--output", required=True, type=Path)
    promote = subparsers.add_parser("promote-search-stage")
    promote.add_argument("--ledger", required=True, type=Path)
    promote.add_argument("--stage", required=True, type=int)
    freeze = subparsers.add_parser("freeze-search")
    freeze.add_argument("--ledger", required=True, type=Path)
    freeze.add_argument("--candidate-id", required=True)
    freeze.add_argument("--stage", required=True, type=int)
    freeze.add_argument("--output", required=True, type=Path)
    baseline_freeze = subparsers.add_parser("freeze-baseline")
    baseline_freeze.add_argument("--config", required=True, type=Path)
    baseline_freeze.add_argument("--dataset-manifest", required=True, type=Path)
    baseline_freeze.add_argument("--source-commit", required=True)
    baseline_freeze.add_argument("--output", required=True, type=Path)
    tune = subparsers.add_parser("tune-paper")
    tune.add_argument("--config", required=True, type=Path)
    tune.add_argument("--method", required=True, choices=["standard_dpo", "multilevel_dpo", "matched_dpo", "grpo"])
    tune.add_argument("--candidate-id", required=True)
    tune.add_argument("--stage", required=True, type=int)
    tune.add_argument("--data", required=True, type=Path)
    tune.add_argument("--validation", required=True, type=Path)
    tune.add_argument("--output-dir", required=True, type=Path)
    tune.add_argument("--ledger", required=True, type=Path)
    paper_train = subparsers.add_parser("train-paper")
    paper_train.add_argument("--config", required=True, type=Path)
    paper_train.add_argument("--method", required=True, choices=["standard_dpo", "multilevel_dpo", "matched_dpo", "grpo", "dapo_sensitivity"])
    paper_train.add_argument("--seed", required=True, type=int)
    paper_train.add_argument("--data", required=True, type=Path)
    paper_train.add_argument("--freeze-manifest", required=True, type=Path)
    paper_train.add_argument("--output-dir", required=True, type=Path)
    paper_eval = subparsers.add_parser("evaluate-paper")
    paper_eval.add_argument("--config", required=True, type=Path)
    paper_eval.add_argument("--checkpoint", type=Path)
    paper_eval.add_argument("--checkpoint-kind", required=True, choices=["base", "adapter"])
    paper_eval.add_argument("--data", required=True, type=Path)
    paper_eval.add_argument("--split", required=True, choices=["validation", "test"])
    paper_eval.add_argument("--output-dir", required=True, type=Path)
    paper_eval.add_argument("--freeze-manifest", type=Path)
    paper_eval.add_argument("--source-commit", required=True)
    paper_eval.add_argument("--shard-index", required=True, type=int)
    paper_eval.add_argument("--num-shards", required=True, type=int)
    merge_eval = subparsers.add_parser("merge-evaluations")
    merge_eval.add_argument("--config", required=True, type=Path)
    merge_eval.add_argument("--data", required=True, type=Path)
    merge_eval.add_argument("--split", required=True, choices=["validation", "test"])
    merge_eval.add_argument("--shard-root", required=True, type=Path)
    merge_eval.add_argument("--expected-shards", required=True, type=int)
    merge_eval.add_argument("--output-dir", required=True, type=Path)
    merge_eval.add_argument("--checkpoint-kind", required=True, choices=["base", "adapter"])
    merge_eval.add_argument("--freeze-manifest", required=True, type=Path)
    merge_eval.add_argument("--source-commit", required=True)
    audit_eval = subparsers.add_parser("audit-evaluation")
    audit_eval.add_argument("--config", required=True, type=Path)
    audit_eval.add_argument("--predictions", required=True, type=Path)
    audit_eval.add_argument("--labels", required=True, type=Path)
    audit_eval.add_argument("--output-dir", required=True, type=Path)
    audit_eval.add_argument("--minimum-labels", required=True, type=int)
    rescore_eval = subparsers.add_parser("rescore-evaluation")
    rescore_eval.add_argument("--predictions", required=True, type=Path)
    rescore_eval.add_argument("--examples", required=True, type=Path)
    rescore_eval.add_argument("--output-dir", required=True, type=Path)
    rescore_eval.add_argument("--source-commit", required=True)
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
    elif args.command == "audit-dataset":
        result = run_audit_dataset(args.config, args.dataset_dir, args.output_path)
    elif args.command == "materialize-preflight-subset":
        result = run_materialize_preflight_subset(
            source_path=args.source_path,
            output_path=args.output_path,
            count=args.count,
            seed=args.seed,
        )
    elif args.command == "collect-shard":
        result = run_collect_shard(
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split=args.split,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            source_commit=args.source_commit,
        )
    elif args.command == "merge-collection":
        result = run_merge_collection(
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            collection_dir=args.collection_dir,
            expected_shards=args.expected_shards,
            output_path=args.output,
            source_commit=args.source_commit,
        )
    elif args.command == "build-preferences":
        result = run_build_preferences(
            collection_path=args.collection,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.command == "init-search-ledger":
        result = run_init_search_ledger(
            config_path=args.config,
            method=args.method,
            output_path=args.output,
            dataset_manifest_hash=args.dataset_manifest_hash,
        )
    elif args.command == "promote-search-stage":
        result = run_promote_search_stage(ledger_path=args.ledger, stage=args.stage)
    elif args.command == "freeze-search":
        result = run_freeze_search(
            ledger_path=args.ledger,
            candidate_id=args.candidate_id,
            stage=args.stage,
            output_path=args.output,
        )
    elif args.command == "freeze-baseline":
        result = run_freeze_baseline(
            config_path=args.config,
            dataset_manifest_path=args.dataset_manifest,
            source_commit=args.source_commit,
            output_path=args.output,
        )
    elif args.command == "tune-paper":
        result = run_tune_paper(
            config_path=args.config,
            method=args.method,
            candidate_id=args.candidate_id,
            stage=args.stage,
            data_path=args.data,
            validation_path=args.validation,
            output_dir=args.output_dir,
            ledger_path=args.ledger,
        )
    elif args.command == "train-paper":
        result = run_train_paper(
            config_path=args.config,
            method=args.method,
            seed=args.seed,
            data_path=args.data,
            freeze_manifest_path=args.freeze_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "evaluate-paper":
        result = run_evaluate_paper(
            config_path=args.config,
            checkpoint=args.checkpoint,
            checkpoint_kind=args.checkpoint_kind,
            data_path=args.data,
            split=args.split,
            output_dir=args.output_dir,
            freeze_manifest=args.freeze_manifest,
            source_commit=args.source_commit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.command == "merge-evaluations":
        result = run_merge_evaluations(
            config_path=args.config,
            data_path=args.data,
            split=args.split,
            shard_root=args.shard_root,
            expected_shards=args.expected_shards,
            output_dir=args.output_dir,
            checkpoint_kind=args.checkpoint_kind,
            freeze_manifest=args.freeze_manifest,
            source_commit=args.source_commit,
        )
    elif args.command == "audit-evaluation":
        result = run_audit_evaluation(
            config_path=args.config,
            predictions_path=args.predictions,
            labels_path=args.labels,
            output_dir=args.output_dir,
            minimum_labels=args.minimum_labels,
        )
    elif args.command == "rescore-evaluation":
        result = run_rescore_evaluation(
            predictions_path=args.predictions,
            examples_path=args.examples,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
        )
    else:
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
