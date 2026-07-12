from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from text_feedback_dpo.prompts import build_student_prompt
from text_feedback_dpo.scoring import evaluate_rollout
from text_feedback_dpo.guidance_policy import validate_accumulated_guidance, validate_guidance_surface
from text_feedback_dpo.evaluators import ModelOutputParseError
from text_feedback_dpo.models import ModelGeneration, normalize_model_generation


Example = dict[str, Any]
Generate = Callable[[str], str | ModelGeneration]
NativeEvaluate = Callable[[Example, str], dict[str, Any]]
NativeGuidance = Callable[[Example, str, dict[str, Any], int, int, list[dict[str, Any]]], str]
GuidanceGuard = Callable[[Example, str, dict[str, Any], int], dict[str, Any]]
GuidanceCritic = Callable[[Example, str, dict[str, Any], int], dict[str, Any]]


def _pair(*, example: Example, prompt: str, chosen: str, rejected: str, metadata: dict[str, Any]) -> dict:
    return {
        "id": str(example["id"]),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {"domain": example["domain"], **metadata},
    }


def _format_sft(*, example: Example, prompt: str, completion: str) -> dict:
    return {
        "id": str(example["id"]),
        "prompt": prompt,
        "completion": completion,
        "domain": example["domain"],
    }


def _base_prompt(example: Example) -> str:
    required = ("id", "domain", "problem", "gold_answer")
    missing = [key for key in required if not example.get(key)]
    if missing:
        raise ValueError(f"example is missing required fields: {', '.join(missing)}")
    return build_student_prompt(str(example["problem"]), str(example["domain"]))


def build_exact_dpo_pairs(
    *,
    examples: list[Example],
    student_generate: Generate,
    teacher_correct: Callable[[Example, str, dict], dict[str, str]],
) -> dict:
    """Build the PDF V1 teacher-corrected DPO and format-SFT datasets."""

    pairs: list[dict] = []
    format_sft: list[dict] = []
    failures: list[dict] = []
    base_prompts: dict[str, str] = {}

    for example in examples:
        prompt = _base_prompt(example)
        example_id = str(example["id"])
        base_prompts[example_id] = prompt
        original = student_generate(prompt)
        original_result = evaluate_rollout(original, str(example["gold_answer"]))
        correction = teacher_correct(example, original, original_result)
        feedback = correction.get("feedback")
        corrected = correction.get("corrected_rollout")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError(f"teacher correction for {example_id} is missing feedback")
        if not isinstance(corrected, str) or not corrected.strip():
            raise ValueError(f"teacher correction for {example_id} is missing corrected_rollout")
        corrected_result = evaluate_rollout(corrected, str(example["gold_answer"]))

        if original_result["score"] < corrected_result["score"] and corrected_result["format_valid"]:
            pairs.append(
                _pair(
                    example=example,
                    prompt=prompt,
                    chosen=corrected,
                    rejected=original,
                    metadata={
                        "method": "exact_textual_feedback_dpo",
                        "feedback": feedback,
                        "original_score": original_result["score"],
                        "corrected_score": corrected_result["score"],
                    },
                )
            )
            format_sft.append(_format_sft(example=example, prompt=prompt, completion=corrected))
        else:
            failures.append(
                {
                    "id": example_id,
                    "error_code": "teacher_correction_not_useful",
                    "original_result": original_result,
                    "corrected_result": corrected_result,
                }
            )

    return {
        "pairs": pairs,
        "format_sft": format_sft,
        "failures": failures,
        "base_prompts": base_prompts,
        "metrics": {
            "examples_total": len(examples),
            "accepted_pairs": len(pairs),
            "rejected_examples": len(failures),
        },
    }


def build_iterative_guidance_dpo_pairs(
    *,
    examples: list[Example],
    student_generate: Generate,
    teacher_guidance: Callable[[Example, str, dict, int], str],
    max_guidance_steps: int,
) -> dict:
    """Pair every failed attempt with the first correct post-guidance student rollout."""

    if max_guidance_steps <= 0:
        raise ValueError("max_guidance_steps must be positive")

    pairs: list[dict] = []
    format_sft: list[dict] = []
    failures: list[dict] = []
    base_prompts: dict[str, str] = {}
    first_correct_examples = 0
    wrong_attempts = 0

    for example in examples:
        prompt = _base_prompt(example)
        example_id = str(example["id"])
        base_prompts[example_id] = prompt
        failed_rollouts: list[tuple[int, str, dict]] = []
        attempt = 0
        rollout = student_generate(prompt)

        while True:
            result = evaluate_rollout(rollout, str(example["gold_answer"]))
            if result["score"] == 1.0 and result["format_valid"]:
                if failed_rollouts:
                    first_correct_examples += 1
                    for failed_attempt, failed_rollout, failed_result in failed_rollouts:
                        pairs.append(
                            _pair(
                                example=example,
                                prompt=prompt,
                                chosen=rollout,
                                rejected=failed_rollout,
                                metadata={
                                    "method": "iterative_guidance_dpo",
                                    "failed_attempt": failed_attempt,
                                    "first_correct_attempt": attempt,
                                    "original_score": failed_result["score"],
                                    "corrected_score": result["score"],
                                },
                            )
                        )
                    format_sft.append(_format_sft(example=example, prompt=prompt, completion=rollout))
                break

            failed_rollouts.append((attempt, rollout, result))
            wrong_attempts += 1
            if attempt >= max_guidance_steps:
                failures.append(
                    {
                        "id": example_id,
                        "error_code": "no_correct_rollout_within_guidance_budget",
                        "attempts": attempt + 1,
                        "last_result": result,
                    }
                )
                break

            guidance = teacher_guidance(example, rollout, result, attempt + 1)
            if not isinstance(guidance, str) or not guidance.strip():
                raise ValueError(f"teacher guidance for {example_id} attempt {attempt + 1} is empty")
            # Guidance is available only during generation. The final DPO prompt remains `prompt`.
            rollout = student_generate(f"{prompt}\n\nTeacher guidance for this retry:\n{guidance}\n")
            attempt += 1

    return {
        "pairs": pairs,
        "format_sft": format_sft,
        "failures": failures,
        "base_prompts": base_prompts,
        "metrics": {
            "examples_total": len(examples),
            "accepted_pairs": len(pairs),
            "first_correct_examples": first_correct_examples,
            "wrong_attempts": wrong_attempts,
            "unresolved_examples": len(failures),
        },
    }


def build_native_iterative_guidance_pairs(
    *,
    examples: list[Example],
    base_prompt_builder: Callable[[Example], str],
    retry_prompt_builder: Callable[[str, str], str],
    student_generate: Generate,
    evaluate: NativeEvaluate,
    teacher_guidance: NativeGuidance,
    guidance_guard: GuidanceGuard,
    max_guidance_steps: int,
    max_guidance_regenerations: int,
    guidance_critic: GuidanceCritic | None = None,
) -> dict[str, Any]:
    """Collect native-thinking trajectories and all wrong-vs-first-correct pairs."""

    if max_guidance_steps <= 0:
        raise ValueError("max_guidance_steps must be positive")
    if max_guidance_regenerations < 0:
        raise ValueError("max_guidance_regenerations must be non-negative")

    pairs: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    response_sft: list[dict[str, Any]] = []
    base_prompts: dict[str, str] = {}
    first_correct_attempt: dict[str, int] = {}
    success_by_attempt: dict[str, int] = {}

    for example in examples:
        example_id = str(example["id"])
        base_prompt = base_prompt_builder(example)
        base_prompts[example_id] = base_prompt
        failed: list[tuple[int, str, dict[str, Any]]] = []
        guidance_history: list[str] = []
        current_prompt = base_prompt
        attempt = 0

        while True:
            generation = normalize_model_generation(student_generate(current_prompt))
            response = generation.text
            result = evaluate(example, response)
            if generation.truncated is True:
                result = {
                    **result,
                    "correct": False,
                    "student_truncation_override": True,
                    "student_finish_reason": generation.finish_reason,
                }
            attempt_row = {
                "id": example_id,
                "attempt": attempt,
                "prompt": current_prompt,
                "response": response,
                "result": result,
                "generation": asdict(generation),
            }
            attempts.append(attempt_row)
            if bool(result.get("correct")):
                first_correct_attempt[example_id] = attempt
                success_by_attempt[str(attempt)] = success_by_attempt.get(str(attempt), 0) + 1
                response_sft.append(
                    _format_sft(example=example, prompt=base_prompt, completion=response)
                )
                if failed:
                    for failed_attempt, failed_response, failed_result in failed:
                        pairs.append(
                            _pair(
                                example=example,
                                prompt=base_prompt,
                                chosen=response,
                                rejected=failed_response,
                                metadata={
                                    "method": "native_iterative_guidance_dpo",
                                    "failed_attempt": failed_attempt,
                                    "first_correct_attempt": attempt,
                                    "failed_result": failed_result,
                                    "chosen_result": result,
                                },
                            )
                        )
                break

            failed.append((attempt, response, result))
            if attempt >= max_guidance_steps:
                failures.append(
                    {
                        "id": example_id,
                        "error_code": "no_correct_rollout_within_guidance_budget",
                        "attempts": attempt + 1,
                        "last_result": result,
                    }
                )
                break

            guidance_attempts: list[dict[str, Any]] = []
            safe_guidance: str | None = None
            safe_accumulated: str | None = None
            for regeneration in range(max_guidance_regenerations + 1):
                try:
                    guidance = teacher_guidance(
                        example,
                        response,
                        result,
                        attempt + 1,
                        regeneration,
                        guidance_attempts,
                    )
                except ModelOutputParseError as exc:
                    guidance_attempts.append(
                        {
                            "regeneration": regeneration,
                            "error_code": "teacher_output_parse_failed",
                            "role": exc.role,
                            "message": str(exc),
                            "raw_output": exc.raw,
                            "raw_outputs": exc.raw_outputs,
                            "parse_failures": exc.parse_failures,
                        }
                    )
                    continue
                if not isinstance(guidance, str) or not guidance.strip():
                    raise ValueError(f"teacher guidance for {example_id} attempt {attempt + 1} is empty")
                surface_result = validate_guidance_surface(
                    guidance,
                    problem=str(example["problem"]),
                    gold_answer=str(example["gold_answer"]),
                    evidence=example.get("evidence", []),
                )
                candidate_history = [*guidance_history, guidance]
                accumulated_result = validate_accumulated_guidance(
                    candidate_history,
                    problem=str(example["problem"]),
                    gold_answer=str(example["gold_answer"]),
                    evidence=example.get("evidence", []),
                )
                guard_result: dict[str, Any] | None = None
                critic_result: dict[str, Any] | None = None
                if bool(surface_result["valid"]) and bool(accumulated_result["valid"]):
                    review_result = {**result, "response": response}
                    if guidance_critic is not None:
                        critic_result = guidance_critic(
                            example,
                            str(accumulated_result["accumulated"]),
                            review_result,
                            attempt + 1,
                        )
                    guard_result = guidance_guard(
                        example,
                        str(accumulated_result["accumulated"]),
                        review_result,
                        attempt + 1,
                    )
                guidance_attempts.append(
                    {
                        "regeneration": regeneration,
                        "guidance": guidance,
                        "surface": surface_result,
                        "accumulated": accumulated_result,
                        "critic": critic_result,
                        "guard": guard_result,
                    }
                )
                critic_valid = critic_result is None or bool(critic_result.get("valid"))
                if guard_result is not None and bool(guard_result.get("safe")) and critic_valid:
                    safe_guidance = guidance
                    safe_accumulated = str(accumulated_result["accumulated"])
                    guidance_history = candidate_history
                    break
            if safe_guidance is None:
                critic_rejected = any(
                    record.get("critic") is not None and not bool(record["critic"].get("valid"))
                    for record in guidance_attempts
                )
                guard_rejected = any(
                    record.get("guard") is not None and not bool(record["guard"].get("safe"))
                    for record in guidance_attempts
                )
                parse_rejected = any(
                    record.get("error_code") == "teacher_output_parse_failed"
                    for record in guidance_attempts
                )
                if parse_rejected and not critic_rejected and not guard_rejected:
                    error_code = "teacher_output_parse_failed"
                elif critic_rejected and not guard_rejected:
                    error_code = "invalid_guidance"
                elif guard_rejected:
                    error_code = "unsafe_guidance"
                else:
                    error_code = "invalid_guidance_surface"
                failures.append(
                    {
                        "id": example_id,
                        "error_code": error_code,
                        "attempt": attempt,
                        "guidance_attempts": len(guidance_attempts),
                        "guidance_records": guidance_attempts,
                    }
                )
                break

            attempts[-1]["guidance_records"] = guidance_attempts
            if safe_accumulated is None:
                raise RuntimeError("safe guidance was missing after guidance policy acceptance")
            current_prompt = retry_prompt_builder(base_prompt, safe_accumulated)
            attempt += 1

    unresolved = len(failures)
    return {
        "pairs": pairs,
        "response_sft": response_sft,
        "attempts": attempts,
        "failures": failures,
        "base_prompts": base_prompts,
        "metrics": {
            "examples_total": len(examples),
            "accepted_pairs": len(pairs),
            "first_correct_attempt": first_correct_attempt,
            "success_by_attempt": success_by_attempt,
            "unresolved_examples": unresolved,
            "attempts_total": len(attempts),
            "wrong_attempts": sum(1 for row in attempts if not row["result"].get("correct")),
        },
    }
