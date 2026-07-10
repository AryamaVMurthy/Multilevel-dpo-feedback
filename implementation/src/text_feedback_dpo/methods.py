from __future__ import annotations

from collections.abc import Callable
from typing import Any

from text_feedback_dpo.prompts import build_student_prompt
from text_feedback_dpo.scoring import evaluate_rollout


Example = dict[str, Any]
Generate = Callable[[str], str]


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
