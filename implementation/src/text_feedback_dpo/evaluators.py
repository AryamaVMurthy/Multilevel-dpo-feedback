from __future__ import annotations

import json
from typing import Any, Callable

from text_feedback_dpo.prompts import build_native_student_prompt


def _parse_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if raw[index + end :].strip():
            continue
        return value
    raise ValueError("evaluator output does not end with one valid JSON object")


def parse_evaluator_output(raw: str) -> dict[str, Any]:
    value = _parse_json_object(raw.strip())
    required = ("correct", "answer", "confidence", "reason")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"evaluator output is missing required fields: {', '.join(missing)}")
    if not isinstance(value["correct"], bool):
        raise ValueError("evaluator field correct must be boolean")
    if not isinstance(value["answer"], str):
        raise ValueError("evaluator field answer must be a string")
    if not isinstance(value["confidence"], (int, float)) or not 0 <= float(value["confidence"]) <= 1:
        raise ValueError("evaluator confidence must be between 0 and 1")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("evaluator reason must be non-empty")
    return value


def parse_guidance_guard_output(raw: str) -> dict[str, Any]:
    value = _parse_json_object(raw.strip())
    required = ("safe", "confidence", "reason")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"guidance guard output is missing required fields: {', '.join(missing)}")
    if not isinstance(value["safe"], bool):
        raise ValueError("guidance guard field safe must be boolean")
    if not isinstance(value["confidence"], (int, float)) or not 0 <= float(value["confidence"]) <= 1:
        raise ValueError("guidance guard confidence must be between 0 and 1")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("guidance guard reason must be non-empty")
    return value


def build_evaluator_prompt(*, example: dict[str, Any], response: str) -> str:
    evidence = example.get("evidence", [])
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "(none supplied)"
    return f"""Evaluate a model response to a {example["domain"]} problem.

Use the gold answer and controlled evidence only for evaluation. The model response may contain
private reasoning followed by an answer. Decide whether its final answer is correct, not whether
its style matches another model.

Return exactly one JSON object with these fields:
correct: boolean
answer: string containing the answer you evaluated
confidence: number from 0 to 1
reason: short explanation
For SearchQA also include answer_type and evidence_supported.

Problem:
{example["problem"]}

Gold answer:
{example["gold_answer"]}

Controlled evidence:
{evidence_text}

Model response:
{response}
"""


def build_guidance_guard_prompt(*, example: dict[str, Any], guidance: str) -> str:
    return f"""Check whether teacher guidance leaks the answer to a student.

The guidance is unsafe if it states the exact answer, an equivalent expression, a decisive named
entity, or enough information to copy the answer without solving. A hint that only identifies a
mistake or next verification is safe.

Return exactly one JSON object with fields:
safe: boolean
confidence: number from 0 to 1
reason: short explanation

Problem:
{example["problem"]}

Gold answer:
{example["gold_answer"]}

Teacher guidance:
{guidance}
"""


def make_model_evaluator(
    *,
    generate: Callable[..., str],
    generation_kwargs: dict[str, Any],
) -> Callable[[dict[str, Any], str], dict[str, Any]]:
    def evaluate(example: dict[str, Any], response: str) -> dict[str, Any]:
        raw = generate("evaluator", build_evaluator_prompt(example=example, response=response), **generation_kwargs)
        parsed = parse_evaluator_output(raw)
        parsed["raw_evaluator_output"] = raw
        return parsed

    return evaluate


def make_model_guidance_guard(
    *,
    generate: Callable[..., str],
    generation_kwargs: dict[str, Any],
) -> Callable[[dict[str, Any], str, dict[str, Any], int], dict[str, Any]]:
    def guard(
        example: dict[str, Any],
        guidance: str,
        _result: dict[str, Any],
        _attempt: int,
    ) -> dict[str, Any]:
        raw = generate(
            "evaluator",
            build_guidance_guard_prompt(example=example, guidance=guidance),
            **generation_kwargs,
        )
        parsed = parse_guidance_guard_output(raw)
        parsed["guidance"] = guidance
        parsed["raw_guard_output"] = raw
        return parsed

    return guard
