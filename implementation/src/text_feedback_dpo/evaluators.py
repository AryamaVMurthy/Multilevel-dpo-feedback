from __future__ import annotations

import json
import time
from typing import Any, Callable

from text_feedback_dpo.answer_evaluation import evaluate_domain_answer


class ModelOutputParseError(ValueError):
    def __init__(self, *, role: str, raw: str, message: str) -> None:
        super().__init__(message)
        self.role = role
        self.raw = raw


def _parse_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        parsed = value
    if parsed is not None:
        return parsed
    raise ValueError("model output does not contain a valid JSON object")


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
    if not value["answer"].strip():
        raise ValueError("evaluator field answer must be non-empty")
    if not isinstance(value["confidence"], (int, float)) or not 0 <= float(value["confidence"]) <= 1:
        raise ValueError("evaluator confidence must be between 0 and 1")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("evaluator reason must be non-empty")
    return value


def parse_guidance_guard_output(raw: str) -> dict[str, Any]:
    token = raw.strip().upper()
    if token in {"SAFE", "UNSAFE"}:
        return {
            "safe": token == "SAFE",
            "confidence": None,
            "reason": "explicit single-token guidance guard verdict",
            "serialization": "single_token",
        }
    if token and "{" not in token:
        raise ValueError("guidance guard output must be exactly SAFE or UNSAFE")
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

Do not provide analysis, reasoning, headings, or a preamble. The first character of your response
must be '{{' and the last character must be '}}'. Return exactly one JSON object with these fields:
correct: boolean
answer: string containing the answer you evaluated
confidence: number from 0 to 1
reason: short single-line explanation, at most 160 characters, with no quotation marks or braces
Do not use quotation marks or braces inside the reason string.
Use a single line for the reason string.
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

Do not provide analysis, reasoning, headings, or a preamble. Return exactly one token: SAFE or UNSAFE.
Return SAFE only when the guidance cannot disclose the answer; otherwise return UNSAFE.

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
        start = time.monotonic_ns()
        raw = generate("evaluator", build_evaluator_prompt(example=example, response=response), **generation_kwargs)
        try:
            parsed = parse_evaluator_output(raw)
        except ValueError as exc:
            raise ModelOutputParseError(role="evaluator", raw=raw, message=str(exc)) from exc
        actual_answer_type = parsed.get("answer_type", "unknown")
        if not isinstance(actual_answer_type, str) or not actual_answer_type.strip():
            raise ModelOutputParseError(
                role="evaluator",
                raw=raw,
                message="evaluator field answer_type must be a non-empty string when supplied",
            )
        model_evidence_supported = parsed.get("evidence_supported")
        if model_evidence_supported is not None and not isinstance(model_evidence_supported, bool):
            raise ModelOutputParseError(
                role="evaluator",
                raw=raw,
                message="evaluator field evidence_supported must be boolean when supplied",
            )
        try:
            deterministic = evaluate_domain_answer(
                domain=str(example["domain"]),
                prediction=parsed["answer"],
                example=example,
                actual_answer_type=actual_answer_type,
                evidence_supported=model_evidence_supported,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelOutputParseError(
                role="evaluator",
                raw=raw,
                message=f"deterministic answer evaluation failed: {exc}",
            ) from exc
        model_correct = bool(parsed["correct"])
        requires_model_judgment = bool(deterministic.get("requires_model_judgment"))
        parsed["model_correct"] = model_correct
        parsed["deterministic"] = deterministic
        parsed["deterministic_correct"] = bool(deterministic["correct"])
        parsed["requires_model_judgment"] = requires_model_judgment
        # Deterministic checks act as a consistency gate for clear cases. Ambiguous cases remain
        # under the evaluator model's judgment and are visible in the result for auditability.
        parsed["correct"] = model_correct if requires_model_judgment else model_correct and bool(deterministic["correct"])
        parsed["raw_evaluator_output"] = raw
        parsed["latency_ms"] = (time.monotonic_ns() - start) // 1_000_000
        parsed["generated_tokens_estimate"] = len(raw.split())
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
        start = time.monotonic_ns()
        raw = generate(
            "evaluator",
            build_guidance_guard_prompt(example=example, guidance=guidance),
            **generation_kwargs,
        )
        try:
            parsed = parse_guidance_guard_output(raw)
        except ValueError as exc:
            raise ModelOutputParseError(role="guidance_guard", raw=raw, message=str(exc)) from exc
        parsed["guidance"] = guidance
        parsed["raw_guard_output"] = raw
        parsed["latency_ms"] = (time.monotonic_ns() - start) // 1_000_000
        parsed["generated_tokens_estimate"] = len(raw.split())
        return parsed

    return guard
