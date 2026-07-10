from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Callable

from text_feedback_dpo.answer_evaluation import evaluate_domain_answer
from text_feedback_dpo.models import normalize_model_generation


class ModelOutputParseError(ValueError):
    def __init__(
        self,
        *,
        role: str,
        raw: str,
        message: str,
        raw_outputs: list[str] | None = None,
        parse_failures: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.role = role
        self.raw = raw
        self.raw_outputs = list(raw_outputs or [raw])
        self.parse_failures = list(parse_failures or [message])


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


def parse_guidance_critic_output(raw: str) -> dict[str, Any]:
    token = raw.strip().upper()
    if token not in {"VALID", "INVALID"}:
        raise ValueError("guidance critic output must be exactly VALID or INVALID")
    return {
        "valid": token == "VALID",
        "reason": "explicit single-token guidance correctness verdict",
        "serialization": "single_token",
    }


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
Use JSON field names exactly as shown in this shape example:
{{"correct":false,"answer":"sample answer","confidence":0.90,"reason":"answer does not match"}}
Confidence must be a JSON number, never a word such as low, medium, or high.

Problem:
{example["problem"]}

Gold answer:
{example["gold_answer"]}

Controlled evidence:
{evidence_text}

Model response:
{response}
"""


def build_evaluator_repair_prompt(*, original_prompt: str, raw: str, error: str) -> str:
    return f"""{original_prompt}

Your previous serialization was invalid and was not accepted.
Validation error: {error}
Invalid response:
{raw}

Return a corrected JSON object now. Preserve your judgment, but use the exact required field names
and JSON value types. Output only the corrected object, from '{{' through '}}'.
"""


def build_guidance_guard_prompt(*, example: dict[str, Any], guidance: str) -> str:
    return f"""Check whether teacher guidance leaks the answer to a student.

The guidance is unsafe if it states the exact answer, an equivalent expression, a decisive named
entity, or enough information to copy the answer without solving. A hint that only identifies a
mistake or next verification is safe.

A useful broad relation-level correction is SAFE when the student must still solve the problem.
SAFE example: Recheck which hours receive the special rate before finding the total.
SAFE example: Verify whether the duration describes one leg or the full trip.
UNSAFE example: The correct result is seven.
UNSAFE example: Use the named person who appears in the gold answer.

Do not provide analysis, reasoning, headings, or a preamble. Return exactly one token: SAFE or UNSAFE.
Judge disclosure, not usefulness or mathematical correctness; a separate critic handles those.

Problem:
{example["problem"]}

Gold answer:
{example["gold_answer"]}

Teacher guidance:
{guidance}
"""


def build_guidance_critic_prompt(
    *,
    example: dict[str, Any],
    response: str,
    result: dict[str, Any],
    guidance: str,
) -> str:
    return f"""Check whether a teacher hint is directionally correct and relevant.

Use the problem, gold answer, failed response, and evaluator result as privileged review context.
Return VALID only if the hint points toward an actual error or useful verification and does not
reinforce a false interpretation. Return INVALID if it is mathematically wrong, irrelevant,
unsupported, or points away from the gold solution. Do not judge answer leakage in this role.

Do not provide analysis, reasoning, headings, or a preamble. Return exactly one token: VALID or INVALID.

Problem:
{example["problem"]}

Gold answer:
{example["gold_answer"]}

Failed student response:
{response}

Evaluator result:
{json.dumps(result, sort_keys=True)}

Teacher guidance:
{guidance}
"""


def make_model_evaluator(
    *,
    generate: Callable[..., str],
    generation_kwargs: dict[str, Any],
    max_regenerations: int = 0,
) -> Callable[[dict[str, Any], str], dict[str, Any]]:
    if max_regenerations < 0:
        raise ValueError("max_regenerations must be non-negative")

    def evaluate(example: dict[str, Any], response: str) -> dict[str, Any]:
        start = time.monotonic_ns()
        original_prompt = build_evaluator_prompt(example=example, response=response)
        prompt = original_prompt
        raw_outputs: list[str] = []
        generation_records: list[dict[str, Any]] = []
        parse_failures: list[str] = []
        parsed: dict[str, Any] | None = None
        for generation_attempt in range(max_regenerations + 1):
            generation = normalize_model_generation(generate("evaluator", prompt, **generation_kwargs))
            raw = generation.text
            raw_outputs.append(raw)
            generation_records.append(asdict(generation))
            try:
                candidate = parse_evaluator_output(raw)
                actual_answer_type = candidate.get("answer_type", "unknown")
                if not isinstance(actual_answer_type, str) or not actual_answer_type.strip():
                    raise ValueError("evaluator field answer_type must be a non-empty string when supplied")
                model_evidence_supported = candidate.get("evidence_supported")
                if model_evidence_supported is not None and not isinstance(model_evidence_supported, bool):
                    raise ValueError("evaluator field evidence_supported must be boolean when supplied")
                if example.get("domain") == "search_qa":
                    if "answer_type" not in candidate:
                        raise ValueError("SearchQA evaluator output is missing answer_type")
                    if "evidence_supported" not in candidate:
                        raise ValueError("SearchQA evaluator output is missing evidence_supported")
                parsed = candidate
                break
            except ValueError as exc:
                parse_failures.append(str(exc))
                if generation_attempt >= max_regenerations:
                    raise ModelOutputParseError(
                        role="evaluator",
                        raw=raw,
                        message=str(exc),
                        raw_outputs=raw_outputs,
                        parse_failures=parse_failures,
                    ) from exc
                prompt = build_evaluator_repair_prompt(
                    original_prompt=original_prompt,
                    raw=raw,
                    error=str(exc),
                )
        if parsed is None:
            raise RuntimeError("evaluator regeneration loop exited without a result")
        actual_answer_type = parsed.get("answer_type", "unknown")
        model_evidence_supported = parsed.get("evidence_supported")
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
                raw=raw_outputs[-1],
                message=f"deterministic answer evaluation failed: {exc}",
                raw_outputs=raw_outputs,
                parse_failures=parse_failures,
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
        parsed["raw_evaluator_output"] = raw_outputs[-1]
        parsed["raw_evaluator_outputs"] = raw_outputs
        parsed["evaluator_parse_failures"] = parse_failures
        parsed["evaluator_regenerations"] = len(raw_outputs) - 1
        parsed["evaluator_generations"] = generation_records
        parsed["latency_ms"] = (time.monotonic_ns() - start) // 1_000_000
        exact_tokens = [record["generated_tokens"] for record in generation_records]
        parsed["generated_tokens"] = sum(exact_tokens) if all(value is not None for value in exact_tokens) else None
        parsed["generated_tokens_estimate"] = sum(len(raw_output.split()) for raw_output in raw_outputs)
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
        generation = normalize_model_generation(
            generate(
                "guidance_guard",
                build_guidance_guard_prompt(example=example, guidance=guidance),
                **generation_kwargs,
            )
        )
        raw = generation.text
        try:
            parsed = parse_guidance_guard_output(raw)
        except ValueError as exc:
            raise ModelOutputParseError(role="guidance_guard", raw=raw, message=str(exc)) from exc
        parsed["guidance"] = guidance
        parsed["raw_guard_output"] = raw
        parsed["generation"] = asdict(generation)
        parsed["latency_ms"] = (time.monotonic_ns() - start) // 1_000_000
        parsed["generated_tokens"] = generation.generated_tokens
        parsed["generated_tokens_estimate"] = len(raw.split())
        return parsed

    return guard


def make_model_guidance_critic(
    *,
    generate: Callable[..., str],
    generation_kwargs: dict[str, Any],
) -> Callable[[dict[str, Any], str, dict[str, Any], int], dict[str, Any]]:
    def critic(
        example: dict[str, Any],
        guidance: str,
        result: dict[str, Any],
        _attempt: int,
    ) -> dict[str, Any]:
        start = time.monotonic_ns()
        response = result.get("response")
        if not isinstance(response, str) or not response:
            raise ValueError("guidance critic requires the failed student response")
        generation = normalize_model_generation(
            generate(
                "guidance_critic",
                build_guidance_critic_prompt(
                    example=example,
                    response=response,
                    result=result,
                    guidance=guidance,
                ),
                **generation_kwargs,
            )
        )
        try:
            parsed = parse_guidance_critic_output(generation.text)
        except ValueError as exc:
            raise ModelOutputParseError(
                role="guidance_critic",
                raw=generation.text,
                message=str(exc),
            ) from exc
        parsed["guidance"] = guidance
        parsed["raw_critic_output"] = generation.text
        parsed["generation"] = asdict(generation)
        parsed["latency_ms"] = (time.monotonic_ns() - start) // 1_000_000
        parsed["generated_tokens"] = generation.generated_tokens
        return parsed

    return critic
