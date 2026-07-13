from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from text_feedback_dpo.batch_generation import (
    EVALUATOR_VERSION,
    FIXED_B,
    FIXED_K1,
    FIXED_TOP_K,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    _zero_cited_score,
    canonical_cited_score,
    parse_search_query,
)
from text_feedback_dpo.feedback import FeedbackFormatError, diagnose_attempt, parse_feedback
from text_feedback_dpo.prompts import build_cited_response_prompt, build_search_query_prompt, build_teacher_prompt
from text_feedback_dpo.responses import CitedResponseFormatError, parse_cited_response, render_cited_response
from text_feedback_dpo.retrieval import FixedBM25Retriever, retrieval_metrics
from text_feedback_dpo.searchqa import SOURCE_SCHEMA_VERSION


class TrajectoryError(ValueError):
    """Raised when a trajectory cannot satisfy the explicit intervention contract."""


_REPAIR_SCOPE_COST = {
    "query/retrieval": 1,
    "response grammar/truncation": 2,
    "answer": 3,
    "lexical support proxy/citation selection": 4,
}


def repair_scope_cost(region: str | None) -> int:
    if region not in _REPAIR_SCOPE_COST:
        raise ValueError(f"unknown or successful repair region: {region}")
    return _REPAIR_SCOPE_COST[region]


def rank_interventions(interventions: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Rank only interventions with an explicitly computed sibling-efficiency score."""
    ranked: list[dict[str, object]] = []
    for intervention in interventions:
        score = intervention.get("efficiency_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("intervention efficiency score is unavailable until sibling gain is verified")
        ranked.append(dict(intervention))
    return sorted(ranked, key=lambda item: (-float(item["efficiency_score"]), int(item["level"])))


def _structured_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def retrieval_context_hash(ranked_results: Sequence[Mapping[str, object]]) -> str:
    return _structured_hash(list(ranked_results))


def _require_exact_field(artifact: Mapping[str, object], key: str, expected: object, *, example_id: str) -> None:
    if artifact.get(key) != expected:
        raise TrajectoryError(f"active artifact {example_id} {key} does not match canonical recomputation")


def validate_active_artifact(
    artifact: object,
    *,
    example: Mapping[str, object],
    hints: Sequence[str],
) -> dict[str, Any]:
    """Recompute and validate a Task 5 active-search artifact without repair."""
    if not isinstance(artifact, Mapping):
        raise TrajectoryError("active artifact must be a mapping")
    example_id = str(example.get("id", ""))
    if not example_id:
        raise TrajectoryError("active artifact validation requires an example id")
    if not isinstance(hints, Sequence) or isinstance(hints, (str, bytes)) or not all(isinstance(hint, str) and hint.strip() for hint in hints):
        raise TrajectoryError(f"active artifact {example_id} hints must be an explicit sequence of non-empty strings")
    required = (
        "id", "raw_query", "ranked_search_results", "raw_response", "truncation",
        "cited_score", "policy_hash", "prompt_version", "response_schema_version",
        "source_schema_version", "input_hash", "query_prompt", "query_prompt_hash",
        "response_prompt", "response_prompt_hash", "retrieval_context_hash",
        "canonical_ranked_search_results", "provenance", "no_hint", "evaluator_version",
        "parsed_response", "rendered_visible_response", "error_code",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise TrajectoryError(f"active artifact for {example_id} is missing {missing[0]}")
    if str(artifact["id"]) != example_id:
        raise TrajectoryError(f"active artifact ID mismatch: expected {example_id}, got {artifact['id']}")
    if artifact["provenance"] != "student":
        raise TrajectoryError(f"active artifact {example_id} requires explicit student provenance")
    if not isinstance(artifact["no_hint"], bool) or artifact["no_hint"] != (len(hints) == 0):
        raise TrajectoryError(f"active artifact {example_id} no_hint disagrees with exact hint context")
    if artifact["prompt_version"] != PROMPT_VERSION:
        raise TrajectoryError(f"active artifact {example_id} prompt_version must be {PROMPT_VERSION}")
    if artifact["evaluator_version"] != EVALUATOR_VERSION:
        raise TrajectoryError(f"active artifact {example_id} evaluator_version must be {EVALUATOR_VERSION}")
    if artifact["response_schema_version"] != RESPONSE_SCHEMA_VERSION:
        raise TrajectoryError(f"active artifact {example_id} response_schema_version must be {RESPONSE_SCHEMA_VERSION}")
    if artifact["source_schema_version"] != SOURCE_SCHEMA_VERSION:
        raise TrajectoryError(f"active artifact {example_id} source_schema_version must be {SOURCE_SCHEMA_VERSION}")
    if not isinstance(artifact["raw_query"], str):
        raise TrajectoryError(f"active artifact {example_id} raw_query must be text")
    if not isinstance(artifact["ranked_search_results"], list):
        raise TrajectoryError(f"active artifact {example_id} ranked_search_results must be a list")
    if not isinstance(artifact["truncation"], Mapping) or any(not isinstance(artifact["truncation"].get(key), bool) for key in ("query", "response")):
        raise TrajectoryError(f"active artifact {example_id} requires explicit query/response truncation booleans")
    if not isinstance(artifact["cited_score"], Mapping):
        raise TrajectoryError(f"active artifact {example_id} cited_score must be a mapping")
    for key in ("policy_hash",):
        if not isinstance(artifact[key], str) or not artifact[key].strip():
            raise TrajectoryError(f"active artifact {example_id} requires non-empty {key}")

    expected_input_hash = _structured_hash({"id": example_id, "question": example.get("question"), "sources": example.get("sources")})
    _require_exact_field(artifact, "input_hash", expected_input_hash, example_id=example_id)
    expected_query_prompt = build_search_query_prompt(dict(example), list(hints))
    _require_exact_field(artifact, "query_prompt", expected_query_prompt, example_id=example_id)
    _require_exact_field(artifact, "query_prompt_hash", _structured_hash(expected_query_prompt), example_id=example_id)

    truncation = artifact["truncation"]
    expected_ranked: list[dict[str, object]] = []
    query_valid = False
    if truncation["query"] is False:
        try:
            normalized_query = parse_search_query(artifact["raw_query"])
        except ValueError:
            normalized_query = None
        if normalized_query is not None:
            query_valid = True
            retriever = FixedBM25Retriever(example["sources"], k1=FIXED_K1, b=FIXED_B)
            expected_ranked = retriever.search(normalized_query, top_k=FIXED_TOP_K)
    if artifact["ranked_search_results"] != expected_ranked:
        raise TrajectoryError(f"active artifact {example_id} ranked retrieval does not match canonical BM25 recomputation")
    if artifact["canonical_ranked_search_results"] != expected_ranked:
        raise TrajectoryError(f"active artifact {example_id} canonical ranked retrieval records do not match recomputation")
    _require_exact_field(artifact, "retrieval_context_hash", retrieval_context_hash(expected_ranked), example_id=example_id)

    expected_response_prompt = build_cited_response_prompt(dict(example), expected_ranked, list(hints)) if query_valid else None
    _require_exact_field(artifact, "response_prompt", expected_response_prompt, example_id=example_id)
    expected_response_hash = _structured_hash(expected_response_prompt) if expected_response_prompt is not None else None
    _require_exact_field(artifact, "response_prompt_hash", expected_response_hash, example_id=example_id)
    if not query_valid:
        if artifact["raw_response"] is not None:
            raise TrajectoryError(f"active artifact {example_id} query-stage failure must not contain a response")
        expected_error = "query_truncated" if truncation["query"] else "query_invalid_format"
        _require_exact_field(
            artifact, "retrieval_metrics", retrieval_metrics([], example["gold_answer"]),
            example_id=example_id,
        )
        _require_exact_field(
            artifact, "cited_score",
            _zero_cited_score(expected_error, truncated=truncation["query"]),
            example_id=example_id,
        )
        _require_exact_field(artifact, "parsed_response", None, example_id=example_id)
        _require_exact_field(artifact, "rendered_visible_response", None, example_id=example_id)
        _require_exact_field(artifact, "error_code", expected_error, example_id=example_id)
        return dict(artifact)

    expected_retrieval_metrics = retrieval_metrics(expected_ranked, example["gold_answer"])
    _require_exact_field(artifact, "retrieval_metrics", expected_retrieval_metrics, example_id=example_id)
    raw_response = artifact["raw_response"]
    if not isinstance(raw_response, str):
        raise TrajectoryError(f"active artifact {example_id} response-stage artifact requires raw_response text")
    if stored_score := artifact["cited_score"]:
        if isinstance(stored_score, Mapping) and stored_score.get("correct") is True and ("<" in raw_response or ">" in raw_response):
            raise TrajectoryError(f"active artifact {example_id} successful response contains XML or angle markup")
    recomputed_score = canonical_cited_score(
        raw_response, example["gold_answer"], expected_ranked, truncated=truncation["response"]
    )
    stored_score = artifact["cited_score"]
    _require_exact_field(artifact, "cited_score", recomputed_score, example_id=example_id)
    expected_error = "response_truncated" if truncation["response"] else recomputed_score["error_code"]
    _require_exact_field(artifact, "error_code", expected_error, example_id=example_id)
    if recomputed_score["parse_valid"] and not truncation["response"]:
        if "<" in raw_response or ">" in raw_response:
            raise TrajectoryError(f"active artifact {example_id} successful response contains XML or angle markup")
        try:
            parsed = parse_cited_response(raw_response, expected_ranked)
        except CitedResponseFormatError as exc:
            raise TrajectoryError(f"active artifact {example_id} claimed success with invalid cited response: {exc}") from exc
        expected_parsed = {"answer": parsed.answer, "reasoning": parsed.reasoning, "source_ids": list(parsed.source_ids)}
        _require_exact_field(artifact, "parsed_response", expected_parsed, example_id=example_id)
        _require_exact_field(artifact, "rendered_visible_response", render_cited_response(parsed, expected_ranked), example_id=example_id)
    else:
        _require_exact_field(artifact, "parsed_response", None, example_id=example_id)
        _require_exact_field(artifact, "rendered_visible_response", None, example_id=example_id)
    normalized = dict(artifact)
    return normalized


def _success(artifact: Mapping[str, object]) -> bool:
    score = artifact["cited_score"]
    return (
        isinstance(score, Mapping)
        and score.get("correct") is True
        and score.get("parse_valid") is True
        and score.get("answer_correct") is True
        and score.get("lexical_cited_answer_support") == 1.0
        and artifact["truncation"]["query"] is False
        and artifact["truncation"]["response"] is False
    )


def _intervention_metadata(*, attempt_index: int, feedback_hint: str, level: int, diagnostics: Mapping[str, object]) -> dict[str, object]:
    region = diagnostics.get("responsible_region")
    cost = repair_scope_cost(region if isinstance(region, str) else None)
    hint_tokens = len(feedback_hint.split())
    if hint_tokens <= 0:
        raise TrajectoryError("teacher hint token count must be positive")
    return {
        "attempt_index": attempt_index,
        "hint": feedback_hint,
        "level": level,
        "escalation_level": level,
        "responsible_region": region,
        "diagnostics": dict(diagnostics),
        "hint_token_count": hint_tokens,
        "privilege_cost": hint_tokens,
        "repair_scope": region,
        "repair_scope_cost": cost,
        "efficiency_numerator": None,
        "efficiency_denominator": hint_tokens + cost,
        "efficiency_score": None,
        "efficiency_components": {"privilege_tokens": hint_tokens, "repair_scope_cost": cost},
    }


def _verify_siblings(
    *,
    example: dict,
    chosen: Mapping[str, object],
    sibling_generate: Callable[..., list[object]],
    sibling_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    if not sibling_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in sibling_seeds):
        raise TrajectoryError("no-hint sibling seeds must be a non-empty sequence of integers")
    requests = [
        {
            "id": str(example["id"]),
            "example": example,
            "hints": [],
            "no_hint": True,
            "seed": seed,
            "query_prompt": build_search_query_prompt(example, []),
        }
        for seed in sibling_seeds
    ]
    outputs = sibling_generate(requests, seeds=list(sibling_seeds), no_hint=True)
    if not isinstance(outputs, list) or len(outputs) != len(requests):
        raise TrajectoryError(f"no-hint sibling cardinality mismatch for {example['id']}")
    siblings: list[dict[str, Any]] = []
    for request, output in zip(requests, outputs, strict=True):
        sibling = validate_active_artifact(output, example=example, hints=[])
        if sibling["query_prompt"] != request["query_prompt"]:
            raise TrajectoryError(f"no-hint sibling query prompt mismatch for {example['id']}")
        if "Hints:" in str(sibling.get("query_prompt", "")) or "Hints:" in str(sibling.get("response_prompt", "")):
            raise TrajectoryError(f"no-hint sibling contains a hinted prompt for {example['id']}")
        for key in ("policy_hash", "prompt_version", "response_schema_version"):
            if sibling.get(key) != chosen.get(key):
                raise TrajectoryError(f"no-hint sibling {key} mismatch for {example['id']}")
        if "evaluator_version" not in chosen or "evaluator_version" not in sibling:
            raise TrajectoryError(f"no-hint sibling evaluator version is not explicit for {example['id']}")
        if sibling["evaluator_version"] != chosen["evaluator_version"]:
            raise TrajectoryError(f"no-hint sibling evaluator version mismatch for {example['id']}")
        sibling["seed"] = request["seed"]
        sibling["future_sibling_gain"] = 1.0 if _success(sibling) else 0.0
        sibling["verified_no_hint_success"] = _success(sibling)
        siblings.append(sibling)
    success_count = sum(int(item["verified_no_hint_success"]) for item in siblings)
    denominator = len(siblings)
    gain = success_count / denominator
    verification = {
        "status": "verified",
        "sibling_count": denominator,
        "success_count": success_count,
        "future_sibling_gain": gain,
        "future_sibling_gain_numerator": success_count,
        "future_sibling_gain_denominator": denominator,
        "seeds": [item["seed"] for item in siblings],
        "evaluator_version": chosen["evaluator_version"],
        "eligible": bool(success_count),
    }
    return siblings, verification


def _finish_trajectory(
    *,
    example: dict,
    query_prompt: str,
    attempts: list[dict[str, Any]],
    interventions: list[dict[str, object]],
    chosen: dict[str, Any] | None,
    sibling_generate: Callable[..., list[object]] | None,
    sibling_seeds: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": example["id"],
        "prompt": query_prompt,
        "query_prompt": query_prompt,
        "attempts": attempts,
        "interventions": interventions,
        "chosen": chosen,
        "resolved": chosen is not None,
        "no_hint_siblings": [],
        "sibling_verification": {"status": "not_required"},
        "training_eligible": False,
        "ranked_interventions": [],
    }
    if chosen is None:
        result["sibling_verification"] = {"status": "unresolved", "eligible": False}
        return result
    hinted_success = bool(interventions)
    if not hinted_success:
        result["training_eligible"] = True
        result["sibling_verification"] = {"status": "not_required", "eligible": True}
        return result
    if sibling_generate is None:
        result["sibling_verification"] = {"status": "missing_sibling_generator", "eligible": False}
        return result
    siblings, verification = _verify_siblings(
        example=example, chosen=chosen, sibling_generate=sibling_generate, sibling_seeds=sibling_seeds
    )
    result["no_hint_siblings"] = siblings
    result["sibling_verification"] = verification
    result["training_eligible"] = verification["eligible"]
    for intervention in interventions:
        intervention["future_sibling_gain"] = verification["future_sibling_gain"]
        intervention["future_sibling_gain_numerator"] = verification["future_sibling_gain_numerator"]
        intervention["future_sibling_gain_denominator"] = verification["future_sibling_gain_denominator"]
        intervention["efficiency_numerator"] = verification["future_sibling_gain"]
        intervention["efficiency_score"] = verification["future_sibling_gain"] / intervention["efficiency_denominator"]
    result["ranked_interventions"] = rank_interventions(interventions)
    return result


def collect_trajectory(
    *,
    example: dict,
    student_generate: Callable[[str, int], object],
    teacher_generate: Callable[[str], str],
    max_interventions: int,
    sibling_generate: Callable[..., list[object]] | None = None,
    sibling_seeds: Sequence[int] = (),
) -> dict:
    if not isinstance(max_interventions, int) or max_interventions < 0:
        raise ValueError("max_interventions must be a nonnegative integer")
    if not isinstance(example.get("sources"), list) or not example["sources"]:
        raise TrajectoryError("active trajectory requires complete SearchQA source records")
    query_prompt = build_search_query_prompt(example, [])
    hints: list[str] = []
    attempts: list[dict[str, Any]] = []
    interventions: list[dict[str, object]] = []
    for attempt_index in range(max_interventions + 1):
        current_prompt = build_search_query_prompt(example, hints)
        raw_artifact = student_generate(current_prompt, attempt_index)
        artifact = validate_active_artifact(raw_artifact, example=example, hints=hints)
        diagnostics = diagnose_attempt(artifact)
        correct = _success(artifact)
        attempts.append({
            "attempt_index": attempt_index,
            "artifact": artifact,
            "response": artifact.get("raw_response"),
            "correct": correct,
            "score": dict(artifact["cited_score"]),
            "diagnostics": diagnostics,
            "responsible_region": diagnostics["responsible_region"],
            "no_hint": not hints,
            "provenance": "student",
        })
        if correct:
            return _finish_trajectory(
                example=example, query_prompt=query_prompt, attempts=attempts,
                interventions=interventions, chosen=artifact,
                sibling_generate=sibling_generate, sibling_seeds=sibling_seeds,
            )
        if attempt_index == max_interventions:
            break
        try:
            raw_feedback = teacher_generate(
                build_teacher_prompt(
                    example,
                    str(artifact.get("raw_response") or ""),
                    interventions,
                    raw_query=artifact["raw_query"],
                    retrieved_sources=artifact["ranked_search_results"],
                    diagnostics=diagnostics,
                    repair_region=diagnostics["responsible_region"],
                    escalation_level=len(interventions) + 1,
                )
            )
            feedback = parse_feedback(raw_feedback, gold_answer=example["gold_answer"])
        except FeedbackFormatError as exc:
            raise TrajectoryError(
                f"invalid teacher feedback for {example['id']} at attempt {attempt_index}; diagnostics={diagnostics}: {exc}"
            ) from exc
        intervention = _intervention_metadata(
            attempt_index=attempt_index,
            feedback_hint=feedback.hint,
            level=len(interventions) + 1,
            diagnostics=diagnostics,
        )
        interventions.append(intervention)
        hints.append(feedback.hint)
    return _finish_trajectory(
        example=example, query_prompt=query_prompt, attempts=attempts,
        interventions=interventions, chosen=None,
        sibling_generate=sibling_generate, sibling_seeds=sibling_seeds,
    )
