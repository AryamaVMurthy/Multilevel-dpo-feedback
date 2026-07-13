from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from text_feedback_dpo.feedback import FeedbackFormatError, diagnose_attempt, parse_feedback
from text_feedback_dpo.prompts import build_search_query_prompt, build_teacher_prompt


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


def retrieval_context_hash(artifact: Mapping[str, object]) -> str:
    ranked = artifact.get("ranked_search_results")
    if not isinstance(ranked, list):
        raise TrajectoryError("active artifact requires ranked_search_results for context hashing")
    payload = json.dumps(ranked, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_active_artifact(artifact: object, *, example_id: str, no_hint: bool | None = None) -> dict[str, Any]:
    """Validate the committed Task 5 active-search artifact without repairing it."""
    if not isinstance(artifact, Mapping):
        raise TrajectoryError(f"active artifact for {example_id} must be a mapping")
    required = (
        "id", "raw_query", "ranked_search_results", "raw_response", "truncation",
        "cited_score", "policy_hash", "prompt_version", "response_schema_version",
        "query_prompt_hash", "response_prompt_hash",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise TrajectoryError(f"active artifact for {example_id} is missing {missing[0]}")
    if str(artifact["id"]) != str(example_id):
        raise TrajectoryError(f"active artifact ID mismatch: expected {example_id}, got {artifact['id']}")
    if not isinstance(artifact["raw_query"], str):
        raise TrajectoryError(f"active artifact {example_id} raw_query must be text")
    if not isinstance(artifact["ranked_search_results"], list):
        raise TrajectoryError(f"active artifact {example_id} ranked_search_results must be a list")
    if not isinstance(artifact["truncation"], Mapping) or any(not isinstance(artifact["truncation"].get(key), bool) for key in ("query", "response")):
        raise TrajectoryError(f"active artifact {example_id} requires explicit query/response truncation booleans")
    if not isinstance(artifact["cited_score"], Mapping):
        raise TrajectoryError(f"active artifact {example_id} cited_score must be a mapping")
    for key in ("policy_hash", "prompt_version"):
        if not isinstance(artifact[key], str) or not artifact[key].strip():
            raise TrajectoryError(f"active artifact {example_id} requires non-empty {key}")
    for key in ("response_schema_version",):
        if isinstance(artifact[key], bool) or not isinstance(artifact[key], int):
            raise TrajectoryError(f"active artifact {example_id} requires integer {key}")
    for key in ("query_prompt_hash", "response_prompt_hash"):
        if artifact[key] is not None and (not isinstance(artifact[key], str) or not artifact[key].strip()):
            raise TrajectoryError(f"active artifact {example_id} {key} must be a non-empty string or null")
    existing_provenance = artifact.get("provenance")
    if existing_provenance is not None and existing_provenance != "student":
        raise TrajectoryError(f"active artifact {example_id} has non-student provenance: {existing_provenance}")
    existing_no_hint = artifact.get("no_hint")
    if existing_no_hint is not None and not isinstance(existing_no_hint, bool):
        raise TrajectoryError(f"active artifact {example_id} no_hint must be boolean")
    if no_hint is not None and existing_no_hint is not None and existing_no_hint != no_hint:
        raise TrajectoryError(f"active artifact {example_id} no_hint metadata disagrees with requested prompt")
    normalized = dict(artifact)
    normalized["provenance"] = "student"
    if no_hint is not None:
        normalized["no_hint"] = no_hint
    normalized["retrieval_context_hash"] = artifact.get("retrieval_context_hash", retrieval_context_hash(artifact))
    if not isinstance(normalized["retrieval_context_hash"], str) or not normalized["retrieval_context_hash"].strip():
        raise TrajectoryError(f"active artifact {example_id} requires retrieval_context_hash")
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
        sibling = validate_active_artifact(output, example_id=str(example["id"]), no_hint=True)
        if sibling.get("query_prompt") and sibling["query_prompt"] != request["query_prompt"]:
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
        if sibling["response_prompt_hash"] != chosen["response_prompt_hash"]:
            raise TrajectoryError(f"no-hint sibling response prompt context mismatch for {example['id']}")
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
        artifact = validate_active_artifact(raw_artifact, example_id=str(example["id"]), no_hint=not hints)
        artifact["query_prompt"] = current_prompt
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
