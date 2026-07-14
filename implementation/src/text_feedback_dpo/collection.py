from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from text_feedback_dpo.feedback import FeedbackFormatError, diagnose_attempt, parse_feedback
from text_feedback_dpo.preferences import build_preference_rows
from text_feedback_dpo.prompts import build_search_query_prompt, build_teacher_prompt
from text_feedback_dpo.retrieval import validate_source_records
from text_feedback_dpo.trajectories import (
    TrajectoryError,
    _seal_supervision,
    _structured_hash,
    _intervention_metadata,
    _success,
    rank_interventions,
    validate_active_artifact,
)


def _validate_ids(examples: Sequence[Mapping[str, object]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("each collection example requires a non-empty string id")
        if example_id in seen:
            raise ValueError(f"duplicate collection example id: {example_id}")
        try:
            validate_source_records(example.get("sources"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"collection example {example_id} has invalid canonical sources: {exc}") from exc
        seen.add(example_id)
        ids.append(example_id)
    return ids


def _batch_siblings(
    states: dict[str, dict[str, Any]],
    resolved_ids: list[str],
    sibling_generate_batch: Callable[..., list[object]],
    sibling_seeds: Sequence[int],
) -> None:
    if not sibling_seeds:
        raise TrajectoryError("sibling seeds are required after a hinted success")
    requests = [
        {
            "id": example_id,
            "example": states[example_id]["example"],
            "hints": [],
            "no_hint": True,
            "seed": seed,
            "query_prompt": build_search_query_prompt(states[example_id]["example"], []),
        }
        for example_id in resolved_ids
        for seed in sibling_seeds
    ]
    outputs = sibling_generate_batch(requests, seeds=list(sibling_seeds), no_hint=True)
    if not isinstance(outputs, list) or len(outputs) != len(requests):
        raise ValueError(f"sibling batch cardinality mismatch: expected {len(requests)}, got {len(outputs) if isinstance(outputs, list) else type(outputs).__name__}")
    output_by_id: dict[str, list[object]] = {example_id: [] for example_id in resolved_ids}
    for request, output in zip(requests, outputs, strict=True):
        output_by_id[request["id"]].append(output)
    for example_id in resolved_ids:
        state = states[example_id]
        chosen = state["chosen"]
        siblings = []
        for request, output in zip(
            [item for item in requests if item["id"] == example_id], output_by_id[example_id], strict=True
        ):
            artifact = validate_active_artifact(output, example=state["example"], hints=[])
            if artifact["query_prompt"] != request["query_prompt"]:
                raise TrajectoryError(f"no-hint sibling query prompt mismatch for {example_id}")
            if "Hints:" in str(artifact.get("query_prompt", "")) or "Hints:" in str(artifact.get("response_prompt", "")):
                raise TrajectoryError(f"no-hint sibling contains a hinted prompt for {example_id}")
            for key in ("policy_hash", "prompt_version", "response_schema_version", "evaluator_version"):
                if key not in chosen or artifact.get(key) != chosen.get(key):
                    raise TrajectoryError(f"no-hint sibling {key} mismatch for {example_id}")
            artifact["seed"] = request["seed"]
            artifact["future_sibling_gain"] = 1.0 if _success(artifact) else 0.0
            artifact["verified_no_hint_success"] = _success(artifact)
            siblings.append(artifact)
        success_count = sum(int(item["verified_no_hint_success"]) for item in siblings)
        denominator = len(siblings)
        gain = success_count / denominator
        state["no_hint_siblings"] = siblings
        state["sibling_verification"] = {
            "status": "verified", "sibling_count": denominator, "success_count": success_count,
            "future_sibling_gain": gain, "future_sibling_gain_numerator": success_count,
            "future_sibling_gain_denominator": denominator,
            "seeds": [item["seed"] for item in siblings], "eligible": bool(success_count),
            "sft_eligible": bool(success_count),
            "preference_eligible": 0 < success_count < denominator,
            "evaluator_version": chosen["evaluator_version"],
        }
        state["training_eligible"] = bool(success_count)
        state["sft_eligible"] = bool(success_count)
        state["preference_eligible"] = 0 < success_count < denominator
        for intervention in state["interventions"]:
            intervention["future_sibling_gain"] = gain
            intervention["future_sibling_gain_numerator"] = success_count
            intervention["future_sibling_gain_denominator"] = denominator
            intervention["efficiency_numerator"] = gain
            intervention["efficiency_score"] = gain / intervention["efficiency_denominator"]
            _seal_supervision(intervention)
        state["ranked_interventions"] = rank_interventions(state["interventions"])


def collect_dataset_batchwise(
    *,
    examples: list[dict],
    student_generate_batch: Callable[..., list[object]],
    teacher_generate_batch: Callable[..., list[str]],
    max_interventions: int,
    teacher_max_new_tokens: int = 1024,
    teacher_temperature: float = 0.0,
    teacher_top_p: float = 1.0,
    teacher_top_k: int | None = None,
    sibling_generate_batch: Callable[..., list[object]] | None = None,
    sibling_seeds: Sequence[int] = (),
    student_seed: int,
    checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    resume_checkpoint: Mapping[str, object] | None = None,
) -> list[dict]:
    if not isinstance(max_interventions, int) or max_interventions < 0:
        raise ValueError("max_interventions must be a nonnegative integer")
    if isinstance(teacher_max_new_tokens, bool) or not isinstance(teacher_max_new_tokens, int) or teacher_max_new_tokens <= 0:
        raise ValueError("teacher_max_new_tokens must be a positive integer")
    if not isinstance(teacher_temperature, (int, float)) or teacher_temperature < 0:
        raise ValueError("teacher_temperature must be nonnegative")
    if not isinstance(teacher_top_p, (int, float)) or not 0 < teacher_top_p <= 1:
        raise ValueError("teacher_top_p must be in (0, 1]")
    if teacher_top_k is not None and (
        isinstance(teacher_top_k, bool) or not isinstance(teacher_top_k, int) or teacher_top_k <= 0
    ):
        raise ValueError("teacher_top_k must be a positive integer when supplied")
    if isinstance(student_seed, bool) or not isinstance(student_seed, int) or student_seed < 0:
        raise ValueError("student_seed must be a nonnegative integer")
    ids = _validate_ids(examples)
    states: dict[str, dict[str, Any]] = {
        example_id: {
            "id": example_id, "example": example, "hints": [], "attempts": [], "interventions": [],
            "resolved": False, "chosen": None, "query_prompt": build_search_query_prompt(example, []),
            "no_hint_siblings": [], "training_eligible": False,
            "sft_eligible": False, "preference_eligible": False,
            "ranked_interventions": [],
        }
        for example_id, example in zip(ids, examples, strict=True)
    }
    active = ids[:]
    start_attempt_index = 0
    if resume_checkpoint is not None:
        if resume_checkpoint.get("schema_version") != 1:
            raise ValueError("collection checkpoint schema_version must be 1")
        checkpoint_states = resume_checkpoint.get("states")
        checkpoint_ids = resume_checkpoint.get("active_ids")
        next_attempt_index = resume_checkpoint.get("next_attempt_index")
        if not isinstance(checkpoint_states, Mapping) or set(checkpoint_states) != set(ids):
            raise ValueError("collection checkpoint state ids do not match input examples")
        if not isinstance(checkpoint_ids, list) or any(example_id not in ids for example_id in checkpoint_ids):
            raise ValueError("collection checkpoint active_ids are invalid")
        if isinstance(next_attempt_index, bool) or not isinstance(next_attempt_index, int) or not 0 <= next_attempt_index <= max_interventions + 1:
            raise ValueError("collection checkpoint next_attempt_index is invalid")
        states = deepcopy(dict(checkpoint_states))
        active = list(checkpoint_ids)
        start_attempt_index = next_attempt_index

    def checkpoint(next_attempt_index: int, active_ids: list[str]) -> None:
        if checkpoint_callback is not None:
            checkpoint_callback({
                "schema_version": 1,
                "next_attempt_index": next_attempt_index,
                "active_ids": list(active_ids),
                "states": deepcopy(states),
            })

    for attempt_index in range(start_attempt_index, max_interventions + 1):
        requests = [{
            "id": example_id,
            "example": states[example_id]["example"],
            "hints": list(states[example_id]["hints"]),
            "no_hint": not states[example_id]["hints"],
            "seed": student_seed + attempt_index,
            "query_prompt": build_search_query_prompt(states[example_id]["example"], states[example_id]["hints"]),
        } for example_id in active]
        outputs = student_generate_batch(
            requests, attempt_index=attempt_index, seed=student_seed + attempt_index,
            max_new_tokens=32, temperature=0.7, top_p=0.9,
        )
        if not isinstance(outputs, list) or len(outputs) != len(active):
            raise ValueError(f"student batch cardinality mismatch at attempt {attempt_index}")
        failed_ids: list[str] = []
        teacher_prompts: list[str] = []
        for request, output in zip(requests, outputs, strict=True):
            example_id = request["id"]
            state = states[example_id]
            artifact = validate_active_artifact(output, example=state["example"], hints=state["hints"])
            diagnostics = diagnose_attempt(artifact)
            correct = _success(artifact)
            attempt = {
                "attempt_index": attempt_index, "artifact": artifact, "response": artifact.get("raw_response"),
                "correct": correct, "score": dict(artifact["cited_score"]), "diagnostics": diagnostics,
                "responsible_region": diagnostics["responsible_region"], "no_hint": not state["hints"],
                "provenance": "student",
            }
            attempt["supervision_hash"] = _structured_hash({
                key: value for key, value in attempt.items() if key not in {"attempt_index", "artifact"}
            })
            state["attempts"].append(attempt)
            if correct:
                state["resolved"] = True
                state["chosen"] = artifact
            elif attempt_index < max_interventions:
                failed_ids.append(example_id)
                teacher_prompts.append(build_teacher_prompt(
                    state["example"], str(artifact.get("raw_response") or ""), state["interventions"],
                    raw_query=artifact["raw_query"], retrieved_sources=artifact["ranked_search_results"],
                    diagnostics=diagnostics, repair_region=diagnostics["responsible_region"],
                    escalation_level=len(state["interventions"]) + 1,
                ))
        if not failed_ids:
            checkpoint(attempt_index + 1, [])
            break
        feedback_rows = teacher_generate_batch(
            teacher_prompts,
            gold_answers=[states[example_id]["example"]["gold_answer"] for example_id in failed_ids],
            max_new_tokens=teacher_max_new_tokens,
            temperature=float(teacher_temperature),
            top_p=float(teacher_top_p),
            top_k=teacher_top_k,
            seed=student_seed + attempt_index,
        )
        if not isinstance(feedback_rows, list) or len(feedback_rows) != len(failed_ids):
            raise ValueError(f"teacher batch cardinality mismatch at attempt {attempt_index}")
        for example_id, teacher_prompt, raw_feedback in zip(
            failed_ids, teacher_prompts, feedback_rows, strict=True
        ):
            if not isinstance(raw_feedback, str):
                raise ValueError(f"teacher output for {example_id} must be strict JSON text")
            state = states[example_id]
            try:
                feedback = parse_feedback(raw_feedback, gold_answer=state["example"]["gold_answer"])
            except FeedbackFormatError as exc:
                diagnostic = state["attempts"][-1]["diagnostics"]
                raise ValueError(f"invalid teacher feedback for {example_id}; diagnostics={diagnostic}: {exc}") from exc
            diagnostic = state["attempts"][-1]["diagnostics"]
            state["interventions"].append(_intervention_metadata(
                attempt_index=attempt_index, feedback_hint=feedback.hint,
                level=len(state["interventions"]) + 1, diagnostics=diagnostic,
                teacher_prompt=teacher_prompt, raw_teacher_response=raw_feedback,
            ))
            state["hints"].append(feedback.hint)
        active = failed_ids
        checkpoint(attempt_index + 1, active)
    resolved_after_hint = [example_id for example_id in ids if states[example_id]["resolved"] and states[example_id]["interventions"]]
    if resolved_after_hint:
        if sibling_generate_batch is not None:
            _batch_siblings(states, resolved_after_hint, sibling_generate_batch, sibling_seeds)
        else:
            for example_id in resolved_after_hint:
                states[example_id]["sibling_verification"] = {"status": "missing_sibling_generator", "eligible": False}
                states[example_id]["ranked_interventions"] = []
    rows = []
    for example_id in ids:
        state = states[example_id]
        if not state["resolved"]:
            state["sibling_verification"] = {"status": "unresolved", "eligible": False}
        elif not state["interventions"]:
            state["sibling_verification"] = {"status": "not_required", "eligible": True}
        trajectory = {
            "id": state["id"], "prompt": state["query_prompt"], "query_prompt": state["query_prompt"],
            "example_identity": _structured_hash(state["example"]),
            "query_prompt_hash": _structured_hash(state["query_prompt"]),
            "attempts": state["attempts"], "interventions": state["interventions"],
            "ranked_interventions": state["ranked_interventions"],
            "chosen": state["chosen"], "resolved": state["resolved"],
            "no_hint_siblings": state["no_hint_siblings"],
            "sibling_verification": state.get("sibling_verification", {"status": "not_required"}),
            "training_eligible": state["training_eligible"] if state["interventions"] else state["resolved"],
            "sft_eligible": state["sft_eligible"] if state["interventions"] else state["resolved"],
            "preference_eligible": state["preference_eligible"] if state["interventions"] else False,
        }
        if trajectory["chosen"]:
            chosen = trajectory["chosen"]
            for key in ("policy_hash", "response_prompt_hash", "evaluator_version"):
                if key in chosen:
                    trajectory[key] = chosen[key]
        trajectory["preference_rows"] = build_preference_rows(trajectory) if trajectory["preference_eligible"] else []
        rows.append(trajectory)
    return rows
