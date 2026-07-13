from __future__ import annotations

from collections.abc import Mapping


def _completion(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} completion must be a non-empty student string")
    return f" {value}"


def _student_no_hint(item: Mapping[str, object]) -> None:
    if item.get("provenance") != "student":
        raise ValueError("preference completion provenance must be student")
    if item.get("no_hint") is not True:
        raise ValueError("preference completion must come from a no-hint prompt")
    if item.get("teacher_output") is True or item.get("fabricated") is True:
        raise ValueError("teacher or fabricated provenance cannot enter preferences")
    if not isinstance(item.get("verified_no_hint_success"), bool):
        raise ValueError("preference candidate requires explicit no-hint verification status")


def _gain(item: Mapping[str, object]) -> float:
    value = item.get("future_sibling_gain")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("preference candidate requires future_sibling_gain")
    gain = float(value)
    if not 0.0 <= gain <= 1.0:
        raise ValueError("future_sibling_gain must be between zero and one")
    return gain


def _pair_metadata(trajectory: Mapping[str, object], chosen: Mapping[str, object], rejected: Mapping[str, object], *, kind: str) -> dict[str, object]:
    return {
        "example_id": trajectory["id"],
        "pair_type": kind,
        "no_hint": True,
        "provenance": "student",
        "chosen_seed": chosen.get("seed"),
        "rejected_seed": rejected.get("seed"),
        "chosen_future_sibling_gain": _gain(chosen),
        "rejected_future_sibling_gain": _gain(rejected),
        "policy_hash": trajectory.get("policy_hash", chosen.get("policy_hash")),
        "query_prompt_hash": chosen.get("query_prompt_hash"),
        "response_prompt_hash": chosen.get("response_prompt_hash"),
        "retrieval_context_hash": chosen.get("retrieval_context_hash"),
    }


def _ranked_pairs(candidates: list[Mapping[str, object]], *, completion_field: str, kind: str, trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    if len(candidates) < 2:
        return []
    ordered = sorted(candidates, key=lambda item: (-_gain(item), int(item.get("seed", 0))))
    rows: list[dict[str, object]] = []
    best = ordered[0]
    if best.get("verified_no_hint_success") is not True:
        raise ValueError("preference chosen completion must be a verified no-hint success")
    for rejected in ordered[1:]:
        if _gain(best) <= _gain(rejected):
            continue
        chosen_text = _completion(best.get(completion_field), field=completion_field)
        rejected_text = _completion(rejected.get(completion_field), field=completion_field)
        if chosen_text == rejected_text:
            raise ValueError("preference completions must not be identical")
        prompt = trajectory["query_prompt"] if kind == "query" else best.get("response_prompt")
        if not isinstance(prompt, str) or not prompt.strip() or "Hints:" in prompt:
            raise ValueError("preference prompt must be an explicit no-hint prompt")
        rows.append({
            "id": f"{trajectory['id']}::{kind}::{best.get('seed')}::{rejected.get('seed')}",
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "metadata": _pair_metadata(trajectory, best, rejected, kind=kind),
        })
    return rows


def build_query_preference_rows(trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    """Rank student-generated no-hint query options by future sibling gain."""
    if not trajectory.get("training_eligible"):
        return []
    prompt = trajectory.get("query_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or "Hints:" in prompt:
        raise ValueError("query preferences require one non-empty no-hint query prompt")
    siblings = trajectory.get("no_hint_siblings")
    if not isinstance(siblings, list):
        raise ValueError("query preferences require no-hint siblings")
    candidates: list[Mapping[str, object]] = []
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise ValueError("query sibling must be a mapping")
        _student_no_hint(sibling)
        if not isinstance(trajectory.get("query_prompt_hash"), str) or sibling.get("query_prompt_hash") != trajectory["query_prompt_hash"]:
            raise ValueError("query preference candidates do not share the no-hint query prompt")
        candidates.append(sibling)
    return _ranked_pairs(candidates, completion_field="raw_query", kind="query", trajectory=trajectory)


def build_response_preference_rows(trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    """Rank only same-context, student-generated, no-hint cited responses."""
    if not trajectory.get("training_eligible"):
        return []
    siblings = trajectory.get("no_hint_siblings")
    if not isinstance(siblings, list):
        raise ValueError("response preferences require no-hint siblings")
    candidates: list[Mapping[str, object]] = []
    context: tuple[object, object, object, object] | None = None
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise ValueError("response sibling must be a mapping")
        _student_no_hint(sibling)
        score = sibling.get("cited_score")
        if not isinstance(score, Mapping) or score.get("parse_valid") is not True:
            raise ValueError("response preferences require parse-valid sibling responses")
        if sibling.get("verified_no_hint_success") is not True and sibling.get("future_sibling_gain") != 0.0:
            raise ValueError("response preference sibling has inconsistent verification and gain")
        current = (
            sibling.get("response_prompt_hash"), sibling.get("retrieval_context_hash"),
            sibling.get("policy_hash"), sibling.get("query_prompt_hash"),
        )
        if any(value is None or value == "" for value in current):
            raise ValueError("response preferences require explicit prompt, retrieval, and policy context hashes")
        if context is None:
            context = current
        elif current != context:
            raise ValueError("response preference candidates cross response or retrieval context")
        candidates.append(sibling)
    return _ranked_pairs(candidates, completion_field="raw_response", kind="response", trajectory=trajectory)


def build_preference_rows(trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    """Build query and response preference rows independently, then concatenate."""
    # Explicit archival compatibility for Task 5's plain-answer artifact.  The
    # active Task 6 collector always supplies ``query_prompt`` and sibling
    # records, so this branch cannot silently admit active rows without the
    # new provenance/context gates.
    if (
        "query_prompt" not in trajectory
        and isinstance(trajectory.get("prompt"), str)
        and isinstance(trajectory.get("chosen"), str)
        and isinstance(trajectory.get("attempts"), list)
    ):
        prompt = trajectory["prompt"]
        if "Hints:" in prompt:
            raise ValueError("archival preference prompt must not contain hints")
        chosen = _completion(trajectory["chosen"], field="chosen")
        rows: list[dict[str, object]] = []
        for attempt in trajectory["attempts"]:
            if not isinstance(attempt, Mapping):
                raise ValueError("archival preference attempt must be a mapping")
            if attempt.get("correct"):
                break
            rejected = _completion(attempt.get("response"), field="rejected")
            if chosen == rejected:
                raise ValueError("preference completions must not be identical")
            rows.append({
                "id": f"{trajectory['id']}::archival-attempt-{attempt.get('attempt_index')}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "example_id": trajectory["id"],
                    "pair_type": "archival-response",
                    "legacy_compatibility": True,
                    "no_hint": True,
                    "provenance": "student-archival",
                    "rejected_attempt_index": attempt.get("attempt_index"),
                },
            })
        return rows
    return build_query_preference_rows(trajectory) + build_response_preference_rows(trajectory)
