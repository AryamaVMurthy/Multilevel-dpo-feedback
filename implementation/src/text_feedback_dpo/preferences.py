from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _pair_metadata(
    trajectory: Mapping[str, object],
    chosen: Mapping[str, object],
    rejected: Mapping[str, object],
    *,
    kind: str,
    canonical_context: Mapping[str, object],
) -> dict[str, object]:
    return {
        "example_id": trajectory["id"],
        "pair_type": kind,
        "no_hint": True,
        "provenance": "student",
        "chosen_seed": chosen.get("seed"),
        "rejected_seed": rejected.get("seed"),
        "chosen_future_sibling_gain": _gain(chosen),
        "rejected_future_sibling_gain": _gain(rejected),
        "policy_hash": chosen.get("policy_hash"),
        "query_prompt_hash": chosen.get("query_prompt_hash"),
        "response_prompt_hash": chosen.get("response_prompt_hash"),
        "retrieval_context_hash": chosen.get("retrieval_context_hash"),
        "canonical_context": dict(canonical_context),
    }


def _record_exclusion(
    trajectory: Mapping[str, object], *, kind: str, reason: str,
    chosen: Mapping[str, object], rejected: Mapping[str, object],
) -> None:
    if not isinstance(trajectory, dict):
        raise ValueError("preference exclusion accounting requires a mutable trajectory mapping")
    exclusions = trajectory.setdefault("preference_exclusions", [])
    counts = trajectory.setdefault("preference_exclusion_counts", {})
    if not isinstance(exclusions, list) or not isinstance(counts, dict):
        raise ValueError("preference exclusion accounting fields have invalid types")
    exclusion_reason = f"{reason}_{kind}_completion"
    exclusions.append({
        "pair_type": kind,
        "reason": exclusion_reason,
        "chosen_seed": chosen.get("seed"),
        "rejected_seed": rejected.get("seed"),
    })
    counts[exclusion_reason] = int(counts.get(exclusion_reason, 0)) + 1


def _ranked_pairs(
    candidates: list[Mapping[str, object]],
    *,
    completion_field: str,
    kind: str,
    trajectory: Mapping[str, object],
    prompt: str,
    canonical_context: Mapping[str, object],
) -> list[dict[str, object]]:
    if len(candidates) < 2:
        return []
    ordered = sorted(candidates, key=lambda item: (-_gain(item), int(item.get("seed", 0))))
    best = ordered[0]
    if best.get("verified_no_hint_success") is not True:
        raise ValueError("preference chosen completion must be a verified no-hint success")
    rows: list[dict[str, object]] = []
    for rejected in ordered[1:]:
        chosen_text = _completion(best.get(completion_field), field=completion_field)
        rejected_text = _completion(rejected.get(completion_field), field=completion_field)
        if chosen_text == rejected_text:
            _record_exclusion(
                trajectory, kind=kind, reason="identical", chosen=best, rejected=rejected
            )
            continue
        if _gain(best) <= _gain(rejected):
            continue
        rows.append({
            "id": f"{trajectory['id']}::{kind}::{best.get('seed')}::{rejected.get('seed')}",
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "metadata": _pair_metadata(
                trajectory, best, rejected, kind=kind, canonical_context=canonical_context
            ),
        })
    return rows


def _query_context(trajectory: Mapping[str, object]) -> tuple[str, str]:
    prompt = trajectory.get("query_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or "Hints:" in prompt:
        raise ValueError("query preferences require one non-empty no-hint query prompt")
    prompt_hash = trajectory.get("query_prompt_hash")
    if not isinstance(prompt_hash, str) or prompt_hash != _hash(prompt):
        raise ValueError("query preferences require the canonical no-hint query prompt hash")
    return prompt, prompt_hash


def _preference_eligible(trajectory: Mapping[str, object]) -> bool:
    explicit = trajectory.get("preference_eligible")
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ValueError("preference_eligible must be boolean")
        return explicit
    return trajectory.get("training_eligible") is True


def build_query_preference_rows(trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    """Rank student queries generated from byte-identical no-hint prompts."""
    if not _preference_eligible(trajectory):
        return []
    prompt, prompt_hash = _query_context(trajectory)
    siblings = trajectory.get("no_hint_siblings")
    if not isinstance(siblings, list):
        raise ValueError("query preferences require no-hint siblings")
    candidates: list[Mapping[str, object]] = []
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise ValueError("query sibling must be a mapping")
        _student_no_hint(sibling)
        if sibling.get("query_prompt") != prompt:
            raise ValueError("query preference candidates require byte-identical query prompt bytes")
        if sibling.get("query_prompt_hash") != prompt_hash:
            raise ValueError("query preference candidates do not share the canonical query prompt hash")
        candidates.append(sibling)
    context = {"query_prompt": prompt, "query_prompt_hash": prompt_hash}
    return _ranked_pairs(
        candidates, completion_field="raw_query", kind="query", trajectory=trajectory,
        prompt=prompt, canonical_context=context,
    )


def _response_context(candidate: Mapping[str, object]) -> tuple[tuple[str, str, str, str], dict[str, object]]:
    prompt = candidate.get("response_prompt")
    ranked = candidate.get("canonical_ranked_search_results")
    if not isinstance(prompt, str) or not prompt.strip() or "Hints:" in prompt:
        raise ValueError("response preferences require explicit no-hint response prompt bytes")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("response preferences require canonical ranked retrieval records")
    prompt_hash = candidate.get("response_prompt_hash")
    retrieval_hash = candidate.get("retrieval_context_hash")
    if prompt_hash != _hash(prompt):
        raise ValueError("response preference response prompt bytes do not match their hash")
    if retrieval_hash != _hash(ranked):
        raise ValueError("response preference canonical retrieval records do not match their hash")
    policy_hash = candidate.get("policy_hash")
    query_prompt_hash = candidate.get("query_prompt_hash")
    if not all(isinstance(value, str) and value for value in (policy_hash, query_prompt_hash)):
        raise ValueError("response preferences require explicit policy and query prompt hashes")
    key = (prompt_hash, retrieval_hash, policy_hash, query_prompt_hash)
    context = {
        "query_prompt": candidate.get("query_prompt"),
        "query_prompt_hash": query_prompt_hash,
        "response_prompt": prompt,
        "response_prompt_hash": prompt_hash,
        "ranked_search_results": json.loads(json.dumps(ranked, ensure_ascii=False, sort_keys=True)),
        "retrieval_context_hash": retrieval_hash,
    }
    return key, context


def build_response_preference_rows(trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    """Rank responses only within byte-identical canonical retrieval contexts."""
    if not _preference_eligible(trajectory):
        return []
    siblings = trajectory.get("no_hint_siblings")
    if not isinstance(siblings, list):
        raise ValueError("response preferences require no-hint siblings")
    groups: dict[tuple[str, str, str, str], tuple[dict[str, object], list[Mapping[str, object]]]] = {}
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise ValueError("response sibling must be a mapping")
        _student_no_hint(sibling)
        score = sibling.get("cited_score")
        if not isinstance(score, Mapping):
            raise ValueError("response preferences require a canonical cited score")
        truncation = sibling.get("truncation")
        canonical_success = bool(
            score.get("correct") is True
            and score.get("parse_valid") is True
            and score.get("answer_correct") is True
            and score.get("lexical_cited_answer_support") == 1.0
            and isinstance(truncation, Mapping)
            and truncation.get("query") is False
            and truncation.get("response") is False
        )
        if sibling.get("verified_no_hint_success") is not canonical_success:
            raise ValueError("response preference candidate contains an unverified success or forged verification")
        if _gain(sibling) != float(canonical_success):
            raise ValueError("response preference sibling has inconsistent verification and gain")
        key, context = _response_context(sibling)
        if key in groups and groups[key][0] != context:
            raise ValueError("response preference candidates have same hashes but different prompt bytes or retrieval records")
        groups.setdefault(key, (context, []))[1].append(sibling)
    rows: list[dict[str, object]] = []
    for context, candidates in groups.values():
        rows.extend(_ranked_pairs(
            candidates, completion_field="raw_response", kind="response", trajectory=trajectory,
            prompt=str(context["response_prompt"]), canonical_context=context,
        ))
    return rows


def build_preference_rows(trajectory: Mapping[str, object]) -> list[dict[str, object]]:
    """Build active query and response preferences; archival rows are unsupported."""
    required = ("query_prompt", "query_prompt_hash", "no_hint_siblings", "training_eligible")
    missing = [field for field in required if field not in trajectory]
    if missing:
        raise ValueError(f"active trajectory is missing required field: {missing[0]}")
    if isinstance(trajectory, dict):
        trajectory["preference_exclusions"] = []
        trajectory["preference_exclusion_counts"] = {}
    return build_query_preference_rows(trajectory) + build_response_preference_rows(trajectory)
