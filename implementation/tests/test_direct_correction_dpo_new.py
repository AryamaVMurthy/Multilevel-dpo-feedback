import hashlib
import json

import pytest

from text_feedback_dpo.preferences import build_direct_correction_preference_row


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact(response, *, correct, no_hint):
    prompt = "Evidence:\n[S001] Ada wrote it.\nQuestion: Who wrote it?\nAnswer:"
    return {
        "provenance": "student",
        "no_hint": no_hint,
        "raw_response": response,
        "response_prompt": prompt if no_hint else prompt + "\nHint: inspect the name",
        "response_prompt_hash": _hash(prompt if no_hint else prompt + "\nHint: inspect the name"),
        "query_prompt_hash": "q",
        "retrieval_context_hash": "r",
        "policy_hash": "p",
        "evaluator_version": "e",
        "cited_score": {"correct": correct},
    }


def trajectory():
    wrong = artifact("Answer: Grace\nReasoning: x [S001].\nSources: S001", correct=False, no_hint=True)
    right = artifact("Answer: Ada\nReasoning: x [S001].\nSources: S001", correct=True, no_hint=False)
    return {
        "id": "q1",
        "attempts": [{"correct": False, "artifact": wrong}],
        "interventions": [{"hint": "inspect the name"}],
        "chosen": right,
        "resolved": True,
    }


def test_builds_original_prompt_wrong_vs_post_feedback_correct_pair():
    row = build_direct_correction_preference_row(trajectory())
    assert row is not None
    assert row["prompt"] == trajectory()["attempts"][0]["artifact"]["response_prompt"]
    assert "Answer: Ada" in row["chosen"]
    assert "Answer: Grace" in row["rejected"]
    assert row["metadata"]["teacher_output_role"] == "hint_only_not_candidate"


def test_returns_none_for_unresolved_initial_failure():
    item = trajectory()
    item.update(chosen=None, resolved=False)
    assert build_direct_correction_preference_row(item) is None


def test_rejects_unverified_chosen_answer():
    item = trajectory()
    item["chosen"]["cited_score"]["correct"] = False
    with pytest.raises(ValueError, match="verified correct"):
        build_direct_correction_preference_row(item)


def test_rejects_hinted_original_prompt():
    item = trajectory()
    original = item["attempts"][0]["artifact"]
    original["response_prompt"] += "\nHints: leaked"
    original["response_prompt_hash"] = _hash(original["response_prompt"])
    with pytest.raises(ValueError, match="original no-hint"):
        build_direct_correction_preference_row(item)
