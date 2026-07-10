from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _example_index(examples: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("preference example is missing a non-empty id")
        if example_id in indexed:
            raise ValueError(f"preference examples contain duplicate id: {example_id}")
        indexed[example_id] = dict(example)
    if not indexed:
        raise ValueError("preference examples must not be empty")
    return indexed


def _attempt_groups(attempts: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        example_id = attempt.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("collection attempt is missing a non-empty id")
        number = attempt.get("attempt")
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"collection attempt {example_id} has invalid attempt number")
        response = attempt.get("response")
        result = attempt.get("result")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"collection attempt {example_id}:{number} has empty response")
        if not isinstance(result, dict) or not isinstance(result.get("correct"), bool):
            raise ValueError(f"collection attempt {example_id}:{number} has invalid evaluator result")
        groups[example_id].append(dict(attempt))
    for example_id, group in groups.items():
        group.sort(key=lambda row: int(row["attempt"]))
        numbers = [int(row["attempt"]) for row in group]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"collection group {example_id} contains duplicate attempt numbers")
    return dict(groups)


def _pair(
    *,
    example: Mapping[str, Any],
    prompt: str,
    failed: Mapping[str, Any],
    chosen: Mapping[str, Any],
    matched: bool,
) -> dict[str, Any]:
    failed_response = str(failed["response"])
    chosen_response = str(chosen["response"])
    metadata = {
        "domain": str(example["domain"]),
        "method_family": "multilevel_feedback_dpo",
        "failed_attempt": int(failed["attempt"]),
        "first_correct_attempt": int(chosen["attempt"]),
        "attempts_before_first_correct": int(chosen["attempt"]),
        "failed_result": dict(failed["result"]),
        "chosen_result": dict(chosen["result"]),
        "prompt_hash": _stable_hash(prompt),
        "chosen_hash": _stable_hash(chosen_response),
        "rejected_hash": _stable_hash(failed_response),
        "matched": matched,
    }
    return {
        "id": f"{example['id']}::attempt-{failed['attempt']}",
        "group_id": str(example["id"]),
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": failed_response,
        "metadata": metadata,
    }


def _matched_sample(rows: list[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError("matched sample count must be between zero and candidate count")
    strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[int(row["metadata"]["failed_attempt"])].append(row)
    for attempt, group in strata.items():
        group.sort(key=lambda row: _stable_hash({"seed": seed, "attempt": attempt, "id": row["id"]}))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for attempt in sorted(strata):
            if strata[attempt] and len(selected) < count:
                selected.append(strata[attempt].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError("matched sampling exhausted candidates before reaching requested count")
    output: list[dict[str, Any]] = []
    for row in selected:
        copied = {**row, "metadata": {**row["metadata"], "matched": True}}
        output.append(copied)
    return output


def build_preference_datasets(
    *,
    attempts: Iterable[Mapping[str, Any]],
    examples: Iterable[Mapping[str, Any]],
    seed: int,
    base_prompt_builder: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    """Build standard, all-level, and pair-budget-matched preference datasets."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("preference seed must be an integer")
    example_index = _example_index(examples)
    groups = _attempt_groups(attempts)
    standard: list[dict[str, Any]] = []
    multilevel: list[dict[str, Any]] = []
    unresolved = 0
    successful_groups = 0
    for example_id, example in example_index.items():
        group = groups.get(example_id, [])
        if not group:
            raise ValueError(f"collection attempts are missing example group: {example_id}")
        correct = [row for row in group if bool(row["result"]["correct"])]
        if not correct:
            unresolved += 1
            continue
        chosen = min(correct, key=lambda row: int(row["attempt"]))
        prior_wrong = [row for row in group if int(row["attempt"]) < int(chosen["attempt"]) and not row["result"]["correct"]]
        if not prior_wrong:
            continue
        successful_groups += 1
        prompt = base_prompt_builder(example)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"base prompt builder returned empty prompt for {example_id}")
        if "teacher guidance" in prompt.casefold():
            raise ValueError(f"base prompt for {example_id} contains teacher guidance")
        all_pairs = [_pair(example=example, prompt=prompt, failed=row, chosen=chosen, matched=False) for row in prior_wrong]
        multilevel.extend(all_pairs)
        first_attempt = next((row for row in prior_wrong if int(row["attempt"]) == 0), None)
        if first_attempt is not None:
            standard.append(_pair(example=example, prompt=prompt, failed=first_attempt, chosen=chosen, matched=False))

    matched = _matched_sample(multilevel, len(standard), seed=seed)
    for row in matched:
        row["metadata"] = {**row["metadata"], "method_family": "pair_budget_matched_multilevel_feedback_dpo"}
    return {
        "standard": standard,
        "multilevel": multilevel,
        "matched": matched,
        "metrics": {
            "groups_total": len(example_index),
            "successful_groups": successful_groups,
            "unresolved_groups": unresolved,
            "standard_pairs": len(standard),
            "multilevel_pairs": len(multilevel),
            "matched_pairs": len(matched),
            "matched_attempt_distribution": {
                str(attempt): sum(1 for row in matched if row["metadata"]["failed_attempt"] == attempt)
                for attempt in sorted({int(row["metadata"]["failed_attempt"]) for row in matched})
            },
        },
    }
