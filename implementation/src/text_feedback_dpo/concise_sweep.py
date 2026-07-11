from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from typing import Any, Iterable


_COMMON = {
    "enable_thinking": False,
    "max_new_tokens": 4096,
    "stop_after_final_answer": True,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}

PROFILES: dict[str, dict[str, Any]] = {
    name: {
        **_COMMON,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": penalty,
    }
    for name, penalty in (
        ("presence-0", 0.0),
        ("presence-0.5", 0.5),
        ("presence-1", 1.0),
        ("presence-1.5", 1.5),
        ("presence-2", 2.0),
    )
}


def protocol_valid_correct(
    *,
    symbolic_correct: bool,
    terminated: bool | None,
    truncated: bool | None,
) -> bool:
    return symbolic_correct and terminated is True and truncated is False


def _rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def stratified_subset(rows: Iterable[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("subset count must be positive")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        level = row.get("difficulty_level")
        subject = row.get("source_subject")
        row_id = row.get("id")
        if level not in {4, 5}:
            continue
        if not isinstance(subject, str) or not subject or not isinstance(row_id, str) or not row_id:
            raise ValueError("eligible sweep rows require id, source_subject, and difficulty_level")
        groups[(subject, level)].append(row)
    if sum(map(len, groups.values())) < count:
        raise ValueError("not enough MATH Level 4-5 training rows for requested subset")
    for values in groups.values():
        values.sort(key=lambda row: _rank(str(row["id"]), seed))
    keys = sorted(groups, key=lambda key: _rank(f"{key[0]}:{key[1]}", seed))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("stratified selection exhausted unexpectedly")
    return selected


def validate_sweep_records(
    records: Iterable[dict[str, Any]],
    *,
    profiles: Iterable[str],
    example_ids: Iterable[str],
) -> None:
    expected = {(str(profile), str(row_id)) for profile in profiles for row_id in example_ids}
    actual: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("profile", "")), str(record.get("id", "")))
        if key in actual:
            raise ValueError(f"duplicate sweep profile/example record: {key}")
        actual.add(key)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"incomplete sweep records: missing={missing}, unexpected={unexpected}")


def summarize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["profile"])].append(record)
    summaries = []
    for profile, values in sorted(groups.items()):
        tokens = [int(value["generated_tokens"]) for value in values]
        latencies = [float(value["latency_seconds"]) for value in values]
        total_tokens = sum(tokens)
        if total_tokens <= 0:
            raise ValueError(f"profile {profile} has no generated tokens")
        correct = sum(bool(value["correct"]) for value in values)
        summaries.append({
            "profile": profile,
            "examples": len(values),
            "correct": correct,
            "accuracy": correct / len(values),
            "correct_per_million_tokens": correct * 1_000_000 / total_tokens,
            "median_tokens": statistics.median(tokens),
            "mean_tokens": statistics.fmean(tokens),
            "max_tokens": max(tokens),
            "truncated": sum(value["finish_reason"] == "length" for value in values),
            "truncation_rate": sum(value["finish_reason"] == "length" for value in values) / len(values),
            "unevaluable": sum(not bool(value.get("evaluable", True)) for value in values),
            "final_answer_stops": sum(value["finish_reason"] == "final_answer" for value in values),
            "mean_latency": statistics.fmean(latencies),
        })
    return summaries


def promote_profiles(summaries: Iterable[dict[str, Any]], *, count: int) -> list[str]:
    values = list(summaries)
    if count <= 0 or count > len(values):
        raise ValueError("promotion count must be within available profiles")
    values.sort(key=lambda value: (
        -int(value["correct"]),
        int(value["truncated"]),
        -float(value["correct_per_million_tokens"]),
        float(value["median_tokens"]),
        float(value["mean_latency"]),
        str(value["profile"]),
    ))
    return [str(value["profile"]) for value in values[:count]]
