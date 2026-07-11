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
    "qwen-reasoning": {**_COMMON, "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 40, "presence_penalty": 2.0},
    "qwen-general": {**_COMMON, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    "conservative": {**_COMMON, "do_sample": True, "temperature": 0.6, "top_p": 0.9, "top_k": 20, "presence_penalty": 0.0},
    "low-diversity": {**_COMMON, "do_sample": True, "temperature": 0.5, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0},
    "mild-repeat": {**_COMMON, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.5, "repetition_penalty": 1.05},
    "greedy": {**_COMMON, "do_sample": False, "presence_penalty": 0.0},
}


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


def summarize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["profile"])].append(record)
    summaries = []
    for profile, values in sorted(groups.items()):
        tokens = [int(value["generated_tokens"]) for value in values]
        latencies = [float(value["latency_seconds"]) for value in values]
        summaries.append({
            "profile": profile,
            "examples": len(values),
            "correct": sum(bool(value["correct"]) for value in values),
            "accuracy": sum(bool(value["correct"]) for value in values) / len(values),
            "median_tokens": statistics.median(tokens),
            "mean_tokens": statistics.fmean(tokens),
            "max_tokens": max(tokens),
            "truncated": sum(value["finish_reason"] == "length" for value in values),
            "final_answer_stops": sum(value["finish_reason"] == "final_answer" for value in values),
            "mean_latency": statistics.fmean(latencies),
        })
    return summaries


def promote_profiles(summaries: Iterable[dict[str, Any]], *, count: int) -> list[str]:
    values = list(summaries)
    if count <= 0 or count > len(values):
        raise ValueError("promotion count must be within available profiles")
    best_correct = max(int(value["correct"]) for value in values)
    eligible = [value for value in values if int(value["correct"]) >= best_correct - 1]
    eligible.sort(key=lambda value: (
        float(value["median_tokens"]),
        int(value["truncated"]),
        float(value["mean_latency"]),
        str(value["profile"]),
    ))
    if len(eligible) < count:
        remainder = [value for value in values if value not in eligible]
        remainder.sort(key=lambda value: (-int(value["correct"]), float(value["median_tokens"]), str(value["profile"])))
        eligible.extend(remainder)
    return [str(value["profile"]) for value in eligible[:count]]
