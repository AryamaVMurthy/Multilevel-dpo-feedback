from __future__ import annotations

import hashlib
import re
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


def build_sweep_prompt(example: dict[str, Any]) -> str:
    from text_feedback_dpo.prompts import build_native_student_prompt

    if example.get("domain") != "math":
        raise ValueError("MATH decoding sweep requires domain=math")
    problem = example.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("MATH decoding sweep example requires a non-empty problem")
    return build_native_student_prompt(problem=problem, domain="math")


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


def validate_screening_context(
    manifest: dict[str, Any],
    *,
    config_sha256: str,
    dataset_manifest_sha256: str,
    dataset_audit_sha256: str,
    model_cache_manifest_sha256: str,
    model: dict[str, str],
) -> None:
    if manifest.get("schema") != "math-decoding-sweep-v1" or manifest.get("stage") != "screening":
        raise ValueError("confirmation requires a MATH screening sweep manifest")
    expected = {
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_audit_sha256": dataset_audit_sha256,
        "model_cache_manifest_sha256": model_cache_manifest_sha256,
        "model": model,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"screening context mismatch in {field}")


def build_decoding_freeze(
    *,
    screening_manifest: dict[str, Any],
    screening_selection: dict[str, Any],
    confirmation_manifest: dict[str, Any],
    confirmation_selection: dict[str, Any],
    student_generation: dict[str, Any],
    frozen_config_sha256: str,
    source_commit: str,
    screening_selection_sha256: str,
    confirmation_selection_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("screening_selection_sha256", screening_selection_sha256),
        ("confirmation_selection_sha256", confirmation_selection_sha256),
        ("frozen_config_sha256", frozen_config_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be an immutable lowercase Git SHA")
    for stage, manifest, selection, count, ceiling in (
        ("screening", screening_manifest, screening_selection, 12, 4096),
        ("confirmation", confirmation_manifest, confirmation_selection, 32, 8192),
    ):
        if manifest.get("schema") != "math-decoding-sweep-v1" or manifest.get("stage") != stage:
            raise ValueError(f"{stage} sweep manifest is invalid")
        if manifest.get("count") != count or manifest.get("max_new_tokens") != ceiling:
            raise ValueError(f"{stage} sweep size or token ceiling is invalid")
        if (
            selection.get("schema") != "math-decoding-sweep-selection-v1"
            or selection.get("stage") != stage
            or selection.get("status") != "passed"
        ):
            raise ValueError(f"{stage} selection is not passed")
        ids = selection.get("example_ids")
        if not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"{stage} selection example IDs are invalid")
    if set(screening_selection["example_ids"]) & set(confirmation_selection["example_ids"]):
        raise ValueError("screening and confirmation examples overlap")
    promoted = screening_selection.get("promoted")
    if not isinstance(promoted, list) or len(promoted) != 3 or set(promoted) != set(confirmation_manifest.get("profiles", {})):
        raise ValueError("confirmation profiles do not match screening promotion")
    selected = confirmation_selection.get("selected_profile")
    if confirmation_selection.get("promoted") != [selected] or selected not in promoted:
        raise ValueError("confirmation did not select exactly one promoted profile")
    expected_generation = {**PROFILES[str(selected)], "max_new_tokens": 8192}
    if student_generation != expected_generation:
        raise ValueError("frozen student generation does not match the selected profile")
    context_fields = (
        "dataset_manifest_sha256",
        "dataset_audit_sha256",
        "model_cache_manifest_sha256",
        "model",
        "prompt_protocol",
    )
    for field in context_fields:
        if confirmation_manifest.get(field) != screening_manifest.get(field):
            raise ValueError(f"decoding stages differ in {field}")
    return {
        "schema": "math-decoding-freeze-v1",
        "source_commit": source_commit,
        "frozen_config_sha256": frozen_config_sha256,
        "selection_config_sha256": confirmation_manifest["config_sha256"],
        "screening_selection_sha256": screening_selection_sha256,
        "confirmation_selection_sha256": confirmation_selection_sha256,
        "selected_profile": selected,
        "student_generation": student_generation,
        "model": confirmation_manifest["model"],
        "prompt_protocol": confirmation_manifest["prompt_protocol"],
        "dataset_manifest_sha256": confirmation_manifest["dataset_manifest_sha256"],
        "dataset_audit_sha256": confirmation_manifest["dataset_audit_sha256"],
        "model_cache_manifest_sha256": confirmation_manifest["model_cache_manifest_sha256"],
        "screening_example_ids": screening_selection["example_ids"],
        "confirmation_example_ids": confirmation_selection["example_ids"],
    }


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
