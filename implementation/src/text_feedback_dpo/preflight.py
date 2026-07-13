from __future__ import annotations

import hashlib

from text_feedback_dpo.scoring import normalize_answer, score_searchqa


def select_preflight_rows(rows: list[dict], *, sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(rows):
        raise ValueError(f"sample_size {sample_size} exceeds available rows {len(rows)}")
    keyed = []
    seen = set()
    for row in rows:
        example_id = str(row.get("id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"preflight rows require unique non-empty ids: {example_id!r}")
        seen.add(example_id)
        key = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
        keyed.append((key, example_id, row))
    return [row for _key, _id, row in sorted(keyed)[:sample_size]]


def summarize_response_quality(examples: list[dict], predictions: list[dict]) -> dict:
    by_id = {}
    for prediction in predictions:
        example_id = str(prediction.get("id", ""))
        if not example_id or example_id in by_id:
            raise ValueError(f"predictions require unique non-empty ids: {example_id!r}")
        if not isinstance(prediction.get("truncated"), bool):
            raise ValueError(f"prediction {example_id} requires explicit boolean truncated metadata")
        by_id[example_id] = prediction
    expected_ids = [str(example["id"]) for example in examples]
    if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
        raise ValueError("prediction/example ID parity mismatch")
    scored = []
    copied = []
    markup = []
    lengths = []
    for example in examples:
        prediction = by_id[str(example["id"])]
        response = prediction.get("response")
        if not isinstance(response, str):
            raise ValueError(f"prediction {example['id']} response must be a string")
        score = score_searchqa(response, example["gold_answer"], example["packed_evidence"])
        scored.append(score)
        normalized = normalize_answer(response)
        word_count = len(normalized.split())
        lengths.append(word_count)
        copied.append(bool(word_count > 8 and normalized and normalized in normalize_answer(example["packed_evidence"])))
        markup.append(any(marker in response for marker in ("<", ">", "{", "}", "```")))
    count = len(scored)
    if count == 0:
        raise ValueError("preflight requires at least one example")
    sorted_lengths = sorted(lengths)
    p95_index = min(count - 1, int(0.95 * count))
    return {
        "examples": count,
        "exact_match": sum(row["exact_match"] for row in scored) / count,
        "f1": sum(row["f1"] for row in scored) / count,
        "nonempty_rate": sum(bool(row["answer"]) for row in scored) / count,
        "copying_rate": sum(copied) / count,
        "markup_rate": sum(markup) / count,
        "truncation_rate": sum(by_id[example_id]["truncated"] for example_id in expected_ids) / count,
        "answer_words": {
            "min": min(lengths),
            "mean": sum(lengths) / count,
            "p95": sorted_lengths[p95_index],
            "max": max(lengths),
        },
    }


def assess_preflight(metrics: dict) -> dict:
    thresholds = {"nonempty_rate": (">=", 0.95), "copying_rate": ("<=", 0.05), "truncation_rate": ("<=", 0.05), "markup_rate": ("<=", 0.0)}
    failures = {}
    for name, (operator, threshold) in thresholds.items():
        if name not in metrics:
            raise ValueError(f"preflight metric is missing: {name}")
        value = float(metrics[name])
        passed = value >= threshold if operator == ">=" else value <= threshold
        if not passed:
            failures[name] = {"value": value, "required": f"{operator}{threshold}"}
    return {"promote": not failures, "failures": failures, "thresholds": thresholds}


def select_thinking_mode(summaries: dict[str, dict]) -> dict:
    if set(summaries) != {"direct", "two_pass"}:
        raise ValueError("thinking-mode selection requires exactly direct and two_pass summaries")
    eligible = [name for name, metrics in summaries.items() if assess_preflight(metrics)["promote"]]
    if not eligible:
        raise ValueError("no thinking mode passed structural preflight gates")
    selected = max(eligible, key=lambda name: (summaries[name]["exact_match"], summaries[name]["f1"], name == "direct"))
    return {"selected": selected, "selection_metric": ["exact_match", "f1", "prefer_direct_on_tie"], "eligible": eligible}
