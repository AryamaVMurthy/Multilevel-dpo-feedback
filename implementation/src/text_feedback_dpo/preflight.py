from __future__ import annotations

import hashlib

from text_feedback_dpo.scoring import normalize_answer, score_searchqa


ACTIVE_SEARCH_ONLY_FIELDS = frozenset({
    "raw_query", "ranked_search_results", "raw_response", "parsed_response",
    "rendered_visible_response", "cited_score", "retrieval_metrics",
    "query_truncated", "response_truncated", "timings_ms", "prompt_version",
    "response_schema_version", "source_schema_version",
})


def has_active_search_fields(prediction: dict) -> bool:
    return bool(ACTIVE_SEARCH_ONLY_FIELDS.intersection(prediction))


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


def summarize_response_quality(examples: list[dict], predictions: list[dict], *, protocol: str = "archival") -> dict:
    if protocol not in {"archival", "active-search"}:
        raise ValueError("preflight protocol must be archival or active-search")
    example_ids = [str(example.get("id", "")) for example in examples]
    if any(not example_id for example_id in example_ids) or len(set(example_ids)) != len(example_ids):
        raise ValueError("example rows require unique non-empty ids")
    by_id = {}
    for prediction in predictions:
        example_id = str(prediction.get("id", ""))
        if not example_id or example_id in by_id:
            raise ValueError(f"predictions require unique non-empty ids: {example_id!r}")
        if protocol == "archival" and not isinstance(prediction.get("truncated"), bool):
            raise ValueError(f"prediction {example_id} requires explicit boolean truncated metadata")
        by_id[example_id] = prediction
    expected_ids = [str(example["id"]) for example in examples]
    if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
        raise ValueError("prediction/example ID parity mismatch")
    has_active_fields = any(has_active_search_fields(prediction) for prediction in predictions)
    if protocol == "active-search" and not all("raw_query" in prediction and "ranked_search_results" in prediction for prediction in predictions):
        raise ValueError("active-search preflight requires the complete active-search prediction schema")
    if protocol == "archival" and has_active_fields:
        raise ValueError("archival preflight received active-search fields; choose protocol=active-search")
    if protocol == "active-search":
        return summarize_active_evaluation(evaluate_active_predictions(examples, predictions))
    scored = []
    copied = []
    markup = []
    verbose = []
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
        verbose.append(word_count > 8)
    count = len(scored)
    if count == 0:
        raise ValueError("preflight requires at least one example")
    sorted_lengths = sorted(lengths)
    p95_index = min(count - 1, int(0.95 * count))
    summary = {
        "examples": count,
        "exact_match": sum(row["exact_match"] for row in scored) / count,
        "f1": sum(row["f1"] for row in scored) / count,
        "nonempty_rate": sum(bool(row["answer"]) for row in scored) / count,
        "copying_rate": sum(copied) / count,
        "markup_rate": sum(markup) / count,
        "verbose_rate": sum(verbose) / count,
        "truncation_rate": sum(by_id[example_id]["truncated"] for example_id in expected_ids) / count,
        "answer_words": {
            "min": min(lengths),
            "mean": sum(lengths) / count,
            "p95": sorted_lengths[p95_index],
            "max": max(lengths),
        },
    }
    return summary


_ACTIVE_TIMING_FIELDS = (
    "query_generation_batch_wall_ms",
    "query_generation_amortized_per_item_ms",
    "retrieval_individual_ms",
    "response_generation_batch_wall_ms",
    "response_generation_amortized_per_item_ms",
    "pipeline_wall_ms",
)


def _zero_active_score(error_code: str, *, truncated: bool) -> dict:
    from text_feedback_dpo.batch_generation import _zero_cited_score

    return _zero_cited_score(error_code, truncated=truncated)


def _validate_active_timings(example_id: str, prediction: dict) -> dict:
    timings = prediction.get("timings_ms")
    if not isinstance(timings, dict) or set(timings) != set(_ACTIVE_TIMING_FIELDS):
        raise ValueError(f"active prediction {example_id} requires exact truthful timing fields")
    for field in _ACTIVE_TIMING_FIELDS:
        value = timings[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"active prediction {example_id} timing {field} must be nonnegative")
    return timings


def evaluate_active_predictions(examples: list[dict], predictions: list[dict]) -> list[dict]:
    """Recompute active-search retrieval, scoring, parsing, and rendering from dataset-owned inputs."""
    from text_feedback_dpo.batch_generation import FIXED_B, FIXED_K1, FIXED_TOP_K, parse_search_query
    from text_feedback_dpo.responses import parse_cited_response, render_cited_response
    from text_feedback_dpo.retrieval import FixedBM25Retriever, retrieval_metrics, tokenize_query
    from text_feedback_dpo.scoring import score_cited_response

    example_by_id: dict[str, dict] = {}
    for example in examples:
        example_id = str(example.get("id", ""))
        if not example_id or example_id in example_by_id:
            raise ValueError("example rows require unique non-empty ids")
        example_by_id[example_id] = example
    prediction_by_id: dict[str, dict] = {}
    for prediction in predictions:
        example_id = str(prediction.get("id", ""))
        if not example_id or example_id in prediction_by_id:
            raise ValueError(f"predictions require unique non-empty ids: {example_id!r}")
        prediction_by_id[example_id] = prediction
    if set(example_by_id) != set(prediction_by_id) or len(example_by_id) != len(prediction_by_id):
        raise ValueError("prediction/example ID parity mismatch")

    rows = []
    for example_id, example in example_by_id.items():
        prediction = prediction_by_id[example_id]
        raw_query = prediction.get("raw_query")
        if not isinstance(raw_query, str):
            raise ValueError(f"active prediction {example_id} requires raw_query string")
        stored_ranked = prediction.get("ranked_search_results")
        if not isinstance(stored_ranked, list):
            raise ValueError(f"active prediction {example_id} requires ranked_search_results")
        stored_metrics = prediction.get("retrieval_metrics")
        if not isinstance(stored_metrics, dict):
            raise ValueError(f"active prediction {example_id} requires retrieval_metrics")
        truncation = prediction.get("truncation")
        if not isinstance(truncation, dict) or not isinstance(truncation.get("query"), bool) or not isinstance(truncation.get("response"), bool):
            raise ValueError(f"active prediction {example_id} requires explicit query/response truncation flags")
        for field, stage in (("query_truncated", "query"), ("response_truncated", "response")):
            if not isinstance(prediction.get(field), bool) or prediction[field] != truncation[stage]:
                raise ValueError(f"active prediction {example_id} {field} mismatch with truncation.{stage}")
        expected_truncated = bool(truncation["query"] or truncation["response"])
        if not isinstance(prediction.get("truncated"), bool) or prediction["truncated"] != expected_truncated:
            raise ValueError(f"active prediction {example_id} has inconsistent aggregate truncation flag")
        timings = _validate_active_timings(example_id, prediction)

        query_format_valid = True
        try:
            normalized_query = parse_search_query(raw_query)
        except ValueError:
            normalized_query = None
            query_format_valid = False
        query_stage_error = truncation["query"] or not query_format_valid
        expected_error = "query_truncated" if truncation["query"] else "query_invalid_format"
        if query_stage_error:
            if stored_ranked:
                raise ValueError(f"active prediction {example_id} retrieval artifact mismatch: query-stage failures require empty ranked results")
            canonical_ranked = []
            canonical_metrics = retrieval_metrics([], example["gold_answer"])
            if stored_metrics != canonical_metrics:
                raise ValueError(f"active prediction {example_id} retrieval metrics mismatch")
            if prediction.get("error_code") != expected_error:
                raise ValueError(f"active prediction {example_id} query-stage error_code mismatch")
            if prediction.get("raw_response") is not None:
                raise ValueError(f"active prediction {example_id} query-stage model error requires raw_response null")
            if truncation["response"]:
                raise ValueError(f"active prediction {example_id} query-stage error cannot also mark response truncation")
            score = _zero_active_score(expected_error, truncated=truncation["query"])
            parsed = None
            rendered = None
            error_code = expected_error
        else:
            if "sources" not in example:
                raise ValueError(f"active example {example_id} requires dataset-owned sources")
            canonical_ranked = FixedBM25Retriever(example["sources"], k1=FIXED_K1, b=FIXED_B).search(
                normalized_query, top_k=FIXED_TOP_K
            )
            if stored_ranked != canonical_ranked:
                raise ValueError(f"active prediction {example_id} retrieval artifact mismatch")
            canonical_metrics = retrieval_metrics(canonical_ranked, example["gold_answer"])
            if stored_metrics != canonical_metrics:
                raise ValueError(f"active prediction {example_id} retrieval metrics mismatch")
            raw_response = prediction.get("raw_response")
            if raw_response is None:
                raise ValueError(f"active prediction {example_id} raw_response null is allowed only for query-stage model errors")
            if not isinstance(raw_response, str):
                raise ValueError(f"active prediction {example_id} raw_response must be a string or null")
            score = score_cited_response(raw_response, example["gold_answer"], canonical_ranked, truncated=truncation["response"])
            capability_exact = float(score["exact_match"])
            capability_f1 = float(score["f1"])
            protocol_valid = bool(score["parse_valid"] and not truncation["response"])
            score.update({
                "answer_capability_exact_match": capability_exact,
                "answer_capability_f1": capability_f1,
                "protocol_exact_match": capability_exact if protocol_valid else 0.0,
                "protocol_f1": capability_f1 if protocol_valid else 0.0,
                "correct": bool(score["correct"] and protocol_valid),
            })
            parsed = None
            rendered = None
            if truncation["response"]:
                score.update({
                    "parse_valid": False,
                    "correct": False,
                    "malformed_response": True,
                    "error_code": "response_truncated",
                    "lexical_cited_answer_support": 0.0,
                    "citation_precision": 0.0,
                    "citation_recall": 0.0,
                })
                error_code = "response_truncated"
            elif score["parse_valid"]:
                parsed_object = parse_cited_response(raw_response, canonical_ranked)
                parsed = {"answer": parsed_object.answer, "reasoning": parsed_object.reasoning, "source_ids": list(parsed_object.source_ids)}
                rendered = render_cited_response(parsed_object, canonical_ranked)
                error_code = None
            else:
                error_code = score["error_code"]

        rows.append({
            "id": example_id,
            "raw_query": raw_query,
            "query_words": len(tokenize_query(raw_query)),
            "query_format_valid": query_format_valid,
            "ranked_search_results": canonical_ranked,
            "raw_response": prediction.get("raw_response"),
            "parsed_response": parsed,
            "rendered_visible_response": rendered,
            "cited_score": score,
            "retrieval_metrics": canonical_metrics,
            "truncation": dict(truncation),
            "truncated": expected_truncated,
            "error_code": error_code,
            "timings_ms": dict(timings),
        })
    return rows


def summarize_active_evaluation(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("active SearchQA evaluation requires at least one row")
    count = len(rows)

    def mean(values):
        return sum(values) / count

    scores = [row["cited_score"] for row in rows]
    pipeline_walls = {float(row["timings_ms"]["pipeline_wall_ms"]) for row in rows}
    pipeline_wall_ms = next(iter(pipeline_walls)) if len(pipeline_walls) == 1 else None
    summary = {
        "examples": count,
        "answer_capability_exact_match": mean([float(score["answer_capability_exact_match"]) for score in scores]),
        "answer_capability_f1": mean([float(score["answer_capability_f1"]) for score in scores]),
        "protocol_exact_match": mean([float(score["protocol_exact_match"]) for score in scores]),
        "protocol_f1": mean([float(score["protocol_f1"]) for score in scores]),
        "exact_match": mean([float(score["protocol_exact_match"]) for score in scores]),
        "f1": mean([float(score["protocol_f1"]) for score in scores]),
        "nonempty_rate": mean([bool(str(score.get("answer", "")).strip()) for score in scores]),
        "valid_format_rate": mean([bool(score["parse_valid"]) for score in scores]),
        "valid_citation_rate": mean([float(score.get("valid_citation_rate", 0.0)) for score in scores]),
        "citation_precision": mean([float(score.get("citation_precision", 0.0)) for score in scores]),
        "citation_recall": mean([float(score.get("citation_recall", 0.0)) for score in scores]),
        "citation_coverage": mean([float(score.get("citation_coverage", 0.0)) for score in scores]),
        "duplicate_citation_rate": mean([float(score.get("duplicate_citation_rate", 0.0)) for score in scores]),
        "lexical_cited_answer_support_rate": mean([float(score.get("lexical_cited_answer_support", 0.0)) for score in scores]),
        "unsupported_source_rate": mean([float(score.get("unsupported_source_rate", 0.0)) for score in scores]),
        "rendered_visible_rate": mean([row["rendered_visible_response"] is not None for row in rows]),
        "empty_query_rate": mean([not bool(row["raw_query"].strip()) for row in rows]),
        "invalid_query_rate": mean([not row["query_format_valid"] for row in rows]),
        "query_truncation_rate": mean([row["truncation"]["query"] for row in rows]),
        "response_truncation_rate": mean([row["truncation"]["response"] for row in rows]),
        "truncation_rate": mean([row["truncated"] for row in rows]),
        "any_stage_truncation_rate": mean([row["truncated"] for row in rows]),
        "malformed_rate": mean([not score["parse_valid"] for score in scores]),
        "correct": sum(bool(score["correct"]) for score in scores),
    }
    for k in (1, 3, 5, 8):
        summary[f"retrieval_recall@{k}"] = mean([float(row["retrieval_metrics"][f"recall@{k}"]) for row in rows])
    summary["retrieval_mrr"] = mean([float(row["retrieval_metrics"]["mrr"]) for row in rows])
    query_words = [row["query_words"] for row in rows]
    reasoning_words = [int(score.get("reasoning_words", 0)) for score in scores]
    summary["query_words"] = {"mean": mean(query_words), "max": max(query_words)}
    summary["reasoning_words"] = {"mean": mean(reasoning_words), "max": max(reasoning_words)}
    summary["timing_ms"] = {
        "pipeline_wall": pipeline_wall_ms,
        "pipeline_wall_mean": mean([float(row["timings_ms"]["pipeline_wall_ms"]) for row in rows]),
        "pipeline_wall_max": max(float(row["timings_ms"]["pipeline_wall_ms"]) for row in rows),
        "query_generation_batch_wall": max(float(row["timings_ms"]["query_generation_batch_wall_ms"]) for row in rows),
        "response_generation_batch_wall": max(float(row["timings_ms"]["response_generation_batch_wall_ms"]) for row in rows),
        "retrieval_individual_mean": mean([float(row["timings_ms"]["retrieval_individual_ms"]) for row in rows]),
    }
    summary["throughput_examples_per_second"] = (
        count * 1000.0 / pipeline_wall_ms if pipeline_wall_ms is not None and pipeline_wall_ms > 0 else None
    )
    return summary


def assess_preflight(metrics: dict) -> dict:
    if "valid_format_rate" in metrics:
        thresholds = {
            "nonempty_rate": (">=", 0.95),
            "valid_format_rate": (">=", 0.95),
            "valid_citation_rate": (">=", 0.95),
            "lexical_cited_answer_support_rate": (">=", 0.95),
            "retrieval_recall@8": (">=", 0.95),
            "rendered_visible_rate": (">=", 0.95),
            "truncation_rate": ("<=", 0.05),
            "query_truncation_rate": ("<=", 0.05),
            "response_truncation_rate": ("<=", 0.05),
            "malformed_rate": ("<=", 0.05),
        }
    else:
        thresholds = {"nonempty_rate": (">=", 0.95), "copying_rate": ("<=", 0.05), "truncation_rate": ("<=", 0.05), "markup_rate": ("<=", 0.0), "verbose_rate": ("<=", 0.05)}
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
