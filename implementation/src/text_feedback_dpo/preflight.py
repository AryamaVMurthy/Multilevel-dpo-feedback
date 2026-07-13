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
        if not isinstance(prediction.get("truncated"), bool):
            raise ValueError(f"prediction {example_id} requires explicit boolean truncated metadata")
        by_id[example_id] = prediction
    expected_ids = [str(example["id"]) for example in examples]
    if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
        raise ValueError("prediction/example ID parity mismatch")
    has_active_fields = any("raw_query" in prediction or "ranked_search_results" in prediction for prediction in predictions)
    if protocol == "active-search" and not all("raw_query" in prediction and "ranked_search_results" in prediction for prediction in predictions):
        raise ValueError("active-search preflight requires the complete active-search prediction schema")
    if protocol == "archival" and has_active_fields:
        raise ValueError("archival preflight received active-search fields; choose protocol=active-search")
    if protocol == "active-search":
        return _summarize_cited_quality(examples, predictions, by_id)
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


def _summarize_cited_quality(examples: list[dict], predictions: list[dict], by_id: dict[str, dict]) -> dict:
    from text_feedback_dpo.scoring import score_cited_response

    format_valid = []
    citation_valid = []
    support = []
    rendered = []
    query_truncated = []
    response_truncated = []
    retrieval_recall = []
    reasoning_lengths = []
    exact_matches = []
    f1_scores = []
    nonempty = []
    citation_precision = []
    citation_recall = []
    citation_coverage = []
    unsupported_source_rate = []
    for example in examples:
        prediction = by_id[str(example["id"])]
        example_id = str(example["id"])
        if not isinstance(prediction.get("raw_query"), str):
            raise ValueError(f"cited prediction {example_id} requires raw_query")
        ranked = prediction.get("ranked_search_results")
        if not isinstance(ranked, list):
            raise ValueError(f"cited prediction {example_id} requires ranked_search_results")
        truncation = prediction.get("truncation")
        if not isinstance(truncation, dict) or not isinstance(truncation.get("query"), bool) or not isinstance(truncation.get("response"), bool):
            raise ValueError(f"cited prediction {example_id} requires explicit query/response truncation flags")
        raw_response = prediction.get("raw_response")
        if raw_response is None:
            cited_score = prediction.get("cited_score")
            if not isinstance(cited_score, dict):
                error_code = prediction.get("error_code")
                if not isinstance(error_code, str) or not error_code:
                    raise ValueError(f"cited prediction {example_id} requires raw_response, cited_score, or explicit error_code")
                cited_score = {
                    "parse_valid": False,
                    "error_code": error_code,
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "answer": "",
                    "citation_count": 0,
                }
        elif not isinstance(raw_response, str):
            raise ValueError(f"cited prediction {example_id} raw_response must be a string or null")
        else:
            cited_score = score_cited_response(raw_response, example["gold_answer"], ranked, truncated=truncation["response"])
            if truncation["response"]:
                cited_score = {
                    **cited_score,
                    "parse_valid": False,
                    "correct": False,
                    "malformed_response": True,
                    "error_code": "response_truncated",
                    "lexical_cited_answer_support": 0.0,
                    "citation_precision": 0.0,
                    "citation_recall": 0.0,
                }
        format_valid.append(bool(cited_score.get("parse_valid")))
        citation_valid.append(bool(cited_score.get("parse_valid") and cited_score.get("citation_count", 0) > 0))
        support_value = cited_score.get("lexical_cited_answer_support")
        if not isinstance(support_value, (int, float)):
            raise ValueError(f"cited prediction {example_id} score lacks explicit cited-answer support metric")
        support.append(float(support_value))
        visible = prediction.get("rendered_visible_response")
        rendered.append(isinstance(visible, str) and bool(visible.strip()) and not truncation["response"])
        query_truncated.append(truncation["query"])
        response_truncated.append(truncation["response"])
        retrieval = prediction.get("retrieval_metrics")
        if retrieval is None:
            if isinstance(prediction.get("error_code"), str) and prediction["error_code"].startswith("query_"):
                retrieval_recall.append(0.0)
            else:
                raise ValueError(f"cited prediction {example_id} requires retrieval_metrics.recall@8")
        elif not isinstance(retrieval, dict) or not isinstance(retrieval.get("recall@8"), (int, float)):
            raise ValueError(f"cited prediction {example_id} requires retrieval_metrics.recall@8")
        else:
            retrieval_recall.append(float(retrieval["recall@8"]))
        reasoning = prediction.get("parsed_response", {}).get("reasoning", "") if isinstance(prediction.get("parsed_response"), dict) else ""
        reasoning_lengths.append(len(str(reasoning).split()))
        exact_matches.append(float(cited_score.get("exact_match", 0.0)))
        f1_scores.append(float(cited_score.get("f1", 0.0)))
        nonempty.append(bool(str(cited_score.get("answer", "")).strip()))
        for field, values in (
            ("citation_precision", citation_precision),
            ("citation_recall", citation_recall),
            ("citation_coverage", citation_coverage),
            ("unsupported_source_rate", unsupported_source_rate),
        ):
            value = cited_score.get(field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"cited prediction {example_id} score lacks explicit {field} metric")
            values.append(float(value))
    count = len(examples)
    return {
        "examples": count,
        "exact_match": sum(exact_matches) / count,
        "f1": sum(f1_scores) / count,
        "nonempty_rate": sum(nonempty) / count,
        "valid_format_rate": sum(format_valid) / count,
        "valid_citation_rate": sum(citation_valid) / count,
        "citation_precision": sum(citation_precision) / count,
        "citation_recall": sum(citation_recall) / count,
        "citation_coverage": sum(citation_coverage) / count,
        "lexical_cited_answer_support_rate": sum(support) / count,
        "unsupported_source_rate": sum(unsupported_source_rate) / count,
        "rendered_visible_rate": sum(rendered) / count,
        "query_truncation_rate": sum(query_truncated) / count,
        "response_truncation_rate": sum(response_truncated) / count,
        "truncation_rate": sum(query or response for query, response in zip(query_truncated, response_truncated, strict=True)) / count,
        "retrieval_recall@8": sum(retrieval_recall) / count,
        "malformed_rate": 1.0 - sum(format_valid) / count,
        "reasoning_words": {"max": max(reasoning_lengths), "mean": sum(reasoning_lengths) / count},
    }


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
