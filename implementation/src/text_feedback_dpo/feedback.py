from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

class FeedbackFormatError(ValueError):
    """Raised when teacher feedback violates the minimal-hint contract."""


@dataclass(frozen=True)
class MinimalFeedback:
    hint: str


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise FeedbackFormatError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _normalized_tokens(value: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[\w]+", value, flags=re.UNICODE))


def _gold_leak_kind(hint: str, gold_answer: str) -> str | None:
    gold_tokens = _normalized_tokens(gold_answer)
    hint_tokens = _normalized_tokens(hint)
    if not gold_tokens:
        return None
    if any(hint_tokens[index : index + len(gold_tokens)] == gold_tokens for index in range(len(hint_tokens) - len(gold_tokens) + 1)):
        return "full"
    if set(gold_tokens).intersection(hint_tokens):
        return "token"
    return None


def is_feedback_shape_valid(text: str) -> bool:
    """Check teacher JSON shape before the caller performs gold-leak checks."""
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (FeedbackFormatError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {"hint"}:
        return False
    hint = payload["hint"]
    return isinstance(hint, str) and bool(hint.strip()) and len(hint.strip().split()) <= 24


def parse_feedback(text: str, *, gold_answer: str) -> MinimalFeedback:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except FeedbackFormatError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise FeedbackFormatError(f"invalid JSON feedback: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"hint"}:
        raise FeedbackFormatError("feedback must contain exactly one field: hint")
    hint = payload["hint"]
    if not isinstance(hint, str) or not hint.strip():
        raise FeedbackFormatError("hint must be a non-empty string")
    hint = hint.strip()
    if len(hint.split()) > 24:
        raise FeedbackFormatError("hint exceeds 24 words")
    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise FeedbackFormatError("gold answer is required for leakage checks")
    if not _normalized_tokens(gold_answer):
        raise FeedbackFormatError("gold answer must contain meaningful normalized tokens")
    leak_kind = _gold_leak_kind(hint, gold_answer)
    if leak_kind == "token":
        raise FeedbackFormatError("hint contains a normalized gold answer token")
    if leak_kind == "full":
        raise FeedbackFormatError("hint contains the gold answer")
    return MinimalFeedback(hint=hint)


def diagnose_attempt(artifact: Mapping[str, object]) -> dict[str, object]:
    """Locate the first deterministic failure in the active artifact.

    ``lexical_support_proxy`` is intentionally reported as an evaluator proxy;
    it is not a semantic entailment judgment.
    """
    if not isinstance(artifact, Mapping):
        raise TypeError("active pipeline artifact must be a mapping")
    truncation = artifact.get("truncation")
    if not isinstance(truncation, Mapping):
        raise ValueError("active pipeline artifact requires truncation metadata")
    query_truncated = truncation.get("query") is True
    response_truncated = truncation.get("response") is True
    raw_query = artifact.get("raw_query")
    ranked = artifact.get("ranked_search_results")
    retrieval_metrics = artifact.get("retrieval_metrics")
    error_code = artifact.get("error_code")
    cited_score = artifact.get("cited_score")
    if not isinstance(cited_score, Mapping):
        cited_score = {}
    query_failure = (
        query_truncated
        or not isinstance(raw_query, str)
        or not raw_query.strip()
        or (isinstance(error_code, str) and error_code.startswith("query_"))
        or not isinstance(ranked, list)
        or (isinstance(ranked, list) and not ranked)
        or (isinstance(retrieval_metrics, Mapping) and retrieval_metrics.get("recall@8") == 0.0 and isinstance(ranked, list) and bool(ranked))
    )
    parse_valid = cited_score.get("parse_valid") is True
    answer_correct = cited_score.get("answer_correct") is True
    lexical_support = cited_score.get("lexical_cited_answer_support")
    citation_precision = cited_score.get("citation_precision")
    if query_failure:
        region = "query/retrieval"
    elif response_truncated or not parse_valid or artifact.get("raw_response") is None:
        region = "response grammar/truncation"
    elif not answer_correct:
        region = "answer"
    elif lexical_support != 1.0 or citation_precision != 1.0:
        region = "lexical support proxy/citation selection"
    else:
        region = None
    return {
        "responsible_region": region,
        "error_code": error_code,
        "query_truncated": query_truncated,
        "response_truncated": response_truncated,
        "parse_valid": parse_valid,
        "answer_correct": answer_correct,
        "lexical_support_proxy": lexical_support if isinstance(lexical_support, (int, float)) else 0.0,
        "lexical_support_is_proxy": True,
        "citation_precision": citation_precision if isinstance(citation_precision, (int, float)) else 0.0,
        "retrieval_metrics": dict(retrieval_metrics) if isinstance(retrieval_metrics, Mapping) else None,
    }
