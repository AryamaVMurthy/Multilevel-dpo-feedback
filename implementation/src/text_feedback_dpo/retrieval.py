from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from text_feedback_dpo.scoring import normalize_answer


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CANONICAL_SOURCE_FIELDS = ("source_id", "original_rank", "title", "url", "snippet", "related_links")


def tokenize_query(text: str) -> tuple[str, ...]:
    """Tokenize retrieval text deterministically by case-folding and splitting punctuation."""
    if not isinstance(text, str):
        raise TypeError("retrieval text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_PATTERN.findall(normalized))


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(sources, (str, bytes)) or not isinstance(sources, (list, tuple)):
        raise TypeError("sources must be a list of source mappings")
    if not sources:
        raise ValueError("sources must contain at least one source")

    normalized: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    original_ranks: set[int] = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            raise TypeError(f"source {index} must be a mapping")
        missing = [field for field in ("source_id", "original_rank", "title", "url", "snippet") if field not in source]
        if missing:
            raise ValueError(f"source {index} is missing required fields: {', '.join(missing)}")
        source_id = source["source_id"]
        original_rank = source["original_rank"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"source {index} requires a non-empty source_id")
        source_id = source_id.strip()
        if source_id in source_ids:
            raise ValueError(f"source_id must be unique: {source_id}")
        if isinstance(original_rank, bool) or not isinstance(original_rank, int) or original_rank < 1:
            raise ValueError(f"source {index} requires a positive integer original_rank")
        if original_rank in original_ranks:
            raise ValueError(f"original_rank must be unique: {original_rank}")
        for field in ("title", "url", "snippet"):
            value = source[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source {index} requires a non-empty {field}")
        related_links = source.get("related_links")
        if related_links is not None and not isinstance(related_links, str):
            raise ValueError(f"source {index} related_links must be null or a string")
        source_copy = dict(source)
        source_copy["source_id"] = source_id
        source_copy["title"] = source["title"].strip()
        source_copy["url"] = source["url"].strip()
        source_copy["snippet"] = source["snippet"].strip()
        source_copy["related_links"] = related_links.strip() if isinstance(related_links, str) else None
        if not tokenize_query(source_copy["snippet"]):
            raise ValueError(f"source {index} snippet must contain at least one token")
        normalized.append(source_copy)
        source_ids.add(source_id)
        original_ranks.add(original_rank)
    return normalized


def _canonical_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(sources, key=lambda source: (source["original_rank"], source["source_id"]))
    return [{field: source.get(field) for field in _CANONICAL_SOURCE_FIELDS} for source in ordered]


def _validate_top_k(top_k: int, source_count: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_k > source_count:
        raise ValueError(f"top_k must be <= source count ({source_count})")


def _validate_parameters(k1: float, b: float) -> None:
    if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 must be a finite number greater than zero")
    if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("b must be a finite number between zero and one")


class FixedBM25Retriever:
    """Deterministic BM25 search over one example's fixed canonical sources."""

    def __init__(self, sources: Sequence[Mapping[str, Any]], *, k1: float = 1.2, b: float = 0.75) -> None:
        _validate_parameters(k1, b)
        self._sources = sorted(_validate_sources(sources), key=lambda source: (source["original_rank"], source["source_id"]))
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens = [tokenize_query(source["snippet"]) for source in self._sources]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequencies = Counter(term for tokens in self._tokens for term in set(tokens))
        self._average_document_length = sum(map(len, self._tokens)) / len(self._tokens)
        self._corpus_hash = _hash_json(_canonical_sources(self._sources))

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        query_tokens = tokenize_query(query)
        if not query_tokens:
            raise ValueError("query must contain at least one token")
        _validate_top_k(top_k, len(self._sources))
        query_hash = _hash_json(list(query_tokens))
        document_count = len(self._sources)
        ranked: list[tuple[float, int, str, dict[str, Any]]] = []
        unique_query_terms = tuple(dict.fromkeys(query_tokens))

        for source, frequencies, document_length in zip(self._sources, self._term_frequencies, self._tokens, strict=True):
            score = 0.0
            matched_terms: list[str] = []
            for term in unique_query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                matched_terms.append(term)
                document_frequency = self._document_frequencies[term]
                idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                normalization = 1 - self.b + self.b * (len(document_length) / self._average_document_length)
                score += idf * (term_frequency * (self.k1 + 1)) / (term_frequency + self.k1 * normalization)
            result = dict(source)
            result.update(
                {
                    "retrieval_rank": 0,
                    "bm25_score": float(score),
                    "matched_query_terms": matched_terms,
                    "query_hash": query_hash,
                    "corpus_hash": self._corpus_hash,
                }
            )
            ranked.append((score, source["original_rank"], source["source_id"], result))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        results = []
        for rank, (_, _, _, result) in enumerate(ranked[:top_k], start=1):
            result["retrieval_rank"] = rank
            results.append(result)
        return results


def retrieve(
    query: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 8,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Search fixed sources without exposing a gold answer to the search API."""
    return FixedBM25Retriever(sources, k1=k1, b=b).search(query, top_k=top_k)


def _contains_answer(result: Mapping[str, Any], normalized_gold: str) -> bool:
    answer_tokens = normalized_gold.split()
    source_text = " ".join(str(result.get(field, "")) for field in ("title", "snippet"))
    source_tokens = normalize_answer(source_text).split()
    width = len(answer_tokens)
    return any(source_tokens[index : index + width] == answer_tokens for index in range(len(source_tokens) - width + 1))


def _validate_ranked_results(ranked_results: Sequence[Mapping[str, Any]]) -> None:
    source_ids: set[str] = set()
    original_ranks: set[int] = set()
    expected_query_hash: str | None = None
    expected_corpus_hash: str | None = None
    required_fields = ("retrieval_rank", "source_id", "original_rank", "bm25_score", "query_hash", "corpus_hash", "title", "snippet")

    for position, result in enumerate(ranked_results, start=1):
        missing = [field for field in required_fields if field not in result]
        if missing:
            raise ValueError(f"ranked result {position} is missing required field: {missing[0]}")

        retrieval_rank = result["retrieval_rank"]
        if isinstance(retrieval_rank, bool) or not isinstance(retrieval_rank, int) or retrieval_rank != position:
            raise ValueError(f"ranked result {position} retrieval_rank must equal its list position")

        source_id = result["source_id"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"ranked result {position} requires a non-empty source_id")
        normalized_source_id = source_id.strip()
        if normalized_source_id in source_ids:
            raise ValueError(f"ranked result source_id must be unique: {normalized_source_id}")
        source_ids.add(normalized_source_id)

        original_rank = result["original_rank"]
        if isinstance(original_rank, bool) or not isinstance(original_rank, int) or original_rank <= 0:
            raise ValueError(f"ranked result {position} requires a positive integer original_rank")
        if original_rank in original_ranks:
            raise ValueError(f"ranked result original_rank must be unique: {original_rank}")
        original_ranks.add(original_rank)

        bm25_score = result["bm25_score"]
        if isinstance(bm25_score, bool) or not isinstance(bm25_score, Real) or not math.isfinite(bm25_score) or bm25_score < 0:
            raise ValueError(f"ranked result {position} bm25_score must be a finite nonnegative number")

        for field in ("title", "snippet"):
            value = result[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ranked result {position} requires a non-empty {field}")

        for field in ("query_hash", "corpus_hash"):
            value = result[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ranked result {position} requires a non-empty {field}")
        if expected_query_hash is None:
            expected_query_hash = result["query_hash"]
            expected_corpus_hash = result["corpus_hash"]
        elif result["query_hash"] != expected_query_hash:
            raise ValueError(f"ranked result {position} query_hash does not match prior rows")
        elif result["corpus_hash"] != expected_corpus_hash:
            raise ValueError(f"ranked result {position} corpus_hash does not match prior rows")


def retrieval_metrics(
    ranked_results: Sequence[Mapping[str, Any]],
    gold_answer: str,
    ks: Sequence[int] = (1, 3, 5, 8),
) -> dict[str, float | int | None]:
    """Report evaluator-only answer-bearing retrieval metrics for one row."""
    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise ValueError("gold answer is required for retrieval metrics")
    normalized_gold = normalize_answer(gold_answer)
    if not normalized_gold:
        raise ValueError("gold answer must contain searchable normalized tokens")
    if isinstance(ks, (str, bytes)):
        raise TypeError("ks must be a sequence of positive integers")
    try:
        requested_ks = tuple(ks)
    except TypeError as exc:
        raise TypeError("ks must be a sequence of positive integers") from exc
    if not requested_ks:
        raise ValueError("ks must not be empty")
    for k in requested_ks:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("each k must be a positive integer")

    if isinstance(ranked_results, (str, bytes)) or not isinstance(ranked_results, Sequence):
        raise TypeError("ranked_results must be a sequence of mappings")
    for index, result in enumerate(ranked_results, start=1):
        if not isinstance(result, Mapping):
            raise TypeError(f"ranked result {index} must be a mapping")
    _validate_ranked_results(ranked_results)
    answer_bearing_ranks = [index for index, result in enumerate(ranked_results, start=1) if _contains_answer(result, normalized_gold)]
    first_rank = answer_bearing_ranks[0] if answer_bearing_ranks else None
    metrics: dict[str, float | int | None] = {
        f"recall@{k}": float(bool(first_rank and first_rank <= k)) for k in requested_ks
    }
    reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0
    metrics.update({"reciprocal_rank": reciprocal_rank, "mrr": reciprocal_rank, "first_answer_rank": first_rank})
    return metrics
