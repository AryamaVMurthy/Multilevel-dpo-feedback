from __future__ import annotations

from collections.abc import Callable, Iterable


SOURCE_SCHEMA = "searchqa.search_results.v1"
SOURCE_SCHEMA_VERSION = 1


class NoUsableSearchQASourcesError(ValueError):
    """Raised when a valid SearchQA source schema contains no usable source."""

    def __init__(self, message: str, *, source_filter_stats: dict | None = None) -> None:
        self.source_filter_stats = source_filter_stats
        super().__init__(message)


def _source_arrays(search_results: object) -> tuple[list[object], list[object], list[object], list[object]]:
    if isinstance(search_results, dict):
        required = ("snippets", "titles", "urls")
        missing = [field for field in required if field not in search_results]
        if missing:
            raise ValueError(f"SearchQA source provenance mapping is missing: {', '.join(missing)}")
        arrays = {field: search_results[field] for field in required}
        related_links = search_results.get("related_links")
        if related_links is not None:
            arrays["related_links"] = related_links
        else:
            arrays["related_links"] = [None] * len(arrays["snippets"]) if isinstance(arrays["snippets"], list) else None
    elif isinstance(search_results, list):
        if not search_results:
            raise NoUsableSearchQASourcesError(
                "SearchQA row has no usable source records",
                source_filter_stats={
                    "input_records": 0,
                    "usable_records": 0,
                    "dropped_records": 0,
                    "drop_reasons": {},
                },
            )
        if not all(isinstance(record, dict) for record in search_results):
            raise ValueError("unsupported SearchQA source provenance schema: expected mapping or records with metadata")
        arrays = {"snippets": [], "titles": [], "urls": [], "related_links": []}
        for record in search_results:
            if "snippet" not in record:
                raise ValueError("unsupported SearchQA source provenance schema: record is missing snippet")
            arrays["snippets"].append(record["snippet"])
            arrays["titles"].append(record.get("title"))
            arrays["urls"].append(record.get("url"))
            arrays["related_links"].append(record.get("related_links"))
    else:
        raise ValueError("unsupported SearchQA source provenance schema: expected search_results mapping or records")

    for field in ("snippets", "titles", "urls", "related_links"):
        values = arrays[field]
        if not isinstance(values, list):
            raise ValueError(f"unsupported SearchQA source provenance schema: {field} must be an array")
    lengths = {field: len(arrays[field]) for field in ("snippets", "titles", "urls", "related_links")}
    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{field}={length}" for field, length in lengths.items())
        raise ValueError(f"SearchQA source array length mismatch: {details}")
    return arrays["snippets"], arrays["titles"], arrays["urls"], arrays["related_links"]


def _text_value(value: object, *, field: str, original_rank: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"unsupported SearchQA source provenance schema: {field} at original rank {original_rank} must be a string")
    return value.strip()


def _related_links_value(value: object, *, original_rank: int) -> str | list[str] | None:
    """Preserve the two related-link shapes present in SearchQA source records."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        if not all(isinstance(link, str) for link in value):
            raise ValueError(
                "unsupported SearchQA source provenance schema: "
                f"related_links at original rank {original_rank} must contain only strings"
            )
        return [link.strip() for link in value]
    raise ValueError(
        "unsupported SearchQA source provenance schema: "
        f"related_links at original rank {original_rank} must be null, a string, or an array of strings"
    )


def _sources(row: dict) -> tuple[list[dict], dict]:
    if "search_results" not in row:
        raise ValueError("SearchQA row requires search_results source provenance")
    snippets, titles, urls, related_links = _source_arrays(row["search_results"])
    sources = []
    drop_reasons = {"blank_snippet": 0, "missing_title": 0, "missing_url": 0}
    for original_index, (snippet_value, title_value, url_value, related_links_value) in enumerate(
        zip(snippets, titles, urls, related_links, strict=True),
        start=1,
    ):
        snippet = _text_value(snippet_value, field="snippet", original_rank=original_index)
        title = _text_value(title_value, field="title", original_rank=original_index)
        url = _text_value(url_value, field="url", original_rank=original_index)
        if not snippet:
            drop_reasons["blank_snippet"] += 1
            continue
        if not title:
            drop_reasons["missing_title"] += 1
            continue
        if not url:
            drop_reasons["missing_url"] += 1
            continue
        related = _related_links_value(related_links_value, original_rank=original_index)
        sources.append(
            {
                "source_id": f"S{original_index:03d}",
                "original_rank": original_index,
                "title": title,
                "url": url,
                "snippet": snippet,
                "related_links": related,
            }
        )
    nonzero_reasons = {reason: count for reason, count in drop_reasons.items() if count}
    source_filter_stats = {
        "input_records": len(snippets),
        "usable_records": len(sources),
        "dropped_records": len(snippets) - len(sources),
        "drop_reasons": nonzero_reasons,
    }
    if not sources:
        raise NoUsableSearchQASourcesError(
            "SearchQA row has no usable source with non-empty snippet, title, and URL",
            source_filter_stats=source_filter_stats,
        )
    return sources, source_filter_stats


def materialize_row(row: dict, *, split: str, index: int) -> dict:
    question = row.get("question")
    answer = row.get("answer", row.get("gold_answer"))
    if answer is None and isinstance(row.get("answers"), list) and row["answers"]:
        answer = row["answers"][0]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("SearchQA row requires non-empty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("SearchQA row requires non-empty answer")
    sources, source_filter_stats = _sources(row)
    return {
        "id": f"{split}-{index}",
        "question": question.strip(),
        "gold_answer": answer.strip(),
        "sources": sources,
        "snippets": [source["snippet"] for source in sources],
        "source_filter_stats": source_filter_stats,
    }


def pack_evidence(snippets: Iterable[str], *, max_tokens: int, token_count: Callable[[str], int]) -> str:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    selected: list[str] = []
    used = 0
    for snippet in snippets:
        cost = token_count(snippet)
        if cost < 0:
            raise ValueError("token_count returned a negative value")
        if used + cost > max_tokens:
            break
        selected.append(snippet)
        used += cost
    if not selected:
        raise ValueError("no evidence snippet fits the configured token budget")
    return "\n".join(selected)
