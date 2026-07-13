from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from text_feedback_dpo.scoring import normalize_answer


_SOURCE_ID_PATTERN = re.compile(r"S\d{3}\Z")
_CITATION_PATTERN = re.compile(r"\[(S\d{3})\]")
_URL_PATTERN = re.compile(
    r"(?ix)(?:"
    r"\b[a-z][a-z0-9+.-]*://\S*|"
    r"\bmailto:\S+|"
    r"\bwww\.\S+|"
    r"(?<![\w@])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?::\d+)?(?:[/?#]\S*)?"
    r")"
)
_XML_PATTERN = re.compile(r"<[^>]*>")


class CitedResponseFormatError(ValueError):
    """A student response violated the cited-response grammar."""

    def __init__(self, error_code: str, message: str | None = None) -> None:
        self.error_code = error_code
        super().__init__(message or error_code)

    @property
    def code(self) -> str:
        return self.error_code


@dataclass(frozen=True)
class CitedResponse:
    answer: str
    reasoning: str
    source_ids: tuple[str, ...]

    @property
    def citations(self) -> tuple[str, ...]:
        return self.source_ids


def validate_retrieved_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the source identity and metadata required by parsing and rendering."""
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise TypeError("retrieved_sources must be a sequence of source mappings")
    if not sources:
        raise ValueError("retrieved_sources must not be empty")

    normalized: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            raise TypeError(f"retrieved source {index} must be a mapping")
        missing = [field for field in ("source_id", "title", "url", "snippet") if field not in source]
        if missing:
            raise ValueError(f"retrieved source {index} is missing required field: {missing[0]}")
        source_id = source["source_id"]
        if not isinstance(source_id, str) or not _SOURCE_ID_PATTERN.fullmatch(source_id.strip()):
            raise ValueError(f"retrieved source {index} has non-canonical source_id")
        source_id = source_id.strip()
        if source_id in source_ids:
            raise ValueError(f"retrieved source_id must be unique: {source_id}")
        copy = dict(source)
        for field in ("title", "url", "snippet"):
            value = copy[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"retrieved source {source_id} requires non-empty {field}")
            copy[field] = value.strip()
        copy["source_id"] = source_id
        normalized.append(copy)
        source_ids.add(source_id)
    return normalized


def _format_error(error_code: str, message: str) -> CitedResponseFormatError:
    return CitedResponseFormatError(error_code, message)


def parse_cited_response(text: str, retrieved_sources: Sequence[Mapping[str, Any]]) -> CitedResponse:
    """Parse exactly three nonblank plain-text lines with canonical citations."""
    source_records = validate_retrieved_sources(retrieved_sources)
    available_ids = {source["source_id"] for source in source_records}
    if not isinstance(text, str):
        raise _format_error("not_text", "student response must be text")
    if _URL_PATTERN.search(text):
        raise _format_error("url_forbidden", "student response must not contain URLs")
    if "```" in text or "{" in text or "}" in text or _XML_PATTERN.search(text):
        raise _format_error("markup_forbidden", "student response must not contain XML, JSON, or code fences")

    lines = text.splitlines()
    if len(lines) != 3 or any(not line.strip() for line in lines):
        raise _format_error("line_count", "student response must contain exactly three nonblank lines")
    if not lines[0].startswith("Answer: ") or not lines[1].startswith("Reasoning: ") or not lines[2].startswith("Sources: "):
        raise _format_error("label_order", "student response labels must be Answer, Reasoning, Sources in order")

    answer = lines[0][len("Answer: ") :].strip()
    reasoning = lines[1][len("Reasoning: ") :].strip()
    source_text = lines[2][len("Sources: ") :]
    if not answer:
        raise _format_error("empty_answer", "Answer must be nonempty")
    if not reasoning:
        raise _format_error("empty_reasoning", "Reasoning must be nonempty")
    reasoning_without_citations = _CITATION_PATTERN.sub("", reasoning)
    if any(bracket in answer + reasoning_without_citations + source_text for bracket in ("[", "]")):
        raise _format_error("markup_forbidden", "student response must not contain unsupported bracket markup")

    answer_words = normalize_answer(answer).split()
    if not answer_words:
        raise _format_error("empty_answer", "Answer must contain normalized words")
    if len(answer_words) > 16:
        raise _format_error("answer_too_long", "Answer must contain at most 16 normalized words")
    reasoning_words = reasoning.split()
    if len(reasoning_words) > 96:
        raise _format_error("reasoning_too_long", "Reasoning must contain at most 96 words")

    source_match = re.fullmatch(r"(S\d{3}(?:, S\d{3})*)", source_text)
    if source_match is None:
        raise _format_error("invalid_sources", "Sources must contain canonical IDs separated by comma-space")
    source_ids = tuple(source_match.group(1).split(", "))
    if len(set(source_ids)) != len(source_ids):
        raise _format_error("duplicate_citation", "Sources must not contain duplicate IDs")

    citations = tuple(_CITATION_PATTERN.findall(reasoning))
    if not citations:
        raise _format_error("missing_citation", "Reasoning must contain at least one bracket citation")
    if len(set(citations)) != len(citations):
        raise _format_error("duplicate_citation", "Reasoning must not repeat citation IDs")
    unknown = [source_id for source_id in (*source_ids, *citations) if source_id not in available_ids]
    if unknown:
        raise _format_error("unknown_citation", f"citation ID is not in retrieved sources: {unknown[0]}")
    if citations != source_ids:
        raise _format_error("citation_mismatch", "Sources must equal reasoning citation order of first appearance")
    return CitedResponse(answer=answer, reasoning=reasoning, source_ids=source_ids)


def render_cited_response(
    response: CitedResponse | str,
    retrieved_sources: Sequence[Mapping[str, Any]],
) -> str:
    """Render model citations using canonical retrieved title and URL metadata."""
    source_records = validate_retrieved_sources(retrieved_sources)
    if isinstance(response, str):
        parsed = parse_cited_response(response, source_records)
    elif isinstance(response, CitedResponse):
        if (
            not isinstance(response.answer, str)
            or not isinstance(response.reasoning, str)
            or not isinstance(response.source_ids, tuple)
            or not all(isinstance(source_id, str) for source_id in response.source_ids)
        ):
            raise CitedResponseFormatError("invalid_response_object", "CitedResponse fields have invalid types")
        serialized = "\n".join(
            (
                f"Answer: {response.answer}",
                f"Reasoning: {response.reasoning}",
                f"Sources: {', '.join(response.source_ids)}",
            )
        )
        parsed = parse_cited_response(serialized, source_records)
    else:
        raise TypeError("response must be a CitedResponse or strict response text")
    by_id = {source["source_id"]: source for source in source_records}
    rendered_sources = []
    for source_id in parsed.source_ids:
        if source_id not in by_id:
            raise ValueError(f"canonical metadata unavailable for cited source: {source_id}")
        source = by_id[source_id]
        rendered_sources.append(f"[{source_id}] {source['title']} — {source['url']}")
    return "\n".join(
        (
            f"Answer: {parsed.answer}",
            f"Reasoning: {parsed.reasoning}",
            "Sources:",
            *rendered_sources,
        )
    )
