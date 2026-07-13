from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence

from text_feedback_dpo.responses import validate_retrieved_sources

EMPTY_RESPONSE_SENTINEL = "__EMPTY_RESPONSE__"


def _implementation_hash(*objects: object, constants: Sequence[object] = ()) -> str:
    source = "\n".join(inspect.getsource(item) for item in objects)
    payload = json.dumps(
        {"source": source, "constants": list(constants)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_short_answer_prompt(example: dict, hints: list[str]) -> str:
    """Build the archival short-answer prompt used by the legacy pipeline."""
    question = str(example.get("question", "")).strip()
    evidence = str(example.get("packed_evidence", "")).strip()
    if not question:
        raise ValueError("student prompt requires a non-empty question")
    if not evidence:
        raise ValueError("student prompt requires non-empty evidence")
    sections = [
        "Use the evidence to answer the question.",
        "Think through the evidence carefully, then return only the short answer in plain text only with no explanation.",
        "The entire answer must be at most 8 words.",
        "Return a noun phrase, not a sentence. Never restate the question or clue; if uncertain, give your best short guess.",
        "Do not use XML, JSON, tags, code fences, labels, or evidence quotations.",
        "",
        "Evidence:",
        evidence,
        "",
        f"Question: {question}",
    ]
    if hints:
        sections.extend(["", "Hints:", *(f"- {str(hint).strip()}" for hint in hints)])
    sections.extend(["", "Answer:"])
    return "\n".join(sections)


def build_student_prompt(example: dict, hints: list[str]) -> str:
    """Compatibility wrapper for archival callers; use explicit builders for new work."""
    return build_short_answer_prompt(example, hints)


def _question(example: Mapping[str, object]) -> str:
    question = example.get("question", "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("prompt requires a non-empty question")
    return question.strip()


def build_search_query_prompt(example: dict, hints: list[str]) -> str:
    """Build a question-only prompt for a single plain-text search query."""
    sections = [
        "Generate one concise search query that will help retrieve evidence for the question.",
        "Return exactly one-line plain text, with no answer, explanation, XML, JSON, or code fence.",
        "",
        "Example question: What gas do plants absorb from the air?",
        "Example query: plants absorb carbon dioxide gas",
        "",
        f"Question: {_question(example)}",
    ]
    if hints:
        sections.extend(["", "Hints:", *(f"- {str(hint).strip()}" for hint in hints)])
    sections.extend(["", "Search query:"])
    return "\n".join(sections)


def build_cited_response_prompt(
    example: dict,
    retrieved_sources: Sequence[Mapping[str, object]],
    hints: list[str],
) -> str:
    """Build the strict normal-text response prompt from retrieved source metadata."""
    sources = validate_retrieved_sources(retrieved_sources)
    sections = [
        "Answer the question using the retrieved sources.",
        "Return exactly three nonblank lines in this order and grammar:",
        "Answer: <nonempty concise answer>",
        "Reasoning: <nonempty concise reasoning with bracket citations such as [S001]>",
        "Sources: S001, S003",
        "Answer must contain at most 16 normalized words.",
        "Reasoning must contain at most 96 words.",
        "Use concise reasoning, cite every reasoning claim, and list each citation once in first-appearance order.",
        "Do not use XML, JSON, code fences, or URLs. Never reproduce URLs or source titles in your response.",
        "",
        "Worked format example:",
        "Question: Which planet is known as the Red Planet?",
        "Example source: Mars is known as the Red Planet [S001].",
        "Answer: Mars",
        "Reasoning: The source identifies Mars as the Red Planet [S001].",
        "Sources: S001",
        "Use this only as a format example; cite only IDs from the retrieved sources below.",
        "",
        "Retrieved sources:",
    ]
    for source in sources:
        sections.extend([f"[{source['source_id']}] {source['title']}", f"Snippet: {source['snippet']}"])
    sections.extend(["", f"Question: {_question(example)}"])
    if hints:
        sections.extend(["", "Hints:", *(f"- {str(hint).strip()}" for hint in hints)])
    sections.extend(["", "Response:"])
    return "\n".join(sections)


def build_teacher_prompt(
    example: dict,
    failed_response: str,
    interventions: list[dict],
    *,
    raw_query: str | None = None,
    retrieved_sources: Sequence[Mapping[str, object]] | None = None,
    diagnostics: Mapping[str, object] | None = None,
    repair_region: str | None = None,
    escalation_level: int | None = None,
) -> str:
    """Build the private, answer-free Qwen3 teacher request."""
    sources = example.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("teacher prompt requires complete example source records")
    if retrieved_sources is None or not isinstance(retrieved_sources, Sequence) or isinstance(retrieved_sources, (str, bytes)):
        raise ValueError("teacher prompt requires retrieved source records")
    if diagnostics is None or not isinstance(diagnostics, Mapping):
        raise ValueError("teacher prompt requires deterministic diagnostics")
    if not isinstance(failed_response, str):
        raise TypeError("failed response must be text")
    level = len(interventions) + 1 if escalation_level is None else escalation_level
    if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
        raise ValueError("escalation level must be a positive integer")
    prior = [{"level": item.get("level"), "hint": item.get("hint")} for item in interventions]
    compact_retrieved_records = [
        {
            "source_id": source["source_id"],
            "title": source["title"],
            "snippet": source["snippet"],
        }
        for source in validate_retrieved_sources(retrieved_sources)
    ]
    request = {
        "available_source_count": len(sources),
        "retrieved_records": compact_retrieved_records,
        "question": example.get("question"),
        "query": raw_query,
        "private_gold_answer": example.get("gold_answer"),
        "failed_response": failed_response.strip() or EMPTY_RESPONSE_SENTINEL,
        "deterministic_diagnostics": dict(diagnostics),
        "prior_hints": prior,
        "escalation_level": level,
        "repair_region": repair_region or diagnostics.get("responsible_region"),
    }
    return """You are a privileged Qwen3 instruct tutor. The fields below are private.
Localize the earliest responsible region and provide one minimal, answer-free directional hint.
At level 1 be slight and directional. Only after failed retries may specificity and repair scope increase.
Never reveal, quote, spell, restate, encode, or imply the private gold answer.
Never provide a critique, answer, solution, source text, citation choice, or extra field.
Return exactly one strict JSON object with exactly this shape: {\"hint\":\"...\"}.
The hint must be non-empty and at most 24 words.

Private request:
""" + json.dumps(request, ensure_ascii=False, sort_keys=True, indent=2)


def prompt_builder_identity() -> dict[str, dict[str, str]]:
    """Return implementation-bound identities for all active prompt builder mappings."""
    return {
        "query": {
            "builder": "text_feedback_dpo.prompts.build_search_query_prompt",
            "implementation_sha256": _implementation_hash(build_search_query_prompt, _question),
        },
        "response": {
            "builder": "text_feedback_dpo.prompts.build_cited_response_prompt",
            "implementation_sha256": _implementation_hash(
                build_cited_response_prompt, _question, validate_retrieved_sources
            ),
        },
        "teacher": {
            "builder": "text_feedback_dpo.prompts.build_teacher_prompt",
            "implementation_sha256": _implementation_hash(
                build_teacher_prompt, constants=(EMPTY_RESPONSE_SENTINEL,)
            ),
        },
    }
