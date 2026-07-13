from __future__ import annotations

from collections.abc import Mapping, Sequence

from text_feedback_dpo.responses import validate_retrieved_sources

EMPTY_RESPONSE_SENTINEL = "__EMPTY_RESPONSE__"


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
        "Retrieved sources:",
    ]
    for source in sources:
        sections.extend([f"[{source['source_id']}] {source['title']}", f"Snippet: {source['snippet']}"])
    sections.extend(["", f"Question: {_question(example)}"])
    if hints:
        sections.extend(["", "Hints:", *(f"- {str(hint).strip()}" for hint in hints)])
    sections.extend(["", "Response:"])
    return "\n".join(sections)


def build_teacher_prompt(example: dict, failed_response: str, interventions: list[dict]) -> str:
    level = len(interventions) + 1
    failed = failed_response.strip() or EMPTY_RESPONSE_SENTINEL
    prior = "\n".join(f"- Level {item['level']}: {item['hint']}" for item in interventions) or "None"
    return f"""You are a privileged tutor. Localize the earliest reason the student's answer is wrong and give only the smallest useful correction hint.

Do not reveal, quote, spell, or restate the gold answer. Do not provide a complete solution or critique. Use at most 24 words. At level 1, give a very slight directional hint. At later levels, increase specificity only enough to repair the remaining error.

Question: {example['question']}
Evidence: {example['packed_evidence']}
Gold answer (private): {example['gold_answer']}
Failed student answer: {failed}
Previous hints:
{prior}
Escalation level: {level}

Return exactly one JSON object: {{"hint":"..."}}"""
