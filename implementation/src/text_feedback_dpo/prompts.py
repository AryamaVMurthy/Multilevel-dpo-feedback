from __future__ import annotations

EMPTY_RESPONSE_SENTINEL = "__EMPTY_RESPONSE__"


def build_student_prompt(example: dict, hints: list[str]) -> str:
    question = str(example.get("question", "")).strip()
    evidence = str(example.get("packed_evidence", "")).strip()
    if not question:
        raise ValueError("student prompt requires a non-empty question")
    if not evidence:
        raise ValueError("student prompt requires non-empty evidence")
    sections = [
        "Use the evidence to answer the question.",
        "Think through the evidence carefully, then return only the short answer in plain text only with no explanation.",
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
