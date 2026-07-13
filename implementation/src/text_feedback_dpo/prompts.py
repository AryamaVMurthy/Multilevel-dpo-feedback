from __future__ import annotations

from xml.sax.saxutils import escape

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
        "Think through the evidence carefully, then return only the short answer with no explanation.",
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
    prior = "".join(f"<prior><hint>{escape(item['hint'])}</hint><scope>{escape(item['scope'])}</scope></prior>" for item in interventions)
    failed_response_xml = escape(failed_response) if failed_response else EMPTY_RESPONSE_SENTINEL
    return f"""<teacher_task>
  <instructions>Use the gold answer only to localize the earliest responsible error. Return one minimal answer-free intervention. Do not provide a corrected answer or complete solution. If the failed response is empty, use {EMPTY_RESPONSE_SENTINEL} as error_span.</instructions>
  <format><feedback><error_span>exact text from the failed response, or {EMPTY_RESPONSE_SENTINEL} when empty</error_span><hint>short correction hint</hint><scope>entity|relation|evidence|verification</scope></feedback></format>
  <question>{escape(example['question'])}</question>
  <evidence>{escape(example['packed_evidence'])}</evidence>
  <gold_answer>{escape(example['gold_answer'])}</gold_answer>
  <failed_response>{failed_response_xml}</failed_response>
  <previous_interventions>{prior}</previous_interventions>
</teacher_task>"""
