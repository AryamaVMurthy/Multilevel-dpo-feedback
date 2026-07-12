from __future__ import annotations

import json


def _domain_verification(domain: str) -> str:
    if domain == "math":
        return (
            "For Math, verify using arithmetic checks, substitution checks, "
            "constraint checks, sign/range/unit checks, and branch agreement if multiple branches are used. "
            "Use at most 2 branches."
        )
    if domain == "search_qa":
        return (
            "For Search-QA, verify the entity, relation, evidence support, and answer type. "
            "Use controlled search observations only. Use at most 3 branches."
        )
    raise ValueError("domain must be math or search_qa")


def trajectory_format_instructions(domain: str) -> str:
    """Return the single format contract shared by every model role."""
    return f"""Trajectory format contract:
1. Emit one <plan> block, then one or more reasoning blocks using the exact tag <think branch="A">.
2. Branch names must be unique uppercase letters starting at A; use at most the domain limit.
3. A <tool branch="A"> block is optional and may contain only a supplied controlled observation.
4. Emit exactly one <reflect> block after all reasoning and tool blocks.
5. Emit exactly one <final> block after <reflect>.
6. Inside <reflect>, include these literal non-empty headings: Branch comparison:, Evidence / derivation check:, Verification:, Decision:.
7. Put only the answer in <final>; do not put reasoning, headings, or XML tags there.
8. Use <think>, never <thinking>. Do not use <thinking>; close every block with its matching closing tag.
9. Do not emit Markdown fences, placeholder text, TODO markers, or an ellipsis.
10. {_domain_verification(domain)}"""


def build_student_prompt(problem: str, domain: str) -> str:
    return f"""You are solving a problem using a structured reasoning policy.

{trajectory_format_instructions(domain)}

Problem:
{problem}
"""


def build_native_student_prompt(
    *,
    problem: str,
    domain: str,
    guidance: str | None = None,
    evidence: list[str] | None = None,
) -> str:
    if domain not in {"math", "search_qa"}:
        raise ValueError("domain must be math or search_qa")
    evidence_text = ""
    if evidence:
        evidence_text = (
            "\nControlled evidence available for this question:\n"
            + "\n".join(f"- {item}" for item in evidence)
            + "\nUse only this evidence; do not invent searches.\n"
        )
    guidance_text = ""
    if guidance:
        guidance_text = (
            "\nGeneral problem-solving advice:\n"
            f"{guidance}\n"
        )
    if domain == "math":
        task_instruction = (
            "Give a concise derivation using at most 6 numbered steps. End with exactly one line "
            "of the form FINAL: \\boxed{answer}. Do not output anything after the closing brace."
        )
    else:
        task_instruction = (
            "Use only as much explanation as the question needs, give one concise final answer, "
            "and stop immediately after that answer."
        )
    return f"""Solve the following {domain.replace("_", " ")} problem.

{task_instruction}
{evidence_text}{guidance_text}
Problem:
{problem}
"""


def build_privileged_guidance_prompt(
    *,
    problem: str,
    gold_answer: str,
    reference_solution: str,
    rollout: str,
    result: dict,
    domain: str,
    feedback_policy: str,
    prior_reviews: list[dict] | None = None,
) -> str:
    if not isinstance(reference_solution, str) or not reference_solution.strip():
        raise ValueError("reference_solution must be a non-empty teacher-only string")
    policy_instructions = {
        "error_only": (
            "Identify the general error in the approach or inference. Explain why that approach is invalid, "
            "but do not give a next-step hint or supply the correction."
        ),
        "hint_only": (
            "Give one slight directional hint about a check or relation that would help. Do not diagnose the "
            "specific attempted approach and do not supply solution steps."
        ),
        "error_and_hint": (
            "Briefly identify the general error in the approach or inference, then give one slight directional "
            "hint about the next check."
        ),
    }
    if feedback_policy not in policy_instructions:
        raise ValueError(
            "feedback_policy must be error_only, hint_only, or error_and_hint"
        )
    review_text = ""
    if prior_reviews:
        latest = prior_reviews[-1]
        surface = latest.get("surface") or {}
        critic = latest.get("critic") or {}
        guard = latest.get("guard") or {}
        review_text = f"""
Previous rejected hint:
{latest.get("guidance", "")}

Review of that hint:
- Surface issues: {json.dumps(surface.get("reasons", []), sort_keys=True)}
- Directionally correct and relevant: {critic.get("valid")}
- Answer-leakage safe: {guard.get("safe")}

Write different standalone feedback that addresses this review.
"""
    return f"""You are a privileged teacher helping a smaller model improve its next attempt.

You may inspect the gold answer only to understand the mathematical issue. The student-facing text
must not reveal the answer, an equivalent expression, a decisive intermediate value, or a complete
solution. It must be standalone general advice: do not refer to a previous turn, chat thread, earlier
response, retry, teacher, or conversation. Use impersonal wording that remains meaningful when paired
only with the original problem.

Feedback policy: {feedback_policy}
{policy_instructions[feedback_policy]}

Return concise feedback of at most forty words. Do not copy phrases or numerical results from the gold
solution. The answer-leakage prohibition applies even when revealing the answer would make the feedback
clearer or more helpful.

Domain: {domain}
Problem:
{problem}

Gold answer (teacher-only):
{gold_answer}

Reference solution (teacher-only):
{reference_solution}

Earlier student response:
{rollout}

Earlier evaluator result:
{json.dumps(result, sort_keys=True)}
{review_text}

Return exactly one block and no surrounding text:
<student_feedback>
standalone answer-free feedback
</student_feedback>
"""


def build_teacher_prompt(
    *,
    problem: str,
    gold_answer: str,
    student_rollout: str,
    result: dict,
    domain: str,
    teacher_mode: str,
) -> str:
    if teacher_mode not in {"stronger_model", "same_model_privileged"}:
        raise ValueError("teacher_mode must be stronger_model or same_model_privileged")

    privileged = ""
    if teacher_mode == "same_model_privileged":
        privileged = (
            "\nYou are given privileged training-only information. "
            "This information is never available during student evaluation. "
            "Use it only to write feedback and a corrected rollout.\n"
        )

    return f"""You are a teacher correcting a small language model's structured rollout.
{privileged}
Your job:
1. Give textual feedback.
2. Explain what computation should change: planning, thinking, tool use, branching, reflection, verification.
3. Explain what content should change: wrong route, missing evidence, bad arithmetic, wrong final type.
4. Produce a corrected rollout in the same format.
5. Do not add unnecessary branches.
6. Ensure <reflect> contains real verification.
7. Ensure <final> contains only the final answer.

{trajectory_format_instructions(domain)}

Domain:
{domain}

Problem:
{problem}

Gold answer:
{gold_answer}

Student rollout:
{student_rollout}

Student result:
{json.dumps(result, sort_keys=True)}

Return exactly two top-level blocks in this order: <feedback> followed by
<corrected_rollout>. Put specific repair guidance in <feedback> without repeating the gold answer.
Put one complete trajectory satisfying the shared contract inside <corrected_rollout>. Write content
specific to the supplied problem and rollout. Never output instructional sentences, TODO markers, or
placeholder text inside either block.
"""
