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


def build_student_prompt(problem: str, domain: str) -> str:
    return f"""You are solving a problem using a structured reasoning policy.

You must use this format:

<plan>
High-level meta-plan. Decide the route, number of branches, tools needed, and verification needed before final answer.
</plan>

<think branch="A">
Local reasoning for branch A.
</think>

<tool branch="A">
Optional tool call and observation.
</tool>

<reflect>
Branch comparison:
Evidence / derivation check:
Verification:
Decision:
</reflect>

<final>
Final answer only.
</final>

Rules:
1. Do not give <final> before <reflect>.
2. <reflect> must contain verification.
3. Use tools only when useful.
4. The final answer must be concise.
5. {_domain_verification(domain)}

Problem:
{problem}
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
The student must use:
<plan>, <think>, <tool>, <reflect>, <final>.

Your job:
1. Give textual feedback.
2. Explain what computation should change: planning, thinking, tool use, branching, reflection, verification.
3. Explain what content should change: wrong route, missing evidence, bad arithmetic, wrong final type.
4. Produce a corrected rollout in the same format.
5. Do not add unnecessary branches.
6. Ensure <reflect> contains real verification.
7. Ensure <final> contains only the final answer.

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

Return exactly:

<feedback>
...
</feedback>

<corrected_rollout>
...
</corrected_rollout>
"""
