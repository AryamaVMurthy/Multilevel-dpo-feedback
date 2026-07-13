from __future__ import annotations

from collections.abc import Callable

from text_feedback_dpo.formatting import XMLFormatError, parse_feedback
from text_feedback_dpo.prompts import build_student_prompt, build_teacher_prompt
from text_feedback_dpo.scoring import score_searchqa


class TrajectoryError(ValueError):
    """Raised when a trajectory cannot satisfy the explicit intervention contract."""


def collect_trajectory(*, example: dict, student_generate: Callable[[str, int], str], teacher_generate: Callable[[str], str], max_interventions: int) -> dict:
    prompt = build_student_prompt(example, [])
    hints: list[str] = []
    attempts: list[dict] = []
    interventions: list[dict] = []
    for attempt_index in range(max_interventions + 1):
        response = student_generate(build_student_prompt(example, hints), attempt_index)
        score = score_searchqa(response, example["gold_answer"], example["packed_evidence"])
        attempts.append({"attempt_index": attempt_index, "response": response, "correct": score["correct"], "score": score})
        if score["correct"]:
            return {"id": example["id"], "prompt": prompt, "attempts": attempts, "interventions": interventions, "chosen": response, "resolved": True}
        if attempt_index == max_interventions:
            break
        try:
            feedback = parse_feedback(teacher_generate(build_teacher_prompt(example, response, interventions)), gold_answer=example["gold_answer"])
        except XMLFormatError as exc:
            raise TrajectoryError(f"invalid teacher feedback for {example['id']} at attempt {attempt_index}: {exc}") from exc
        if feedback.error_span not in response:
            raise TrajectoryError(f"teacher error_span is absent from failed response for {example['id']}")
        interventions.append({"attempt_index": attempt_index, "error_span": feedback.error_span, "hint": feedback.hint, "scope": feedback.scope})
        hints.append(feedback.hint)
    return {"id": example["id"], "prompt": prompt, "attempts": attempts, "interventions": interventions, "chosen": None, "resolved": False}
