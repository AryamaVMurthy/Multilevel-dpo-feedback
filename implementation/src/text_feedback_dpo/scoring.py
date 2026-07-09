from __future__ import annotations

import re

from text_feedback_dpo.parsing import Trajectory, TrajectoryParseError, parse_trajectory


def normalize_answer(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def exact_score(final_answer: str, gold_answer: str) -> float:
    return 1.0 if normalize_answer(final_answer) == normalize_answer(gold_answer) else 0.0


def evaluate_rollout(rollout: str, gold_answer: str) -> dict:
    try:
        trajectory: Trajectory = parse_trajectory(rollout)
    except TrajectoryParseError as exc:
        return {
            "format_valid": False,
            "verification_present": False,
            "score": 0.0,
            "error_code": exc.error_code,
        }

    return {
        "format_valid": True,
        "verification_present": trajectory.verification_present,
        "score": exact_score(trajectory.final, gold_answer),
        "final_answer": trajectory.final,
        "error_code": None,
    }

