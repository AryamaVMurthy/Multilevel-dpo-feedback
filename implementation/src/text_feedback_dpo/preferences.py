from __future__ import annotations

from text_feedback_dpo.prompts import EMPTY_RESPONSE_SENTINEL


def build_preference_rows(trajectory: dict) -> list[dict]:
    if not trajectory.get("resolved"):
        return []
    chosen = trajectory.get("chosen")
    prompt = trajectory.get("prompt")
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("resolved trajectory is missing chosen XML response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("resolved trajectory is missing hint-free prompt")
    rows = []
    for attempt in trajectory.get("attempts", []):
        if attempt.get("correct"):
            break
        rejected = attempt.get("response")
        if rejected == "":
            rejected = EMPTY_RESPONSE_SENTINEL
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError("trajectory contains empty rejected response")
        rows.append({
            "id": f"{trajectory['id']}::attempt-{attempt['attempt_index']}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "example_id": trajectory["id"],
                "rejected_attempt_index": attempt["attempt_index"],
                "intervention_count": len(trajectory.get("interventions", [])),
            },
        })
    return rows
