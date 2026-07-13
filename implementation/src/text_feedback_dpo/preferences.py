from __future__ import annotations

def build_preference_rows(trajectory: dict) -> list[dict]:
    if not trajectory.get("resolved"):
        return []
    chosen = trajectory.get("chosen")
    prompt = trajectory.get("prompt")
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("resolved trajectory is missing chosen student answer")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("resolved trajectory is missing hint-free prompt")
    rows = []
    for attempt in trajectory.get("attempts", []):
        if attempt.get("correct"):
            break
        rejected = attempt.get("response")
        if isinstance(rejected, str) and not rejected.strip():
            continue
        if not isinstance(rejected, str):
            raise ValueError("trajectory contains a non-string rejected response")
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
