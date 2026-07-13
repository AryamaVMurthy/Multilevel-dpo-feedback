from __future__ import annotations

from collections.abc import Callable

from text_feedback_dpo.formatting import XMLFormatError, parse_feedback
from text_feedback_dpo.preferences import build_preference_rows
from text_feedback_dpo.prompts import EMPTY_RESPONSE_SENTINEL, build_student_prompt, build_teacher_prompt
from text_feedback_dpo.scoring import score_searchqa


def collect_dataset_batchwise(*, examples: list[dict], student_generate_batch: Callable[..., list[str]], teacher_generate_batch: Callable[..., list[str]], max_interventions: int) -> list[dict]:
    states = {
        str(example["id"]): {"id": example["id"], "example": example, "hints": [], "attempts": [], "interventions": [], "resolved": False}
        for example in examples
    }
    active = [str(example["id"]) for example in examples]
    for attempt_index in range(max_interventions + 1):
        prompts = [build_student_prompt(states[example_id]["example"], states[example_id]["hints"]) for example_id in active]
        responses = student_generate_batch(prompts, max_new_tokens=32, temperature=0.7, top_p=0.9)
        if len(responses) != len(active):
            raise ValueError(f"student batch cardinality mismatch at attempt {attempt_index}")
        failed_ids = []
        teacher_prompts = []
        for example_id, response in zip(active, responses, strict=True):
            state = states[example_id]
            score = score_searchqa(response, state["example"]["gold_answer"], state["example"]["packed_evidence"])
            state["attempts"].append({"attempt_index": attempt_index, "response": response, "correct": score["correct"], "score": score})
            if score["correct"]:
                state["resolved"] = True
                state["chosen"] = response
            elif attempt_index < max_interventions:
                failed_ids.append(example_id)
                teacher_prompts.append(build_teacher_prompt(state["example"], response, state["interventions"]))
        if not failed_ids:
            break
        feedback_rows = teacher_generate_batch(teacher_prompts, max_new_tokens=96, temperature=0.0, top_p=1.0)
        if len(feedback_rows) != len(failed_ids):
            raise ValueError(f"teacher batch cardinality mismatch at attempt {attempt_index}")
        for example_id, raw_feedback in zip(failed_ids, feedback_rows, strict=True):
            state = states[example_id]
            failed_response = state["attempts"][-1]["response"]
            try:
                feedback = parse_feedback(raw_feedback, gold_answer=state["example"]["gold_answer"])
            except XMLFormatError as exc:
                raise ValueError(f"invalid teacher feedback for {example_id}: {exc}") from exc
            if failed_response:
                error_span_valid = feedback.error_span in failed_response
            else:
                error_span_valid = feedback.error_span == EMPTY_RESPONSE_SENTINEL
            if not error_span_valid:
                raise ValueError(f"teacher error_span absent from failed response for {example_id}")
            state["interventions"].append({"attempt_index": attempt_index, "error_span": feedback.error_span, "hint": feedback.hint, "scope": feedback.scope})
            state["hints"].append(feedback.hint)
        active = [example_id for example_id in failed_ids]
    rows = []
    for example in examples:
        state = states[str(example["id"])]
        prompt = build_student_prompt(example, [])
        trajectory = {key: state.get(key) for key in ("id", "attempts", "interventions", "chosen", "resolved")}
        trajectory["prompt"] = prompt
        trajectory["preference_rows"] = build_preference_rows(trajectory)
        rows.append(trajectory)
    return rows
