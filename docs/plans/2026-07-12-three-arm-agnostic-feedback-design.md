# Three-Arm Agnostic Feedback Pilot Design

## Objective

Compare three privileged-teacher feedback policies on the same frozen MATH
student failures:

1. error-only: identify the general error in the attempted approach;
2. hint-only: provide one slight direction for another attempt;
3. error-and-hint: identify the general error and provide one slight direction.

The feedback is training data for DPO and must be usable independently of a
chat transcript. It must not reveal the gold answer or make the student prompt
depend on a previous conversational turn.

## Teacher output contract

The teacher returns exactly one tagged block:

```text
<student_feedback>
Standalone, answer-free guidance.
</student_feedback>
```

The text inside the block may be multiline and may contain normal mathematical
notation. The parser rejects missing tags, duplicate blocks, empty feedback,
or non-whitespace outside the block. It does not repair malformed output or
substitute a default.

## Feedback policies

All three prompts prohibit the gold answer, equivalent expressions, decisive
intermediate values, complete solution steps, and language tied to a chat turn.
They also prohibit phrases such as "your previous response", "last time", and
"try again". Feedback must instead state a reusable mathematical check or error
pattern in impersonal language.

- **Error-only:** explain which approach or inference is invalid, without
  supplying the correction or a next-step hint.
- **Hint-only:** give one slight next check, without diagnosing the attempted
  approach or supplying solution steps.
- **Error-and-hint:** briefly diagnose the invalid approach and give one slight
  next check.

## Data flow and isolation

The original problem remains the DPO prompt. Privileged inputs, including the
gold answer, failed rollout, teacher prompt, and reviewer decisions, are stored
only in trajectory metadata. The student retry prompt receives only the
original problem and approved `<student_feedback>` text, presented as general
advice rather than chat history.

Every feedback candidate is checked by independent leakage and mathematical
correctness reviewers. Reviewer rejection triggers an explicit regeneration
request containing the review result. Exhausted attempts remain unresolved;
they never produce fabricated feedback or preference pairs.

## Paired pilot

Run all three policies over the same frozen 16-example MATH preflight set, the
same initial student rollouts, and the same frozen model and decoding settings.
Record raw generations, prompts, reviewer decisions, token counts, finish
reasons, latency, and artifact hashes.

Compare malformed-output rate, leakage-review acceptance, mathematical-review
acceptance, first-retry correctness, generated tokens, and unresolved rate.
No arm is promoted unless it has zero malformed outputs, no accepted leakage,
at least 95 percent manual reviewer agreement, and complete provenance.

## Failure behavior

Any malformed teacher output, reviewer disagreement, prompt leakage, missing
metadata, mixed protocol, or artifact mismatch stops promotion. The system must
surface the raw output and exact failure; it must not silently sanitize, infer,
or fall back to another feedback mode.
