# Native Iterative Guidance DPO

Status: active experiment specification

This document adapts the V1 textual-feedback DPO research method to Qwen's native
thinking behavior. The research object is the preference-data construction method,
not a particular XML response format.

## Research invariants

1. The student is a 2B model and the teacher has privileged access to the gold answer.
2. The teacher's guidance must not reveal the answer.
3. A student trajectory is generated from the base problem, evaluated, and then may
   receive teacher guidance for a retry.
4. Iteration stops at the first correct student response.
5. Every earlier wrong response is a rejected completion against that first correct
   response as the chosen completion.
6. The DPO prompt contains the original student problem and instructions only. It
   does not contain the gold answer, teacher guidance, or evaluator-only data.
7. Unresolved examples produce explicit failure artifacts and no fabricated pairs.
8. Math and SearchQA use the same collection loop with domain-specific evaluation.

## Qwen adaptation

The student is not required to emit XML tags, hidden reasoning markers, branch names,
or a prescribed reflection format. The prompt asks Qwen to reason naturally and give
a concise final answer. Raw model output is retained exactly.

Evaluation is a separate role. It returns a small structured judgment containing
correctness, extracted answer, confidence, and reason. This structure belongs to the
evaluator contract, not to the student's answer-generation contract.

## Guidance safety policy

The teacher receives the problem, gold answer, failed response, and evaluator result.
It must produce a short next-step hint. The hint cannot state the answer, an equivalent
expression, a decisive entity, or an answer-bearing phrase.

An evaluator-model guidance guard checks the hint. A bounded regeneration is attempted
when the guard rejects it. If all attempts are unsafe, the group is marked
`unsafe_guidance` and no pair is produced.

## Collection artifacts

- `examples.jsonl`: input problems and teacher-only gold answers
- `attempts.jsonl`: every student attempt and evaluator result
- `guidance.jsonl`: every teacher hint, including rejected hints
- `pairs.jsonl`: all wrong-versus-first-correct preference pairs
- `response_sft.jsonl`: first-correct responses for optional temporary SFT anchoring
- `failures.jsonl`: unresolved or unsafe groups
- `events.jsonl`: structured lifecycle and metric events
- `metrics.json`: aggregate experiment metrics
- `report.html`: human-readable summary

## Planned comparisons

1. Prompt-only student baseline.
2. Standard DPO with one initial wrong versus first-correct pair.
3. Native multilevel DPO with all wrong attempts versus first correct.
4. Standard GRPO using the shared evaluator.
5. On-policy distillation using the privileged teacher as target policy.

The first training smoke will use the same prompt-only artifact split. Training is
not started until the collection artifacts have been manually inspected.

## Loss policy

The default objective is DPO only. No formatting loss is used. If prompt-only
collection shows that natural responses cannot be evaluated reliably, a temporary
response-SFT anchor may be enabled on valid natural responses and annealed to zero.
This anchor must be logged separately from DPO loss.

## Decision log

### 2026-07-10

- Replaced mandatory XML trajectory formatting with native Qwen responses.
- Kept the PDF's preference-data and privileged-teacher invariants.
- Made answer-free teacher guidance a hard collection invariant.
- Added a separate evaluator and guidance-guard role; roles may share the 9B weights.
- Set the initial prompt-only smoke to two examples and at most three guidance steps.
