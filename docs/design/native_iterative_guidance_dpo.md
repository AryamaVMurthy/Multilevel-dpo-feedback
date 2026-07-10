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

Role-level generation controls are explicit: the student keeps the configured native
generation behavior, while teacher guidance and evaluator judgments use Qwen's
supported non-thinking chat-template mode so short guidance and machine-readable
judgments do not spend their entire output budget on internal reasoning. This is a
serialization control for those roles, not a student-format constraint.

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

- Immutable dataset roles are stored as `*.jsonl.zst` with a manifest and content hash.
- Collection shards store one compressed record per immutable example ID, with atomic
  progress and completion markers; attempt records reference the example rather than
  duplicating the base prompt and evidence.
- Merged collection records retain every raw student response, evaluator result, teacher
  output, guidance-policy decision, guard result, token estimate, and latency.
- `standard.jsonl`, `multilevel.jsonl`, and `matched.jsonl` are separate preference
  artifacts built after merge, each with prompt and response hashes.
- Each paper run also writes canonical JSONL events, a required run manifest, metrics,
  GPU telemetry for GPU phases, optional TensorBoard scalars, and an HTML report.

## Planned comparisons

1. Prompt-only student baseline.
2. Standard DPO with one initial wrong versus first-correct pair.
3. Native multilevel DPO with all wrong attempts versus first correct.
4. Pair-budget-matched multilevel DPO.
5. Original GRPO using the shared evaluator and reward semantics.
6. One-seed DAPO-loss sensitivity analysis, labeled separately from GRPO.

On-policy distillation remains a completed smoke baseline but is outside the primary
paper-scale GSM8K and SearchQA-8K comparison. Paper training is not started until the
collection artifacts and the optimizer/LoRA preflight have been manually inspected.

## Paper training policy

The canonical optimizer, architecture-aware LoRA coverage, hyperparameter search,
selection, and freeze rules are in `docs/design/training_hyperparameter_protocol.md`.
Historical one-step smoke constants are not paper hyperparameters.

The Qwen3.5 text backbone mixes linear-attention and full-attention layers. Paper LoRA
therefore uses an inventoried text-only linear target set rather than assuming that
`q_proj`, `k_proj`, `v_proj`, and `o_proj` cover the architecture. Vision and output
modules remain frozen. Every comparison method uses the same verified target set and
parameter budget.

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
- Frozen paper sampling uses temperature `1.0`, top-p `0.95`, top-k `20`, presence
  penalty `1.5`, and a 2048-token completion budget; the student remains in native
  Qwen thinking mode while structured teacher/evaluator roles use non-thinking mode.
- SearchQA GRPO reward weights are fixed at exact `0.55`, token F1 `0.25`, evidence
  support `0.10`, and answer type `0.10`; unknown answer type is neutral `0.5`.
- Prompt-only run qwen35-native-smoke-r2 completed on one RTX 6000 Ada with both
  examples correct on attempt 0; it produced no pairs, so the next smoke uses four
  harder but still controlled examples to test the guidance loop.
- Benchmark run qwen35-native-benchmark-smoke-r2 failed explicitly because the 9B
  evaluator exhausted its reasoning budget before emitting JSON. The raw output is
  preserved in `model_failures.jsonl`; the corrective change passes
  `enable_thinking=false` as a top-level local Transformers chat-template option for
  teacher/evaluator roles. The serving-API `chat_template_kwargs` spelling is not
  interchangeable with the local tokenizer API.
- Benchmark run qwen35-native-benchmark-smoke-r3 reached SearchQA but failed on an
  evaluator object with unescaped quotation marks inside `reason`. The evaluator
  contract now bounds and character-restricts that field; malformed output remains a
  hard failure with its raw text preserved.
- The next r3 retry reached the guidance guard and failed on a literal newline inside
  its JSON `reason` string. Structured-role reasons are now explicitly single-line;
  parsing remains strict rather than repairing malformed model output.
- The following r3 retry reached the guidance guard but omitted a JSON string delimiter.
  The guard is now an explicit one-token `SAFE`/`UNSAFE` role contract; evaluator
  judgments remain JSON because they require answer and confidence fields.
- Collection r3 completed with valid artifacts but no pairs: five wrong attempts,
  three unresolved groups, and all rejected guidance contained answer-bearing numbers
  or entities. The teacher prompt now forbids digits, proper nouns, quoted spans, and
  answer-bearing words so safe abstract hints can be tested without relaxing the guard.
- Collection r3 retry completed and passed native artifact validation: eight examples,
  nine attempts, five correct at attempt zero, one correct at attempt one, two unsafe
  guidance groups, one accepted pair, and no active Slurm jobs.
- One-step training smokes completed on that real collection. Standard DPO and native
  multilevel DPO each used one pair and both reported loss `0.693147`; on-policy
  distillation reported loss `1.352725`; GRPO completed with loss `0.0` but reward
  standard deviation `0.0` and all completions clipped at the configured limit, so
  it is a runtime smoke only and not a gain claim. Aggregate artifacts are in
  `runs/qwen35-method-comparison-r3/`.
- Approved deterministic paper hyperparameter tuning. DPO searches learning rate and
  DPO beta; GRPO searches learning rate and KL beta. Optimizer, warmup, decay, clipping,
  architecture coverage, candidate promotion, and model-selection evidence are logged.
- Reserved official test splits exclusively for final frozen evaluation. GSM8K tuning
  uses nested train/validation subsets; SearchQA-8K uses an additional disjoint
  2,000/500 auxiliary tuning pool from unused original official rows.
- Defined original `loss_type="grpo"` as the primary GRPO baseline and relegated the
  current DAPO loss to a separately labeled one-seed sensitivity run.
