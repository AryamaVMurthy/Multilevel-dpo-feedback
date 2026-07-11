# Native Iterative Guidance DPO

Status: active experiment specification

This document adapts the V1 textual-feedback DPO research method to Qwen3.5's native
chat protocol. The research object is the preference-data construction method, not a
particular XML response format or extended-thinking behavior.

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
or a prescribed reflection format. The MATH prompt asks Qwen to use only as much
reasoning as needed, emit one boxed final answer, and stop immediately afterward. Raw
model output is retained exactly.

Role-level generation controls are explicit: every role uses Qwen's supported
non-thinking chat-template mode. The student remains sampled, while teacher guidance
and evaluator judgments are greedy so short guidance and machine-readable judgments
do not spend their output budget on internal reasoning.

The student uses sampled non-thinking decoding with temperature `1.0`, top-p `1.0`,
top-k `20`, presence penalty `2.0`, balanced final-box stopping, and at most 16,384
new tokens. The teacher,
evaluator, leakage guard, and guidance critic use explicit non-thinking greedy profiles
with maximum output budgets of 64, 256, eight, and eight tokens respectively. These
profiles are independently hashed and logged; structured roles never inherit student
sampling settings.

Evaluation is a separate role. It returns a small structured judgment containing
correctness, extracted answer, confidence, and reason. This structure belongs to the
evaluator contract, not to the student's answer-generation contract.

## Guidance safety policy

The teacher receives the problem, gold answer, failed response, and evaluator result.
It must produce a short next-step hint. The hint cannot state the answer, an equivalent
expression, a decisive entity, or an answer-bearing phrase.

A separate critic checks whether the hint is directionally correct and relevant. A
separate evaluator-model leakage guard checks whether the accumulated hints disclose
the answer. Bounded deterministic regeneration includes the prior rejection review.
If all attempts are invalid or unsafe, the group is explicitly unresolved and no pair
is produced.

## Collection artifacts

- Immutable dataset roles are stored as `*.jsonl.zst` with a manifest and content hash.
- Collection shards store one compressed record per immutable example ID, with atomic
  progress and completion markers; attempt records reference the example rather than
  duplicating the base prompt and evidence.
- Merged collection records retain every raw student response, evaluator result, teacher
  output, guidance-policy decision, critic and guard results, exact prompt/generated
  token counts, finish reason, EOS/length truncation, and latency.
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
paper-scale GSM8K and SearchQA-8K comparison. Before collection or training, the frozen
base checkpoint is evaluated teacher-free on a manually audited validation preflight,
the full validation split, and the official test split once. Paper training is not
started until both that baseline and the collection/optimizer/LoRA preflights pass.

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
  penalty `1.5`, and an 8192-token completion budget; the student remains in native
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
- Reserved official test splits from all tuning and method changes. The immutable base
  checkpoint receives one frozen pre-research test evaluation; adapted checkpoints are
  evaluated only after selection. GSM8K tuning uses nested train/validation subsets;
  SearchQA-8K uses an additional disjoint 2,000/500 auxiliary tuning pool.
- Defined original `loss_type="grpo"` as the primary GRPO baseline and relegated the
  current DAPO loss to a separately labeled one-seed sensitivity run.
- Materialized and validated the complete pinned GSM8K paper split: 6,726 train, 747
  validation, and 1,319 official test rows, with validation partitioned 500/247.
- Started an immutable 64-example GSM8K R1 preflight. Partial evidence showed repeated
  malformed evaluator outputs before bounded evaluator repair was added, and an
  over-conservative guidance guard rejecting broad relation-level hints. R1 must finish
  unchanged; if its final gate fails, R2 will use explicit role-specific decoding and
  audited model-guard calibration under a new artifact path and protocol hash.
- R1 finished with 64 examples, 66 attempts, 54 first-attempt evaluator successes, one
  post-guidance success, nine unresolved groups, and one pair. It failed the paper gate:
  records span two evaluator schemas, generated-token finish metadata is absent, many
  responses end mid-reasoning, the surface policy rejects useful language, the guard is
  over-conservative, and no independent critic checks hint correctness. R2 therefore
  starts from zero with source/protocol fingerprinting, exact generation metadata, a
  8,192-token student ceiling, flexible slight hints, and separate correctness and
  leakage reviews.
- Added a mandatory pre-research base-checkpoint evaluation. Its immutable freeze binds
  source, dataset, Qwen3.5-2B, Qwen3.5-9B evaluator, prompt, decoding, and seed. Full
  validation and test generation are teacher-free, hash-merged Slurm shards with exact
  token/finish metadata, failure ledgers, manual evaluator-agreement gates, GPU
  telemetry, and one-time test markers. Any later student-prompt or inference-profile
  change invalidates and blocks on a complete baseline rerun.

### 2026-07-11

- MATH thinking-mode diagnostic job `13021` completed without runtime failures but
  truncated 11 of 16 generations at 8,192 tokens. Nine truncated responses already
  contained deterministically gold-equivalent answers, isolating termination as the
  dominant nuisance variable.
- Changed the primary paper student protocol to `qwen-nonthinking-r1`: explicit
  non-thinking mode, sampled `1.0/1.0/20/2.0` decoding, and a MATH boxed-answer-and-stop
  prompt. Thinking mode remains a separately labeled ablation.
- Non-thinking comparison job `13038` improved valid accuracy from 5/16 to 7/16 and
  reduced truncation from 11/16 to 8/16 on the same examples, but still failed the 5%
  truncation gate. Non-thinking remains the primary mode; its sampling profile requires
  a disjoint train-only termination study before DPO collection or training.
- Replaced the prompt-only stop request with `qwen-nonthinking-final-r2`: at most six
  numbered steps, one exact `FINAL: \boxed{answer}` line, tokenizer-level balanced-box
  stopping, and a 16,384-token emergency ceiling with 18,432-token training context.
- Final-r2 comparison job `13053` produced 13 balanced final-answer stops, three EOS
  stops, and zero truncations on the same 16 examples. Automated accuracy was 9/16,
  but manual audit corrected it to 8/16 after finding one confident evaluator false
  positive on a nested fraction/exponent gold answer. Deterministic LaTeX scorer repair
  commit `9df4d13` and CPU job `13058` then produced an immutable rescore with 8/16
  corrected accuracy, zero model-judgment fallbacks, and 16/16 manual agreement. The
  complete validation baseline is the current gate.
- Full-validation array `13061` was stopped at user request after shard 0 completed.
  Shard 0 contains 58 unique predictions, zero failures/truncations, 53 final-answer
  stops, and five EOS stops under source `9df4d13`; its immutable hashes and provenance
  are preserved. Shards 1-6 remain and must reuse `NUM_SHARDS=7`.
