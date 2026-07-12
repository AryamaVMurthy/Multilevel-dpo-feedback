# Canonical Qwen3 MATH-Then-SearchQA Execution Plan

Status: active and authoritative as of 2026-07-12.

This document supersedes every earlier experiment plan in this repository. Git
history and `docs/status/turing-usage-log.md` preserve historical diagnostics;
they are not active protocol instructions.

## 1. Objective and execution bias

Test whether a Qwen3-4B student benefits more from all genuine failures in a
privileged-feedback trajectory than from one conventional preference pair,
while controlling for pair count and comparing against online RL.

Execution is MATH-first. Infrastructure work is limited to what is required to
make a real baseline, collection, training, or evaluation job valid. After one
small end-to-end smoke passes, use Slurm arrays and spend compute on model runs
rather than additional scaffolding.

SearchQA starts only after the MATH report and selected hyperparameters are
frozen.

## 2. Frozen foundations

- Student: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Teacher and evaluator roles: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Post-trained checkpoints only; never `-Base`.
- `enable_thinking: false` for every role.
- LoRA rank 16, alpha 32, dropout 0.05, BF16, no quantization.
- Maximum generated completion: 8,192 tokens.
- Student decoding begins from Qwen's non-thinking profile and uses the already
  selected train-only termination setting recorded in the active MATH config.
- Final paper seeds: 17, 31, and 47.
- Official test data never selects prompts, policies, hyperparameters, or
  checkpoints.

Any change to these foundations creates a new protocol and new output root.

## 3. Model-output contracts

### Evaluator

The evaluator emits exactly:

```text
<verdict>CORRECT|WRONG</verdict>
<evaluated_answer>answer copied from the student response</evaluated_answer>
```

Normal mathematical notation and multiline text require no JSON escaping.
Missing, duplicated, nested, or surrounding content is an explicit parse
failure. The evaluator does not emit confidence or a free-form reason.

### Privileged teacher

The teacher emits exactly:

```text
<student_feedback>
Standalone answer-free mathematical advice.
</student_feedback>
```

The student-facing text cannot mention a previous turn, chat, retry, teacher,
or conversation. It cannot reveal the answer, an equivalent expression, a
decisive intermediate value, or a complete solution.

Raw generations, prompts, parse errors, token counts, finish reasons, latency,
and model revisions are always retained. No malformed output is silently
repaired or replaced.

## 4. Data and role separation

Use the pinned official `EleutherAI/hendrycks_math` snapshot with all 7,500
train and 5,000 test rows across seven subjects. Levels 4-5 are primary.

Deterministically divide eligible official training rows into 90% train and 10%
validation. Divide validation into tune and confirmation roles at 2:1. Preserve
the official test split untouched. Quarantine duplicates and source errata with
provenance rather than silently dropping or rewriting them.

The original problem is the only DPO inference prompt. Gold answers, teacher
prompts, feedback, reviewer decisions, and guided retry prompts remain
privileged collection metadata.

## 5. Phase A: finish, verify, and deploy the minimal protocol change

1. Finish the tagged evaluator and tagged teacher integration.
2. Add three explicit feedback-policy configs and a paired-config validator.
3. Bind feedback policy, prompt protocol, code commit, models, decoding, dataset
   hash, and seeds into immutable manifests.
4. Run focused tests, the complete local suite, Ruff, compile checks, shell
   syntax checks, and `git diff --check`.
5. Manually inspect prompts and successful/failing parses for all roles.
6. Commit and push one clean source revision.
7. Fast-forward the standalone Turing clone to exactly that revision.

Do not add another abstraction layer, service, or orchestration framework in
this phase.

## 6. Phase B: frozen teacher-free baseline before training

Run in this order:

1. Previously failing evaluator examples.
2. One-example end-to-end canary.
3. Frozen 16-example validation preflight with manual inspection.
4. Full teacher-free validation baseline.
5. One-time official-test baseline after the validation baseline is frozen.

Required gate:

- zero malformed evaluator outputs;
- at least 95% manual evaluator agreement;
- truncation at most 5%;
- complete generation metadata and hashes;
- no teacher information in prompts.

Report Levels 4-5 accuracy as primary and all-level accuracy as secondary,
including subject/level breakdowns, generated-token mean/median/P90,
truncation, latency, GPU memory, GPU-hours, and wall time.

No main training starts before the baseline artifact is frozen.

## 7. Phase C: paired three-policy feedback experiment

Use the same frozen initial student failures for all arms:

1. `error_only`: identify why the mathematical approach or inference is
   invalid, without giving a next step.
2. `hint_only`: give one slight direction, without explicitly diagnosing the
   attempted approach.
3. `error_and_hint`: briefly diagnose the error and give one slight direction.

Each arm may use at most three feedback levels. At level `k`, the teacher sees
the current failed response plus privileged context. The student sees the
original problem plus all approved standalone advice accumulated through level
`k`. Review both the new feedback alone and the accumulated feedback for answer
leakage, contradiction, and mathematical correctness.

Run the 16-example paired pilot first. Compare malformed rate, leakage-review
acceptance, correctness-review acceptance, success at each guidance step,
attempts to first correct, unresolved rate, pair yield, tokens, and latency.
Manually inspect all 48 initial feedback items.

Select one policy using validation data only, in this priority:

1. zero accepted answer leakage;
2. zero malformed output;
3. reviewer agreement at least 95%;
4. highest retry accuracy;
5. lowest unresolved rate;
6. lowest teacher-token cost.

Freeze one policy for full collection. Do not mix policies in the primary
dataset.

## 8. Phase D: full privileged-guidance collection

For every Level 4-5 training problem:

1. Generate the Qwen3-4B initial response.
2. Evaluate symbolic/model correctness and termination.
3. If wrong, generate feedback under the frozen policy.
4. Run independent leakage and mathematical-correctness reviewers.
5. Regenerate rejected feedback only with explicit review results.
6. Retry the student with original problem plus accumulated approved advice.
7. Stop at the first correct response or after three feedback steps.
8. Preserve unresolved trajectories without fabricating pairs.

Use sharded collection. Validate every shard before merge. A failed or
mixed-protocol shard is diagnostic and must be rerun into a new directory.

## 9. Preference datasets

For a trajectory with wrong responses `W0, W1, ..., Wn` followed by first
correct response `C`:

- Standard LN-DPO: one pair, `C > W0`.
- Multilevel LN-DPO: all pairs, `C > Wi` for every earlier wrong response.
- Matched LN-DPO: deterministic subset of the multilevel pool with the same
  total pair count as Standard. Stratify selection by failure depth where
  possible and freeze the selection seed.
- Response SFT: `C` only. This is a secondary control and cannot delay the main
  methods.

All LN-DPO variants use the same length-normalized sigmoid objective, frozen
reference model, optimizer search, LoRA inventory, and data roles. Only pair
construction differs.

Audit correctness, pair provenance, prompt equality, duplicate isolation,
response lengths, and hashes before training.

## 10. Main training methods

All methods start from the identical frozen Qwen3-4B checkpoint.

### Offline preference methods

1. Standard LN-DPO.
2. Multilevel LN-DPO.
3. Matched LN-DPO.

DPO search:

- learning rate: `2e-6`, `5e-6`, `1e-5`;
- beta: `0.05`, `0.1`, `0.3`, `0.5`;
- weight decay: `0`, `0.01`;
- warmup fraction: `0.05`, `0.10`;
- scheduler: linear or cosine;
- fused AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`;
- gradient clipping `1.0`.

Use successive halving on tune validation and one confirmation run for promoted
candidates. Then train the frozen candidate with seeds 17, 31, and 47.

### GRPO

- four generations per prompt;
- correctness requires a correct, evaluable, properly terminated response;
- truncated completions cannot receive positive correctness reward;
- symmetric ratio clipping: `epsilon_low=0.2`, `epsilon_high=0.2`;
- at least two optimizer iterations per rollout batch so clipping is active;
- learning rate: `2e-6`, `5e-6`, `1e-5`;
- KL beta: `0`, `0.001`, `0.01`, `0.04`;
- do not reward verbosity or reasoning length.

### DAPO

- four generations per prompt;
- token-level loss normalization;
- decoupled clipping: `epsilon_low=0.20`, `epsilon_high=0.28`;
- dynamic sampling explicitly identifies zero-variance all-correct/all-wrong
  groups and resamples only up to a frozen limit;
- exhausted groups are logged, never silently replaced;
- truncated responses receive no positive correctness reward;
- any overlength shaping is explicit, bounded, and cannot reward verbosity.

Log reward, reward standard deviation, KL, entropy, gradient norm, low/high/total
clip ratios, zero-variance groups, resampling, truncation, rollout throughput,
GPU memory, and wall time for both online methods.

Primary comparison: Base, Standard LN-DPO, Multilevel LN-DPO, Matched LN-DPO,
GRPO, and DAPO. Response SFT, on-policy distillation, and LD-DPO are secondary
and run after the primary jobs are safely launched.

## 11. Fast execution strategy

1. Run one short non-paper smoke for each trainer using a real validated shard.
2. Once model load, forward/backward, metrics, and adapter save/load pass, launch
   real tuning arrays; do not add more smoke layers.
3. Run collection shards and independent trainer smokes concurrently when they
   use distinct immutable output paths.
4. Use one 48 GB GPU by default. Use two only after a measured one-GPU OOM or a
   measured colocated-rollout failure.
5. Keep weights, environments, caches, and temporary rollouts on node-local
   scratch. Keep compressed final artifacts and small adapters in home.

## 12. Adaptive failure plan

### Malformed evaluator or teacher output

Preserve the raw output and exact parse error. Reproduce on the failing example.
Change the contract or prompt only after classifying the failure. Re-run the
known failure, then 16-example preflight, then resume with a new output root.
Never sanitize malformed text into a verdict or feedback.

### Evaluator agreement below 95%

Audit disagreements by category: extraction, symbolic equivalence, ambiguous
gold, or model judgment. Fix the responsible layer and rescore immutable
predictions when the repair is deterministic. Otherwise rerun evaluation. Do
not lower the threshold.

### Feedback leakage or incorrect advice

Reject the item and regenerate with reviewer feedback within the frozen retry
budget. If an arm repeatedly fails the 16-example gate, drop that arm rather
than weakening the guard. If all arms fail, revise the teacher contract once
and rerun the paired pilot before collection.

### Low successful-trajectory or pair yield

Report it as a method result first. Inspect yield by subject, level, base
correctness, and feedback step. Do not fabricate negatives or use official test
data. A validation-only change to feedback policy or retry budget requires a
new frozen collection protocol and new output root.

### Collection shard failure

Stop that shard, retain diagnostics, correct the root cause, and rerun it from
the start into a clean path. Continue independent valid shards if their
protocol is unaffected.

### One-GPU OOM

First verify memory attribution. Reduce per-device batch to one, use gradient
accumulation, gradient checkpointing, and measured sequence packing without
changing effective batch or token ceiling. Use two GPUs only if the valid
one-GPU profile still fails. Record the change in the run manifest.

### NaN/Inf training

Stop immediately. Inspect the first bad step, batch, reward, log probabilities,
gradient norm, BF16 scaling, and optimizer state. Reproduce with the same batch
before changing one variable. Never skip or replace the batch silently.

### GRPO/DAPO zero-variance groups

Log their rate by prompt and reward component. DAPO may use its frozen dynamic
sampling policy. GRPO retains them as measured zero-gradient groups. If the
rate is high, improve prompt sampling or reward evaluability using validation
data; do not add arbitrary verbosity rewards.

### Excessive clipping or KL

Inspect clip-low, clip-high, total clip ratio, KL, and advantage scale. Reduce
learning rate or optimizer iterations through tune-validation selection. Do not
change clipping based on official-test performance.

### Job/environment failure

Separate SSH, Slurm, storage, model cache, environment, CUDA, data, and code
failures. Re-run only the smallest failing layer. Environment failures never
justify fake data or CPU training hidden inside a GPU job.

## 13. Evaluation and statistics

Evaluate every final method and seed with the frozen baseline protocol.

Report exact/symbolic accuracy with paired-bootstrap 95% confidence intervals,
subject/level accuracy, token mean/median/P90, truncation, guidance-step success,
unresolved rate, pair yield, teacher tokens, optimizer metrics, GPU memory,
GPU-hours, wall time, failures, retries, and storage.

Run paired tests against Base and Standard LN-DPO. Apply Holm correction across
primary comparisons. Emphasize effect sizes and confidence intervals; do not
select models using test results.

Produce immutable JSON/JSONL, TensorBoard logs, plots, tables, an HTML report,
artifact hashes, and a command ledger.

## 14. SearchQA-8K

After the MATH report is frozen, materialize the pinned original SearchQA
release into 5,000 train, 1,000 validation, and 2,000 official-test examples,
with separate unused prompt/hyperparameter reserves. Use controlled packaged
evidence and no live web search during evaluation.

Repeat Base, the selected feedback collection, Standard/Multilevel/Matched
LN-DPO, GRPO, and DAPO. Select SearchQA settings only from SearchQA tune and
confirmation data. Produce a separate report and cross-domain comparison.

## 15. Completion criteria

MATH and then SearchQA each require frozen data, baseline scores, audited
trajectories and pairs, primary six-method comparison, three final seeds,
statistical tests, compute accounting, immutable manifests, HTML reports, and a
reproducible command ledger pushed with a clean Git tree.
