# Paper-Scale GSM8K and SearchQA-8K Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and execute a reproducible paper-grade comparison of standard DPO, multilevel-feedback DPO, pair-budget-matched multilevel DPO, and GRPO on full GSM8K followed by SearchQA-8K using Qwen3.5-2B with a privileged Qwen3.5-9B slight-hint teacher.

**Architecture:** The implementation separates immutable dataset manifests, resumable collection shards, deterministic preference construction, LoRA training, adapter-aware held-out generation, domain evaluation, and paper reporting. Large artifacts and model caches live on node-local Turing scratch; only manifests, metrics, reports, and final adapters are copied to `/home`.

**Tech Stack:** Python 3.12, Hugging Face Datasets/Transformers, Qwen3.5, TRL, PEFT LoRA, PyTorch, TensorBoard, Slurm, JSONL/Zstandard, HTML/SVG reporting.

---

The canonical optimizer and search specification is
`docs/design/training_hyperparameter_protocol.md`. Historical smoke settings are not
paper defaults.

## Current Execution State

Checkpoint: 2026-07-10, branch `agent/qwen35-pretest`. Baseline/R2 protocol commit
`ea45bd7b1a47ea32c1a9dc3df330d593829da5ff` passed a one-example GPU micro; the
post-micro observability correction is locally verified and awaiting its source commit.

| Scope | State | Evidence or next gate |
| --- | --- | --- |
| Tasks 1-12 | implemented | strict configs, data, collection, preference, LoRA, DPO/GRPO, held-out, observability, and Slurm surfaces exist |
| Task 13 | verified locally | 150 tests pass in the locked local `.venv`; compile, lock, shell syntax, documentation integrity, and diff checks pass |
| Task 14 | complete | GSM8K manifest hash `61a7a7f82f0ff75491b7b363504f85b0543628a860084cc0be66a01cf6f9eb6c` with 6,726/747/1,319 and nested 500/247 roles |
| Task 15 | one-example micro passed | correct, EOS-terminated 3,058-token response; final 16-example freeze/audit remains required |
| Task 16 R1 | diagnostic complete, gate failed | 64 records across jobs `12948_0`, `12952_0`, and `12964_0`; mixed source protocols, unverified truncation, guidance failures, and one pair |
| Tasks 17-27 | blocked | require a passing baseline gate, passing R2 collection preflight, and all later domain gates |

R1 observations are not results and must not be used to start training. Its artifacts
remain intact, and the corrected protocol runs from example zero under a new R2 path,
source commit, and protocol hash.

## Canonical Paths

- Local worktree: `multilevel-feedback-dpo/.worktrees/qwen35-pretest/`.
- Turing code: `/home/aryama.murthy/multilevel-feedback-dpo/implementation`.
- Turing persistent metadata: `/home/aryama.murthy/tfdpo-runs`.
- Turing large active artifacts: revision-keyed directories under the allocated
  compute node's `/scratch`.
- Local durable archive: `/home/aryamavmurthy/work/SLM-Research/multilevel-feedback-dpo-artifacts/`, outside Git.

Code sync is restricted to `implementation/` and selected docs. Never use a whole
remote worktree as an `rsync --delete` destination because experiment artifacts are not
source files.

## Execution Rules

- Work in the existing `agent/qwen35-pretest` worktree.
- Follow test-driven development for every behavior change.
- Commit after each task; push only verified commits.
- Never run ML imports, dataset processing, inference, or training on the Turing login node.
- Never use the official test split for pair collection, prompt changes, reward changes,
  hyperparameter selection, stopping, or model selection. The immutable base checkpoint
  is evaluated once before research after its evaluation protocol freezes; its test
result cannot change any later experimental decision.
- The frozen baseline student prompt and inference profile are shared by every later
  checkpoint. Any later change to either invalidates the baseline and requires a new
  freeze and complete baseline rerun before research resumes.
- Never use SearchQA auxiliary tuning rows in final SearchQA-8K training or reporting.
- Never merge a missing or failed shard as an empty shard.
- Do not start SearchQA until the GSM8K completion gate passes.
- Do not start a full job after a failed preflight.
- Before and after each Slurm phase, run `squeue -u "$USER"` and record the result.
- Before every phase, record `df`, `du`, Slurm accounting, node-local cache identity,
  and projected output size. Require at least 8 GB free and less than 85% persistent
  storage utilization before submission.
- Student, teacher, evaluator, leakage-guard, and guidance-critic generation profiles
  are separate immutable config objects. No role inherits another role's decoding
  settings implicitly.

## Task 1: Add Paper Experiment Configuration Schema

**Files:**
- Create: `implementation/src/text_feedback_dpo/experiment_config.py`
- Create: `implementation/tests/test_experiment_config.py`
- Create: `implementation/configs/paper/gsm8k.yaml`
- Create: `implementation/configs/paper/searchqa8k.yaml`
- Modify: `implementation/src/text_feedback_dpo/cli.py`

**Step 1: Write failing schema tests**

Test that a paper config requires dataset revision, source counts, split manifest,
seeds, architecture-audited LoRA settings, optimizer fields, deterministic candidate
matrix, promotion budgets, nested validation partitions, generation settings, shard
size, retry budget, and final-test freeze flag. Test that unknown keys, missing
revisions, overlap-prone split settings, deprecated warmup fields, implicit optimizer
defaults, and a non-16384 completion budget fail explicitly.

**Step 2: Verify the tests fail**

Run from `implementation/`:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_experiment_config.py'
```

Expected: import or assertion failure because `experiment_config.py` does not exist.

**Step 3: Implement strict dataclass-backed parsing**

Expose:

```python
def load_paper_experiment(path: Path) -> PaperExperimentConfig: ...
def validate_paper_experiment(config: PaperExperimentConfig) -> None: ...
```

No missing value receives a default that changes experiment behavior. Config parsing
must fail with the exact field path and remediation.

**Step 4: Add frozen GSM8K and SearchQA-8K configs**

GSM8K contains source counts `7473/1319`, split counts `6726/747/1319`, nested
validation counts `500/247`, seeds, BF16 LoRA rank 16, alpha 32, the approved DPO and
GRPO candidate matrices, and the approved sampling settings. SearchQA-8K contains
source counts `99820/13393/27248`, sample counts `5000/1000/2000`, and disjoint
auxiliary tuning counts `2000/500`.

**Step 5: Run tests and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_experiment_config.py'
git add implementation/src/text_feedback_dpo/experiment_config.py implementation/tests/test_experiment_config.py implementation/configs/paper implementation/src/text_feedback_dpo/cli.py
git commit -m "feat: add paper experiment configuration"
```

## Task 2: Build Immutable Dataset Manifests

**Files:**
- Create: `implementation/src/text_feedback_dpo/dataset_manifests.py`
- Create: `implementation/tests/test_dataset_manifests.py`
- Modify: `implementation/src/text_feedback_dpo/benchmarks.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`

**Step 1: Write failing tests for stable hashes and splits**

Use small fixtures to prove that:

- canonical row hashes are stable;
- a fixed seed produces identical manifests;
- GSM8K train and validation counts sum to the source train count;
- official test rows remain test rows;
- GSM8K tuning-development and confirmation-development exactly partition validation;
- SearchQA sampling stays inside each official source split;
- SearchQA auxiliary tuning rows are disjoint from every SearchQA-8K split;
- normalized question or source-key overlap fails;
- an unexpected source count fails before writing output.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_dataset_manifests.py'
```

**Step 3: Implement manifest generation**

Expose pure functions:

```python
def canonical_row_hash(row: Mapping[str, object]) -> str: ...
def split_gsm8k_train(rows, seed, validation_count=747): ...
def sample_searchqa8k(train, validation, test, seed): ...
def validate_disjoint_splits(manifest_rows): ...
```

Use stable hashing and deterministic ordering, not the process-global random state.

**Step 4: Add `materialize-dataset` CLI**

The command runs under Slurm, pins dataset revisions, writes `manifest.json`,
`train.jsonl.zst`, `validation.jsonl.zst`, `test.jsonl.zst`, and a duplicate audit.
GSM8K additionally writes nested validation-role manifests. SearchQA additionally
writes `hparam_train.jsonl.zst` and `hparam_validation.jsonl.zst`; final trainers must
reject these roles.

**Step 5: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_dataset_manifests.py'
git add implementation/src/text_feedback_dpo/dataset_manifests.py implementation/src/text_feedback_dpo/benchmarks.py implementation/src/text_feedback_dpo/cli.py implementation/tests/test_dataset_manifests.py
git commit -m "feat: materialize immutable paper dataset manifests"
```

## Task 3: Load the Original SearchQA Release

**Files:**
- Create: `implementation/src/text_feedback_dpo/searchqa.py`
- Create: `implementation/tests/test_searchqa.py`
- Modify: `implementation/src/text_feedback_dpo/benchmarks.py`
- Modify: `implementation/pyproject.toml`
- Modify: `implementation/uv.lock`

**Step 1: Write fixture-based parser tests**

Test original SearchQA metadata, snippets, answer aliases, year, source key, and split
conversion. Test that missing snippets, answer aliases, source keys, or source split
fail explicitly.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_searchqa.py'
```

**Step 3: Implement original-format parsing and token-budgeted evidence packaging**

Preserve snippet order. Store full raw snippets in the materialized dataset. Generate
the model prompt from a configurable token budget and record snippet count, input token
count, truncation, and whether an answer alias remains in packaged evidence.

**Step 4: Prohibit the MRQA mirror in paper configs**

Paper config validation must reject `lucadiliello/searchqa` with an error pointing to
the original SearchQA loader.

**Step 5: Test, lock dependencies, and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_searchqa.py'
uv lock
git add implementation/src/text_feedback_dpo/searchqa.py implementation/src/text_feedback_dpo/benchmarks.py implementation/tests/test_searchqa.py implementation/pyproject.toml implementation/uv.lock
git commit -m "feat: load official SearchQA splits"
```

## Task 4: Enforce the Very-Slight Hint Contract

**Files:**
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/src/text_feedback_dpo/evaluators.py`
- Modify: `implementation/src/text_feedback_dpo/methods.py`
- Create: `implementation/src/text_feedback_dpo/guidance_policy.py`
- Create: `implementation/tests/test_guidance_policy.py`
- Modify: `implementation/tests/test_native_pipeline.py`

**Step 1: Write failing tests**

Test that the teacher prompt requires one 8-to-15-word sentence and prohibits digits,
quantities, equations, explicit arithmetic operations, proper-noun disclosure, answer
length, initials, and copied evidence. Test that surface-invalid hints never reach the
student. Test that the semantic guard receives all accumulated hints and can return
`unsafe_accumulated_guidance`.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_guidance_policy.py'
```

**Step 3: Implement two-layer safety**

Use deterministic checks only for the measurable surface contract. Use the 9B guard
for semantic leakage and combinations of prior hints. Do not implement semantic safety
with keyword matching.

**Step 4: Preserve auditability**

Store raw teacher output, surface-policy result, accumulated guidance, guard raw output,
and rejection reason for every hint generation.

**Step 5: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_guidance_policy.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_native_pipeline.py'
git add implementation/src/text_feedback_dpo/prompts.py implementation/src/text_feedback_dpo/evaluators.py implementation/src/text_feedback_dpo/methods.py implementation/src/text_feedback_dpo/guidance_policy.py implementation/tests/test_guidance_policy.py implementation/tests/test_native_pipeline.py
git commit -m "feat: enforce very slight accumulated guidance"
```

## Task 5: Add Native Domain Evaluation

**Files:**
- Create: `implementation/src/text_feedback_dpo/answer_evaluation.py`
- Create: `implementation/tests/test_answer_evaluation.py`
- Modify: `implementation/src/text_feedback_dpo/evaluation.py`
- Modify: `implementation/src/text_feedback_dpo/evaluators.py`

**Step 1: Write failing GSM8K evaluation tests**

Cover commas, currency symbols, negative values, decimals, equivalent numeric strings,
missing answers, and ambiguous multiple answers. Deterministic checks validate numeric
equivalence after evaluator-backed answer extraction; they do not parse private reasoning.

**Step 2: Write failing SearchQA evaluation tests**

Cover alias exact match, normalized exact match, token F1, expected answer type,
evidence support, ambiguity routing to the model evaluator, and unsupported answers.

**Step 3: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_answer_evaluation.py'
```

**Step 4: Implement shared evaluator results**

Return a schema containing extracted answer, correctness, confidence, evaluator source,
exact match, F1, answer type, evidence support, ambiguity, raw judgment, latency, and
token count. Malformed evaluator output remains a hard failure.

**Step 5: Replace native calls to the legacy XML evaluator**

Keep legacy functions only for legacy commands. Paper collection, validation, test,
and GRPO must import the new evaluator.

**Step 6: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_answer_evaluation.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_evaluation.py'
git add implementation/src/text_feedback_dpo/answer_evaluation.py implementation/src/text_feedback_dpo/evaluation.py implementation/src/text_feedback_dpo/evaluators.py implementation/tests/test_answer_evaluation.py
git commit -m "feat: add native domain evaluation"
```

## Task 6: Make Collection Sharded, Resumable, and Compressed

**Files:**
- Create: `implementation/src/text_feedback_dpo/sharding.py`
- Create: `implementation/tests/test_sharding.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/src/text_feedback_dpo/io.py`
- Modify: `implementation/src/text_feedback_dpo/methods.py`

**Step 1: Write failing sharding tests**

Test deterministic shard membership, complete coverage, no overlap, atomic completion
marker, refusal to overwrite a completed shard with another config hash, resume from
the last complete example, and hard failure for a missing shard during merge.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_sharding.py'
```

**Step 3: Implement shard commands**

Add:

```text
collect-shard --config ... --split train --shard-index N --num-shards M
merge-collection --config ... --expected-shards M
```

Write compressed records incrementally and fsync before advancing the progress marker.
Do not duplicate full base prompts and evidence in every attempt record; reference the
immutable example ID.

**Step 4: Add merge invariants**

Merge validates config hash, dataset manifest hash, model revisions, seed, shard index,
row counts, unique IDs, and all expected completion markers.

**Step 5: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_sharding.py'
git add implementation/src/text_feedback_dpo/sharding.py implementation/src/text_feedback_dpo/cli.py implementation/src/text_feedback_dpo/io.py implementation/src/text_feedback_dpo/methods.py implementation/tests/test_sharding.py
git commit -m "feat: add resumable compressed collection shards"
```

## Task 7: Build Standard, Multilevel, and Matched Datasets

**Files:**
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Create: `implementation/src/text_feedback_dpo/preference_data.py`
- Create: `implementation/tests/test_preference_data.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`

**Step 1: Write failing construction tests**

Test one standard pair per successful group, every prior wrong response for multilevel,
no pair for unresolved groups, original-prompt-only rows, deterministic matched sampling,
attempt-level stratification, and equal matched/standard pair counts.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_preference_data.py'
```

**Step 3: Implement and add `build-preferences` CLI**

Write separate immutable datasets and manifests for `standard`, `multilevel`, and
`multilevel_matched`. Include group count, pair count, attempt distribution, prompt
hashes, chosen/rejected hashes, and leakage audit.

**Step 4: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_preference_data.py'
git add implementation/src/text_feedback_dpo/preference_data.py implementation/src/text_feedback_dpo/training.py implementation/src/text_feedback_dpo/cli.py implementation/tests/test_preference_data.py
git commit -m "feat: build fair paper preference datasets"
```

## Task 8: Add Architecture-Audited LoRA And Hyperparameter Search

**Files:**
- Create: `implementation/src/text_feedback_dpo/lora_coverage.py`
- Create: `implementation/src/text_feedback_dpo/hyperparameter_search.py`
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/tests/test_training.py`
- Create: `implementation/tests/test_training_profiles.py`
- Create: `implementation/tests/test_lora_coverage.py`
- Create: `implementation/tests/test_hyperparameter_search.py`

**Step 1: Write failing architecture-coverage tests**

Use a hybrid-model fixture with text linear-attention, full-attention, MLP, vision,
embedding, and output modules. Assert that discovery covers all text projection classes,
excludes vision/embedding/output modules, fails on an empty or unexpected inventory,
and writes exact matched names, shapes, and trainable parameter counts.

**Step 2: Write failing optimizer and search-ledger tests**

Assert BF16, no quantization, fused AdamW, Adam coefficients `0.9/0.999`, epsilon
`1e-8`, weight decay `0.01`, maximum gradient norm `1.0`, integer 5% warmup, cosine
scheduling, effective DPO batch 16, and identical LoRA targets across methods. Assert
the exact DPO and GRPO candidate matrices, deterministic successive-halving promotion,
invalid-run rejection, prespecified tie-breakers, equal per-method budgets, resumable
ledgers, and refusal to overwrite a frozen selection.

**Step 3: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_training_profiles.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_lora_coverage.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_hyperparameter_search.py'
```

**Step 4: Implement architecture-driven profiles**

Inventory the pinned Qwen3.5 model structurally and create a text-backbone target list.
Do not use a static `q/k/v/o` assumption. Apply rank 16, alpha 32, and dropout 0.05,
then assert the recorded target inventory before constructing any trainer.

**Step 5: Implement explicit optimizer and deterministic search APIs**

Expose immutable candidate, stage, observation, promotion, and freeze records. Compute
integer warmup steps from total optimizer updates. The CLI must support creating a
search ledger, registering a completed run, promoting a stage, and writing a signed
freeze manifest. Missing metrics and failed artifacts are explicit invalid candidates,
never worst-score placeholders.

**Step 6: Expand profile-driven training observability**

Training must write config, dataset manifest hash, parameter counts, trainable parameter
percentage, exact target modules, optimizer/scheduler, learning rate by step, warmup,
gradient norms, clipping, beta/KL, package versions, history, candidate ID, search-ledger
hash, best validation checkpoint, and final adapter. Standard and matched runs must use
equal optimizer-update budgets.

**Step 7: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_training*.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_lora_coverage.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_hyperparameter_search.py'
git add implementation/src/text_feedback_dpo/lora_coverage.py implementation/src/text_feedback_dpo/hyperparameter_search.py implementation/src/text_feedback_dpo/training.py implementation/src/text_feedback_dpo/cli.py implementation/tests/test_training.py implementation/tests/test_training_profiles.py implementation/tests/test_lora_coverage.py implementation/tests/test_hyperparameter_search.py
git commit -m "feat: add audited LoRA and hyperparameter search"
```

## Task 9: Add Adapter-Aware Held-Out Generation

**Files:**
- Modify: `implementation/src/text_feedback_dpo/models.py`
- Create: `implementation/src/text_feedback_dpo/heldout.py`
- Create: `implementation/tests/test_heldout.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`

**Step 1: Write failing tests**

Test base generation, PEFT adapter loading, base-model compatibility validation,
teacher-free prompts, no gold in model inputs, deterministic seed handling, raw output
storage, and refusal to evaluate a mismatched adapter.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_heldout.py'
```

**Step 3: Implement `evaluate-checkpoint`**

The command generates validation or test predictions from a base model or adapter,
then scores after generation. Test execution requires a frozen-run manifest and must
write a one-time test marker preventing accidental repeated test-based tuning.

**Step 4: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_heldout.py'
git add implementation/src/text_feedback_dpo/models.py implementation/src/text_feedback_dpo/heldout.py implementation/src/text_feedback_dpo/cli.py implementation/tests/test_heldout.py
git commit -m "feat: evaluate base and LoRA checkpoints"
```

## Task 10: Replace the Degenerate GRPO Reward

**Files:**
- Create: `implementation/src/text_feedback_dpo/rewards.py`
- Create: `implementation/tests/test_rewards.py`
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Modify: `implementation/tests/test_training.py`

**Step 1: Write failing reward tests**

Test that incidental answer substrings do not score, exact GSM8K numeric answers do,
SearchQA components receive approved weights, truncated responses are masked, evaluator
failures stop training, and a mixed completion group has nonzero variance.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_rewards.py'
```

**Step 3: Implement shared reward functions**

Use the same evaluation result schema as held-out scoring. Configure four generations,
generation batch divisibility, max completion length 16384, non-thinking chat templating,
temperature 1.0, top-p 1.0, top-k 20, presence penalty 2.0 through supported generation kwargs, completion logging,
and truncated-completion masking. Configure the primary baseline explicitly as original
`loss_type="grpo"`, one policy iteration, clipping epsilon `0.2`, and within-group reward
scaling. Configure DAPO only through the separately named `dapo_sensitivity` profile.

**Step 4: Add preflight metric gates**

Abort a full GRPO run when zero-variance group rate exceeds 50%, truncation exceeds 5%,
or reward/evaluator agreement on the audited pilot falls below 95%.

**Step 5: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_rewards.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_training.py'
git add implementation/src/text_feedback_dpo/rewards.py implementation/src/text_feedback_dpo/training.py implementation/tests/test_rewards.py implementation/tests/test_training.py
git commit -m "fix: add nondegenerate shared GRPO rewards"
```

The fixed SearchQA reward weights are exact match `0.55`, token F1 `0.25`, controlled
evidence support `0.10`, and answer-type correctness `0.10`; unknown official answer
types receive a neutral type component of `0.5`. Missing evaluator fields and failed
reward evaluation are hard errors.

## Task 11: Expand Observability and Run Validation

**Files:**
- Modify: `implementation/src/text_feedback_dpo/observability.py`
- Modify: `implementation/src/text_feedback_dpo/validation.py`
- Modify: `implementation/src/text_feedback_dpo/report.py`
- Modify: `implementation/tests/test_observability.py`
- Modify: `implementation/tests/test_validation.py`
- Modify: `implementation/tests/test_report.py`

**Step 1: Write failing artifact tests**

Require Git commit, config hash, dataset hash, seed, source revision, model revisions,
package versions, Slurm metadata, GPU telemetry, token counts, latency, throughput,
peak memory, pair metrics, evaluator confidence, training metrics, architecture target
inventory, optimizer fields, learning-rate schedule, warmup, gradient norms, clipping,
candidate ID, promotion stage, selection evidence, search-ledger hash, and failure ledger.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_observability.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_validation.py'
```

**Step 3: Implement TensorBoard plus canonical JSONL**

JSONL remains the source of truth. TensorBoard is an additional view. HTML reports are
generated from canonical metrics and contain success-step distributions, pair yield,
loss, reward margin, accuracy, truncation, throughput, memory, and wall time.

**Step 4: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_observability.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_validation.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_report.py'
git add implementation/src/text_feedback_dpo/observability.py implementation/src/text_feedback_dpo/validation.py implementation/src/text_feedback_dpo/report.py implementation/tests/test_observability.py implementation/tests/test_validation.py implementation/tests/test_report.py
git commit -m "feat: add paper experiment observability"
```

## Task 12: Add Turing Dataset, Collection, Training, and Evaluation Scripts

**Files:**
- Create: `implementation/scripts/turing_materialize_dataset.sh`
- Create: `implementation/scripts/turing_collect_array.sh`
- Create: `implementation/scripts/turing_merge_collection.sh`
- Create: `implementation/scripts/turing_tune_paper.sh`
- Create: `implementation/scripts/turing_train_paper.sh`
- Create: `implementation/scripts/turing_evaluate_paper.sh`
- Modify: `implementation/tests/test_turing_scripts.py`

**Step 1: Write failing script tests**

Assert `set -euo pipefail`, account/config requirements, one-node allocation, expected
GPU count, CUDA visibility, scratch-only caches and environments, conservative uv
settings, GPU telemetry, trap cleanup, config copying, no login-node training, and
explicit shard environment variables.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_turing_scripts.py'
```

**Step 3: Implement scripts**

Collection and tuning arrays use one GPU per task. Dataset and model caches use a
node-local shared cache keyed by revision. Job-specific environments and temporary
outputs use job scratch. Tuning jobs require candidate and stage IDs and atomically
write ledger observations. Critical copies and merges must not use `|| true`.

**Step 4: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_turing_scripts.py'
git add implementation/scripts implementation/tests/test_turing_scripts.py
git commit -m "feat: add paper-scale Turing workflows"
```

The script commands map to `materialize-dataset`, `collect-shard`, `merge-collection`,
`tune-paper`, `train-paper`, and `evaluate-paper`. Search-ledger initialization,
promotion, and freezing are explicit CLI operations between tuning and final training.

## Task 13: Run the Complete Local Verification Gate

**Files:**
- Modify only files required by discovered failures.

**Step 1: Run all tests**

```bash
cd implementation
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests pass with no warnings promoted to errors.

**Step 2: Run static repository checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intentional changes.

**Step 3: Commit any verification-only corrections and push**

```bash
git push origin agent/qwen35-pretest
```

## Task 14: Materialize and Verify Full GSM8K on Turing

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/data/`

**Step 1: Verify cluster state without allocating a GPU**

```bash
ssh aryama.murthy@turing.iiit.ac.in
squeue -u "$USER"
sacctmgr show assoc user="$USER" format=Account,QOS,MaxJobs,MaxSubmitJobs,MaxTRES
df -h "$HOME" /scratch/node01
```

Expected: no unintended active jobs, account `priyesh.shukla` is available, and enough
scratch capacity exists. Do not expose credentials in logs.

**Step 2: Sync the verified branch and submit dataset materialization under Slurm**

Use `turing_materialize_dataset.sh` with the GSM config. Record the job ID.

**Step 3: Verify materialized counts and hashes**

Run the manifest validator. Expected counts: 6,726 train, 747 validation, 1,319 test,
500 tuning-development, 247 confirmation-development, zero overlap, pinned revision,
and successful duplicate audit. Nested development roles must exactly partition the
747 validation rows.

**Step 4: Record accounting and queue state**

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,AllocTRES,MaxRSS,NodeList
squeue -u "$USER"
```

## Task 15: Freeze and Evaluate the Teacher-Free GSM8K Baseline

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/baseline/`
- Use: `implementation/scripts/turing_evaluate_paper.sh`
- Use: `implementation/scripts/turing_merge_evaluations.sh`
- Use: `implementation/scripts/turing_materialize_preflight_subset.sh`
- Use: `implementation/scripts/turing_freeze_baseline.sh`
- Use CLI: `materialize-preflight-subset`, `freeze-baseline`, `evaluate-paper`, `merge-evaluations`, and
  `audit-evaluation`

**Step 1: Freeze the baseline evaluation identity**

After the corrected evaluation code is committed and synced, write a
`baseline-evaluation-freeze-v1` manifest. It binds source commit, full config hash,
dataset-manifest hash, Qwen3.5-2B revision, Qwen3.5-9B evaluator revision, native
student prompt protocol, role-specific generation profiles, and generation seed.
Teacher generation is disabled. A mismatched source, model, config, dataset, prompt,
or decoding profile is a hard error.
The paper config is schema v3; schema v2 files are rejected because they do not bind
the mandatory baseline protocol or the 16,384/18,432-token contracts.

**Step 2: Run a small validation preflight and manually audit every response**

Materialize 16 hash-selected validation IDs with the frozen evaluation seed and retain
the subset manifest, source/output hashes, and exact selected IDs. Run that immutable
subset through the base checkpoint. Store raw responses,
exact prompt and generated token counts, EOS/length finish reasons, truncation,
generation/evaluator latency, evaluator raw turns, failure ledger, GPU telemetry, and
Slurm accounting. Manually label all 16 evaluator decisions. Require 100% artifact
coverage, no missing metadata, no teacher/gold prompt context, no unexplained failures,
at least 95% evaluator/manual agreement, at most 5% truncation, and peak memory below
90%. If this fails, revise the evaluation protocol under a new freeze before any full
baseline or collection job.

**Step 3: Run and merge the full 747-example validation baseline**

Choose the validation shard count from measured 16,384-token preflight throughput, with
at least 25% wall-time headroom; six shards of approximately 125 examples are only the
initial estimate. Merge only complete
shards whose prediction hashes, freeze hash, checkpoint identity, seed, split, and
canonical ID order match. Audit at least 64 stratified validation predictions and
enforce the same agreement, truncation, metadata, teacher-free, memory, and failure
gates. Report exact numerical accuracy, evaluator confidence, response lengths,
finish reasons, latency, throughput, peak GPU memory, wall time, and GPU-hours.

**Step 4: Run the base checkpoint once on all 1,319 official test examples**

Only after Steps 1-3 pass, choose the test shard count from measured throughput (11
shards is the initial estimate) and merge them under the same freeze. This is the one
pre-research exception to delayed test execution because
the base checkpoint and evaluation policy are immutable. The baseline test result is
descriptive only and is prohibited from changing prompts, collection policy, rewards,
search spaces, promotion, stopping, or model selection. Preserve the one-time test
markers; do not regenerate the base test predictions later.

**Step 5: Publish the baseline artifacts before preference collection**

Write JSON/CSV metrics, per-example predictions, audit labels and disagreements, GPU
telemetry, Slurm accounting, plots, and an HTML report. Record hashes in the living
execution log. No teacher-guidance collection or training job may start until this
baseline gate passes.

## Task 16: Run the Corrected GSM8K Collection Preflight

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/collection-preflight/`

**Step 1: Preserve and classify immutable R1 as diagnostic**

R1 already completed on the first deterministic 64 training examples with three
slight-hint rounds and failed its paper gate. Preserve jobs `12948`, `12952`, and
`12964`, raw outputs, model failures, progress, telemetry, accounting, and local archive.
Never append corrected records to R1 or use its one preference pair for training.

**Step 2: Inspect every raw attempt and hint**

Manually audit all 64 examples, all hints, accumulated guard inputs, and evaluator
decisions. Record corrections in an audit JSONL, never by editing raw artifacts.

**Step 3: Enforce gates**

Require evaluator parse success at least 99%, audit agreement at least 95%, nonzero
pair yield, zero accepted answer leakage, guidance-guard agreement at least 95%, memory
below 90%, and truncation at most 5%. Report first-attempt accuracy, success by guidance
step, unresolved rate, mean attempts, pair yield, surface-rejection rate, guard
confusion matrix, evaluator regeneration rate, latency, tokens, and peak memory.

**Step 4: Diagnose a failed R1 without modifying its artifacts**

If a gate fails, capture exact artifacts, reproduce the smallest failing example, add a
failing local test, and write the root cause to the failure ledger. A failed R1 does not
permit full collection or training.

**Step 5: Add explicit role-specific decoding for R2 when required**

**Files:**
- Modify: `implementation/src/text_feedback_dpo/experiment_config.py`
- Modify: `implementation/src/text_feedback_dpo/collection.py`
- Modify: `implementation/src/text_feedback_dpo/models.py`
- Modify: `implementation/src/text_feedback_dpo/evaluators.py`
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/configs/paper/gsm8k.yaml`
- Modify: `implementation/configs/paper/searchqa8k.yaml`
- Modify: `implementation/tests/test_experiment_config.py`
- Modify: `implementation/tests/test_collection.py`
- Modify: `implementation/tests/test_models.py`
- Modify: `implementation/tests/test_guidance_policy.py`

Write failing tests proving that the student uses explicit non-thinking mode plus
sampled `1.0/1.0/20/2.0` decoding, balanced final-box stopping, and 16,384 tokens. Require teacher greedy non-thinking
decoding with 64 tokens, evaluator greedy non-thinking decoding with 256 tokens, and
guard greedy non-thinking decoding with eight tokens. Greedy profiles must omit
temperature, top-p, top-k, and presence penalty instead of passing ignored values.
Missing or inherited profiles fail config validation.

Calibrate the model-based guard with explicit safe relation-level hints and unsafe
answer-bearing hints from the audited R1 failures. The guard's sole question is whether
the accumulated guidance discloses the answer or an equivalent, not whether the hint is
useful. Keep strict parsing, zero answer leakage, bounded repair turns, and full raw
logging; do not replace semantic judgment with keywords or regexes.

Run focused tests, the entire local suite, compile checks, and shell syntax checks.
Commit and push the protocol change before syncing only `implementation/` to Turing.

**Step 6: Run R2 from scratch and freeze the passing protocol**

Write R2 to `collection-preflight-r2/`; never append it to R1. Re-run all 64 examples,
audit every record, and enforce every Step 3 gate. If R2 fails, repeat the same
test-first failure workflow under R3 rather than relaxing a gate. When a preflight
passes, freeze its config, prompts, role profiles, evaluator, guard, package lock,
model revisions, and artifact schema hashes for full GSM8K collection.

## Task 17: Run Full GSM8K Collection and Merge

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/collection/`

**Step 1: Estimate shard time from preflight**

Choose the largest shard size projected below three hours. Start with 256 examples per
shard, yielding 27 shards, and adjust only from measured throughput.

Before submission, restore the persistent-storage gate: at least 8 GB free and less
than 85% utilization. Stage and hash the pinned model cache on every eligible node, or
constrain jobs to a node whose cache has been verified. A missing node-local cache is a
hard scheduling error, not permission to download an untracked replacement.

**Step 2: Submit the array at verified account concurrency**

Example only if four concurrent GPUs are allowed:

```bash
sbatch --array=0-26%4 --export=ALL,CONFIG=configs/paper/gsm8k.yaml scripts/turing_collect_array.sh
```

**Step 3: Monitor jobs and disk**

Track queue, Slurm states, shard progress, GPU telemetry, scratch usage, and errors.
Do not submit training while collection is incomplete.

**Step 4: Merge only after every shard completes**

Submit `turing_merge_collection.sh`. Validate exact train coverage, unique IDs, config
hash, dataset hash, expected shard count, and raw artifact schema.

**Step 5: Build preference datasets**

Generate standard, multilevel, and matched rows. Report group count, pair count, pair
yield, attempt distribution, unresolved rate, and unsafe-guidance rate.

## Task 18: Train and Select GSM8K DPO Models

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/{standard_dpo,multilevel_dpo,multilevel_matched}/`

**Step 1: Run equal-budget validation tuning**

Create separate immutable ledgers for standard, multilevel, and matched DPO. Each
method receives the same 12 candidates formed by learning rate
`{2e-6, 5e-6, 1e-5}` and beta `{0.05, 0.1, 0.3, 0.5}`, one epoch maximum, identical
pilot subsets, identical tuning seeds, and identical promotion budgets. Use the
500-example tuning-development role for ranking. Never inspect test predictions.

**Step 2: Run optimizer finalist checks**

For the top four candidates, execute the fixed fractional comparison of weight decay
`{0.0, 0.01}`, warmup `{5%, 10%}`, and scheduler `{linear, cosine}`. Promote the top
two to full pilot data and two tuning seeds. Evaluate the selected candidate once on
the 247-example confirmation-development role.

**Step 3: Freeze selected hyperparameters**

Write a signed freeze manifest containing selected settings, validation evidence,
complete candidate-ledger hash, tie-break evidence, adapter-selection rule, and test
command.

**Step 4: Run three seeds per selected method**

Train standard, multilevel, and matched LoRA adapters. Log losses, margins, preference
accuracy, throughput, memory, wall time, GPU-hours, and adapter hashes.

**Step 5: Run the shared-profile sensitivity**

Run one seed of all three DPO methods with the same optimizer and DPO beta selected by
the prespecified aggregate validation rule. Report it separately from the independently
tuned primary results.

**Step 6: Generate validation predictions and validate artifacts**

No test job starts until every selected run and validation prediction artifact passes.

## Task 19: Run GSM8K GRPO

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/grpo/`

**Step 1: Run 32-prompt reward and optimizer preflight**

Generate four completions per prompt with original `loss_type="grpo"`, one policy
iteration, clipping epsilon `0.2`, within-group scaling, and truncation masking. Audit
every completion, reward, optimizer field, and gradient update.
Use the isolated frozen vLLM environment so presence penalty `2.0` is actually applied;
non-thinking mode, temperature `1.0`, top-p `1.0`, top-k `20`, balanced final-answer stopping, and the 16,384-token ceiling are explicit.
Start with colocated vLLM at 25% device memory. If and only if the measured preflight
fails the memory gate, run the documented two-GPU server-mode profile and then use that
same profile for all GRPO candidates and seeds. Never omit presence penalty as a hidden
Transformers fallback.

**Step 2: Enforce GRPO gates**

Zero-variance groups at most 50%, truncation at most 5%, evaluator agreement at least
95%, and nonzero gradient/reward signal.

**Step 3: Run deterministic GRPO successive halving**

Screen the 12 candidates formed by learning rate `{2e-6, 5e-6, 1e-5}` and KL beta
`{0.0, 0.001, 0.01, 0.04}` on fixed training subsets. Promote four, then two, using
the same tuning-development and confirmation-development roles and prespecified
invalid-run and tie-break rules. Freeze settings before test evaluation.

**Step 4: Run three seeds and validate adapters**

Record reward components, variance, KL, clipping, entropy, length, throughput, memory,
and GPU-hours.

**Step 5: Run the labeled DAPO sensitivity**

Run one seed with the frozen GRPO optimizer profile but `loss_type="dapo"`. Store it
under `runs/paper/gsm8k/dapo_sensitivity/`; never substitute it for an invalid GRPO run.

## Task 20: Run the Frozen GSM8K Adapted-Method Test Evaluation

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/test/`

**Step 1: Confirm freeze manifests and no prior test marker**

**Step 2: Generate full 1,319-example predictions**

Reuse the already frozen base predictions from Task 15 without regenerating them.
Evaluate standard DPO, multilevel DPO, matched DPO, and original GRPO for every primary
seed. Evaluate the one-seed DAPO sensitivity in a separately labeled artifact. Teacher
guidance is disabled.

**Step 3: Score and validate**

Produce per-example predictions and canonical correctness. Validate complete ID coverage,
no duplicates, no teacher/gold prompt leakage, and adapter/config hashes.

**Step 4: Mark GSM8K complete**

SearchQA work remains blocked until GSM statistics and report pass Task 21.

## Task 21: Produce GSM8K Statistics and Paper Report

**Files:**
- Create: `implementation/src/text_feedback_dpo/statistics.py`
- Create: `implementation/tests/test_statistics.py`
- Modify: `implementation/src/text_feedback_dpo/report.py`
- Create remotely: `reports/paper/gsm8k/`

**Step 1: Write failing statistics tests**

Test bootstrap confidence intervals, paired McNemar inputs, seed aggregation, effect
sizes, Holm correction, and deterministic bootstrap seeds.

**Step 2: Implement and verify locally**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_statistics.py'
```

**Step 3: Generate paper artifacts**

Produce CSV, JSON, LaTeX tables, SVG/PNG/PDF plots, HTML report, failure ledger,
compute table, and exact reproduction commands.

**Step 4: Validate the report and commit statistics code**

Only after report validation may SearchQA begin.

## Task 22: Materialize, Baseline, and Preflight SearchQA-8K

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/data/`
- Create remotely: `runs/paper/searchqa8k/collection-preflight/`

**Step 1: Materialize official-source manifests under Slurm**

**Step 2: Freeze and run the teacher-free SearchQA-8K baseline before collection**

Apply Task 15 to SearchQA-8K: first a manually audited validation micro-preflight,
then full 1,000-example validation and one-time 2,000-example test inference under a
SearchQA-specific freeze. The test result remains descriptive and cannot influence
SearchQA prompts, rewards, collection, tuning, or model selection.

Expected sample counts: 5,000 train, 1,000 validation, 2,000 test. Validate source
split membership, stratification, disjointness, hashes, evidence packaging, answer
aliases, and duplicate audit.

Also materialize 2,000 auxiliary hyperparameter-train and 500 auxiliary
hyperparameter-validation rows from otherwise unused original official rows. Validate
that both auxiliary roles are disjoint from every SearchQA-8K role and are rejected by
final-training and final-evaluation commands.

**Step 3: Run 64-example collection preflight**

Audit every response, slight hint, accumulated guard decision, evidence package, and
evaluator result.

**Step 4: Enforce the same gates plus evidence coverage**

Require answer-alias evidence coverage and report any deterministic truncation. Verify
the already frozen `0.55/0.25/0.10/0.10` SearchQA reward implementation during
preflight; do not tune reward weights from preflight outcomes.

## Task 23: Run Full SearchQA-8K Collection

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/collection/`

Repeat Task 17 with 5,000 training examples. Start with 250 examples per shard and 20
shards, then adjust only from measured preflight throughput. Merge only complete shards,
validate all IDs, and build standard, multilevel, and matched preference datasets.

## Task 24: Train SearchQA-8K DPO and GRPO

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/{standard_dpo,multilevel_dpo,multilevel_matched,grpo}/`

Repeat Tasks 18 and 19 using SearchQA exact match, token F1, answer type, and evidence
support. Use the approved composite GRPO reward. Keep the same LoRA architecture and
three-seed policy. Run tuning pilots on the disjoint 2,000/500 auxiliary pool, then
evaluate the chosen configuration once on the main 1,000-example validation split
before freezing it. Final training uses only the main 5,000-example training split.

## Task 25: Run Frozen SearchQA-8K Adapted-Method Test and Statistics

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/test/`
- Create remotely: `reports/paper/searchqa8k/`

Generate teacher-free predictions for the fixed 2,000-example test subset, validate
coverage and hashes, then report EM, token F1, answer type, evidence support, bootstrap
confidence intervals, paired bootstrap differences, compute, and failures. Label every
artifact and table as `SearchQA-8K`.

## Task 26: Optional One-Seed GSM8K Full-Fine-Tuning Ablation

**Files:**
- Create: `implementation/configs/paper/gsm8k-fullft-standard.yaml`
- Create: `implementation/configs/paper/gsm8k-fullft-multilevel.yaml`
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Modify: `implementation/tests/test_training_profiles.py`

Run only after both LoRA domain studies are complete and only if a preflight proves the
full optimizer state and sequence length fit safely. Run one seed for standard and
multilevel DPO with identical updates. Report this as an ablation, not part of the main
three-seed table. If either method cannot use the identical setup, omit the ablation
rather than use asymmetric settings.

This task is non-blocking and is not part of the paper completion definition. The main
study is LoRA because full fine-tuning would change the compute and optimization budget.

## Task 27: Final Cross-Domain Report and Completion Audit

**Files/Artifacts:**
- Create remotely: `reports/paper/cross-domain/`
- Modify: `docs/design/native_iterative_guidance_dpo.md`

**Step 1: Generate cross-domain tables and figures**

Include base, standard, multilevel, matched, original GRPO, and separately labeled DAPO
sensitivity results, uncertainty, significance, effect sizes, pair distributions,
guidance outcomes, hyperparameter sensitivity, promotion ledgers, compute, and
limitations.

**Step 2: Audit every explicit requirement**

Verify datasets, split hashes, auxiliary tuning-role isolation, no test contamination,
slight-hint policy, LoRA target coverage, optimizer manifests, candidate ledgers,
freeze signatures, all methods, all seeds, adapters, test predictions, statistics,
raw logs, telemetry, reports, Git commits, and reproduction commands.

**Step 3: Verify repository and cluster state**

```bash
cd implementation
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
git diff --check
git status --short --branch
ssh aryama.murthy@turing.iiit.ac.in 'squeue -u "$USER"; df -h "$HOME" /scratch/node01'
```

Expected: all tests pass, repository clean and pushed, all required artifacts present,
no active Slurm jobs, and storage usage recorded.

**Step 4: Update the living design and commit final documentation**

```bash
git add docs/design docs/plans
git commit -m "docs: record paper experiment results"
git push origin agent/qwen35-pretest
```

## Review Checkpoints

Stop and report for review after:

1. Tasks 1-5: data, hint, and evaluator foundations.
2. Tasks 6-13: sharding, training, GRPO, observability, and local verification.
3. Tasks 14-15: GSM8K materialization and preflight.
4. Tasks 16-20: complete GSM8K experiment and paper report.
5. Tasks 21-24: complete SearchQA-8K experiment and paper report.
6. Tasks 25-26: optional full-FT ablation and final completion audit.
