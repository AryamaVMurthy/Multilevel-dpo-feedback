# Paper-Scale GSM8K and SearchQA-8K Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and execute a reproducible paper-grade comparison of standard DPO, multilevel-feedback DPO, pair-budget-matched multilevel DPO, and GRPO on full GSM8K followed by SearchQA-8K using Qwen3.5-2B with a privileged Qwen3.5-9B slight-hint teacher.

**Architecture:** The implementation separates immutable dataset manifests, resumable collection shards, deterministic preference construction, LoRA training, adapter-aware held-out generation, domain evaluation, and paper reporting. Large artifacts and model caches live on node-local Turing scratch; only manifests, metrics, reports, and final adapters are copied to `/home`.

**Tech Stack:** Python 3.12, Hugging Face Datasets/Transformers, Qwen3.5, TRL, PEFT LoRA, PyTorch, TensorBoard, Slurm, JSONL/Zstandard, HTML/SVG reporting.

---

## Execution Rules

- Work in the existing `agent/qwen35-pretest` worktree.
- Follow test-driven development for every behavior change.
- Commit after each task; push only verified commits.
- Never run ML imports, dataset processing, inference, or training on the Turing login node.
- Never use the official test split for pair collection or model selection.
- Never merge a missing or failed shard as an empty shard.
- Do not start SearchQA until the GSM8K completion gate passes.
- Do not start a full job after a failed preflight.
- Before and after each Slurm phase, run `squeue -u "$USER"` and record the result.

## Task 1: Add Paper Experiment Configuration Schema

**Files:**
- Create: `implementation/src/text_feedback_dpo/experiment_config.py`
- Create: `implementation/tests/test_experiment_config.py`
- Create: `implementation/configs/paper/gsm8k.yaml`
- Create: `implementation/configs/paper/searchqa8k.yaml`
- Modify: `implementation/src/text_feedback_dpo/cli.py`

**Step 1: Write failing schema tests**

Test that a paper config requires dataset revision, source counts, split manifest,
seeds, LoRA settings, generation settings, shard size, retry budget, and final-test
freeze flag. Test that unknown keys, missing revisions, overlap-prone split settings,
and a non-2048 completion budget fail explicitly.

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

GSM8K contains source counts `7473/1319`, split counts `6726/747/1319`, seeds, BF16
LoRA rank 16, alpha 32, and the approved sampling settings. SearchQA-8K contains
source counts `99820/13393/27248` and sample counts `5000/1000/2000`.

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
- SearchQA sampling stays inside each official source split;
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

## Task 8: Add Fixed BF16 LoRA Training Profiles

**Files:**
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/tests/test_training.py`
- Create: `implementation/tests/test_training_profiles.py`

**Step 1: Write failing profile tests**

Assert rank 16, alpha 32, dropout 0.05, attention projection targets, BF16, no
quantization, equal profiles across methods, seed propagation, TensorBoard logging,
validation strategy, and explicit effective batch/update budget.

**Step 2: Verify red**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_training_profiles.py'
```

**Step 3: Implement profile-driven training**

Training must write config, dataset manifest hash, parameter counts, trainable parameter
percentage, optimizer/scheduler, package versions, history, best validation checkpoint,
and final adapter. Standard and matched runs must use equal optimizer-update budgets.

**Step 4: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_training*.py'
git add implementation/src/text_feedback_dpo/training.py implementation/src/text_feedback_dpo/cli.py implementation/tests/test_training.py implementation/tests/test_training_profiles.py
git commit -m "feat: add paper BF16 LoRA profiles"
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
generation batch divisibility, max completion length 2048, temperature 1.0, top-p 0.95,
top-k 20, presence penalty 1.5 through supported generation kwargs, completion logging,
and truncated-completion masking.

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
peak memory, pair metrics, evaluator confidence, training metrics, and failure ledger.

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

Collection arrays use one GPU per task. Dataset and model caches use a node-local shared
cache keyed by revision. Job-specific environments and temporary outputs use job scratch.
Critical copies and merges must not use `|| true`.

**Step 4: Test and commit**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_turing_scripts.py'
git add implementation/scripts implementation/tests/test_turing_scripts.py
git commit -m "feat: add paper-scale Turing workflows"
```

## Task 13: Run the Complete Local Verification Gate

**Files:**
- Modify only files required by discovered failures.

**Step 1: Run all tests**

```bash
cd implementation
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
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
zero overlap, pinned revision, and successful duplicate audit.

**Step 4: Record accounting and queue state**

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,AllocTRES,MaxRSS,NodeList
squeue -u "$USER"
```

## Task 15: Run GSM8K Collection Preflight

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/collection-preflight/`

**Step 1: Submit 64-train-example collection**

Use the first deterministic preflight shard. Allow three slight-hint rounds.

**Step 2: Inspect every raw attempt and hint**

Manually audit all 64 examples, all hints, accumulated guard inputs, and evaluator
decisions. Record corrections in an audit JSONL, never by editing raw artifacts.

**Step 3: Enforce gates**

Require evaluator parse success at least 99%, audit agreement at least 95%, nonzero
pair yield, no leakage, memory below 90%, and truncation at most 5%.

**Step 4: Stop on failure**

If a gate fails, capture exact artifacts, reproduce the smallest failing example, add a
failing local test, fix, verify, commit, push, and rerun only the preflight.

## Task 16: Run Full GSM8K Collection and Merge

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/collection/`

**Step 1: Estimate shard time from preflight**

Choose the largest shard size projected below three hours. Start with 256 examples per
shard, yielding 27 shards, and adjust only from measured throughput.

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

## Task 17: Train and Select GSM8K DPO Models

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/{standard_dpo,multilevel_dpo,multilevel_matched}/`

**Step 1: Run equal-budget validation tuning**

Each method receives the same grid: beta `{0.05, 0.1}`, learning rate
`{2e-6, 5e-6}`, one epoch maximum, identical seeds for tuning, and validation-only
selection. Never inspect test predictions.

**Step 2: Freeze selected hyperparameters**

Write a signed freeze manifest containing selected settings, validation evidence,
adapter-selection rule, and test command.

**Step 3: Run three seeds per selected method**

Train standard, multilevel, and matched LoRA adapters. Log losses, margins, preference
accuracy, throughput, memory, wall time, GPU-hours, and adapter hashes.

**Step 4: Generate validation predictions and validate artifacts**

No test job starts until every selected run and validation prediction artifact passes.

## Task 18: Run GSM8K GRPO

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/grpo/`

**Step 1: Run 32-prompt reward preflight**

Generate four completions per prompt. Audit every completion and reward.

**Step 2: Enforce GRPO gates**

Zero-variance groups at most 50%, truncation at most 5%, evaluator agreement at least
95%, and nonzero gradient/reward signal.

**Step 3: Tune on validation with the same budget policy**

Freeze GRPO settings before test evaluation.

**Step 4: Run three seeds and validate adapters**

Record reward components, variance, KL, clipping, entropy, length, throughput, memory,
and GPU-hours.

## Task 19: Run the Frozen GSM8K Test Evaluation

**Files/Artifacts:**
- Create remotely: `runs/paper/gsm8k/test/`

**Step 1: Confirm freeze manifests and no prior test marker**

**Step 2: Generate full 1,319-example predictions**

Evaluate base, standard DPO, multilevel DPO, matched DPO, and GRPO for every primary
seed. Teacher guidance is disabled.

**Step 3: Score and validate**

Produce per-example predictions and canonical correctness. Validate complete ID coverage,
no duplicates, no teacher/gold prompt leakage, and adapter/config hashes.

**Step 4: Mark GSM8K complete**

SearchQA work remains blocked until GSM statistics and report pass Task 20.

## Task 20: Produce GSM8K Statistics and Paper Report

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

## Task 21: Materialize and Preflight SearchQA-8K

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/data/`
- Create remotely: `runs/paper/searchqa8k/collection-preflight/`

**Step 1: Materialize official-source manifests under Slurm**

Expected sample counts: 5,000 train, 1,000 validation, 2,000 test. Validate source
split membership, stratification, disjointness, hashes, evidence packaging, answer
aliases, and duplicate audit.

**Step 2: Run 64-example collection preflight**

Audit every response, slight hint, accumulated guard decision, evidence package, and
evaluator result.

**Step 3: Enforce the same gates plus evidence coverage**

Require answer-alias evidence coverage and report any deterministic truncation. Freeze
the SearchQA reward only after preflight audit.

## Task 22: Run Full SearchQA-8K Collection

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/collection/`

Repeat Task 16 with 5,000 training examples. Start with 250 examples per shard and 20
shards, then adjust only from measured preflight throughput. Merge only complete shards,
validate all IDs, and build standard, multilevel, and matched preference datasets.

## Task 23: Train SearchQA-8K DPO and GRPO

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/{standard_dpo,multilevel_dpo,multilevel_matched,grpo}/`

Repeat Tasks 17 and 18 using SearchQA exact match, token F1, answer type, and evidence
support. Use the approved composite GRPO reward. Keep the same LoRA architecture and
three-seed policy. Tune on the 1,000-example validation subset only.

## Task 24: Run Frozen SearchQA-8K Test and Statistics

**Files/Artifacts:**
- Create remotely: `runs/paper/searchqa8k/test/`
- Create remotely: `reports/paper/searchqa8k/`

Generate teacher-free predictions for the fixed 2,000-example test subset, validate
coverage and hashes, then report EM, token F1, answer type, evidence support, bootstrap
confidence intervals, paired bootstrap differences, compute, and failures. Label every
artifact and table as `SearchQA-8K`.

## Task 25: Optional One-Seed GSM8K Full-Fine-Tuning Ablation

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

## Task 26: Final Cross-Domain Report and Completion Audit

**Files/Artifacts:**
- Create remotely: `reports/paper/cross-domain/`
- Modify: `docs/design/native_iterative_guidance_dpo.md`

**Step 1: Generate cross-domain tables and figures**

Include base, standard, multilevel, matched, and GRPO results, uncertainty, significance,
effect sizes, pair distributions, guidance outcomes, compute, and limitations.

**Step 2: Audit every explicit requirement**

Verify datasets, split hashes, no test contamination, slight-hint policy, all methods,
all seeds, adapters, test predictions, statistics, raw logs, telemetry, reports, Git
commits, and reproduction commands.

**Step 3: Verify repository and cluster state**

```bash
cd implementation
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
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
