# SearchQA Bootstrap-SFT and Minimal-Intervention DPO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train and evaluate Qwen3-4B-Base end to end with verified student-only bootstrap SFT, Qwen3-32B answer-free minimal interventions, causally valid DPO, and matched SFT/GRPO/DAPO baselines.

**Architecture:** Generate many batched no-hint student trajectories and validate query and response repair regions independently. Use the verified targets for full-parameter SFT, then collect answer-free teacher interventions from the SFT student and construct only same-context student DPO pairs. Every stage is manifest-bound, checkpointed, response-audited, and promoted by strict generation metrics.

**Tech Stack:** Python 3.12, PyTorch 2.13, Transformers 5.13, TRL 1.8, DeepSpeed ZeRO-3, bitsandbytes NF4, pytest, Ruff, Bash, Slurm, Qwen3, SearchQA.

---

### Task 1: Freeze the audited design and status

**Files:**
- Create: `docs/plans/2026-07-13-searchqa-bootstrap-sft-dpo-design.md`
- Create: `docs/plans/2026-07-13-searchqa-bootstrap-sft-dpo.md`
- Existing: `docs/status/2026-07-13-searchqa-current-status.md`

**Steps:**
1. Run `git diff --check` and inspect all three documents.
2. Confirm the documents say no training has run and no teacher-guided trajectory exists.
3. Commit with `git commit -m "docs: plan verified SearchQA bootstrap training"`.

### Task 2: Run and audit the corrected 32-row teacher smoke

**Files:**
- Remote input: `outputs/preflight-v5/prompt-preflight-32.jsonl`
- Remote decision: `outputs/optimization-v5-smoke/generation-optimization-decision.json`
- Remote output: `outputs/collection-v2/smoke/trajectories.jsonl`
- Create: `implementation/scripts/audit_trajectories.py`
- Test: `implementation/tests/test_trajectory_audit_new.py`

**Steps:**
1. Use bounded private teacher context only: compact retrieved records, gold answer, raw attempt, deterministic diagnostics, prior hints, and source count. Never duplicate all 6–99 complete source records and never truncate silently.
2. Freeze teacher output at 96 tokens for the strict at-most-24-word JSON hint. Tokenize every rendered teacher prompt and fail before scale-up if any prompt plus output reserve exceeds 4,096 tokens.
3. Regenerate the sample-bound measured generation decision after every prompt/config/commit change. Validate the 32-row sample, prompt SHA-256 `f76750c597c14f4358df2b3d8fcd60211caa7bf1baa000f3e26de998642fe1b3`, config SHA-256 `080375b466be3cd956f49babb173b011241720dc7946039e96ada975cd41f95b`, and canonical raw-base policy digest.
4. Submit `turing_collect.sh` on node10 with two GPUs, teacher batch 8, student batch 4, four interventions, two smoke siblings, direct student mode, temperature 0.7, top-p 0.9, and the authoritative source-schema hash.
5. Monitor `squeue`, `sacct`, stdout, stderr, GPU telemetry, output cardinality, and cache artifacts until completion.
6. Write a failing test requiring the audit command to emit per-attempt question, gold, query, top sources, raw response, error, teacher hint, retry, sibling, leakage, latency, prompt token count, and eligibility fields.
7. Run `PYTHONPATH=src uv run pytest -q tests/test_trajectory_audit_new.py` and confirm the missing audit implementation fails.
8. Implement `audit_trajectories.py` using canonical trajectory validation; do not infer or repair malformed output.
9. Run the focused test, full suite, Ruff, `python -m py_compile`, shell syntax checks, and `git diff --check`.
10. Generate JSONL, CSV, and HTML smoke audits and manually inspect all 32 rows.
11. Commit with `git commit -m "feat: audit minimal intervention trajectories"`.

### Task 3: Add no-hint bootstrap rollout collection

**Files:**
- Create: `implementation/src/text_feedback_dpo/bootstrap.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Create: `implementation/tests/test_bootstrap_rollouts_new.py`
- Create: `implementation/scripts/turing_bootstrap_rollouts.sh`
- Modify: `implementation/tests/test_turing_runtime_new.py`

**Steps:**
1. Write failing tests for deterministic example/seed expansion, exact cardinality, student/no-hint provenance, per-example seed identity, canonical artifact validation, duplicate rejection, resumable shards, and zero teacher fields.
2. Run `PYTHONPATH=src uv run pytest -q tests/test_bootstrap_rollouts_new.py` and confirm failure because `bootstrap.py` is absent.
3. Implement a bootstrap collector that batches one seed across active examples, calls the existing fixed-retrieval pipeline, and stores candidates grouped by example ID.
4. Add `bootstrap-rollouts` CLI arguments for data, output, model/revision, seed list, query/response batch sizes, decoding controls, prompt/retrieval identities, and cache manifest.
5. Add a one-GPU Slurm launcher that stores shards in node10 scratch and writes only manifests/summaries to home.
6. Ensure interrupted shards resume only after exact cache-manifest validation.
7. Run focused/full tests and static checks.
8. Commit with `git commit -m "feat: collect verified no-hint bootstrap rollouts"`.

### Task 4: Split query and response SFT eligibility

**Files:**
- Modify: `implementation/src/text_feedback_dpo/dataset.py`
- Modify: `implementation/src/text_feedback_dpo/trajectories.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/tests/test_task7_dataset_new.py`
- Create: `implementation/tests/test_bootstrap_sft_new.py`

**Steps:**
1. Write a failing test where a no-hint query retrieves the gold answer at rank 3 but the response is malformed; assert one query SFT row and zero response rows.
2. Write failing tests requiring response rows to remain exact-correct, parse-valid, supported, cited, student-generated, no-hint, untruncated, and same-context.
3. Run the focused tests and confirm the current all-or-nothing `_candidate_base_reason` excludes the valid query.
4. Introduce separate canonical query and response validators. Query validation recomputes BM25 and answer-bearing recall; response validation reuses the strict active-artifact validator.
5. Select query and response candidates independently across seeds while retaining exact prompt/retrieval hashes.
6. Extend reports with unique-example coverage, per-task exclusions, seed distribution, answer-rank distribution, and leakage counts.
7. Preserve the ban on gold/teacher/fabricated completions.
8. Run focused/full tests and commit `feat: bootstrap query and response SFT independently`.

### Task 5: Add continuous response monitoring

**Files:**
- Create: `implementation/src/text_feedback_dpo/monitoring.py`
- Modify: `implementation/src/text_feedback_dpo/report.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Create: `implementation/tests/test_response_monitoring_new.py`
- Create: `implementation/scripts/turing_checkpoint_monitor.sh`

**Steps:**
1. Write failing tests for fixed-canary selection, deterministic rotating samples, checkpoint-to-checkpoint raw response diffs, failure categories, exact artifact links, and HTML escaping.
2. Implement `monitor-responses` to consume canonical data/predictions and emit JSON, JSONL, CSV, and HTML without changing scores.
3. Add a checkpoint monitor that rejects incomplete checkpoints, generates the fixed and rotating samples, and records checkpoint hash plus generation identities.
4. Require manual-review status and notes in promotion manifests; an unchecked report cannot promote a checkpoint.
5. Run tests/static checks and commit `feat: monitor response changes across checkpoints`.

### Task 6: Run the measured bootstrap pool

**Files:**
- Remote root: `/scratch/node10/aryama.murthy/searchqa-dpo/bootstrap-v1/`
- Home summaries: `outputs/bootstrap-v1/`

**Steps:**
1. Deterministically select 2,048 train examples and record IDs/hash without using official validation or test.
2. Freeze eight seeds and shard the pool across available one-GPU jobs.
3. Run a real batch-size probe for 4/8/16; promote a batch only when cardinality, per-example seed identity, strict metrics, memory, and throughput pass.
4. Launch all bootstrap shards with Slurm dependencies and exact merge parity.
5. Monitor raw outputs and GPU telemetry during every shard; stop on systemic empty-output, prompt leakage, or cache mismatch.
6. Merge and build SFT data with gates `query_unique_examples >= 512` and `response_unique_examples >= 256`.
7. Generate and manually inspect the bootstrap response report.
8. If the response gate fails, stop and execute Task 7; otherwise skip Task 7 with an explicit `fallback_reason=prompt_experiment_not_required`.

### Task 7: Run a controlled prompt/scaffold experiment if required

**Files:**
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/src/text_feedback_dpo/batch_generation.py`
- Modify: `implementation/tests/test_cited_responses_new.py`
- Modify: `implementation/tests/test_batch_generation_new.py`

**Steps:**
1. Write failing tests for versioned prompt candidates and explicit fixed-prefix accounting with no post-generation label insertion.
2. Implement only the candidate(s) needed by the observed failure: current prompt, two-example continuation prompt, and explicit `Answer:` prefix candidate.
3. Evaluate candidates on identical 512 train-derived examples and eight seeds.
4. Compare strict success coverage, answer metrics, retrieval, truncation, raw outputs, latency, and throughput.
5. Promote exactly one prompt or fail with `fallback_reason=verified_student_response_coverage_insufficient`.
6. Rebuild prompt hashes and rerun Task 6 under the promoted prompt.
7. Commit `feat: improve base response continuation contract` only if a candidate passes.

### Task 8: Run SFT overfit and 1% pilot

**Files:**
- Modify: `implementation/src/text_feedback_dpo/trainers.py`
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Modify: `implementation/scripts/turing_train.sh`
- Modify: `implementation/tests/test_task7_training_new.py`
- Remote root: `/scratch/node10/aryama.murthy/searchqa-dpo/sft-pilot-v1/`

**Steps:**
1. Write failing tests for task-balanced sampling, full-parameter mode, completion-only loss, checkpoint manifests, and generation-monitor promotion gates.
2. Implement deterministic task balancing and explicit optimizer/checkpoint identities.
3. Run 32 query plus 32 response rows until overfit; require finite decreasing loss and >=95% format/citation success on the overfit set.
4. Verify checkpoint save, optimizer state, RNG state, resume, and identical next-step loss within tolerance.
5. Run one-variable learning-rate pilots on deterministic 1% bootstrap data.
6. Monitor fixed and rotating responses at every stable checkpoint.
7. Promote only a checkpoint improving answer and structural metrics; commit `feat: gate full SearchQA SFT training`.

### Task 9: Correct and run 4-vs-8 GPU scaling probes

**Files:**
- Modify: `implementation/scripts/turing_scaling_probe.sh`
- Modify: `implementation/scripts/turing_probe_runner.py`
- Modify: `implementation/tests/test_turing_scripts_behavior_new.py`

**Steps:**
1. Write a failing test showing that identical per-device accumulation gives different global batches at four and eight GPUs.
2. Update scale decisions to compare equal global batch, seed, data order, optimizer semantics, and token count while allowing world-size-dependent gradient accumulation.
3. Run 20-step four-GPU and eight-GPU probes on homogeneous node10 A100s.
4. Compare global tokens/s, examples/s, utilization, peak memory, finite loss, and correctness hashes.
5. Freeze the faster valid configuration and commit `perf: compare equal-batch SearchQA scaling`.

### Task 10: Run full SFT

**Files:**
- Remote root: `/scratch/node10/aryama.murthy/searchqa-dpo/sft-full-v1/`
- Home summaries: `outputs/sft-full-v1/`

**Steps:**
1. Validate code/config/data/prompt/optimization/scale hashes and free scratch space.
2. Launch the selected four- or eight-GPU full BF16 ZeRO-3 SFT.
3. Log loss every 10 steps; save resumable checkpoints and run fixed/rotating generation monitors at each stable checkpoint cadence.
4. Inspect responses continuously; stop on collapse, leakage, fabricated citations, non-finite loss, or structural regression.
5. Run full train-dev at epoch boundaries and official validation before promotion.
6. Retain best plus latest after checksum validation; clean only superseded scratch checkpoints.

### Task 11: Collect full minimal-intervention trajectories

**Files:**
- Modify: `implementation/scripts/turing_collect.sh`
- Modify: `implementation/scripts/turing_primary_round.sh`
- Modify: `implementation/tests/test_round_script_new.py`
- Remote root: `/scratch/node10/aryama.murthy/searchqa-dpo/trajectories-v1/`

**Steps:**
1. Write failing tests for per-allocation GPU decision freezing, four sibling seeds, shard resume, compact active batches, and exact merge parity.
2. Run a 512-row SFT-student collection pilot and audit every failure category plus a representative sample.
3. Require zero teacher leakage, nonzero hinted resolution, nonzero no-hint sibling success, and nonzero preference eligibility.
4. Launch deterministic two-GPU shards in parallel, one quantized teacher and one BF16 student per worker.
5. Monitor queries, evidence, answers, hints, retries, siblings, latency, throughput, memory, and intervention efficiency.
6. Merge with exact IDs and canonical revalidation; build query/response preference rows.
7. Commit `perf: scale verified minimal intervention collection`.

### Task 12: Run primary DPO

**Files:**
- Modify: `implementation/scripts/turing_train.sh`
- Modify: `implementation/tests/test_task7_training_new.py`
- Remote root: `/scratch/node10/aryama.murthy/searchqa-dpo/dpo-v1/`

**Steps:**
1. Audit every preference for student provenance, chosen correctness, rejected failure, prompt identity, retrieval-context identity, and non-identical completions.
2. Precompute train/eval reference log probabilities and freeze their manifest.
3. Run a 32-pair DPO overfit, checkpoint/resume test, and response monitor.
4. Run a deterministic 1% beta/learning-rate pilot one variable at a time.
5. Run full DPO from the frozen SFT checkpoint using the selected GPU configuration.
6. Promote only on official validation improvement with structural/citation gates intact.

### Task 13: Run SFT, GRPO, and DAPO comparisons

**Files:**
- Modify: `implementation/scripts/turing_comparisons.sh`
- Modify: `implementation/src/text_feedback_dpo/trainers.py`
- Modify: `implementation/tests/test_method_configs_new.py`
- Remote root: `/scratch/node10/aryama.murthy/searchqa-dpo/comparisons-v1/`

**Steps:**
1. Freeze the SFT initialization, datasets, prompts, retrieval, max length, validation protocol, and compute/token budgets.
2. Preserve SFT-only as the no-preference baseline.
3. Run GRPO and DAPO overfit/pilots before full jobs; log every reward component and zero-variance fraction.
4. Stop reward-hacking runs that improve format rewards without answer accuracy or evidence support.
5. Run matched full GRPO/DAPO arms and generation monitors.

### Task 14: Final evaluation, research report, and cleanup

**Files:**
- Modify: `implementation/scripts/turing_finalize_report.sh`
- Modify: `implementation/src/text_feedback_dpo/report.py`
- Modify: `implementation/tests/test_report_new.py`
- Remote report: `report/final/`

**Steps:**
1. Evaluate raw base, SFT, DPO, GRPO, and DAPO on full official validation.
2. Freeze every choice and record the final test authorization manifest.
3. Materialize the official 43,228-row test and evaluate each promoted method exactly once.
4. Compute paired bootstrap confidence intervals and significance for primary metrics.
5. Generate JSON, JSONL, CSV, and HTML with raw samples, failure categories, intervention efficiency, configs, hashes, logs, telemetry, and checkpoint lineage.
6. Audit every design invariant against authoritative artifacts.
7. Remove obsolete artifacts only after preservation/hash checks; retain the reproducible best models and reports.
8. Run the full verification suite and mark the research goal complete only when every required artifact exists.
