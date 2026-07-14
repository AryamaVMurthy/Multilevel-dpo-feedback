# SearchQA Fixed-Retrieval Cited-Reasoning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and execute an end-to-end SearchQA research pipeline in which Qwen3-4B-Base actively searches fixed dataset sources, gives concise cited reasoning and real source links, and is improved with full SFT and causally valid minimal-intervention DPO.

**Architecture:** Preserve aligned SearchQA source records, expose a deterministic per-example BM25 search tool, and use separate batched model calls for query generation and cited answer generation. Parse a strict normal-text response contract, render canonical source metadata outside the model, and construct only same-context, student-generated SFT/DPO targets verified by no-hint sibling rollouts.

**Tech Stack:** Python 3.12, PyTorch, Transformers, TRL, DeepSpeed ZeRO-3, bitsandbytes NF4, pytest, Ruff, Bash, Slurm, Qwen3, SearchQA.

---

### Task 1: Freeze and retire the obsolete run graph

**Files:**
- Modify: `docs/plans/2026-07-13-plain-answer-thinking-design.md`
- Modify: `docs/plans/2026-07-13-plain-answer-thinking.md`
- Remote artifacts: `~/searchqa-dpo/plain-answer-v2/`

**Steps:**
1. Record `squeue`, `sacct`, job dependencies, remote commit, output counts, partial metrics, and GPU telemetry for jobs 13676-13692.
2. Preserve the legacy raw short-answer baseline predictions/metrics under an explicitly labeled archival-baseline directory.
3. Cancel or permanently hold every pending SFT/trajectory/training job using the obsolete short-answer target; do not cancel a still-useful raw-baseline merge/evaluation chain.
4. Mark the two old plan files as superseded by this design without presenting their incomplete runs as final science.
5. Verify no obsolete training job can become runnable, then commit `docs: supersede short-answer SearchQA protocol`.

### Task 2: Preserve aligned SearchQA source records

**Files:**
- Modify: `implementation/src/text_feedback_dpo/searchqa.py`
- Modify: `implementation/src/text_feedback_dpo/dataset.py`
- Modify: `implementation/tests/test_searchqa_data_new.py`

**Steps:**
1. Write failing tests for parallel `snippets`/`titles`/`urls` arrays, stable `SNNN` IDs, empty-snippet filtering without metadata shift, length mismatch failure, missing-title/URL failure, and deterministic fingerprints.
2. Run `PYTHONPATH=src uv run --frozen pytest tests/test_searchqa_data_new.py -q` and confirm failures are caused by absent source-record preservation.
3. Implement strict source-record normalization and materialize `sources`, retaining `snippets` only as an explicitly deprecated derived field if a migration test requires it.
4. Include source schema/version and drop reasons in manifests; never fabricate missing metadata.
5. Run focused tests, full tests, Ruff, and `git diff --check`; commit `feat: preserve SearchQA source provenance`.

### Task 3: Add deterministic fixed-corpus search

**Files:**
- Create: `implementation/src/text_feedback_dpo/retrieval.py`
- Create: `implementation/tests/test_retrieval_new.py`
- Modify: `implementation/src/text_feedback_dpo/config.py`
- Modify: `implementation/configs/searchqa.yaml`

**Steps:**
1. Write failing tests for tokenization, BM25 ranking, stable tie-breaking, top-k validation, empty-query failure, no-gold access, and retrieval metrics at 1/3/5/8.
2. Verify the tests fail because the retrieval module does not exist.
3. Implement a deterministic in-memory BM25 index over one example's fixed source records and return immutable ranked result objects.
4. Add explicit retrieval config (`backend`, `top_k`, `k1`, `b`, schema version); reject unknown backends and missing values.
5. Add structured search timing/query/result logging and corpus/query hashes.
6. Run focused/full verification; commit `feat: search fixed SearchQA sources`.

### Task 4: Define and score the cited-response contract

**Files:**
- Create: `implementation/src/text_feedback_dpo/responses.py`
- Modify: `implementation/src/text_feedback_dpo/scoring.py`
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/src/text_feedback_dpo/report.py`
- Modify: `implementation/tests/test_searchqa_core_new.py`
- Create: `implementation/tests/test_cited_responses_new.py`

**Steps:**
1. Write failing tests for strict `Answer`/`Reasoning`/`Sources` parsing, malformed/duplicate/unknown citation failures, citation coverage, answer/support metrics, and canonical title/URL rendering.
2. Confirm failures come from the old whole-string short-answer scorer.
3. Implement typed parsed responses and a strict line parser with no recovery path.
4. Build separate query and response prompts; retrieved records include stable IDs, title, URL, and snippet, while hints remain answer-free text.
5. Score EM/F1 only on the parsed answer; add structural, retrieval, citation, support, length, and truncation metrics.
6. Render canonical source lists from cited IDs rather than accepting model-generated URLs.
7. Run focused/full verification and commit `feat: evaluate cited SearchQA reasoning`.

### Task 5: Build batched active-search inference

**Files:**
- Modify: `implementation/src/text_feedback_dpo/runtime.py`
- Modify: `implementation/src/text_feedback_dpo/batch_generation.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/src/text_feedback_dpo/preflight.py`
- Modify: `implementation/tests/test_batch_generation_new.py`
- Modify: `implementation/tests/test_thinking_runtime_new.py`
- Modify: `implementation/tests/test_cli_new.py`
- Modify: `implementation/tests/test_preflight_new.py`

**Steps:**
1. Write failing tests for the query-generation -> search -> response-generation pipeline, exact batch cardinality, stable IDs, prompt rebuilding from raw rows, private scratchpads, explicit truncation, and per-stage timing.
2. Reproduce the stale stored-prompt bug: prepared rows retain obsolete prompts while generation silently uses them.
3. Make generation rebuild versioned prompts from raw structured fields; accept preformatted prompts only through a separate explicit command/input schema.
4. Implement active-example compaction and batched query/response calls with configurable batch sizes and one 4,096-token total budget.
5. Emit raw query, ranked search results, parsed response, rendered visible response, private scratchpad metadata, hashes, timing, and explicit error codes.
6. Update preflight gates and sample artifacts for search, citations, and visible reasoning.
7. Verify red-green behavior, run full checks, and commit `feat: generate batched search trajectories`.

### Task 6: Rebuild minimal-hint trajectories and preferences

**Files:**
- Modify: `implementation/src/text_feedback_dpo/feedback.py`
- Modify: `implementation/src/text_feedback_dpo/collection.py`
- Modify: `implementation/src/text_feedback_dpo/trajectories.py`
- Modify: `implementation/src/text_feedback_dpo/preferences.py`
- Modify: `implementation/src/text_feedback_dpo/offline.py`
- Modify: `implementation/tests/test_feedback_new.py`
- Modify: `implementation/tests/test_collection_batch_new.py`
- Modify: `implementation/tests/test_offline_reuse_new.py`
- Modify: `implementation/tests/test_searchqa_core_new.py`

**Steps:**
1. Write failing tests for earliest-region diagnostics, answer-free escalating hints, hinted retries, no-hint sibling verification, repair-region metadata, privilege/scope cost, and future sibling gain.
2. Write failing preference tests requiring identical prompt and retrieval hashes, student provenance, verified no-hint success, non-identical completions, and separate query/response pair types.
3. Update the teacher prompt to include private diagnostics and source records while returning exactly one answer-free JSON hint.
4. Collect batched query/search/response attempts, compact resolved examples, and run deterministic no-hint siblings after hinted success.
5. Rank interventions by sibling gain per hint token and repair scope; store every component.
6. Build query preferences from future gain and response preferences only from identical retrieved contexts.
7. Version cache manifests with retrieval, response, evaluator, prompt, policy, and source-schema hashes.
8. Run focused/full checks and commit `feat: learn from minimal search interventions`.

### Task 7: Build student-generated SFT and RL datasets

**Files:**
- Modify: `implementation/src/text_feedback_dpo/dataset.py`
- Modify: `implementation/src/text_feedback_dpo/trainers.py`
- Modify: `implementation/src/text_feedback_dpo/training.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/tests/test_trainer_contracts_new.py`
- Modify: `implementation/tests/test_method_configs_new.py`
- Modify: `implementation/tests/test_cli_new.py`

**Steps:**
1. Write failing tests for query SFT rows, response SFT rows, student-provenance requirements, no-hint requirements, exact completion token boundaries, and rejection of teacher/fabricated targets.
2. Implement SFT builders from verified successful student trajectories only and report coverage/failure reasons.
3. Update full SFT and DPO lengths for cited responses while preserving 4,096 total tokens and completion-only loss.
4. Add multi-component GRPO/DAPO rewards with separately logged exact-answer, F1, retrieval, citation, support, format, truncation, and verbosity components.
5. Add exact manifest matching for precomputed DPO reference log probabilities.
6. Run focused/full checks and commit `feat: train search and cited-answer policies`.

### Task 8: Harden and optimize Turing orchestration

**Files:**
- Modify: `implementation/scripts/turing_prepare.sh`
- Modify: `implementation/scripts/turing_generate.sh`
- Modify: `implementation/scripts/turing_collect.sh`
- Modify: `implementation/scripts/turing_train.sh`
- Modify: `implementation/scripts/turing_preflight.sh`
- Modify: `implementation/scripts/turing_primary_round.sh`
- Modify: `implementation/scripts/turing_comparisons.sh`
- Modify: `implementation/scripts/turing_finalize_report.sh`
- Modify: `implementation/tests/test_turing_runtime_new.py`
- Modify: `implementation/tests/test_round_script_new.py`

**Steps:**
1. Write failing script tests for single-node four-GPU training, one process per allocation, exact GPU counts, per-stage batch controls, source/schema hashes, checkpoint/resume gates, and structured non-XML logs.
2. Remove XML-formatted shell logging and emit JSONL/plain key-value observability.
3. Require `--nodes=1` for local `torchrun`, derive `nproc_per_node` from allocated GPUs, and fail on allocation mismatch.
4. Add measured SDPA/FlashAttention2, static-cache, compile, packing, worker-count, and batch-size probes; accept optimizations only when outputs match and throughput improves.
5. Use four GPUs for full training and two-GPU student/teacher shard workers for trajectory collection; permit parallel independent shards with deterministic merge checks.
6. Record versions, commit, config, Slurm state, node, GPU telemetry, throughput, peak memory, checkpoints, and explicit fallback reasons.
7. Run shell syntax, focused/full pytest, Ruff, compileall, and `git diff --check`; commit `perf: optimize cited SearchQA training on Turing`.

### Task 9: Run local and Turing preflights

**Files:**
- Outputs: `~/searchqa-dpo/fixed-retrieval-v1/preflight/`

**Steps:**
1. Sync the verified branch without caches, virtualenvs, logs, checkpoints, or generated data.
2. Materialize 32 real train-dev examples and manually inspect aligned source IDs/titles/URLs/snippets.
3. Run raw Qwen3-4B query/search/cited-response inference and inspect question/query/results/answer/reasoning/sources—not metrics alone.
4. Probe Qwen3-32B 4-bit teacher fit and strict answer-free hint output on one GPU.
5. Run 32-example end-to-end collection, SFT build, DPO build, checkpoint save, resume, cache reuse, and artifact validation.
6. Freeze retrieval `k`, prompt/thinking mode, answer length, batch sizes, attention backend, and policy hashes only after measured gates pass.

### Task 10: Run baselines and full SFT

**Files:**
- Run root: `~/searchqa-dpo/fixed-retrieval-v1/`

**Steps:**
1. Generate the complete raw-base active-search validation baseline in parallel shards; merge by exact IDs and compute all answer/retrieval/citation metrics.
2. Present representative correct, incorrect, malformed, unsupported, and retrieval-failure outputs.
3. Collect verified student-success bootstrap trajectories and build query/response SFT data; fail if coverage is below the frozen gate.
4. Run a 32-example full-parameter overfit; require loss convergence and generation-quality improvement.
5. Run a deterministic 1% pilot with frequent checkpoints/evaluation and a tested resume.
6. Launch full four-GPU Qwen3-4B BF16 SFT; monitor losses, generations, throughput, memory, and fixed-subset validation continuously.
7. Promote only the best verified checkpoint; retain best plus latest and clean superseded nonessential checkpoints after hash verification.

### Task 11: Collect trajectories and run primary DPO

**Files:**
- Run root: `~/searchqa-dpo/fixed-retrieval-v1/dpo/`

**Steps:**
1. Launch deterministic two-GPU collection shards using the frozen SFT student and 32B 4-bit teacher.
2. Monitor live query/result/response/hint samples, resolution curves, no-hint sibling gain, teacher leakage, and GPU utilization.
3. Merge trajectories with exact ID/cache parity and audit preference provenance/context hashes.
4. Count unique verified query and response preference prompts after every gate. Treat the 151,277-row materialized train split only as a source reservoir; do not report it as optimizer data.
5. Freeze a trajectory-disjoint 1,000-prompt model-selection validation set and one deterministic training ordering with stable ID and file hashes.
6. Materialize nested 1,000/5,000/10,000/15,000-prompt training ablations and an optional 20,000-prompt arm when coverage permits. If an arm is unavailable, fail with its exact shortfall and scale collection from new train-reservoir examples; never duplicate, pad, fabricate, or silently substitute rows.
7. Target 15,000-20,000 verified prompts for the primary DPO package. Record the active 4,096-example collection as the yield-measurement stage unless its audited prompt cardinality independently satisfies a named arm. Start with two student no-hint siblings per prompt; generate a third or fourth only for prompts lacking a valid contrast after two siblings and only when audited marginal pair yield per GPU-hour is positive. Never count teacher hints or teacher outputs as trajectories.
8. Precompute reference log probabilities once per exact split manifest and reject cross-arm reference reuse when any data hash differs.
9. Run DPO overfit and 1% pilots, then full four- or eight-GPU DPO selected by the measured equal-batch scaling gate, with regular generation validation, checkpoints, and resume support.
10. Iterate one measured variable at a time only when fixed 1,000-prompt validation diagnostics identify a concrete failure mode; stop when locked validation no longer improves.

### Task 12: Run comparisons, final evaluation, and research audit

**Files:**
- Run root: `~/searchqa-dpo/fixed-retrieval-v1/comparisons/`
- Report: `~/searchqa-dpo/fixed-retrieval-v1/report/`

**Steps:**
1. Freeze initialization, nested 1,000/5,000/10,000/15,000 prompt manifests plus the optional 20,000 arm, the trajectory-disjoint 1,000-prompt model-selection validation set, prompts, retrieval, sequence length, and token/compute budgets.
2. Run matched SFT-only, GRPO, and DAPO arms after the primary DPO result is complete.
3. Evaluate raw base, SFT, DPO, GRPO, and DAPO on validation with answer, retrieval, citation, reasoning, malformed-output, latency, and throughput metrics.
4. Freeze all choices, then run the untouched test exactly once per promoted method.
5. Compute bootstrap confidence intervals and paired significance for primary metrics; include failure-category and intervention-efficiency ablations.
6. Generate JSON, JSONL, CSV, and human-readable HTML with links to configs, hashes, logs, samples, checkpoints, and telemetry.
7. Audit every design success criterion against authoritative artifacts, clean obsolete/failed nonessential outputs, and mark complete only when all evidence is present.
