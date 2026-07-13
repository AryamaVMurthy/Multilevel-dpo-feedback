# SearchQA Plain-Answer Thinking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the invalid XML student contract with plain short answers, add explicit teacher and student thinking modes, and execute a reproducible full-finetuning SearchQA comparison through final test reporting.

**Architecture:** Keep the existing SearchQA, Slurm, offline-cache, and TRL infrastructure, but replace XML student parsing with direct answer scoring and JSON-only internal teacher hints. Add a two-pass private student scratchpad mode, strict cache manifests, staged baseline/SFT/DPO gates, and dependency-safe Turing submission scripts.

**Tech Stack:** Python 3.12, PyTorch, Transformers 5.13, TRL 1.8, DeepSpeed ZeRO-3, bitsandbytes NF4, pytest, Ruff, Bash, Slurm, Qwen3, SearchQA.

---

### Task 1: Replace the student answer contract

**Files:**
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/src/text_feedback_dpo/scoring.py`
- Modify: `implementation/src/text_feedback_dpo/dataset.py`
- Delete: `implementation/src/text_feedback_dpo/formatting.py`
- Modify: `implementation/tests/test_searchqa_core_new.py`
- Modify: `implementation/tests/test_searchqa_data_new.py`
- Modify: `implementation/tests/test_artifacts_new.py`

**Steps:**
1. Write failing tests asserting a plain `Evidence/Question/Answer` prompt, direct plain-answer EM/F1, and prompt-completion SFT rows.
2. Run the focused tests and confirm they fail because XML remains required.
3. Implement plain prompts, strict answer validation, and `prompt`/`completion` SFT rows.
4. Remove XML student parsing and all XML completion construction.
5. Run focused tests, the complete test suite, Ruff, and `git diff --check`.
6. Commit `feat: use plain SearchQA student answers`.

### Task 2: Add internal JSON teacher hints and escalation

**Files:**
- Create: `implementation/src/text_feedback_dpo/feedback.py`
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/src/text_feedback_dpo/collection.py`
- Modify: `implementation/src/text_feedback_dpo/trajectories.py`
- Modify: `implementation/tests/test_collection_batch_new.py`
- Modify: `implementation/tests/test_searchqa_core_new.py`

**Steps:**
1. Write failing tests for one-field JSON hints, maximum hint length, answer leakage rejection, cumulative hints, escalation levels, and unresolved trajectories.
2. Verify failures come from the old XML feedback parser.
3. Implement typed JSON feedback parsing and normalized alias leakage checks.
4. Update sequential and batched collection to emit one student answer per attempt and one teacher hint per failed attempt.
5. Ensure empty answers remain truthful empty failures in trajectory metadata without inventing a DPO completion.
6. Run focused and full tests, then commit `feat: add minimal JSON teacher hints`.

### Task 3: Add teacher native thinking and student two-pass thinking

**Files:**
- Modify: `implementation/src/text_feedback_dpo/runtime.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Modify: `implementation/configs/searchqa.yaml`
- Create: `implementation/tests/test_thinking_runtime_new.py`
- Modify: `implementation/tests/test_cli_new.py`

**Steps:**
1. Write failing tokenizer-fake tests for Qwen chat-template `enable_thinking=True`, final-content extraction, bounded private scratchpads, and answer-only second-pass output.
2. Verify the runtime lacks thinking-mode APIs.
3. Implement separate base-student completion prompts and post-trained teacher chat-template rendering.
4. Implement direct and two-pass student generation modes with explicit metadata and no scratchpad in scored responses.
5. Add CLI/config flags for thinking mode, scratchpad length, answer length, and teacher final-hint length.
6. Run tests and commit `feat: add explicit Qwen thinking modes`.

### Task 4: Build valid DPO and RL datasets

**Files:**
- Modify: `implementation/src/text_feedback_dpo/preferences.py`
- Modify: `implementation/src/text_feedback_dpo/trainers.py`
- Modify: `implementation/src/text_feedback_dpo/offline.py`
- Modify: `implementation/tests/test_searchqa_core_new.py`
- Modify: `implementation/tests/test_offline_reuse_new.py`
- Modify: `implementation/tests/test_method_configs_new.py`

**Steps:**
1. Write failing tests that chosen/rejected are plain answers, chosen is student-generated exact-correct, prompts are hint-free, and empty failures do not become fabricated completions.
2. Add cache-key tests covering model revisions, prompt version, thinking mode, decoding, intervention policy, and seed.
3. Implement deterministic preference rows and strict cache manifests.
4. Update SFT to prompt-completion completion-only loss; update DPO, GRPO, and DAPO lengths and rewards for plain answers.
5. Add exact-match-primary and bounded-F1 reward tests.
6. Run tests and commit `feat: train preferences on plain answers`.

### Task 5: Add prompt preflight, sample review, and metric gates

**Files:**
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Create: `implementation/src/text_feedback_dpo/preflight.py`
- Create: `implementation/tests/test_preflight_new.py`
- Modify: `implementation/src/text_feedback_dpo/report.py`
- Modify: `implementation/src/text_feedback_dpo/comparison.py`

**Steps:**
1. Write failing tests for deterministic train-derived prompt-development samples and response-quality summaries.
2. Implement prompt-variant and direct-versus-thinking evaluation without touching official test data.
3. Emit question/gold/response samples, EM, F1, nonempty rate, copying rate, truncation rate, and answer-length summaries.
4. Add promotion gates for full baseline and SFT stages.
5. Run tests and commit `feat: gate runs on answer-quality preflight`.

### Task 6: Harden Slurm scripts and cleanup workflow

**Files:**
- Modify: `implementation/scripts/turing_generate.sh`
- Modify: `implementation/scripts/turing_collect.sh`
- Modify: `implementation/scripts/turing_train.sh`
- Modify: `implementation/scripts/turing_primary_round.sh`
- Modify: `implementation/scripts/turing_comparisons.sh`
- Modify: `implementation/scripts/turing_preflight.sh`
- Create: `implementation/scripts/turing_prompt_preflight.sh`
- Create: `implementation/scripts/turing_retire_invalid_run.sh`
- Modify: `implementation/tests/test_turing_runtime_new.py`
- Modify: `implementation/tests/test_round_script_new.py`

**Steps:**
1. Write failing script-content tests for 32-token answers, teacher thinking, 4-bit NF4 teacher, explicit fallback reasons, source-root checks, and safe invalid-run retirement.
2. Update scripts to use the corrected project root, conservative `uv` settings, immutable run roots, structured logs, and periodic GPU telemetry.
3. Require exact row/ID parity before merge/evaluation and complete checkpoint manifests before dependent jobs start.
4. Add an archive-then-delete invalid-run script that rejects paths outside the exact superseded run root.
5. Run shell syntax tests, pytest, Ruff, compileall, and `git diff --check`.
6. Commit `feat: orchestrate plain-answer Turing runs`.

### Task 7: Update documentation and run manifests

**Files:**
- Modify: `README.md`
- Modify: `implementation/README.md`
- Replace: `docs/plans/2026-07-13-searchqa-primary-and-comparisons.md`
- Modify: `implementation/configs/searchqa.yaml`

**Steps:**
1. Remove every claim that students generate XML.
2. Document plain answers, private thinking, post-trained instruct teacher, quantized teacher inference, full BF16 student updates, cache keys, validation cadence, and fallback policy.
3. Search the active tree for obsolete XML student contracts and fail if any remain outside archived history.
4. Run all verification commands and commit `docs: finalize plain-answer SearchQA protocol`.

### Task 8: Retire the remote invalid run and sync the verified branch

**Files:**
- Remote archive: `~/searchqa-dpo/invalid-runs/20260713-xml-contract/`
- New remote checkout: `~/multilevel-feedback-dpo-plain-answer/`

**Steps:**
1. Record current Slurm states, source hashes, artifact counts, and representative invalid responses.
2. Cancel only the XML run job graph after confirming every job ID and dependency.
3. Run the verified retirement script; retain logs/manifests/samples and delete derived XML predictions/SFT artifacts.
4. Sync the new branch without caches, virtualenvs, logs, or outputs.
5. Run remote compile, unit, shell, config, and package-import checks.
6. Record quota before and after cleanup.

### Task 9: Run corrected prompt and model preflights

**Files:**
- Outputs: `~/searchqa-dpo/plain-answer-v2/preflight/`

**Steps:**
1. Submit a one-GPU Qwen3-4B-Base prompt/direct-thinking pilot on train-derived examples.
2. Verify CUDA use, output cardinality, response samples, EM/F1, and no XML student output.
3. Present sample question/gold/response rows before full baseline.
4. Submit a one-GPU 4-bit Qwen3-32B native-thinking hint probe.
5. Use 14B only after a recorded 32B OOM/runtime failure.
6. Freeze prompt, thinking mode, and decoding hashes.

### Task 10: Launch baseline, SFT, primary DPO, comparisons, and report

**Files:**
- Run root: `~/searchqa-dpo/plain-answer-v2/`

**Steps:**
1. Generate and evaluate the complete raw-base validation baseline, then test baseline for final reporting.
2. Build plain prompt-completion SFT data and run the 32-example overfit and 1% pilots.
3. Launch full 4B BF16 SFT with regular checkpoints, subset generation validation, epoch-boundary full validation, and resume support.
4. Collect training and validation trajectories with the quantized thinking teacher, build preferences, and verify offline reuse.
5. Run full DPO rounds with promotion/rollback gates.
6. Run SFT, GRPO, and DAPO comparisons from the same SFT checkpoint and comparable budget.
7. Freeze configurations, run untouched final test evaluations, bootstrap comparisons, and generate JSON/CSV/HTML reports.
8. Audit every explicit requirement and only then mark the experiment complete.
