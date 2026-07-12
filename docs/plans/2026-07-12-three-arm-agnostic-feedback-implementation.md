# Three-Arm Agnostic Feedback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fragile evaluator JSON with a strict verdict token and add three independently frozen, thread-agnostic teacher-feedback pilot arms.

**Architecture:** The evaluator emits only `CORRECT` or `WRONG`; deterministic answer checks remain a separate consistency gate using the student response. The privileged teacher emits one `<student_feedback>` block under an explicit error-only, hint-only, or error-and-hint policy. Only parsed, independently reviewed feedback enters a retry prompt, and each policy is part of the immutable collection protocol.

**Tech Stack:** Python 3.11, pytest/unittest, YAML paper configs, existing Qwen3 Transformers provider, Slurm shell scripts.

---

### Task 1: Replace evaluator JSON with an explicit verdict token

**Files:**
- Modify: `implementation/src/text_feedback_dpo/evaluators.py`
- Modify: `implementation/tests/test_native_pipeline.py`
- Modify: `implementation/tests/test_evaluation.py`

1. Write tests requiring exactly `CORRECT` or `WRONG` and rejecting prose, JSON, empty output, and unknown tokens.
2. Run the focused tests with `PYTHONPATH=src uv run --frozen pytest tests/test_native_pipeline.py -q` and confirm the new tests fail for the expected contract mismatch.
3. Replace the evaluator prompt and parser with the single-token contract. Preserve raw generations, retries, timing, and parse failures. Feed the original student response to the existing domain evaluator rather than asking the judge to serialize an answer.
4. Run the focused tests and confirm they pass.
5. Commit the evaluator contract change.

### Task 2: Add strict tagged feedback and three policies

**Files:**
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/src/text_feedback_dpo/evaluators.py`
- Modify: `implementation/tests/test_prompts.py`
- Modify: `implementation/tests/test_native_pipeline.py`

1. Write tests for `error_only`, `hint_only`, and `error_and_hint` prompts, including the answer-leakage prohibition and thread-agnostic wording rules.
2. Write parser tests for exactly one non-empty `<student_feedback>` block and explicit rejection of missing, duplicate, nested, or surrounding content.
3. Run focused tests and confirm the new cases fail.
4. Implement the policy enum validation, prompts, retry-review wording, and strict tagged parser without repair or defaults.
5. Run focused tests and confirm they pass.
6. Commit the feedback-contract change.

### Task 3: Bind policy to collection provenance and student prompts

**Files:**
- Modify: `implementation/src/text_feedback_dpo/experiment_config.py`
- Modify: `implementation/src/text_feedback_dpo/collection.py`
- Modify: `implementation/src/text_feedback_dpo/prompts.py`
- Modify: `implementation/configs/paper/math.yaml`
- Modify: `implementation/configs/paper/gsm8k.yaml`
- Modify: `implementation/configs/paper/searchqa8k.yaml`
- Modify: `implementation/tests/test_experiment_config.py`
- Modify: `implementation/tests/test_collection.py`
- Modify: `implementation/tests/test_prompts.py`

1. Write failing tests requiring an explicit `collection.feedback_policy` and verifying it changes the protocol hash.
2. Write failing tests proving retry prompts contain only general approved advice and contain no chat-history language.
3. Add the mandatory policy field, parse teacher output before review, store raw and parsed forms, and bind the policy into the protocol manifest.
4. Update existing configs explicitly; do not introduce a default.
5. Run focused config, collection, and prompt tests.
6. Commit the provenance integration.

### Task 4: Add three immutable pilot configurations and paired-run validation

**Files:**
- Create: `implementation/configs/pilots/math-feedback-error-only.yaml`
- Create: `implementation/configs/pilots/math-feedback-hint-only.yaml`
- Create: `implementation/configs/pilots/math-feedback-error-and-hint.yaml`
- Create: `implementation/scripts/validate_feedback_pilot_configs.py`
- Modify: `implementation/tests/test_experiment_config.py`
- Modify: `implementation/tests/test_turing_scripts.py`

1. Write failing tests that load all three configs and assert identical models, revisions, decoding, data, seeds, and budgets with only experiment ID and feedback policy differing.
2. Add the three explicit configs and a fail-fast paired-protocol validator.
3. Run the focused tests and validator.
4. Commit the pilot configs.

### Task 5: Verify locally and prepare the controlled cluster pilot

**Files:**
- Modify: `docs/status/turing-usage-log.md`

1. Run `PYTHONPATH=src uv run --frozen pytest -q`.
2. Run `uv run --frozen ruff check .`.
3. Run `PYTHONPATH=src uv run --frozen python -m compileall -q src tests scripts`.
4. Run shell syntax checks and `git diff --check`.
5. Manually inspect representative prompts and parsed outputs for all three policies.
6. Record the exact source commit, verification evidence, and cluster prerequisites. Do not submit Turing jobs without current authorization and a clean remote checkout.

### Task 6: Execute and audit the paired 16-example pilot

1. Re-audit Turing authorization, Slurm account, storage, checkout commit, and GPU allocation.
2. Submit one policy at a time over the identical frozen 16-example set.
3. Preserve raw outputs, reviewer results, generation metadata, hashes, and failures.
4. Stop on any malformed output, leakage, mixed protocol, missing artifact, or reviewer disagreement.
5. Produce a comparison artifact covering reviewer acceptance, first-retry accuracy, malformed rate, token usage, latency, and unresolved rate.
6. Manually audit all 48 feedback items before recommending any policy.
