# Textual Feedback DPO On Turing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the V1 textual-feedback DPO prototype from `multilevel-feedback-dpo/v1_textual_feedback_dpo_design.pdf` end to end on the IIIT Turing GPU cluster.

**Architecture:** Build an offline pipeline that generates structured Qwen3-4B student rollouts, executes controlled tools, scores Search-QA and Math outputs, asks a frozen stronger teacher for textual feedback plus corrected rollouts, filters useful preference pairs, and trains a LoRA/QLoRA DPO adapter with a small format-SFT stabilizer. Turing is used only for GPU-backed rollout, teacher generation, training, and evaluation; login-node work is limited to code, config, sync, and Slurm submission. `/home` stores source, configs, logs, summaries, and final adapters; `/scratch` stores model caches, temporary datasets, retrieval indexes, and large intermediate artifacts.

**Tech Stack:** Turing Slurm `u22`, CUDA 12.4, Python 3.12 via `uv`, PyTorch, Hugging Face Transformers, Datasets, Evaluate, TRL `DPOTrainer`/`DPOConfig`, PEFT LoRA/QLoRA, Accelerate, FAISS or Pyserini/BM25 retrieval, HotpotQA, GSM8K, optional NQ and MATH, JSONL observability, HTML reporting.

---

## Grounding Notes

- Current date: 2026-07-09.
- PDF source: `multilevel-feedback-dpo/v1_textual_feedback_dpo_design.pdf`, 14 pages, generated 2026-07-09.
- Turing local docs source: `instructions/turing_instructions.md`.
- Current TRL docs confirm explicit preference rows with `prompt`, `chosen`, `rejected`, `DPOConfig(beta=0.1)`, and PEFT LoRA via `peft_config`.
- DeepWiki repository grounding for `huggingface/trl` also confirms `prompt`/`chosen`/`rejected` rows and `LoraConfig` passed to `DPOTrainer`.
- Hugging Face search on 2026-07-09 shows `Qwen/Qwen3-4B-Instruct-2507` exists and the base `Qwen/Qwen3-32B` exists. Treat the exact teacher model as a config choice, not a hidden default.

## Non-Negotiable Rules

- Do not train, generate rollouts, run teacher inference, build retrieval indexes, or evaluate models on the login node.
- Do not use live web search in V1. Use controlled retrieval over a fixed local corpus.
- Do not fabricate tool observations. If a `<tool>` block cannot be parsed or executed, fail the example with an explicit error record.
- Do not include teacher feedback in the main DPO prompt. The DPO prompt is only the original problem plus format/tool instructions.
- Do not silently keep malformed corrected rollouts. Filtering must require valid brackets and verification in `<reflect>`.
- Do not add GRPO, PPO, online RL, memory tools, SWE/OpenHands tasks, focus weights, or multiple DPO losses in V1.
- Do not silently choose a Slurm account, CUDA version, model id, dataset split, or fallback teacher. Missing config must raise an explicit error.
- Every job must record config, git or source snapshot metadata, Slurm job id, node, GPU, package versions, random seed, dataset sizes, metrics, and output paths.

## Repository Layout To Create

This workspace is not currently a git repository. Before implementation, either initialize a new repo in `multilevel-feedback-dpo/implementation` or clone the intended remote repo there.

```text
multilevel-feedback-dpo/implementation/
  pyproject.toml
  uv.lock
  README.md
  configs/
    smoke.yaml
    dry_run.yaml
    v1_real.yaml
    eval_e0_base.yaml
    eval_e1_prompt.yaml
    eval_e2_sft.yaml
    eval_e3_dpo_sft.yaml
    eval_e4_no_verification.yaml
  scripts/
    setup_turing.sh
    run_generate_pairs_turing.sh
    run_train_turing.sh
    run_eval_turing.sh
    submit_v1_pipeline.sh
    sync_to_turing.sh
  src/text_feedback_dpo/
    __init__.py
    cli.py
    config.py
    prompts.py
    schema.py
    parsing.py
    tools/
      __init__.py
      search.py
      python_sandbox.py
    data/
      __init__.py
      datasets.py
      retrieval_corpus.py
      pair_builder.py
      io.py
    models/
      __init__.py
      generation.py
      teacher.py
      train_dpo.py
      train_sft.py
    eval/
      __init__.py
      scoring.py
      qa_metrics.py
      math_metrics.py
      format_metrics.py
      run_eval.py
    observability.py
    report.py
  tests/
    test_parsing.py
    test_prompts.py
    test_scoring.py
    test_pair_filtering.py
    test_python_sandbox.py
    test_config.py
```

## Phase 0: Create Implementation Repo And Environment

### Task 0.1: Create or clone the implementation repo

Run locally:

```bash
cd /home/aryamavmurthy/work/SLM-Research/multilevel-feedback-dpo
mkdir -p implementation
cd implementation
git init
```

Expected:

```text
Initialized empty Git repository
```

If a remote already exists, clone that instead and do not run `git init`.

### Task 0.2: Create `pyproject.toml`

**Files:**
- Create: `multilevel-feedback-dpo/implementation/pyproject.toml`

Use explicit dependencies. Do not leave package versions unconstrained during cluster runs.

```toml
[project]
name = "text-feedback-dpo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "accelerate",
  "bitsandbytes",
  "datasets",
  "evaluate",
  "faiss-cpu",
  "jinja2",
  "numpy",
  "pandas",
  "peft",
  "pyyaml",
  "regex",
  "rich",
  "safetensors",
  "scikit-learn",
  "torch",
  "transformers",
  "trl",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[project.scripts]
tfdpo = "text_feedback_dpo.cli:main"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Task 0.3: Verify local test shell

Run:

```bash
cd /home/aryamavmurthy/work/SLM-Research/multilevel-feedback-dpo/implementation
uv sync --extra dev
uv run pytest -q
```

Expected initially:

```text
no tests ran
```

Commit:

```bash
git add pyproject.toml uv.lock
git commit -m "chore: initialize textual feedback dpo project"
```

## Phase 1: Configuration And Fail-Fast Validation

### Task 1.1: Write config schema tests

**Files:**
- Create: `tests/test_config.py`
- Create: `src/text_feedback_dpo/config.py`

Test:

```python
import pytest

from text_feedback_dpo.config import load_config


def test_missing_teacher_model_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("student_model: Qwen/Qwen3-4B-Instruct-2507\n", encoding="utf-8")

    with pytest.raises(ValueError, match="teacher_model"):
        load_config(path)
```

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected: fail because `load_config` does not exist.

### Task 1.2: Implement strict config loading

Implementation requirements:

- Load YAML into dataclasses or typed dictionaries.
- Required keys:
  - `student_model`
  - `teacher_model`
  - `output_root`
  - `seed`
  - `domains`
  - `datasets`
  - `generation`
  - `teacher_generation`
  - `dpo`
  - `sft`
  - `slurm`
- Reject unknown top-level keys.
- Reject missing `slurm.account`; do not default to a faculty account.
- Resolve all paths explicitly.
- Emit a structured config validation log.

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected: pass.

Commit:

```bash
git add src/text_feedback_dpo/config.py tests/test_config.py
git commit -m "feat: add strict experiment config loading"
```

## Phase 2: Prompt Templates And Bracket Parser

### Task 2.1: Write parser tests

**Files:**
- Create: `tests/test_parsing.py`
- Create: `src/text_feedback_dpo/schema.py`
- Create: `src/text_feedback_dpo/parsing.py`

Tests must cover:

- Valid trajectory with `<plan>`, one or more `<think>`, optional `<tool>`, `<reflect>`, `<final>`.
- Missing `<reflect>` fails.
- `<final>` before `<reflect>` fails.
- `<reflect>` without a `Verification:` section fails.
- Duplicate `<final>` fails.
- Malformed teacher output without both `<feedback>` and `<corrected_rollout>` fails.

Run:

```bash
uv run pytest tests/test_parsing.py -q
```

Expected: fail before implementation.

### Task 2.2: Implement parser

Implementation requirements:

- Use structured parsing with explicit tag validation.
- A regex tokenizer for tag boundaries is acceptable only as a lexer; semantic validation must be explicit.
- Return typed objects, not loose strings.
- On failure, raise `TrajectoryParseError` with:
  - `error_code`
  - `message`
  - `raw_text_excerpt`
- Never auto-insert missing tags.

Run:

```bash
uv run pytest tests/test_parsing.py -q
```

Expected: pass.

Commit:

```bash
git add src/text_feedback_dpo/schema.py src/text_feedback_dpo/parsing.py tests/test_parsing.py
git commit -m "feat: parse structured reflective trajectories"
```

### Task 2.3: Implement prompts

**Files:**
- Create: `tests/test_prompts.py`
- Create: `src/text_feedback_dpo/prompts.py`

Tests:

- Student prompt contains required tags and domain-specific verification checklist.
- Search-QA prompt caps branches at 3.
- Math prompt caps branches at 2.
- Teacher prompt includes problem, gold answer, student rollout, result, and exact output contract.
- DPO prompt builder does not include teacher feedback.

Run:

```bash
uv run pytest tests/test_prompts.py -q
```

Expected: fail, then implement and pass.

Commit:

```bash
git add src/text_feedback_dpo/prompts.py tests/test_prompts.py
git commit -m "feat: add student and teacher prompt templates"
```

## Phase 3: Controlled Tools

### Task 3.1: Implement restricted Python verification tool

**Files:**
- Create: `tests/test_python_sandbox.py`
- Create: `src/text_feedback_dpo/tools/python_sandbox.py`

Requirements:

- Allow simple arithmetic and substitution checks.
- Block imports, file access, network access, subprocess, and long-running code.
- Timeout every execution.
- Return explicit `ToolResult(status, output, error_code, elapsed_ms)`.
- On unsafe code, return a failed tool result and record the reason; do not execute.

Run:

```bash
uv run pytest tests/test_python_sandbox.py -q
```

Expected: fail, then implement and pass.

Commit:

```bash
git add src/text_feedback_dpo/tools/python_sandbox.py tests/test_python_sandbox.py
git commit -m "feat: add restricted math verification tool"
```

### Task 3.2: Implement controlled Search-QA retrieval

**Files:**
- Create: `src/text_feedback_dpo/tools/search.py`
- Create: `src/text_feedback_dpo/data/retrieval_corpus.py`

Requirements:

- Use fixed local corpus files built from train/eval datasets or Wikipedia passages.
- Start with BM25 or FAISS over a frozen corpus.
- `search(query, top_k)` must return passage id, title/source, text, and retrieval score.
- If the index path is missing, fail with a clear remediation command.
- Do not call live web search.

Smoke command:

```bash
uv run tfdpo build-index --config configs/smoke.yaml
uv run tfdpo search --config configs/smoke.yaml --query "Where was Marie Curie born?" --top-k 3
```

Expected:

- Index artifacts are created under configured scratch/output path.
- Results include stable passage ids and scores.

Commit:

```bash
git add src/text_feedback_dpo/tools/search.py src/text_feedback_dpo/data/retrieval_corpus.py
git commit -m "feat: add controlled search retrieval"
```

## Phase 4: Dataset Loading And Evaluators

### Task 4.1: Add dataset loaders

**Files:**
- Create: `src/text_feedback_dpo/data/datasets.py`

Requirements:

- Load HotpotQA and GSM8K first.
- Add NQ and MATH only after the smoke pipeline works.
- Support deterministic subset sizes:
  - smoke: 100 HotpotQA + 100 GSM8K
  - dry run: 500 HotpotQA + 500 GSM8K with 100 validation each
  - real V1: 2k HotpotQA + 2k GSM8K + 500 NQ + 500 MATH easy
- Cache datasets in scratch on Turing.
- Save a manifest with dataset name, split, seed, row ids, and fingerprint.

Command:

```bash
uv run tfdpo prepare-data --config configs/smoke.yaml
```

Expected:

- A manifest JSON is written.
- Counts match config exactly.

### Task 4.2: Implement scoring tests

**Files:**
- Create: `tests/test_scoring.py`
- Create: `src/text_feedback_dpo/eval/scoring.py`
- Create: `src/text_feedback_dpo/eval/qa_metrics.py`
- Create: `src/text_feedback_dpo/eval/math_metrics.py`
- Create: `src/text_feedback_dpo/eval/format_metrics.py`

Tests:

- Exact match handles whitespace/case normalization.
- QA F1 handles token overlap.
- Math exact answer accepts equivalent numeric forms where safe.
- Format validity is false if required brackets are missing.
- Verification-present is true only when `<reflect>` contains a real verification section.

Run:

```bash
uv run pytest tests/test_scoring.py -q
```

Expected: fail, then implement and pass.

Commit:

```bash
git add src/text_feedback_dpo/eval tests/test_scoring.py
git commit -m "feat: add qa math and format evaluators"
```

## Phase 5: Rollout Generation And Tool Execution

### Task 5.1: Implement model generation wrapper

**Files:**
- Create: `src/text_feedback_dpo/models/generation.py`

Requirements:

- Load student model from config, defaulting in config files to `Qwen/Qwen3-4B-Instruct-2507`.
- Make generation params explicit:
  - `max_new_tokens: 2048`
  - `temperature: 0.7`
  - `top_p`
  - seed
- Log model id, dtype, device map, GPU name, and memory.
- Fail if CUDA is unavailable during GPU commands unless `allow_cpu_for_unit_tests: true` is explicitly set in a test config.

Command inside Slurm allocation:

```bash
uv run tfdpo generate-one --config configs/smoke.yaml --domain math --index 0
```

Expected:

- Structured rollout JSON is written.
- CUDA is true.

### Task 5.2: Implement tool execution harness

**Files:**
- Create: `src/text_feedback_dpo/data/pair_builder.py`

Requirements:

- Parse `<tool>` requests.
- For Search-QA, execute `search(query)` and insert real observations.
- For Math, execute allowed `python(code)` checks only.
- If a tool request is malformed or unsafe, mark the rollout invalid with explicit error context.
- Never fabricate observations.

Commit:

```bash
git add src/text_feedback_dpo/models/generation.py src/text_feedback_dpo/data/pair_builder.py
git commit -m "feat: generate student rollouts with controlled tools"
```

## Phase 6: Teacher Correction And Preference Pair Filtering

### Task 6.1: Implement teacher generation

**Files:**
- Create: `src/text_feedback_dpo/models/teacher.py`

Requirements:

- Teacher model is required in config, for example `Qwen/Qwen3-32B` or a reachable stronger endpoint.
- If local 32B inference does not fit the allocated GPU, fail with a capacity error and document remediation:
  - request more GPU memory,
  - use quantized teacher,
  - use a configured external inference endpoint,
  - or choose another explicitly configured teacher.
- Teacher temperature starts at `0.2`.
- Parse exactly `<feedback>` and `<corrected_rollout>`.
- Evaluate corrected rollout before storing.

Command inside Slurm allocation:

```bash
uv run tfdpo correct-one --config configs/smoke.yaml --domain qa --index 0
```

Expected:

- Teacher output JSON contains feedback, corrected rollout, corrected score, and parse status.

### Task 6.2: Write pair filtering tests

**Files:**
- Create: `tests/test_pair_filtering.py`

Tests:

- Keep original-wrong/corrected-correct pair.
- Reject corrected rollout with invalid brackets.
- Reject corrected rollout without verification.
- Reject pair where corrected score is not better.
- Metadata retains feedback but DPO row excludes feedback from prompt.

Run:

```bash
uv run pytest tests/test_pair_filtering.py -q
```

Expected: fail, then implement and pass.

Commit:

```bash
git add src/text_feedback_dpo/data/pair_builder.py src/text_feedback_dpo/models/teacher.py tests/test_pair_filtering.py
git commit -m "feat: build filtered dpo preference pairs"
```

## Phase 7: Training Implementation

### Task 7.1: Implement format SFT trainer

**Files:**
- Create: `src/text_feedback_dpo/models/train_sft.py`

Requirements:

- Train only on corrected rollouts as completions.
- Use LoRA/QLoRA from config.
- Save adapter to `outputs/<run_id>/format_sft_adapter`.
- Save trainer state and metrics.
- This is E2 in the evaluation table.

Command:

```bash
uv run tfdpo train-sft --config configs/smoke.yaml --pairs outputs/smoke/pairs/train.jsonl
```

Expected:

- Adapter is written.
- Training metrics JSON exists.

### Task 7.2: Implement DPO trainer

**Files:**
- Create: `src/text_feedback_dpo/models/train_dpo.py`

Requirements:

- Build Hugging Face Dataset rows with explicit:
  - `prompt`
  - `chosen`
  - `rejected`
- Use `trl.DPOTrainer` and `trl.DPOConfig`.
- Set `beta: 0.1`.
- Set learning rate in config, starting `5e-6` to `1e-5`.
- Use PEFT `LoraConfig` via `peft_config`.
- Do not pass separate `ref_model` when PEFT config is used unless explicitly configured and tested.
- Configure `max_prompt_length` and `max_completion_length` from config.
- Save adapter to `outputs/<run_id>/dpo_adapter`.

Important implementation note:

The PDF specifies `L_total = L_DPO + 0.1 * L_FormatSFT`. TRL `DPOTrainer` does not directly implement this combined auxiliary SFT term as a one-line setting. For V1, implement this as two explicit stages unless you choose to subclass the trainer:

1. Format SFT warmup with corrected rollouts.
2. DPO training initialized from the SFT adapter.

If subclassing TRL to add the auxiliary SFT loss, write tests around the loss computation first. Do not hide the approximation.

Command:

```bash
uv run tfdpo train-dpo --config configs/smoke.yaml --pairs outputs/smoke/pairs/train.jsonl
```

Expected:

- DPO adapter is written.
- Metrics include DPO loss, learning rate, beta, and sample count.

Commit:

```bash
git add src/text_feedback_dpo/models/train_sft.py src/text_feedback_dpo/models/train_dpo.py
git commit -m "feat: train format sft and dpo adapters"
```

## Phase 8: Observability And Reporting

### Task 8.1: Add structured logs

**Files:**
- Create: `src/text_feedback_dpo/observability.py`

Every command writes JSONL events with:

- `timestamp`
- `event_name`
- `run_id`
- `slurm_job_id`
- `hostname`
- `config_path`
- `domain`
- `dataset_name`
- `example_id`
- `status`
- `elapsed_ms`
- `error_code`
- `fallback_reason` only when an explicit fallback is intentionally configured

No silent fallback is allowed. Most failures should raise, not fallback.

### Task 8.2: Add report generator

**Files:**
- Create: `src/text_feedback_dpo/report.py`

Inputs:

- `outputs/*/events.jsonl`
- `outputs/*/metrics.json`
- `outputs/*/gpu-*.csv`
- `outputs/*/config.yaml`

Outputs:

- `reports/summary.csv`
- `reports/summary.html`

Columns:

- experiment id
- system: E0/E1/E2/E3/E4
- dataset
- job id
- state
- elapsed
- GPU model
- train pair count
- eval count
- final answer accuracy
- QA EM/F1
- Math exact accuracy
- format validity
- verification-present rate
- verification-valid rate
- average tokens
- branch count
- tool calls
- evidence support rate
- arithmetic error rate
- premature final rate
- adapter path

Commit:

```bash
git add src/text_feedback_dpo/observability.py src/text_feedback_dpo/report.py
git commit -m "feat: add observability and experiment reports"
```

## Phase 9: Turing Slurm Scripts

### Task 9.1: Create setup script

**Files:**
- Create: `scripts/setup_turing.sh`

Script:

```bash
#!/bin/bash
set -euo pipefail

cd "$HOME/text-feedback-dpo"
export PATH="$HOME/.local/bin:$PATH"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy

module load u22/cuda/12.4

uv --version
uv python install 3.12
uv sync --extra dev
uv run python -c "import sys; print(sys.executable); print(sys.version)"
```

Run on Turing login node only for environment setup. If login-node `uv sync` fails from resource limits, rerun inside a short Slurm allocation and record that as the explicit remediation.

### Task 9.2: Create pair generation Slurm script

**Files:**
- Create: `scripts/run_generate_pairs_turing.sh`

Script:

```bash
#!/bin/bash
#SBATCH -A ${TURING_ACCOUNT:?Set TURING_ACCOUNT in sbatch --export}
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"

module load u22/cuda/12.4

cd "$HOME/text-feedback-dpo"
export PATH="$HOME/.local/bin:$PATH"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy

mkdir -p logs outputs/"$RUN_ID"

echo "job_id=${SLURM_JOB_ID}"
echo "job_name=${SLURM_JOB_NAME}"
echo "host=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "config=${CONFIG}"
echo "run_id=${RUN_ID}"
echo "start_time=$(date --iso-8601=seconds)"
nvidia-smi

SCRATCH_DIR="/scratch/$(hostname)/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export TFDPO_SCRATCH_DIR="$SCRATCH_DIR"
export TFDPO_OUTPUT_ROOT="$HOME/text-feedback-dpo/outputs/$RUN_ID"

cp "$CONFIG" "$TFDPO_OUTPUT_ROOT/config.yaml"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!

uv run tfdpo build-pairs --config "$CONFIG" --run-id "$RUN_ID"

kill "$GPU_MONITOR_PID"
cp "logs/gpu-${SLURM_JOB_ID}.csv" "$TFDPO_OUTPUT_ROOT/"
echo "end_time=$(date --iso-8601=seconds)"
```

Submit:

```bash
sbatch --job-name=tfdpo-pairs-smoke \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/smoke.yaml,RUN_ID=smoke-pairs \
  scripts/run_generate_pairs_turing.sh
```

### Task 9.3: Create train Slurm script

**Files:**
- Create: `scripts/run_train_turing.sh`

Same structure as pair generation, with command:

```bash
uv run tfdpo train-pipeline --config "$CONFIG" --run-id "$RUN_ID" --pairs "$PAIRS"
```

Required environment variables:

- `TURING_ACCOUNT`
- `CONFIG`
- `RUN_ID`
- `PAIRS`

Submit:

```bash
sbatch --job-name=tfdpo-train-smoke \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/smoke.yaml,RUN_ID=smoke-train,PAIRS=outputs/smoke-pairs/pairs/train.jsonl \
  scripts/run_train_turing.sh
```

### Task 9.4: Create eval Slurm script

**Files:**
- Create: `scripts/run_eval_turing.sh`

Command:

```bash
uv run tfdpo eval --config "$CONFIG" --run-id "$RUN_ID" --adapter "$ADAPTER"
```

Required environment variables:

- `TURING_ACCOUNT`
- `CONFIG`
- `RUN_ID`
- `ADAPTER`, except for E0/E1 base evaluations where config must explicitly set `adapter: null`.

Commit:

```bash
git add scripts
git commit -m "feat: add turing slurm pipeline scripts"
```

## Phase 10: Configs

### Task 10.1: Create smoke config

**Files:**
- Create: `configs/smoke.yaml`

```yaml
student_model: Qwen/Qwen3-4B-Instruct-2507
teacher_model: Qwen/Qwen3-32B
output_root: outputs
seed: 23

domains:
  - search_qa
  - math

datasets:
  search_qa:
    name: hotpotqa
    split: train
    count: 100
    validation_count: 25
  math:
    name: gsm8k
    split: train
    count: 100
    validation_count: 25

generation:
  max_new_tokens: 2048
  temperature: 0.7
  top_p: 0.95
  max_search_branches: 3
  max_math_branches: 2

teacher_generation:
  max_new_tokens: 3000
  temperature: 0.2
  top_p: 0.95

tools:
  search:
    mode: controlled_bm25
    top_k: 3
  python:
    timeout_seconds: 2

pair_filter:
  require_original_wrong: true
  require_corrected_correct: true
  require_valid_format: true
  require_verification: true

sft:
  enabled: true
  weight_note: "V1 uses SFT warmup as explicit approximation to 0.1 auxiliary FormatSFT unless combined loss is implemented."
  learning_rate: 1.0e-5
  epochs: 1
  per_device_train_batch_size: 1

dpo:
  beta: 0.1
  learning_rate: 5.0e-6
  epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_prompt_length: 4096
  max_completion_length: 2048
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    task_type: CAUSAL_LM

slurm:
  account: null
  partition: u22
  cpus: 16
  gpus: 1
  mem_per_cpu: 4096
```

Important: `slurm.account: null` is intentional in the template. The loader must fail until the user provides a real account or the Slurm script receives `TURING_ACCOUNT`.

### Task 10.2: Create dry run and real configs

**Files:**
- Create: `configs/dry_run.yaml`
- Create: `configs/v1_real.yaml`

Dry run:

- 500 HotpotQA
- 500 GSM8K
- 100 validation each
- same hyperparameters as smoke unless a smoke failure gives evidence for a change.

Real V1:

- 2k HotpotQA
- 2k GSM8K
- 500 NQ
- 500 MATH easy
- top_k can increase from 3 to 5 only after retrieval metrics justify it.

Commit:

```bash
git add configs
git commit -m "feat: add smoke dry-run and v1 configs"
```

## Phase 11: First Turing Smoke Run

### Task 11.1: Sync code to Turing

Run locally:

```bash
rsync -avz \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.ruff_cache' \
  --exclude 'outputs' \
  --exclude 'logs' \
  --exclude 'tmp' \
  /home/aryamavmurthy/work/SLM-Research/multilevel-feedback-dpo/implementation/ \
  <user>@turing.iiit.ac.in:~/text-feedback-dpo/
```

### Task 11.2: Verify interactive GPU runtime

Run on Turing login node:

```bash
sinteractive -c 16 -p u22 -A <your_account> -g 1
```

Inside allocation:

```bash
cd ~/text-feedback-dpo
module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
hostname
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
uv sync --extra dev
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
uv run pytest -q
exit
```

Pass criteria:

- Hostname is a compute node, not the login node.
- `CUDA_VISIBLE_DEVICES` is set.
- `torch.cuda.is_available()` prints `True`.
- Unit tests pass.

### Task 11.3: Submit smoke pair generation

Run on Turing login node:

```bash
cd ~/text-feedback-dpo
mkdir -p logs outputs
sbatch --job-name=tfdpo-pairs-smoke \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/smoke.yaml,RUN_ID=smoke-pairs \
  scripts/run_generate_pairs_turing.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/slurm-tfdpo-pairs-smoke-<jobid>.out
tail -f logs/slurm-tfdpo-pairs-smoke-<jobid>.err
```

Pass criteria:

- Slurm state is `COMPLETED`.
- `outputs/smoke-pairs/config.yaml` exists.
- `outputs/smoke-pairs/events.jsonl` exists.
- `outputs/smoke-pairs/pairs/train.jsonl` exists.
- Pair count is nonzero.
- Invalid parse/tool/correction failures are visible and counted.
- No live web calls occurred.
- No fake tool observations occurred.

### Task 11.4: Manual inspection gate

Run:

```bash
uv run tfdpo inspect-pairs --pairs outputs/smoke-pairs/pairs/train.jsonl --count 20
```

Manual checks:

- Corrected rollouts obey the bracket format.
- `<reflect>` contains real verification.
- Search-QA observations are tied to controlled retrieval.
- Math Python snippets verify rather than solve the whole problem.
- Feedback teaches the repair and does not simply spoon-feed in the feedback block.

If manual inspection fails, stop and fix prompt, parser, teacher, or filtering before training.

## Phase 12: Smoke Training And Evaluation

### Task 12.1: Train smoke E2 and E3

Submit:

```bash
sbatch --job-name=tfdpo-train-smoke \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/smoke.yaml,RUN_ID=smoke-train,PAIRS=outputs/smoke-pairs/pairs/train.jsonl \
  scripts/run_train_turing.sh
```

Pass criteria:

- Format SFT adapter exists if SFT is enabled.
- DPO adapter exists.
- Metrics JSON exists.
- GPU telemetry exists.
- No CUDA/CPU fallback occurred.

### Task 12.2: Evaluate E0 to E4

Systems:

- E0: Base Qwen3-4B normal prompt.
- E1: Base Qwen3-4B bracket prompt, no training.
- E2: Format SFT only.
- E3: DPO initialized from format SFT.
- E4: DPO without mandatory verification filtering.

Submit E3 example:

```bash
sbatch --job-name=tfdpo-eval-e3-smoke \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/eval_e3_dpo_sft.yaml,RUN_ID=eval-e3-smoke,ADAPTER=outputs/smoke-train/dpo_adapter \
  scripts/run_eval_turing.sh
```

Pass criteria:

- Evaluation metrics include all common, Search-QA, and Math metrics from the PDF.
- E3 has measurable results against E1 and E2.
- If E3 is worse, report that honestly and inspect pair quality before scaling.

## Phase 13: Dry Run

### Task 13.1: Generate 500+500 dry-run pairs

Submit:

```bash
sbatch --job-name=tfdpo-pairs-dry \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/dry_run.yaml,RUN_ID=dry-pairs \
  scripts/run_generate_pairs_turing.sh
```

Pass criteria:

- Pair construction completes.
- Pair acceptance rate is reported by domain.
- Teacher correction validity is reported.
- Manual inspection of at least 30 examples passes.

### Task 13.2: Train and evaluate dry run

Submit train:

```bash
sbatch --job-name=tfdpo-train-dry \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/dry_run.yaml,RUN_ID=dry-train,PAIRS=outputs/dry-pairs/pairs/train.jsonl \
  scripts/run_train_turing.sh
```

Generate report:

```bash
uv run tfdpo report --outputs outputs --report-dir reports
```

Exit criteria:

- E3 improves over E1 or E2 on at least one domain accuracy metric, or the report identifies pair quality/training failure as the next blocker.
- Verification-present and verification-valid rates improve for E3.
- Branch count and tool calls do not explode.

## Phase 14: Real V1 Run

Only start this phase after smoke and dry-run gates pass.

### Task 14.1: Generate real V1 pairs

Submit:

```bash
sbatch --job-name=tfdpo-pairs-v1 \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/v1_real.yaml,RUN_ID=v1-pairs \
  scripts/run_generate_pairs_turing.sh
```

### Task 14.2: Train real V1 adapter

Submit:

```bash
sbatch --job-name=tfdpo-train-v1 \
  --export=ALL,TURING_ACCOUNT=<your_account>,CONFIG=configs/v1_real.yaml,RUN_ID=v1-train,PAIRS=outputs/v1-pairs/pairs/train.jsonl \
  scripts/run_train_turing.sh
```

### Task 14.3: Full evaluation table

Run E0/E1/E2/E3/E4 on:

- HotpotQA held-out
- GSM8K held-out/test
- NQ held-out if included
- MATH-500
- 2WikiMultiHopQA
- MuSiQue

Required report comparisons:

- E3 vs E1 answer accuracy.
- E3 vs E2 answer accuracy.
- E3 verification-present rate.
- E3 verification-valid rate.
- Premature final answer rate.
- Search-QA multi-hop completion rate.
- Math arithmetic/substitution/constraint checks.
- Branch count and tool-call count.

## Operational Commands On Turing

Check jobs:

```bash
squeue -u "$USER"
```

Inspect job:

```bash
scontrol show job <jobid>
```

Completed accounting:

```bash
sacct -j <jobid> --format=user,jobid,jobname,partition,state,time,start,end,elapsed,alloctres,ncpus,nodelist
```

Watch logs:

```bash
tail -f logs/slurm-<jobname>-<jobid>.out
tail -f logs/slurm-<jobname>-<jobid>.err
```

Cancel one bad job:

```bash
scancel <jobid>
```

Do not run `scancel -u "$USER"` unless intentionally cancelling every active job.

## Failure Handling

When anything fails:

1. Stop the pipeline.
2. Record command, config, job id, node, Slurm account, dataset row, and exact exception.
3. Classify the layer:
   - SSH/VPN
   - Slurm account/partition/QoS
   - modules/CUDA
   - Python environment
   - storage quota
   - model cache
   - dataset loading
   - retrieval index
   - parser/tool execution
   - teacher generation
   - pair filtering
   - DPO training
   - evaluation/reporting
4. Fix the root cause explicitly.
5. Re-run the smallest failing command.
6. Resume larger jobs only after the smaller command passes.

Never remediate by random account changes, random partitions, unverified CUDA downgrades, fake data, default teacher substitution, live search substitution, or broad retries without evidence.

## Completion Definition

The implementation is complete only when:

- Unit tests pass locally and inside a Turing GPU allocation.
- Smoke pair generation completes and passes manual inspection.
- Smoke training produces adapters and metrics.
- Dry-run E0/E1/E2/E3/E4 evaluation report exists.
- Real V1 run completes or fails with a documented root cause and a precise remediation.
- `reports/summary.html` contains the full evaluation table and links to raw logs/artifacts.
- No large model caches, temporary datasets, or retrieval scratch artifacts are left in `/home`.
