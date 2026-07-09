# Basic Model-Backed Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the current fixture-only observable pipeline into a verified model-backed smoke pipeline that can generate, correct, filter, and inspect a tiny set of Qwen3.5-2B preference pairs before any training starts.

**Architecture:** Keep the current local fixture pipeline as the stable base. Add strict config, prompt construction, model-provider abstraction, structured observability, and Turing Slurm smoke commands in small TDD steps. The first real GPU gate only generates a few student rollouts and teacher corrections; DPO, GRPO, distillation, and sweeps remain blocked until this basic pair-generation report is inspected and approved.

**Tech Stack:** Python 3.12, stdlib tests first, optional `uv`, PyTorch, Transformers, Accelerate, Hugging Face `Qwen/Qwen3.5-2B`, teacher configs for `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-2B` privileged, Turing Slurm `u22`, CUDA 12.4, JSONL logs, JSON metrics, HTML report.

---

## Current State

- Repository root is `multilevel-feedback-dpo`.
- Remote is `origin = git@github.com:AryamaVMurthy/Multilevel-dpo-feedback.git`.
- Basic fixture pipeline already exists under `implementation/`.
- Existing verified commands:

```bash
cd implementation
PYTHONPATH=src python3 -m unittest tests.test_basic_pipeline -v
PYTHONPATH=src python3 -m compileall -q src tests
```

- Existing fixture run output is intentionally ignored by Git under `implementation/runs/basic-fixture/`.

## Hard Gates

- Do not start DPO training.
- Do not start GRPO.
- Do not start on-policy distillation.
- Do not run large datasets.
- Do not run model generation on the Turing login node.
- Do not silently substitute model ids, teacher ids, Slurm account, CUDA version, datasets, or prompts.
- If a fallback is intentionally used in test mode, it must be explicit and logged with `fallback_reason`.

## Success Definition For This Plan

This plan is complete only when:

- Local unit tests pass.
- A local fake-model pipeline test writes all observable artifacts.
- A Turing GPU model-load smoke confirms `Qwen/Qwen3.5-2B` loads and CUDA is used.
- A Turing end-to-end tiny run creates real `events.jsonl`, `metrics.json`, `rollouts.jsonl`, `corrections.jsonl`, `pairs.jsonl`, `rejections.jsonl`, and `report.html`.
- Human inspection of 3-5 examples is possible from the report and raw JSONL files.
- The next training phase remains blocked until the user explicitly approves continuation.

## Task 1: Add Project Metadata And Dependency Plan

**Files:**
- Create: `implementation/pyproject.toml`
- Create: `implementation/tests/test_project_metadata.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_pyproject_declares_cli_and_python_version():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in text
    assert 'tfdpo = "text_feedback_dpo.cli:main"' in text
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd implementation
PYTHONPATH=src python3 -m unittest tests.test_project_metadata -v
```

Expected:

```text
FileNotFoundError: pyproject.toml
```

**Step 3: Add minimal metadata**

Add:

```toml
[project]
name = "text-feedback-dpo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "accelerate",
  "bitsandbytes",
  "datasets",
  "jinja2",
  "peft",
  "pyyaml",
  "torch",
  "transformers",
  "trl",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[project.scripts]
tfdpo = "text_feedback_dpo.cli:main"
```

**Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_project_metadata -v
```

Expected: `OK`

**Step 5: Commit**

```bash
git add implementation/pyproject.toml implementation/tests/test_project_metadata.py
git commit -m "chore: add project metadata for model pipeline"
```

## Task 2: Add Strict Config Loading

**Files:**
- Create: `implementation/src/text_feedback_dpo/config.py`
- Create: `implementation/tests/test_config.py`
- Create: `implementation/configs/basic_smoke.yaml`

**Step 1: Write failing tests**

Test required fields:

```python
import tempfile
import unittest
from pathlib import Path

from text_feedback_dpo.config import load_config


class ConfigTest(unittest.TestCase):
    def test_missing_student_model_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("run_id: bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "student_model"):
                load_config(path)

    def test_smoke_config_loads(self):
        config = load_config(Path("configs/basic_smoke.yaml"))
        self.assertEqual(config["student_model"], "Qwen/Qwen3.5-2B")
        self.assertEqual(config["teacher_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(config["max_examples"], 5)
```

**Step 2: Run test to verify it fails**

```bash
cd implementation
PYTHONPATH=src python3 -m unittest tests.test_config -v
```

Expected: import failure or missing file failure.

**Step 3: Implement minimal config loader**

Requirements:

- Use YAML.
- Require:
  - `run_id`
  - `student_model`
  - `teacher_model`
  - `teacher_mode`
  - `max_examples`
  - `output_dir`
  - `generation`
  - `teacher_generation`
  - `slurm`
- Reject unknown top-level keys.
- Reject `slurm.account: null` for Turing commands.
- Allow `slurm.account: null` only for local dry config validation if `allow_missing_slurm_account_for_local: true`.

**Step 4: Create smoke config**

`configs/basic_smoke.yaml`:

```yaml
run_id: qwen35-basic-smoke
student_model: Qwen/Qwen3.5-2B
teacher_model: Qwen/Qwen3.5-9B
teacher_mode: stronger_model
max_examples: 5
output_dir: runs/qwen35-basic-smoke
allow_missing_slurm_account_for_local: true

generation:
  max_new_tokens: 512
  temperature: 0.2
  top_p: 0.95

teacher_generation:
  max_new_tokens: 1024
  temperature: 0.2
  top_p: 0.95

slurm:
  account: null
  partition: u22
  cpus: 16
  gpus: 1
  mem_per_cpu: 4096
  cuda_module: u22/cuda/12.4
```

**Step 5: Run tests**

```bash
PYTHONPATH=src python3 -m unittest tests.test_config -v
```

Expected: `OK`

**Step 6: Commit**

```bash
git add implementation/src/text_feedback_dpo/config.py implementation/tests/test_config.py implementation/configs/basic_smoke.yaml
git commit -m "feat: add strict smoke config loading"
```

## Task 3: Make Observability Run-Complete Enough For Debugging

**Files:**
- Modify: `implementation/src/text_feedback_dpo/observability.py`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Test: `implementation/tests/test_observability.py`

**Step 1: Write failing tests**

Test that every event includes:

- `event_name`
- `run_id`
- `elapsed_ms`
- `status`
- `timestamp`

Test that failure events include:

- `error_code`
- `message`
- `stage`

**Step 2: Run test to verify it fails**

```bash
cd implementation
PYTHONPATH=src python3 -m unittest tests.test_observability -v
```

Expected: missing fields.

**Step 3: Implement event schema**

Requirements:

- Add ISO timestamp.
- Add `status` defaulting to `"ok"`.
- Add `stage` where relevant.
- Add helper `logger.failure(stage, error_code, message, **fields)`.
- Never catch and hide fatal exceptions without logging an explicit failure event first.

**Step 4: Run full local tests**

```bash
PYTHONPATH=src python3 -m unittest discover -v
PYTHONPATH=src python3 -m compileall -q src tests
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add implementation/src/text_feedback_dpo/observability.py implementation/src/text_feedback_dpo/cli.py implementation/tests/test_observability.py
git commit -m "feat: strengthen pipeline observability events"
```

## Task 4: Add Prompt Builder For Student And Teacher

**Files:**
- Create: `implementation/src/text_feedback_dpo/prompts.py`
- Create: `implementation/tests/test_prompts.py`

**Step 1: Write failing tests**

Tests:

- Student prompt contains `<plan>`, `<reflect>`, `<final>` requirements.
- Math prompt requires arithmetic/substitution/constraint verification.
- Search-QA prompt requires entity/relation/evidence/type verification.
- Teacher prompt contains problem, gold answer, student rollout, evaluator result, and output contract.
- Privileged teacher prompt explicitly marks privileged information as training-only.

**Step 2: Run red test**

```bash
PYTHONPATH=src python3 -m unittest tests.test_prompts -v
```

Expected: import failure.

**Step 3: Implement prompt builders**

Functions:

```python
build_student_prompt(problem: str, domain: str) -> str
build_teacher_prompt(problem: str, gold_answer: str, student_rollout: str, result: dict, domain: str, teacher_mode: str) -> str
```

Allowed `teacher_mode`:

- `stronger_model`
- `same_model_privileged`

Unknown mode must raise `ValueError`.

**Step 4: Run tests and commit**

```bash
PYTHONPATH=src python3 -m unittest tests.test_prompts -v
git add implementation/src/text_feedback_dpo/prompts.py implementation/tests/test_prompts.py
git commit -m "feat: add student and teacher prompt builders"
```

## Task 5: Add Model Provider Interface With Fake Provider First

**Files:**
- Create: `implementation/src/text_feedback_dpo/models.py`
- Create: `implementation/tests/test_models.py`

**Step 1: Write failing tests**

Test a fake provider:

```python
from text_feedback_dpo.models import FakeModelProvider


def test_fake_model_provider_returns_configured_text():
    provider = FakeModelProvider({"student": "hello"})
    assert provider.generate("student", "prompt") == "hello"
```

**Step 2: Run red test**

```bash
PYTHONPATH=src python3 -m unittest tests.test_models -v
```

Expected: import failure.

**Step 3: Implement provider interface**

Requirements:

- `FakeModelProvider` for local tests.
- `TransformersModelProvider` skeleton that raises explicit `ImportError` if dependencies are unavailable.
- No hidden CPU fallback.
- GPU commands must require `torch.cuda.is_available()` unless config explicitly says local fake mode.

**Step 4: Run tests and commit**

```bash
PYTHONPATH=src python3 -m unittest tests.test_models -v
git add implementation/src/text_feedback_dpo/models.py implementation/tests/test_models.py
git commit -m "feat: add model provider abstraction"
```

## Task 6: Add Local End-To-End Fake-Model Generation Pipeline

**Files:**
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Create: `implementation/tests/test_generate_pipeline.py`

**Step 1: Write failing test**

Test command-level function:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from text_feedback_dpo.cli import run_generate_pipeline
from text_feedback_dpo.models import FakeModelProvider


class GeneratePipelineTest(unittest.TestCase):
    def test_fake_pipeline_writes_rollouts_corrections_pairs_and_report(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            provider = FakeModelProvider({
                "student": "<plan>...</plan>...",
                "teacher": "<feedback>...</feedback><corrected_rollout>...</corrected_rollout>",
            })
            result = run_generate_pipeline(
                config_path=Path("configs/basic_smoke.yaml"),
                output_dir=out,
                model_provider=provider,
            )
            self.assertIn("accepted_pairs", result)
            self.assertTrue((out / "events.jsonl").exists())
            self.assertTrue((out / "rollouts.jsonl").exists())
            self.assertTrue((out / "corrections.jsonl").exists())
            self.assertTrue((out / "pairs.jsonl").exists())
            self.assertTrue((out / "report.html").exists())
```

Use complete valid rollout text in the actual test; do not leave ellipses.

**Step 2: Run red test**

```bash
PYTHONPATH=src python3 -m unittest tests.test_generate_pipeline -v
```

Expected: missing `run_generate_pipeline`.

**Step 3: Implement local fake end-to-end path**

Requirements:

- Build prompts.
- Generate fake student rollout.
- Evaluate original.
- Generate fake teacher correction.
- Parse feedback and corrected rollout.
- Evaluate corrected.
- Reuse pair filtering.
- Write all artifacts:
  - `events.jsonl`
  - `rollouts.jsonl`
  - `corrections.jsonl`
  - `pairs.jsonl`
  - `rejections.jsonl`
  - `metrics.json`
  - `report.html`

**Step 4: Run full local checks**

```bash
PYTHONPATH=src python3 -m unittest discover -v
PYTHONPATH=src python3 -m compileall -q src tests
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add implementation/src/text_feedback_dpo implementation/tests
git commit -m "feat: add fake model generation pipeline"
```

## Task 7: Add Turing Model-Load Smoke Script

**Files:**
- Create: `implementation/scripts/turing_model_load_smoke.sh`
- Create: `implementation/tests/test_turing_scripts.py`

**Step 1: Write script tests**

Test that the script contains:

- `#SBATCH -p u22`
- `#SBATCH --gres=gpu:1`
- `set -euo pipefail`
- `module load u22/cuda/12.4`
- `Qwen/Qwen3.5-2B`
- `torch.cuda.is_available()`
- no `|| true` around model load or CUDA checks

**Step 2: Run red test**

```bash
PYTHONPATH=src python3 -m unittest tests.test_turing_scripts -v
```

Expected: missing script.

**Step 3: Implement script**

Script behavior:

- Require `TURING_ACCOUNT`.
- Load CUDA.
- Set conservative `uv` env vars.
- Use scratch for `HF_HOME` and datasets cache.
- Run one Python command that:
  - imports torch
  - verifies CUDA
  - loads tokenizer
  - loads `Qwen/Qwen3.5-2B`
  - generates one short response from a tiny prompt
  - writes `runs/model-load-smoke/events.jsonl`

**Step 4: Run local script tests**

```bash
PYTHONPATH=src python3 -m unittest tests.test_turing_scripts -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add implementation/scripts/turing_model_load_smoke.sh implementation/tests/test_turing_scripts.py
git commit -m "feat: add turing qwen model load smoke script"
```

## Task 8: Add Turing Tiny Pair-Generation Script

**Files:**
- Create: `implementation/scripts/turing_basic_pair_generation.sh`
- Modify: `implementation/src/text_feedback_dpo/cli.py`
- Test: `implementation/tests/test_turing_scripts.py`

**Step 1: Extend failing tests**

Assert the script:

- requires `CONFIG`
- requires `TURING_ACCOUNT`
- writes under configured output dir
- logs GPU telemetry
- runs `tfdpo` or `python -m text_feedback_dpo.cli`
- does not train
- does not call GRPO/DPO trainer

**Step 2: Run red test**

```bash
PYTHONPATH=src python3 -m unittest tests.test_turing_scripts -v
```

Expected: missing script assertions fail.

**Step 3: Implement script**

Script flow:

```bash
sbatch --job-name=tfdpo-basic-pairs \
  --export=ALL,TURING_ACCOUNT=<account>,CONFIG=configs/basic_smoke.yaml \
  scripts/turing_basic_pair_generation.sh
```

Inside Slurm:

- Verify GPU.
- Set scratch caches.
- Run at most 5 examples.
- Generate student rollouts using `Qwen/Qwen3.5-2B`.
- Generate teacher corrections using `Qwen/Qwen3.5-9B`.
- Write all observable artifacts.
- Stop after pair generation.

**Step 4: Run local script tests**

```bash
PYTHONPATH=src python3 -m unittest tests.test_turing_scripts -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add implementation/scripts/turing_basic_pair_generation.sh implementation/src/text_feedback_dpo/cli.py implementation/tests/test_turing_scripts.py
git commit -m "feat: add turing basic pair generation script"
```

## Task 9: Run Verification Before Any Training

**Files:**
- Modify only if a verified bug appears.

**Step 1: Run local verification**

```bash
cd implementation
PYTHONPATH=src python3 -m unittest discover -v
PYTHONPATH=src python3 -m compileall -q src tests
```

Expected: all tests pass.

**Step 2: Run Turing model-load smoke**

On Turing login node:

```bash
cd ~/Multilevel-dpo-feedback/implementation
sbatch --job-name=tfdpo-model-load \
  --export=ALL,TURING_ACCOUNT=<account> \
  scripts/turing_model_load_smoke.sh
```

Inspect:

```bash
squeue -u "$USER"
tail -f logs/slurm-tfdpo-model-load-<jobid>.out
tail -f logs/slurm-tfdpo-model-load-<jobid>.err
sacct -j <jobid> --format=user,jobid,jobname,partition,state,elapsed,alloctres,ncpus,nodelist
```

Pass criteria:

- Slurm state is `COMPLETED`.
- CUDA is true.
- GPU name is logged.
- Model id is logged.
- One short generation is logged.
- No CPU fallback occurred.

**Step 3: Run Turing basic pair generation**

```bash
sbatch --job-name=tfdpo-basic-pairs \
  --export=ALL,TURING_ACCOUNT=<account>,CONFIG=configs/basic_smoke.yaml \
  scripts/turing_basic_pair_generation.sh
```

Pass criteria:

- Slurm state is `COMPLETED`.
- Output dir contains:
  - `events.jsonl`
  - `rollouts.jsonl`
  - `corrections.jsonl`
  - `pairs.jsonl`
  - `rejections.jsonl`
  - `metrics.json`
  - `report.html`
  - `gpu-<jobid>.csv`
- Metrics show `examples_total <= 5`.
- Logs show teacher model used.
- Logs show student model used.
- No DPO/GRPO/distillation training occurs.

**Step 4: Manual inspection**

Inspect:

```bash
cat runs/qwen35-basic-smoke/metrics.json
sed -n '1,20p' runs/qwen35-basic-smoke/events.jsonl
sed -n '1,5p' runs/qwen35-basic-smoke/pairs.jsonl
sed -n '1,5p' runs/qwen35-basic-smoke/rejections.jsonl
```

Open or copy `report.html`.

Manual criteria:

- Rollouts use the bracket format.
- Corrected rollouts include verification.
- Rejected examples have explicit reasons.
- Pair prompt does not include teacher feedback.
- The report makes failures visible.

**Step 5: Stop**

Do not continue to DPO, GRPO, or distillation until the user explicitly approves continuation after reviewing the run artifacts.

## Task 10: Push The Basic Pipeline Gate

**Files:**
- No new files unless fixes were required.

**Step 1: Check status**

```bash
git status -sb
```

**Step 2: Run final local verification**

```bash
cd implementation
PYTHONPATH=src python3 -m unittest discover -v
PYTHONPATH=src python3 -m compileall -q src tests
```

**Step 3: Push**

```bash
git push origin main
```

**Step 4: Verify remote**

```bash
git rev-parse HEAD
git ls-remote --heads origin main
```

Expected: both SHAs match.

## Next Plan After This Gate

Only after user approval:

1. Add small DPO training over accepted pairs.
2. Add E0/E1/E2 baselines.
3. Add GRPO reward-function baseline.
4. Add on-policy distillation baseline.
5. Scale beyond 5 examples.
