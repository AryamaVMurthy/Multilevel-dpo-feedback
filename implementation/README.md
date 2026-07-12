# Implementation

This directory contains the Qwen3 MATH-first collection, LN-DPO, GRPO/DAPO,
evaluation, reporting, and Turing workflows.

## Local verification

```bash
cd implementation
PYTHONPATH=src uv run --frozen pytest -q
uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen python -m compileall -q src tests scripts
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

## Active paper configs

- `configs/paper/math.yaml`: primary MATH Levels 4-5 study.
- `configs/paper/searchqa8k.yaml`: secondary study, used only after MATH freezes.
- `configs/pilots/math-feedback-*.yaml`: paired feedback-policy pilot.

Paper configs pin exact Qwen3 models/revisions, disable thinking for every role,
use BF16 LoRA without quantization, and reject generation above 8,192 tokens.

## Turing

Use the standalone checkout at
`/home/aryama.murthy/multilevel-feedback-dpo-qwen3`. Keep model weights,
environments, temporary generations, and training state on node-local scratch;
keep only compressed final artifacts and small adapters in home.

Before submission, verify the checkout commit, config hash, dataset hash, Slurm
account, queue, storage, and model cache. Start with one 48 GB GPU. Every real
job must write its source commit, protocol hash, model revisions, node, GPU,
peak memory, exit status, and artifact hashes.

The canonical order is baseline, feedback-policy pilot, full collection, pair
audit, trainer smoke, tuning, three-seed final training, and frozen evaluation.
See `../docs/plans/2026-07-12-canonical-math-searchqa-execution.md`.
