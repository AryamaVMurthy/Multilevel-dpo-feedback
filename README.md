# Multilevel DPO Feedback

This repository contains the implementation and paper-scale execution protocol for
multilevel-feedback DPO with the pinned post-trained Qwen3-4B student and Qwen3-8B
privileged roles.

Current status:

- Original design source: `v1_textual_feedback_dpo_design.pdf`.
- Canonical living method: `docs/design/native_iterative_guidance_dpo.md`.
- Paper design and execution plan: `docs/plans/`.
- Tested implementation and Turing workflows: `implementation/`.
- Current state: MATH Levels 4-5 is the primary benchmark and SearchQA-8K is strictly
  secondary. Active paper configs use schema 4, explicit non-thinking
  `qwen3-nonthinking-final-r1`, balanced final-box stopping, and an 8,192-token
  generation ceiling. Earlier Qwen3.5 jobs and reports are preserved as diagnostics
  only and must never be merged into Qwen3 paper artifacts.

Run the basic local pipeline:

```bash
cd implementation
PYTHONPATH=src python3 -m text_feedback_dpo.cli basic-pipeline \
  --examples examples/basic_pipeline/examples.jsonl \
  --rollouts examples/basic_pipeline/rollouts.jsonl \
  --corrections examples/basic_pipeline/corrections.jsonl \
  --output-dir runs/basic-fixture \
  --run-id basic-fixture
```

Verify:

```bash
PYTHONPATH=src python3 -m unittest tests.test_basic_pipeline -v
PYTHONPATH=src python3 -m compileall -q src tests
```

For paper execution, use
`docs/plans/2026-07-10-paper-scale-experiment-implementation.md`. Smoke commands and
their constants are runtime checks only, not paper hyperparameters.

The basic pipeline writes observable artifacts:

- `events.jsonl`
- `metrics.json`
- `pairs.jsonl`
- `rejections.jsonl`
- `report.html`
