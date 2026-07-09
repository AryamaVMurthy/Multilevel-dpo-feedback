# Multilevel DPO Feedback

This repository contains the V1 textual-feedback DPO research prototype design and the first basic observable pipeline.

Current status:

- Design PDF: `v1_textual_feedback_dpo_design.pdf`
- Planning docs: `docs/plans/`
- Basic pipeline implementation: `implementation/`
- Model training, GRPO, on-policy distillation, and Turing jobs have not started yet.

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

The basic pipeline writes observable artifacts:

- `events.jsonl`
- `metrics.json`
- `pairs.jsonl`
- `rejections.jsonl`
- `report.html`

