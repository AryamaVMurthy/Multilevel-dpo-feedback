# Multilevel Feedback DPO Implementation

This folder contains the observable native-Qwen collection, preference construction,
DPO/GRPO training, held-out evaluation, reporting, and Turing Slurm workflows. Paper
runs are fail-fast and remain blocked until the preceding preflight and freeze gates
pass.

Run the basic pipeline smoke check:

```bash
cd /home/aryamavmurthy/work/SLM-Research/multilevel-feedback-dpo/implementation
PYTHONPATH=src python3 -m text_feedback_dpo.cli basic-pipeline \
  --examples examples/basic_pipeline/examples.jsonl \
  --rollouts examples/basic_pipeline/rollouts.jsonl \
  --corrections examples/basic_pipeline/corrections.jsonl \
  --output-dir runs/basic-fixture \
  --run-id basic-fixture
```

Inspect:

```bash
ls -la runs/basic-fixture
cat runs/basic-fixture/events.jsonl
cat runs/basic-fixture/metrics.json
cat runs/basic-fixture/pairs.jsonl
cat runs/basic-fixture/rejections.jsonl
```

Open `runs/basic-fixture/report.html` in a browser for the human-readable summary.

Turing model-load smoke:

```bash
sbatch -A <account> --job-name=tfdpo-model-load \
  --export=ALL,TURING_ACCOUNT=<account> \
  scripts/turing_model_load_smoke.sh
```

Turing tiny pair generation:

```bash
sbatch -A <account> --job-name=tfdpo-basic-pairs \
  --export=ALL,TURING_ACCOUNT=<account>,CONFIG=configs/basic_smoke.yaml \
  scripts/turing_basic_pair_generation.sh
```

These two commands are legacy runtime smokes. They do not authorize paper training.

Current smoke generation settings:

```yaml
temperature: 1.0
top_p: 0.95
top_k: 20
presence_penalty: 1.5
```

`temperature`, `top_p`, and `top_k` are passed directly to Hugging Face Transformers `generate()`. `presence_penalty` is not a native Transformers `GenerationConfig` field, so this repo applies it through a custom logits processor that subtracts the penalty from tokens already present in the sequence. This is explicit and tested; it is not mapped silently to `repetition_penalty`.

The paper plan applies these sampling settings only to student rollouts. Teacher,
evaluator, and guidance-guard roles use separately configured non-thinking decoding
profiles. See `../docs/plans/2026-07-10-paper-scale-experiment-implementation.md` for
the exact execution order, storage gates, and artifact requirements.
