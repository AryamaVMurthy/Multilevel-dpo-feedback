# Textual Feedback DPO Basic Pipeline

This folder starts with the minimal observable pipeline only. It does not train Qwen, run GRPO, or launch Turing jobs yet.

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

These jobs only verify model loading and tiny pair generation. They do not start DPO, GRPO, or distillation training.

Current smoke generation settings:

```yaml
temperature: 1.0
top_p: 0.95
top_k: 20
presence_penalty: 1.5
```

`temperature`, `top_p`, and `top_k` are passed directly to Hugging Face Transformers `generate()`. `presence_penalty` is not a native Transformers `GenerationConfig` field, so this repo applies it through a custom logits processor that subtracts the penalty from tokens already present in the sequence. This is explicit and tested; it is not mapped silently to `repetition_penalty`.
