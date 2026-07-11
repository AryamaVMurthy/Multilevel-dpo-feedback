# Qwen3 MATH-Then-SearchQA Protocol Freeze

Date: 2026-07-11 Asia/Kolkata

Status: local implementation gate passed; Turing execution not yet started

## Frozen identities

| Role | Model | Revision |
| --- | --- | --- |
| student | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| teacher | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| evaluator and review roles | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

All roles are post-trained checkpoints and receive `enable_thinking=false`. Active
schema-4 configs reject any alternate ID, revision, Base checkpoint, missing sampling
field, or generation allowance above 8,192 tokens.

## Student generation start point

- temperature: 0.7
- top-p: 0.8
- top-k: 20
- min-p: 0
- repetition penalty: 1.0
- presence penalty: 0.0 pending the frozen train-only MATH termination study
- maximum new tokens: 8,192
- termination: EOS or a nonempty balanced `FINAL: \boxed{...}` answer

The presence-penalty screen uses 12 MATH Level 4-5 training examples at 4,096 tokens.
Three profiles advance, in a frozen order, to 32 disjoint training examples at 8,192
tokens. Selection orders protocol-valid accuracy, truncation, correct answers per
million generated tokens, median length, mean latency, then stable profile ID. A
truncated answer is never counted as protocol-valid correct.

## Training foundation

- LoRA rank 16, alpha 32, dropout 0.05
- BF16, no quantization
- exact per-layer Qwen3 target inventory: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`
- any missing or unexpected text projection fails before training
- AdamW fused, betas 0.9/0.999, epsilon 1e-8, clipping 1.0
- DPO and GRPO search matrices, final seeds 17/31/47, and test-blind selection remain
  frozen in the active configs

## Isolation and storage

Qwen3.5 results remain historical diagnostics. Qwen3 uses schema `paper-v3`, Qwen3
prompt-protocol identifiers, distinct experiment IDs, a separate Git branch, and a
standalone Turing clone. Turing scripts reject home-directory runtime environments;
weights, environments, dataset caches, and temporary generations must live under
node-local `/scratch`.

## Local evidence

- 187 unit tests passed.
- Ruff passed with no findings.
- Python compilation passed.
- Every shell wrapper passed `bash -n`.
- MATH, SearchQA-8K, and diagnostic GSM8K configs passed strict CLI validation.
- The exact Hugging Face revision pages for both frozen models were verified on
  2026-07-11.

No cluster result is claimed by this report. Dataset, GPU, baseline, collection,
training, evaluation, statistical, and final-report gates remain pending until their
immutable Turing artifacts exist.
