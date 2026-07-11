# Local Concise Generation Sweep Design

## Objective

Select the Qwen3-4B non-thinking presence penalty using only official MATH Level 4-5 training rows. Validation and official test remain untouched.

## Fixed Invariants

- Model: pinned `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Chat template receives `enable_thinking=false` explicitly and receives one fresh user turn.
- Prompt requests step-by-step reasoning and one final `\boxed{}` answer.
- Batch size is one on one Turing 48 GB GPU; CPU inference fallback is prohibited.
- Every run preserves raw text, exact token counts, latency, finish reason, symbolic correctness, seed, config hash, model revision, and GPU telemetry.
- A balanced `FINAL: \\boxed{...}` answer may terminate generation only when the profile explicitly enables `stop_after_final_answer`.

## Candidate Grid

Stage A evaluates 12 deterministic stratified train-only problems at 4,096 tokens:

| ID | temperature | top_p | top_k | min_p | presence | repetition | sampling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| presence-0 | 0.7 | 0.8 | 20 | 0.0 | 0.0 | 1.0 | sampled |
| presence-0.5 | 0.7 | 0.8 | 20 | 0.0 | 0.5 | 1.0 | sampled |
| presence-1 | 0.7 | 0.8 | 20 | 0.0 | 1.0 | 1.0 | sampled |
| presence-1.5 | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 | sampled |
| presence-2 | 0.7 | 0.8 | 20 | 0.0 | 2.0 | 1.0 | sampled |

All candidates use the same balanced final-answer stop. Stage B promotes three candidates to 32 disjoint train-only problems at 8,192 tokens.

## Selection

Reject malformed, failed, truncated-positive, or unevaluable outputs. Rank by protocol-valid accuracy first, then truncation count, correct answers per million generated tokens, median length, mean latency, and stable profile ID. Stage B uses the same order and freezes one presence penalty without consulting validation or test.

## Literature Basis

Recent efficient-reasoning work warns that direct length pressure can collapse correctness and recommends budget-conditioned evaluation, sufficient positive rollouts, and separate accuracy/length reporting. This sweep changes inference only; GRPO length rewards remain a later labeled ablation.

## Failure Policy

OOM, model revision drift, missing CUDA, malformed source data, invalid sampling fields, incomplete artifacts, or evaluator errors stop the run explicitly. The runner resumes only from complete per-example records and never substitutes another model, device, dataset, or decoding profile.
