# Local Concise Generation Sweep Design

## Objective

Select a Qwen3.5-2B non-thinking MATH decoding protocol that materially reduces generated tokens and latency without a material correctness loss. Selection uses only official MATH Level 4-5 training rows. Validation and official test remain untouched.

## Fixed Invariants

- Model: pinned `Qwen/Qwen3.5-2B` revision from `configs/paper/math.yaml`.
- Chat template receives `enable_thinking=false` explicitly and receives one fresh user turn.
- Prompt requests step-by-step reasoning and one final `\boxed{}` answer.
- Batch size is one on the local RTX 3050 6 GB GPU; CPU inference fallback is prohibited.
- Every run preserves raw text, exact token counts, latency, finish reason, symbolic correctness, seed, config hash, model revision, and GPU telemetry.
- A balanced `FINAL: \\boxed{...}` answer may terminate generation only when the profile explicitly enables `stop_after_final_answer`.

## Candidate Grid

Stage A evaluates 12 deterministic stratified train-only problems at 4,096 tokens:

| ID | temperature | top_p | top_k | min_p | presence | repetition | sampling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen-reasoning | 1.0 | 1.0 | 40 | 0.0 | 2.0 | 1.0 | sampled |
| qwen-general | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 | sampled |
| conservative | 0.6 | 0.9 | 20 | 0.0 | 0.0 | 1.0 | sampled |
| low-diversity | 0.5 | 0.8 | 20 | 0.0 | 0.0 | 1.0 | sampled |
| mild-repeat | 0.7 | 0.8 | 20 | 0.0 | 0.5 | 1.05 | sampled |
| greedy | n/a | n/a | n/a | n/a | 0.0 | 1.0 | greedy |

All candidates use box stopping. The existing natural-termination `qwen-nonthinking-r1` result remains the no-stop control. Stage B promotes three candidates to 32 disjoint train-only problems at 8,192 tokens, subject to measured VRAM feasibility.

## Selection

Reject malformed, failed, or unevaluable configurations. Rank by symbolic accuracy, retaining candidates within one correct answer of the best Stage A result. Within that set rank by median tokens, then truncation, mean latency, and P90 tokens. Stage B uses accuracy first and the same efficiency tie-breakers. Report the full Pareto frontier rather than only the winner.

## Literature Basis

Recent efficient-reasoning work warns that direct length pressure can collapse correctness and recommends budget-conditioned evaluation, sufficient positive rollouts, and separate accuracy/length reporting. This sweep changes inference only; GRPO length rewards remain a later labeled ablation.

## Failure Policy

OOM, model revision drift, missing CUDA, malformed source data, invalid sampling fields, incomplete artifacts, or evaluator errors stop the run explicitly. The runner resumes only from complete per-example records and never substitutes another model, device, dataset, or decoding profile.
