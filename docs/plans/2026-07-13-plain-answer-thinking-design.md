# SearchQA Plain-Answer Thinking Design

> **Superseded:** This answer-only design is retained solely to interpret the archival short-answer baseline. The active design is [2026-07-13-fixed-retrieval-cited-reasoning-design.md](2026-07-13-fixed-retrieval-cited-reasoning-design.md). Do not launch new training from this document.

## Objective

Train `Qwen/Qwen3-4B-Base` with full-parameter SFT and minimal-intervention DPO on SearchQA, while keeping every externally scored student response as a plain short answer. Compare the primary method against raw-base, SFT, GRPO, and DAPO baselines under the same teacher-free evaluator.

## Corrected response contract

The student is never asked to produce XML or JSON. Its scored completion is a stripped plain answer such as `Hemingway`. The prompt ends with `Answer:` and reserves at most 32 completion tokens within a 4,096-token total sequence budget.

The deterministic evaluator computes normalized exact match and token F1 directly on the complete decoded answer. It removes tokenizer special tokens and surrounding whitespace only; it does not use regex extraction, fabricate missing answers, or silently repair verbose output.

## Thinking modes

The primary teacher is the post-trained `Qwen/Qwen3-32B`, loaded in 4-bit NF4 with BF16 compute on one GPU. It uses its native chat template with `enable_thinking=True`. The private thinking section is discarded and only a validated final hint object is consumed. `Qwen/Qwen3-14B` is the only teacher fallback and is used only after a recorded 32B load or inference OOM/runtime failure.

The base student does not natively provide the same reliable post-trained chat thinking switch. Student thinking is therefore implemented as an explicit two-pass mode: first generate a bounded private scratchpad from the evidence, then condition a second generation on that scratchpad and emit only a short final answer. Direct answer mode and two-pass thinking mode are compared on a train-derived prompt-development set. The higher-accuracy mode is frozen before official validation. Private scratchpads never become DPO chosen/rejected text and never enter the final evaluator.

## Data and split discipline

Use pinned `kyunghyuncho/search_qa` data and pinned Qwen revisions. Build deterministic evidence packs that reserve completion capacity within 4,096 tokens. A fixed subset drawn only from training data is used for prompt and thinking-mode selection. Official validation is used for checkpoint selection. Test remains untouched until all method and hyperparameter decisions are frozen.

SFT rows use TRL prompt-completion format with completion-only loss:

```json
{"prompt": "Evidence: ...\n\nQuestion: ...\n\nAnswer:", "completion": "Hemingway"}
```

## Minimal-intervention trajectory

For each example, the SFT student generates one answer. If normalized exact match is zero, the instruct teacher receives the question, evidence, gold answer, failed answer, previous hints, and current escalation level. The teacher internally localizes the earliest responsible error and returns only one short JSON hint. The student sees only the plain hint text and retries once. Hints accumulate and escalate from broad entity/relation guidance to a stronger evidence pointer, without stating the gold answer.

The loop stops at the first student-generated exact answer or at the configured hard intervention cap. Unresolved examples produce no preference pair. A valid DPO row uses the original no-hint prompt, the first student-generated correct answer as `chosen`, and a real failed student answer as `rejected`. Teacher answers and hints never appear in the DPO prompt or completions.

Teacher output is internal JSON rather than student markup. It is schema-validated, limited in length, checked against normalized gold-answer aliases, and rejected explicitly on leakage or malformed output. There is no separate critic model.

## Full training and comparisons

The student remains BF16 full-parameter training with DeepSpeed ZeRO-3 and gradient checkpointing. Quantization is inference-only for the teacher. The 4B student is attempted with microbatch one and all suitable GPUs before a recorded OOM can authorize the pinned 1.7B base fallback.

Run order:

1. Plain-answer prompt and student-thinking pilot.
2. Raw 4B base validation baseline with inspected samples.
3. Full-parameter SFT with periodic checkpoints and generation validation.
4. Quantized 32B instruct-teacher trajectory collection.
5. Offline cache verification and preference construction.
6. Full-parameter DPO rounds, stopping when locked validation no longer improves.
7. Fair SFT, GRPO, and DAPO comparisons from the same SFT checkpoint and budget.
8. Frozen final test evaluation and report.

GRPO and DAPO rewards use exact match as the primary reward with bounded token-F1 shaping. There is no format reward and no reward for verbosity. Every comparison uses the same plain-answer prompt, sequence limit, validation protocol, and teacher-free test evaluator.

## Offline reuse

Trajectory caches are immutable and keyed by dataset revision, example ID, student and teacher model hashes, prompt version, thinking mode, decoding settings, intervention policy, and seed. A mismatch fails instead of reusing stale data. Complete matching caches can rebuild DPO data without loading either model. A matched refreshed-versus-cached comparison quantifies any accuracy difference.

## Validation and monitoring

Before each full stage, run a 32-example end-to-end preflight covering model load, CUDA use, generation, scoring, cache writing, checkpoint save, resume, and artifact validation. During training, log loss every 10 steps, save and evaluate at fixed intervals, retain best checkpoints plus latest, and run answer-generation validation on a fixed subset. Full validation runs at epoch boundaries and before promotion.

All jobs record source hash, model/data revisions, resolved config, seed, Slurm ID, node/GPU, library versions, GPU telemetry, timing, checkpoints, and explicit fallback reasons. Missing rows, malformed hints, leakage, OOM, cache mismatch, incomplete checkpoints, or evaluation cardinality mismatch stop downstream execution.

## Invalid-run retirement

The superseded XML experiment is never reported as a scientific baseline. Preserve only its source checkpoint, logs, manifest, and representative failure examples under an invalid-run archive. Cancel its active and dependent jobs, then delete derived XML predictions, XML SFT rows, and incomplete outputs after dependency and path verification.

## Success criteria

- Student scored outputs are plain answers and contain no XML contract.
- Corrected raw-base baseline has a verified nonempty rate and official validation EM/F1.
- SFT materially improves validation accuracy and answer reliability.
- DPO chosen answers are student-generated and teacher hints are answer-free.
- Primary DPO is compared fairly with SFT, GRPO, and DAPO.
- Offline reuse is byte-verifiable and its accuracy impact is measured.
- Untouched test metrics and a reproducible HTML/JSON/CSV report are produced.
