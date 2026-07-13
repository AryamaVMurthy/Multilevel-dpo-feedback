# SearchQA Bootstrap-SFT and Minimal-Intervention DPO Design

## Decision

Use a verified student-only bootstrap before large-scale teacher collection:

```text
raw Qwen3-4B-Base
  -> broad no-hint query/response rollouts
  -> separate verified query and response SFT targets
  -> full-parameter SFT
  -> answer-free Qwen3-32B minimal-hint collection
  -> same-context student-only DPO
  -> matched SFT / GRPO / DAPO comparisons
  -> frozen validation and one-time official test reporting
```

This replaces the earlier assumption that teacher-guided collection should precede SFT. The audited 32-row base preflight produced zero fully correct trajectories, so the current no-hint sibling gate would yield no SFT rows on that sample. The base student nevertheless generated useful queries and several semantically correct answer fragments, making student-only rejection sampling a viable bootstrap source.

## Invariants

- Student remains `Qwen/Qwen3-4B-Base` with full BF16 parameter updates.
- The 1.7B base model is allowed only after a recorded OOM on the intended 4B training configuration.
- Teacher remains `Qwen/Qwen3-32B` in 4-bit NF4 with BF16 compute and native thinking.
- The 14B teacher is allowed only after an explicit 32B load or inference failure.
- Maximum prompt plus completion length remains 4,096 tokens.
- Student-visible output is plain text, never XML.
- Teacher output is internal strict JSON containing exactly one answer-free hint.
- No teacher answer, gold-derived response, repaired string, fabricated target, or hidden parser fallback enters SFT or DPO.
- Official test remains untouched until every model and hyperparameter choice is frozen.
- Every fallback is explicit and records `fallback_reason`; otherwise the stage fails.

## Bootstrap data model

Each train-derived example receives several independently seeded no-hint student rollouts. Every rollout contains the student query, deterministic BM25 top-8 retrieval, raw response, strict parsed response, canonical citation context, score, truncation state, seed, and complete identity hashes.

Query and response eligibility are independent repair regions:

- A query target is eligible when it is student-generated, no-hint, nonempty, one-line, untruncated, contains no answer/policy leakage, and retrieves an answer-bearing source within top 8 under canonical recomputation.
- A response target is eligible only when it is student-generated, no-hint, untruncated, strict-format-valid, exact-answer-correct, citation-valid, lexically supported, and canonically revalidated on the same retrieval context.

One example may therefore provide query SFT without response SFT. Different no-hint seeds may supply the best query and response candidate, but each target retains its own exact prompt and retrieval identity.

The initial bootstrap pool is 2,048 deterministic train-derived examples with eight seeds each. Promotion requires at least 512 unique query targets and 256 unique response targets. Failure does not trigger target fabrication; it triggers a controlled prompt/scaffold experiment on the same examples and seeds.

## Prompt experiment policy

Direct mode remains the only bootstrap student mode. The current custom two-pass path is disabled because it copied policy instructions into queries and emitted mostly empty responses.

If response coverage is below the bootstrap gate, compare prompt candidates using identical rows, seeds, decoding parameters, and retrieval:

1. Current full-response prompt.
2. Continuation-oriented base-model prompt with two diverse worked examples.
3. Explicit fixed non-answer prefix such as `Answer:` represented in both inference and training prompt bytes.

No candidate may prepend labels after generation, repair malformed responses, or use a grammar decoder without reporting it as a different policy. A candidate is promoted only after strict metrics, manual raw-response review, and measured throughput improve without answer leakage.

## SFT architecture

Build a task-balanced prompt-completion dataset:

- Query rows teach concise search actions.
- Response rows teach grounded answer, reasoning, and source citation.
- Completion-only loss excludes prompt/source tokens.
- Sampling balances tasks so abundant query rows do not drown out response rows.

Run a 64-row overfit gate, then a deterministic 1% pilot, then full SFT. All stages use full-parameter Qwen3-4B BF16, DeepSpeed ZeRO-3, TF32, fused AdamW, non-reentrant gradient checkpointing, exact 4,096-token validation, regular checkpointing, and tested resume.

## Minimal-intervention collection after SFT

Run one teacher and one student per two-GPU worker. The student generates one complete query/search/answer attempt. On failure, the teacher localizes the earliest responsible region and returns one slight answer-free hint. Hint scope and strength escalate only after the preceding retry fails. The student always generates the repaired continuation.

After hinted success, run four no-hint siblings. SFT eligibility requires at least one verified no-hint success. Preference eligibility requires both successful and failed siblings so future gain is identifiable. Interventions are ranked by no-hint sibling gain per hint token and repair scope.

## DPO architecture

- Query pairs use byte-identical no-hint query prompts and rank student queries by future no-hint answer gain.
- Response pairs use byte-identical response prompts and canonical retrieval-context hashes.
- Chosen responses are verified correct student generations.
- Rejected responses are student generations from the same context.
- Reference log probabilities are precomputed once and reused only under an exact manifest match.

Run DPO overfit, 1% pilot, and full training from the promoted SFT checkpoint. Do not mix prompt or retrieval contexts to increase pair count.

## Monitoring and promotion

Every model promotion includes:

- A fixed 32-example train-derived canary set.
- A rotating 32-example deterministic random sample.
- Raw question, query, retrieval records, response, parser result, answer score, citations, and rendered sources.
- Diff against the previous checkpoint.
- Failure counts for query, retrieval, answer, format, support, citation, truncation, and leakage.
- Manual inspection of at least 20 representative outputs.

Loss is logged every 10 optimizer steps. Lightweight generation monitoring runs at every stable saved checkpoint during pilots and at a measured cadence during full runs. Full train-dev evaluation runs at epoch boundaries; official validation runs before promotion. A checkpoint is promoted only when answer metrics do not regress and structural/citation gates pass.

## Hardware and storage

Generation begins with the verified SDPA batch-4 baseline. Batch-size, vLLM, FlashAttention, static-cache, compile, packing, and worker candidates are isolated measurements; none is enabled by assumption. Sampled rollouts use stable per-example seed mapping so batch composition does not redefine a trajectory.

Before full training, compare four and eight A100s while holding global batch, data order, seed, and optimizer semantics constant. Gradient accumulation may differ to preserve global batch and must be recorded in the decision artifact.

Turing home is 83% full and cannot hold full optimizer checkpoints. Model caches, datasets used by jobs, rollout shards, reference log probabilities, and checkpoints live under a manifest-bound node10 scratch root. Home retains source, small logs, manifests, summaries, and compact reports. Obsolete large directories are deleted only after lineage and artifact preservation checks.

## Comparison and final reporting

The primary method is SFT followed by minimal-intervention DPO. Matched baselines are raw base, SFT-only, GRPO, and DAPO. GRPO/DAPO begin from the same SFT checkpoint and use comparable data and token/compute budgets. Reward components are logged separately to detect format-only or verbosity reward hacking.

After all choices are frozen, evaluate full validation, then materialize and evaluate the official 43,228-row test once per promoted arm. Produce JSON, JSONL, CSV, and HTML reports with confidence intervals, paired comparisons, failure categories, intervention efficiency, throughput, telemetry, configs, hashes, and representative raw outputs.

