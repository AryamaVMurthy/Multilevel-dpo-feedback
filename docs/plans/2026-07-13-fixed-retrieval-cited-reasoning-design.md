# SearchQA Fixed-Retrieval Cited-Reasoning Design

## Objective

Train `Qwen/Qwen3-4B-Base` with full-parameter SFT and minimal-intervention DPO to actively search each question's fixed SearchQA result set, answer accurately, explain the evidence briefly, and cite real dataset sources. Compare against the raw base model, matched SFT, GRPO, and DAPO under one teacher-free evaluator.

## Non-negotiable model and runtime contract

- Student: pinned `Qwen/Qwen3-4B-Base`, BF16 full-parameter updates. Use the pinned 1.7B base model only after a recorded 4B OOM on the intended multi-GPU training configuration.
- Teacher: largest pinned Qwen3 post-trained instruct model that fits one GPU, starting with `Qwen/Qwen3-32B` in 4-bit NF4 with BF16 compute. Use the pinned 14B instruct model only after an explicit 32B load or inference failure.
- Total prompt plus completion length remains exactly 4,096 tokens.
- Student outputs never use XML. Teacher hints remain internal strict JSON.
- Missing source metadata, malformed actions, invalid citations, cache mismatches, cardinality mismatches, CUDA failures, and incomplete artifacts fail explicitly.

## Fixed retrieval environment

SearchQA supplies parallel snippet, title, URL, and related-link metadata. Materialization preserves one stable source record per original search-result index:

```text
source_id: S001
title: archival search-result title
url: archival search-result URL
snippet: archival search-result snippet
```

Empty snippets are excluded from retrieval but never cause remaining titles or URLs to shift. A usable source requires a non-empty snippet, title, and URL; rows without any usable source fail materialization and are counted by reason. URLs are archival dataset provenance and are not fetched or refreshed.

The student first receives the question and accumulated answer-free hints and generates one plain search query. A deterministic BM25 tool searches only that example's fixed source records and returns the top `k` records with stable source IDs. BM25 is an environment tool, not a substitute for model reasoning. Ties are resolved by original source index. The query, result IDs, ranks, scores, and elapsed time are logged.

The primary configuration uses one search round and `k=8`. Search-round count and `k` are frozen on a train-derived development split. Retrieval evaluation reports answer-bearing recall at 1/3/5/8, mean reciprocal rank, empty-query rate, and query length. Gold answers are used only by training/evaluation oracles and never by the search tool.

## Student response contract

After retrieval, the student receives only the question, retrieved records, and accumulated hints. It generates normal text in this strict grammar:

```text
Answer: Ada Lovelace
Reasoning: The retrieved biography identifies Lovelace as the author of the first published algorithm. [S014]
Sources: S014
```

The answer is concise; reasoning is one or two short evidence-grounded sentences; every reasoning claim has a stable citation; and `Sources` lists exactly the cited IDs. A strict line parser validates the grammar without guessing or repairing malformed output. The renderer expands cited IDs to the canonical dataset title and URL for human-visible predictions and reports. The model never invents or reproduces URLs.

The evaluator scores the parsed answer separately from the explanation. Primary answer metrics are normalized exact match and token F1. Retrieval/citation metrics include answer-bearing recall, valid-citation rate, citation precision, citation coverage, cited-answer support, duplicate citations, unsupported-source rate, malformed-response rate, and truncation rate. A fixed validation subset also receives blinded teacher faithfulness and reasoning-quality judgments; teacher judgments never determine answer EM.

## Thinking mode

The base student uses an explicit bounded private scratchpad pass before query generation and before the final response when the train-derived preflight proves that it improves answer accuracy or retrieval. Scratchpads are logged as private metadata, are never shown in the visible answer, and are never included in DPO chosen/rejected completions.

The post-trained teacher uses its native chat template and thinking mode. Its thinking is discarded. Only one schema-validated answer-free hint is consumed.

## Minimal-intervention trajectory

Each attempt contains two student actions:

1. Generate a search query.
2. Search the fixed source set and generate a cited response from returned records.

The deterministic evaluator identifies the earliest failing region in this order: search query/retrieval, response grammar, answer, reasoning support, citation selection. On failure, the teacher receives the question, private gold answer, complete source set, query, retrieved records, failed response, deterministic diagnostics, prior hints, and escalation level. It returns exactly `{"hint":"..."}` with at most 24 words, without giving the gold answer or a complete solution.

Hint strength and repair scope escalate only when the prior intervention did not resolve the earliest error. The student retries and itself generates every query and response. No teacher answer becomes an SFT or preference target.

After a hinted attempt succeeds, the system runs no-hint sibling rollouts. A successful continuation is eligible for training only when a no-hint sibling independently reproduces a correct, well-cited response. Interventions are ranked by future no-hint sibling gain divided by privilege tokens and repair scope. The cache records the verified repair region, sibling seed, gain, hint length, repair scope, and retrieval context hash.

## SFT bootstrap

SFT is used to teach the dual query and cited-response contracts before preference optimization. Targets come only from verified student-generated successful trajectories:

- Query SFT row: question-only query prompt and the student's successful no-hint query.
- Response SFT row: question plus frozen retrieved records and the student's successful no-hint cited response.

Rows conditioned on hints are not used directly. If the preflight cannot produce enough verified student successes, the stage fails and reports coverage; it does not replace missing targets with teacher answers or fabricated examples. Full SFT starts only after a 32-example overfit gate and a deterministic 1% pilot improve both answer and structural metrics.

## Causally valid DPO data

Query and response preferences are separate because search changes the downstream context.

- Query preference: the same no-hint query prompt, a student-generated chosen query with greater future no-hint answer gain, and a student-generated rejected query with lower gain. Retrieval context and sibling seeds are recorded.
- Response preference: the exact same no-hint response prompt and retrieved-record hash, a student-generated exact-correct supported response, and a student-generated failed response. Cross-context pairs are rejected.

The teacher's hint, answer, reasoning, and source text never appear in chosen/rejected completions. Preference construction rejects empty choices, identical pairs, mismatched prompt hashes, mismatched retrieval hashes, malformed citations, and unverified success labels.

## Full training and matched comparisons

Run order:

1. Preserve the completed legacy short-answer raw-base baseline as a clearly labeled archival baseline.
2. Run a new raw-base active-search/cited-reasoning validation baseline and inspect representative outputs.
3. Build verified student-success SFT data; run 32-example overfit and 1% pilots.
4. Run full 4B BF16 SFT with frequent loss evaluation, generation validation, checkpoints, and resume tests.
5. Collect minimal-intervention trajectories with the 32B quantized instruct teacher.
6. Build and audit query and response preferences, including no-hint sibling gain and context identity.
7. Run full-parameter DPO rounds from the frozen SFT checkpoint.
8. Run matched SFT, GRPO, and DAPO arms from the same initialization and comparable token/compute budgets.
9. Freeze all choices, evaluate validation and untouched test, bootstrap confidence intervals, and generate JSON/JSONL/CSV/HTML reports.

GRPO/DAPO reward is primarily exact answer accuracy, with bounded components for F1, retrieval recall, valid citations, citation support, and concise grounded reasoning. Malformed or fabricated citations receive explicit negative reward; verbosity never earns reward. Reward components and rates are logged separately to detect reward hacking.

## Experimental cardinality contract

The materialized SearchQA train split is a deterministic source reservoir, not the number of prompts consumed by an optimizer. Every reported training size counts unique, canonically validated prompt/completion rows after student-provenance, no-hint, retrieval-context, and method-specific eligibility gates. Repeated seeds, rejected candidates, raw source rows, and teacher hints do not inflate this count.

The primary data target is 15,000-20,000 verified training prompts. Report nested deterministic ablations at 1,000, 5,000, 10,000, and 15,000 prompts, with an optional 20,000-prompt arm when the verified pool has sufficient cardinality. All ablations use a single frozen ordering and stable ID/hash manifests so each smaller arm is a strict subset of the larger arm. If a requested arm is unavailable, fail with the measured eligible count and collect more train-reservoir examples; never pad, duplicate, fabricate, or silently relabel data.

Generate two student no-hint siblings per prompt by default. After the 4,096-prompt pilot is canonically audited, measure valid preference contrasts and future no-hint gain per GPU-hour. Generate a third or fourth student sibling only for prompts whose first two siblings lack a usable success/failure contrast and only if the measured marginal yield justifies the added compute. Teacher hints and teacher outputs are privileged interventions, never chosen or rejected candidates and never counted toward trajectory totals.

Freeze one 1,000-prompt model-selection validation set before DPO hyperparameter selection. It must be trajectory-disjoint from every training arm and remain fixed across SFT, DPO, GRPO, and DAPO comparisons. The full official validation split remains a final promotion evaluation, and the 43,228-example official test remains untouched until all methods and settings are frozen.

The active 4,096-example teacher collection is an initial verified-preference collection, not a claim that the 15,000-20,000-prompt primary target has been reached. Its audit determines preference yield per source example, whether two siblings are sufficient, and therefore the measured scale-up required for the final data package.

## Hardware and throughput strategy

- Baseline and evaluation: one model per GPU, multiple prompts per generation batch, deterministic hash sharding, one process per Slurm allocation, and exact ID merge checks.
- Collection: two-GPU shard workers, with one GPU holding the quantized teacher and one holding the BF16 student. Active examples are compacted between intervention rounds; student and teacher requests are batched independently. Multiple two-GPU shard jobs may run in parallel when resources permit.
- Teacher decoding is an explicit cache and manifest identity. The 4,096-prompt pilot used greedy Qwen3-32B thinking and exposed a high retry rate. Before the 15,000-20,000 scale run, compare that frozen lineage against the Qwen3 model-card sampling candidate (`temperature=0.6`, `top_p=0.95`, `top_k=20`) on the same prompts, seed rule, model revision, token caps, batch size, and evaluator. Promote a candidate only if answer-free contract validity, useful-hint yield, and downstream preference yield are non-inferior while retries, recoveries, or GPU-hours per valid pair improve. Every intervention round resets the teacher RNG to `collection_seed + attempt_index`, so resumed and uninterrupted sampled runs have the same seed state.
- Do not introduce a synthetic thinking-budget shortcut into the primary run. Qwen's documented open-source budget mechanism requires a model-specific two-generation continuation and recommends budgets above 1,024 tokens; any such candidate must first prove final-content parity inside this experiment's stricter 4,096-token total budget.
- Training: single-node four-GPU DeepSpeed ZeRO-3 for the 4B full model, microbatch 1, BF16, TF32, fused AdamW, non-reentrant gradient checkpointing, pinned persistent dataloader workers, and completion-only loss. FlashAttention 2, sequence packing, static KV cache, and `torch.compile` are enabled only after isolated compatibility and throughput probes beat the SDPA reference without output drift.
- DPO reference log probabilities are precomputed once and reused only under an exact manifest match.
- Every performance change is accepted on measured examples/second, tokens/second, peak memory, GPU utilization, and identical evaluation output—not assumed.

## Validation, checkpointing, and monitoring

Before each full stage, run an end-to-end 32-example gate covering data, search, model generation, parser, source rendering, metrics, cache identity, checkpoint save, checkpoint resume, and manual sample inspection. Training logs loss every 10 optimizer steps, evaluates at a dataset-size-derived cadence, saves resumable checkpoints regularly, retains best plus latest checkpoints, and runs generation validation on a fixed train-dev subset. Full validation occurs at epoch boundaries and before promotion.

Promotion requires no regression in answer EM and explicit minimum gates for non-empty responses, valid grammar, valid citations, citation support, retrieval recall, truncation, and no source fabrication. Failures stop dependent jobs and record an actionable reason. Hyperparameter changes are one-variable, measured experiments; there are no random retries or hidden fallbacks.

## Success criteria

- The raw base and every trained checkpoint actively generate a query before receiving search results.
- Human-visible outputs include an answer, concise cited reasoning, and canonical SearchQA titles/URLs.
- Answer accuracy, retrieval, citation, faithfulness, malformed-output, and performance metrics are all reported.
- SFT and DPO targets are student-generated and verified; teacher output is limited to answer-free hints.
- Every response DPO pair shares the exact prompt and retrieval context.
- Qwen3-4B receives full BF16 updates unless a documented intended-configuration OOM authorizes 1.7B.
- DPO is compared fairly against raw base, SFT, GRPO, and DAPO.
- Validation/checkpoint/resume gates and untouched-test reporting are complete and reproducible.
