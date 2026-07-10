# Paper-Scale Multilevel Feedback Experiment Design

Status: approved for implementation; optimizer revision approved on 2026-07-10

The canonical optimizer and search protocol is
`docs/design/training_hyperparameter_protocol.md`. If a historical smoke constant or
older plan conflicts with that document, the canonical protocol controls.

## Research Question

Does training on every wrong rollout preceding the first correct, teacher-guided
rollout improve Qwen3.5-2B more than standard DPO, which uses only the initial wrong
rollout, under a fixed model, data, parameter budget, tuning budget, and evaluation
protocol?

## Frozen Scope

- Student policy: `Qwen/Qwen3.5-2B`.
- Privileged teacher: `Qwen/Qwen3.5-9B`.
- Judgment-sensitive evaluator: `Qwen/Qwen3.5-9B`.
- Primary adaptation: BF16 LoRA, not quantized LoRA.
- Primary domains: full GSM8K followed by SearchQA-8K.
- Methods: prompt-only base, standard DPO, multilevel DPO, pair-budget-matched
  multilevel DPO, and GRPO.
- Three seeds for primary LoRA results.
- Optional full-parameter ablation: one GSM8K seed for standard DPO and multilevel DPO.
- No XML response format and no formatting loss.
- Student sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5.
- Student completion budget: 2048 tokens.
- Original GRPO is the primary online-RL baseline; a DAPO-loss run is labeled only as
  a sensitivity analysis.

## Dataset Protocol

### GSM8K

Use `openai/gsm8k`, configuration `main`, pinned to an explicit repository revision.

- Official train: 7,473 rows.
- Deterministic paper train: 6,726 rows.
- Deterministic validation: 747 rows sampled from official train.
- Final test: all 1,319 official test rows, untouched until model selection is frozen.

The train/validation split is defined by a checked-in manifest containing source row
IDs, source revision, canonical row hashes, seed, and split. The official test split
is never used for pair collection, prompt changes, hyperparameter selection, reward
changes, or early stopping.

The 747 validation examples are deterministically divided into 500 tuning-development
and 247 confirmation-development examples. Pilot training uses fixed subsets of the
6,726 paper-training examples. After hyperparameters are frozen, final seed runs use all
6,726 paper-training examples.

### SearchQA-8K

Use the original SearchQA release and preserve its official train, validation, and test
boundaries. Do not use the current MRQA-derived mirror as the paper dataset.

- Train: 5,000 sampled from the official 99,820-row training split.
- Validation: 1,000 sampled from the official 13,393-row validation split.
- Test: 2,000 sampled from the official 27,248-row test split.

Sampling is deterministic and stratified by answer length and context-length quantile.
Where metadata permits, source-year proportions are preserved. The resulting benchmark
must be named `SearchQA-8K`; results must not be described as full SearchQA test results.

Create a disjoint auxiliary hyperparameter pool from otherwise unused original
SearchQA rows: 2,000 official-train rows and 500 official-validation rows. These rows
are used only for tuning pilots. They never enter the SearchQA-8K final train,
validation, or test artifacts and never contribute to reported benchmark metrics.

### Integrity Requirements

- Exact expected source counts are checked before sampling.
- Every row has a stable source key and SHA256 content hash.
- No source key or normalized question crosses splits.
- Duplicate and near-duplicate audit results are persisted.
- Dataset revisions and split manifests are immutable once the first training run starts.

## Slight-Hint Teacher Policy

The teacher has privileged access to the gold answer only to locate the broad failure.
It never writes a corrected rollout and never reveals the answer.

Each hint must:

- contain one sentence and 8 to 15 words;
- identify only a broad error class or verification location;
- contain no digits, quantities, equations, proper nouns, answer initials, answer
  length, quoted evidence, or copied answer-bearing phrase;
- avoid explicit operations such as add, subtract, multiply, or divide;
- remain equally subtle on every retry; there is no final-step escalation.

Allowed examples:

- `Recheck how the quantities relate before performing the final calculation.`
- `Verify that your response matches the kind of entity being requested.`

The semantic guard evaluates the complete accumulated guidance history. If individually
safe hints combine into answer-bearing guidance, the trajectory ends as
`unsafe_accumulated_guidance` and creates no pair.

## Collection Policy

Collection runs only on each domain's training split.

1. Generate attempt zero from the base student prompt.
2. Evaluate the answer with the domain evaluator.
3. If wrong, ask the privileged teacher for one slight hint.
4. Validate the surface policy and semantic safety of the accumulated hints.
5. Retry from the original problem plus the accumulated safe hints.
6. Stop on the first correct response or after three guidance rounds.
7. Store every raw generation, evaluator decision, hint, guard decision, token count,
   latency, model revision, and attempt index.
8. Create no pair for unresolved or unsafe trajectories.

Validation and test generation receives no teacher hints. Gold answers are used only
after generation for scoring.

## Preference Construction

For a trajectory `wrong_0, wrong_1, ..., correct_k`:

- Standard DPO: `(correct_k, wrong_0)`.
- Multilevel DPO: `(correct_k, wrong_i)` for every `i < k`.
- Pair-budget-matched multilevel DPO: deterministic, attempt-stratified sample of the
  multilevel pairs with the same pair and optimizer-update budget as standard DPO.

Every DPO row uses the original student-facing prompt. Gold answers, teacher hints,
and evaluator outputs remain metadata only.

## Training Policy

Primary training uses BF16 LoRA:

- rank 16;
- alpha 32;
- dropout 0.05;
- architecture-audited text-backbone linear targets covering Qwen3.5 linear attention,
  full attention, and MLP projections;
- explicit exclusion of vision, multimodal projection, embeddings, and output head;
- no 4-bit or 8-bit quantization;
- identical trainable modules for DPO and GRPO;
- identical base checkpoint and chat template for all methods.

The model preflight writes every matched module and fails if module coverage, trainable
parameter count, or memory differs from the approved profile. The initial optimizer is
fused AdamW with beta coefficients `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`,
maximum gradient norm `1.0`, cosine decay, and 5% integer-step warmup.

Use deterministic successive halving. DPO searches learning rate
`{2e-6, 5e-6, 1e-5}` and beta `{0.05, 0.1, 0.3, 0.5}`. GRPO searches learning rate
`{2e-6, 5e-6, 1e-5}` and KL beta `{0.0, 0.001, 0.01, 0.04}`. Finalists receive
prespecified weight-decay, warmup, and scheduler checks. Every DPO method receives an
independent equal tuning budget, plus one shared-profile sensitivity seed. Final
reported LoRA results use three seeds. Full fine-tuning is not part of the main table
because it changes compute, storage, and optimization conditions.

## GRPO Reward Policy

The current substring reward is prohibited.

GSM8K reward uses canonical numeric correctness after evaluator-backed final-answer
extraction. SearchQA reward is frozen after preflight as:

`0.70 * exact_match + 0.15 * token_f1 + 0.10 * evidence_support + 0.05 * answer_type_correct`

Original GRPO uses four generations per prompt, one policy iteration per generation
batch, clipping epsilon `0.2`, within-group reward scaling, a 2048-token completion
limit, the same sampling settings as collection, and the shared domain evaluator
semantics. Truncated completions are masked. A full GRPO run cannot start if the pilot
has more than 50% zero-variance groups or more than 5% truncated completions. A
one-seed DAPO-loss sensitivity run is reported separately and never substituted for
the original-GRPO baseline.

## Evaluation and Statistics

Primary metrics:

- GSM8K exact numerical accuracy;
- SearchQA exact match, token F1, answer-type accuracy, and evidence-support rate;
- first-attempt accuracy, success by hint step, unresolved rate, unsafe-guidance rate;
- attempts to first correct, pair yield, and pairs per successful group;
- DPO loss, preference accuracy, chosen/rejected reward, and reward margin;
- GRPO reward mean/std, zero-variance groups, KL, clipping, and truncation;
- tokens, latency, throughput, peak GPU memory, wall time, and GPU-hours.

Report mean and standard deviation over three seeds, bootstrap 95% confidence intervals,
paired McNemar tests for exact correctness, paired bootstrap for SearchQA F1, Holm
correction across method comparisons, and effect sizes.

## Turing Execution Design

- Login node is used only for Git, queue inspection, submission, and small-file reads.
- All dataset materialization, generation, training, and evaluation runs under Slurm.
- Each collection shard requests one 48 GB GPU and must complete within three hours.
- Query account concurrency before choosing array concurrency; do not assume four GPUs.
- Use node-local scratch for model caches, datasets, environments, and compressed raw
  trajectories.
- Keep only manifests, metrics, reports, and final LoRA adapters in `/home`.
- Every job records commit, config, seed, package versions, dataset revisions, node,
  GPU, CUDA visibility, start/end times, and GPU telemetry.
- Failed shards remain explicit and are never silently skipped during merge.

## Domain Order and Gates

GSM8K must complete collection, DPO, GRPO, three-seed evaluation, statistics, and report
validation before SearchQA-8K starts. SearchQA-8K uses the same gates.

Full collection gates:

- evaluator parse success at least 99%;
- manual audit agreement at least 95%;
- nonzero pair yield;
- zero prompt/guidance/gold leakage;
- peak GPU memory below 90% of device memory;
- completion truncation at most 5%;
- every shard and merged artifact passes schema validation.

Final test runs begin only after prompts, rewards, hyperparameters, stopping rules, and
adapter-selection criteria are frozen in a signed run manifest.

Hyperparameter candidate failures, promotion decisions, validation metrics, tie-breaks,
and GPU-hours are retained in an immutable search ledger. Training loss is never the
model-selection metric.

## Required Deliverables

- Immutable dataset and split manifests.
- Compressed raw collection shards and merged trajectory groups.
- Standard, multilevel, and matched preference datasets.
- LoRA target-module inventories and trainable-parameter coverage reports.
- Complete hyperparameter candidate ledgers, promotion records, and freeze manifests.
- LoRA adapters for each method and seed.
- Base, validation, and final-test predictions for every method and seed.
- JSONL logs, TensorBoard logs, GPU CSV telemetry, metrics JSON, and HTML reports.
- Paper-ready CSV/LaTeX tables and SVG/PNG/PDF figures.
- Failure ledger and exact reproduction commands.
- Updated living design decision log.
