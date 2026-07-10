# Training Hyperparameter Protocol

Status: approved on 2026-07-10

This document is the canonical optimizer, LoRA coverage, hyperparameter search, and
model-selection protocol for the paper-scale GSM8K and SearchQA-8K experiments. The
historical smoke-run constants in `implementation/src/text_feedback_dpo/training.py`
are runtime checks only and are not approved paper settings.

## Verified Sources

| Source | Status | Verified use |
| --- | --- | --- |
| [TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer) | official software documentation | DPO beta semantics and defaults, PEFT learning-rate guidance, logged metrics |
| [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer) | official software documentation | KL beta, clipping, update count, loss variants, reward scaling, truncation masking |
| [Transformers TrainingArguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) | official source | fused AdamW, Adam coefficients, scheduler, warmup, weight decay, clipping |
| [PEFT LoRA guide](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md) | official software documentation | broad linear-module targeting support |
| [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) | official model documentation | model identity, architecture, chat template, supported Transformers integration |
| [Direct Preference Optimization](https://papers.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf) | peer-reviewed NeurIPS 2023 paper | standard DPO objective and beta sensitivity |
| [Beta-DPO](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea888178abdb6fc233226d12321d754f-Paper-Conference.pdf) | peer-reviewed NeurIPS 2024 paper | evidence that DPO performance is beta-sensitive |

The exact Git revisions and installed versions of TRL, Transformers, PEFT, PyTorch,
Accelerate, Datasets, and Qwen model artifacts are frozen in each run manifest. A
full job must fail if its installed API does not expose every configured field.

## Qwen3.5 LoRA Coverage Gate

Qwen3.5-2B has a hybrid text architecture: 18 linear-attention layers and six
full-attention layers. Targeting only `q_proj`, `k_proj`, `v_proj`, and `o_proj`
therefore does not establish adequate text-backbone coverage.

Before training, load the pinned model revision and inventory every named trainable
module. Build an explicit text-backbone-only linear-module target set that:

- includes the linear-attention, full-attention, and MLP projections;
- excludes the vision encoder, multimodal projector, embeddings, and output head;
- resolves to a nonempty, expected module list;
- records every matched module name and shape;
- records total and trainable parameter counts and the trainable percentage;
- is identical across standard DPO, multilevel DPO, matched DPO, and GRPO.

The primary profile uses rank 16, alpha 32, and dropout 0.05. A coverage preflight
compares the resulting profile with an attention-only rank-16 profile. This is a
coverage audit, not a method-specific tuning opportunity. If broad targeting does not
fit one 48 GB GPU at the approved sequence length, the run fails and reports measured
memory; rank reduction or two-GPU execution requires an explicit amended profile used
by every method.

## Fixed Optimizer Foundation

The initial optimizer profile is:

| Field | Value |
| --- | --- |
| optimizer | `adamw_torch_fused` |
| Adam beta1 | `0.9` |
| Adam beta2 | `0.999` |
| Adam epsilon | `1e-8` |
| weight decay | `0.01` |
| maximum gradient norm | `1.0` |
| precision | BF16 |
| scheduler | cosine decay |
| warmup | 5% of optimizer updates |

Warmup is materialized as an integer `warmup_steps` value after the exact optimizer
update count is known. No deprecated or ambiguous warmup argument is written to the
frozen configuration.

## Data Used For Tuning

Official test data is never used for tuning, stopping, prompt changes, reward changes,
or model selection.

### GSM8K

- Final paper train: 6,726 examples from official train.
- Validation: 747 examples from official train.
- Tuning development: deterministic 500-example subset of validation.
- Confirmation development: the remaining 247 validation examples.
- Pilot training uses deterministic subsets of the 6,726 training examples.
- After settings freeze, final seed runs use all 6,726 training examples.

No external GSM8K-like benchmark is introduced by default because distribution shift
and benchmark contamination would weaken the interpretation. An external math set may
be added only as a separately reported transfer diagnostic, never as a replacement for
GSM8K validation.

### SearchQA-8K

In addition to the disjoint 5,000/1,000/2,000 SearchQA-8K split, sample an auxiliary
hyperparameter pool from otherwise unused rows in the original official SearchQA
splits:

- auxiliary tuning train: 2,000 rows from unused official train rows;
- auxiliary tuning validation: 500 rows from unused official validation rows.

These rows must be disjoint by source key, normalized question, and content hash from
SearchQA-8K and from each other. They are used only for pilot training and selection,
never added to a final training run and never included in reported SearchQA-8K metrics.
The chosen configuration receives one confirmation evaluation on the main 1,000-row
validation split before it is frozen.

## Deterministic Search Strategy

Use deterministic successive halving, not an opaque random search. Every candidate,
promotion, rejection, and failure is written to the search ledger. All methods receive
the same candidate set, pilot examples, update budget, seed, and promotion rule.

### DPO Candidates

- learning rate: `2e-6`, `5e-6`, `1e-5`;
- DPO beta: `0.05`, `0.1`, `0.3`, `0.5`;
- loss: sigmoid DPO;
- maximum duration: one epoch;
- effective global batch: 16 preference pairs.

First screen the 12 learning-rate/beta combinations with the fixed optimizer profile.
Promote the top four, then compare weight decay `0.0` versus `0.01`, warmup 5% versus
10%, and linear versus cosine scheduling through a fixed fractional design. Promote
the top two to full pilot data and two tuning seeds. Standard, multilevel, and matched
DPO receive independent but equal-budget selection. Also run one shared-profile seed
for all three methods to isolate the preference-data effect from method-specific
hyperparameter selection.

### GRPO Candidates

- learning rate: `2e-6`, `5e-6`, `1e-5`;
- KL beta: `0.0`, `0.001`, `0.01`, `0.04`;
- clipping epsilon: `0.2`;
- policy iterations per generation batch: one;
- generations per prompt: four;
- original objective: `loss_type="grpo"`;
- reward scaling: within group;
- truncated completion masking: enabled.

The 12 learning-rate/KL combinations use the same successive-halving stages. The main
baseline is explicitly labeled original GRPO. A one-seed sensitivity run uses the
current TRL DAPO loss and is labeled `dapo_sensitivity`; it is not reported as standard
GRPO and cannot replace a failed original-GRPO run.

### Shared SearchQA GRPO Reward

The SearchQA reward is a fixed weighted sum recorded in every reward artifact:

| Component | Weight |
| --- | ---: |
| alias/normalized exact match | 0.55 |
| token F1 | 0.25 |
| controlled-evidence support | 0.10 |
| answer-type correctness | 0.10 |

When the official row has no known answer type, the type component is neutral `0.5`;
it is not treated as an incorrect answer. Ambiguous or evaluator-failed rows are
explicitly routed to the evaluator and never silently receive a reward.

## Promotion And Freeze Rules

A candidate is invalid, not merely low-ranked, if it has a nonfinite loss, nonfinite
gradient norm, zero optimizer updates, schema-invalid artifacts, evaluator failure,
gold leakage, peak memory at or above 90% of device memory, or more than 5% truncated
completions. GRPO also requires nonzero reward variance and audited reward agreement of
at least 95%.

Rank valid candidates by teacher-free held-out task performance:

- GSM8K: exact numerical accuracy;
- SearchQA: exact match, then token F1 as the prespecified tie-breaker.

Training loss is diagnostic and never the selection metric. If candidates are tied
within the confirmation set's resolution, select the lower learning rate, then the
higher regularization beta, then the lower measured GPU-hours. These tie-breakers are
fixed before any pilot result is inspected.

The winner is frozen in a signed manifest containing the complete config, source and
dataset hashes, candidate ledger hash, selection evidence, checkpoint rule, seeds, and
official test command. Three final seeds are run only after this manifest validates.

## Required Hyperparameter Observability

Every run records learning rate by step, scheduler value, optimizer identity, Adam
coefficients, weight decay, warmup steps, gradient norm before clipping, clipping rate,
effective global batch, tokens per update, DPO beta or GRPO KL beta, policy/reference
KL, loss components, preference accuracy or reward distribution, response length,
truncation, peak GPU memory, throughput, wall time, and GPU-hours.

The final HTML report includes candidate tables, promotion paths, validation curves,
failure reasons, selected settings, and sensitivity plots for learning rate and beta.

## Artifact Commands

Paper workflows use the implementation CLI with explicit immutable paths:

- `materialize-dataset` writes the pinned dataset manifest and compressed role files.
- `collect-shard` and `merge-collection` write resumable compressed trajectory records.
- `build-preferences` writes `standard`, `multilevel`, and `matched` datasets.
- `init-search-ledger`, `tune-paper`, `promote-search-stage`, and `freeze-search`
  create the auditable candidate-selection chain.
- `train-paper` accepts only a freeze manifest; `evaluate-paper` requires it for the
  official test split and writes a one-time test marker.

All Turing scripts use scratch for caches and virtual environments. Collection,
training, and evaluation record GPU telemetry; materialization and merge are CPU-only.
