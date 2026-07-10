# Qwen3.5-2B DPO, GRPO, And Distillation Pretest Design

> Historical pretest specification. Paper-scale work is governed by
> `2026-07-10-paper-scale-experiment-design.md` and
> `../design/training_hyperparameter_protocol.md`.

**Status:** Design approved for planning. No implementation has started.

**Current date:** 2026-07-09

**Goal:** Test the textual-feedback DPO method on `Qwen/Qwen3.5-2B`, compare it against standard GRPO and on-policy distillation, and measure whether gains come from stronger teacher capacity or privileged correction context.

## Verified External Facts

- `Qwen/Qwen3.5-2B` exists on Hugging Face and is a 2B post-trained Qwen3.5 model.
- `Qwen/Qwen3.5-9B` exists on Hugging Face and is the recommended stronger teacher for the first full comparison.
- TRL v0.29.0 supports:
  - `DPOTrainer` with explicit `prompt`, `chosen`, `rejected` preference rows.
  - `GRPOTrainer` with prompt datasets and explicit reward functions.
  - PEFT LoRA for both DPO and GRPO via `LoraConfig`.

## Core Experimental Question

Can `Qwen/Qwen3.5-2B` improve structured self-reflective reasoning on Search-QA and Math more from textual-feedback DPO than from prompt-only structure, format SFT, standard GRPO, or on-policy teacher distillation?

Secondary question:

Does the teacher need to be a larger model, or can the same 2B model act as an effective teacher when given privileged training-only information?

## Student

All trained and evaluated student arms use:

```text
Qwen/Qwen3.5-2B
```

Use text-only prompts for this pretest. The model card presents Qwen3.5-2B as a multimodal-capable model, but V1 Search-QA and Math are text-only. The implementation must verify the correct Transformers loading path before training.

## Teacher Conditions

### T9B: Stronger-Model Teacher

Teacher:

```text
Qwen/Qwen3.5-9B
```

Use this as the primary teacher. It is the cleanest version of the original method: a stronger frozen model repairs weaker student failures.

Teacher input:

- problem
- gold answer
- student rollout
- tool observations
- evaluator result
- domain
- correction instructions

Teacher output:

```text
<feedback>
...
</feedback>

<corrected_rollout>
...
</corrected_rollout>
```

### T2B-Priv: Same-Model Privileged Teacher

Teacher:

```text
Qwen/Qwen3.5-2B
```

This teacher uses the same base model as the student but receives privileged training-only context.

Privileged inputs:

- gold answer
- evaluator result
- retrieved supporting evidence for Search-QA
- derivation/check hints for Math
- domain-specific verification checklist
- original student rollout

Purpose:

This tests whether teacher capacity is essential, or whether privileged correction context is enough to create useful DPO pairs.

Important rule:

Privileged information is allowed only during pair generation or distillation data generation. It is never available during student evaluation.

## Experiment Arms

| Arm | Student | Teacher | Method | Purpose |
|---|---|---|---|---|
| E0 | 2B | none | normal prompt | raw baseline |
| E1 | 2B | none | structured bracket prompt | prompt-only structure baseline |
| E2 | 2B | none | format SFT | structure imitation baseline |
| E3a | 2B | 9B | textual-feedback DPO | main method with stronger teacher |
| E3b | 2B | 2B privileged | textual-feedback DPO | same-model privileged-teacher ablation |
| E4 | 2B | none | standard GRPO | online RL baseline |
| E5a | 2B | 9B | on-policy distillation | stronger-teacher imitation baseline |
| E5b | 2B | 2B privileged | on-policy distillation | same-model privileged imitation baseline |

## Data

### Smoke Pretest

- 100 HotpotQA examples
- 100 GSM8K examples
- 25 validation examples per domain

Purpose:

Verify parsing, tool execution, teacher correction quality, reward functions, training scripts, evaluation scripts, and reporting.

### Dry Run

- 500 HotpotQA examples
- 500 GSM8K examples
- 100 validation examples per domain

Purpose:

Produce first meaningful method comparisons.

### Scale Only After Dry Run

Do not scale to NQ, MATH, 2WikiMultiHopQA, or MuSiQue until smoke and dry-run reports identify no pipeline-quality blockers.

## Structured Output Format

All methods that use the structured prompt train or evaluate trajectories in this format:

```text
<plan>
...
</plan>

<think branch="A">
...
</think>

<tool branch="A">
...
</tool>

<reflect>
Branch comparison:
...

Evidence / derivation check:
...

Verification:
...

Decision:
...
</reflect>

<final>
...
</final>
```

Rules:

- `<reflect>` must appear before `<final>`.
- `<reflect>` must contain a real verification section.
- Search-QA allows at most 3 branches.
- Math allows at most 2 branches.
- Tool observations must be real controlled observations, not model-fabricated text.

## Textual-Feedback DPO Arms: E3a And E3b

For each example:

1. Student generates original rollout.
2. Controlled tools execute any valid tool calls.
3. Evaluator scores original rollout.
4. Teacher writes feedback and corrected rollout.
5. Evaluator scores corrected rollout.
6. Pair filter keeps only useful pairs.

DPO row:

```json
{
  "prompt": "problem + format/tool instructions",
  "chosen": "corrected rollout",
  "rejected": "original student rollout"
}
```

Teacher feedback is stored in metadata but excluded from the DPO prompt.

Initial filtering:

- original is wrong
- corrected rollout is correct
- corrected rollout has valid brackets
- corrected rollout has verification inside `<reflect>`

Training:

- LoRA or QLoRA.
- Format SFT warmup first.
- DPO second.
- If exact `DPO + 0.1 FormatSFT` is required, implement a custom tested trainer later. For pretest, use explicit two-stage training.

## Standard GRPO Arm: E4

GRPO uses no teacher-corrected pairs.

Input:

- prompt-only dataset with problem and structured-output instructions.

Reward components:

- final answer correctness
- format validity
- verification present
- verification valid
- no premature `<final>`
- controlled Search-QA evidence support
- Math arithmetic/substitution/constraint check
- small penalty for excessive branches
- small penalty for excessive tool calls

Do not reward malformed outputs through silent repair. If parsing fails, the reward function records the parse failure and gives the explicit configured failure reward.

## On-Policy Distillation Arms: E5a And E5b

For each example:

1. Current 2B student samples an on-policy rollout.
2. Teacher receives the sampled rollout and correction context.
3. Teacher produces corrected rollout.
4. Student trains with SFT on corrected rollout.

This baseline tests whether teacher imitation alone explains the gains, without DPO preference learning.

Use the same teacher conditions as DPO:

- E5a: 9B teacher
- E5b: 2B privileged teacher

## Evaluation

Evaluation prompt access must be identical across trained students.

Evaluation never includes:

- gold answer
- teacher feedback
- privileged hints
- oracle derivations

Metrics:

- final answer accuracy
- Search-QA exact match and F1
- Math exact answer accuracy
- format validity
- verification-present rate
- verification-valid rate
- premature final answer rate
- branch count
- tool-call count
- controlled evidence support rate
- arithmetic error rate
- substitution-check rate
- constraint-check rate

Primary comparisons:

- E3a vs E1: does stronger-teacher DPO beat structured prompting?
- E3a vs E2: does stronger-teacher DPO beat format imitation?
- E3a vs E4: does stronger-teacher DPO beat standard GRPO?
- E3a vs E5a: does DPO beat stronger-teacher distillation?
- E3b vs E3a: how much does teacher capacity matter?
- E5b vs E5a: how much does teacher capacity matter for distillation?
- E3b vs E5b: does same-model privileged DPO beat same-model privileged imitation?

## Turing Execution Assumptions

- Use Turing `u22`.
- Do not train or generate model rollouts on the login node.
- Use `sbatch` for all real jobs.
- Use `sinteractive` only for smoke debugging.
- Store Hugging Face caches, temporary datasets, retrieval indexes, and rollout scratch in `/scratch`.
- Store source, configs, compact logs, summary metrics, reports, and final LoRA adapters in `/home`.
- Slurm account must be explicit in config or job export. No default account is allowed.

## Failure Policy

Fail fast on:

- missing Slurm account
- missing model id
- missing dataset split
- missing retrieval index
- malformed tool call
- fabricated or absent tool observation
- malformed teacher correction
- corrected rollout without verification
- CPU fallback during GPU job
- live web use in controlled Search-QA

If any fallback is intentionally configured, it must be visible in logs with `fallback_reason`.

## Recommended Next Step

Write an implementation plan that modifies the existing end-to-end Turing plan for this pretest:

- replace student with `Qwen/Qwen3.5-2B`
- add teacher configs for `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-2B` privileged
- add GRPO trainer path
- add on-policy distillation path
- add reporting for E0/E1/E2/E3a/E3b/E4/E5a/E5b
