# Multilevel Feedback DPO for Small Reasoning Models

## Comprehensive Research and Execution Plan

- Document date: 2026-07-10
- Project: Multilevel Feedback DPO
- Primary student: Qwen/Qwen3.5-2B, post-trained checkpoint
- Privileged teacher and evaluator: Qwen/Qwen3.5-9B, post-trained checkpoint
- Primary benchmark: MATH Levels 4-5
- Secondary domain: SearchQA-8K
- Execution environment: IIIT Hyderabad Turing GPU cluster
- Current code branch: agent/qwen35-pretest
- Current verified code revision: 23108a63805cb28811df48f6c5cd17bad0fad083
Document status: project blueprint; observed results and planned experiments are explicitly separated

## Executive Summary

This project studies whether a small reasoning model can learn more effectively from an ordered sequence of its own failures than from a single conventional preference pair. The core mechanism is an iterative, privileged-guidance collection loop. A Qwen3.5-2B student first solves a problem naturally. If its answer is wrong, a frozen Qwen3.5-9B teacher receives privileged access to the gold solution and the failed response, but is permitted to return only a very slight answer-free hint. The student retries from the original problem with accumulated safe hints. Collection stops at the first correct response or after a fixed retry budget.

For every successful trajectory, the first correct response is the chosen completion. Standard preference construction pairs it only with the initial wrong response. Multilevel preference construction pairs it with every preceding wrong response. A pair-budget-matched multilevel control separates the benefit of attempt diversity from the benefit of simply having more optimizer examples. Gold answers, hints, evaluator outputs, and retry metadata never enter the DPO prompt. The trained student must solve teacher-free at inference time.

The original design proposed rigid XML trajectories, teacher-written corrected rollouts, GSM8K as the first math benchmark, and ordinary sequence-summed DPO. Empirical work invalidated several of those choices. Qwen3.5 performs more naturally without XML constraints. GSM8K is too easy for the 2B post-trained checkpoint and produces very few useful preference trajectories. A 16-example teacher-free GSM8K preflight reached 81.25 percent exact accuracy while two responses hit the full 8,192-token ceiling. A subsequent 16-example MATH Levels 4-5 thinking-mode diagnostic reached only 31.25 percent protocol-valid accuracy because 11 of 16 generations hit the ceiling, although nine of those 11 already contained a deterministically gold-equivalent extracted answer. This establishes failure to terminate, rather than answer discovery, as a dominant nuisance variable. The primary student protocol therefore uses the same post-trained Qwen3.5-2B checkpoint in explicit non-thinking mode with a boxed-answer-and-stop prompt. Thinking mode remains a labeled secondary ablation. The project uses the official MATH competition dataset, with Levels 4-5 as the primary study, while preserving GSM8K and thinking-mode artifacts as diagnostics only.

The primary preference objective will be length-normalized DPO, implemented with the locked TRL `sigmoid_norm` loss for both standard and multilevel pair construction. Length-desensitized DPO, using `ld_alpha` to downweight verbose response tails, is a separately labeled ablation. The two mechanisms are not silently combined in the primary result. This keeps the paper claim interpretable: the main comparison isolates preference construction under the same length-normalized objective.

The complete experiment suite includes the frozen base model, response SFT, on-policy distillation, standard length-normalized DPO, multilevel length-normalized DPO, pair-budget-matched multilevel DPO, length-desensitized DPO, standard GRPO, and a separately labeled DAPO sensitivity. Every method uses the same Qwen3.5-2B base revision, LoRA target inventory, evaluator semantics, held-out splits, token ceiling, and final seeds. The principal scientific question is whether multilevel preference construction improves teacher-free MATH accuracy, sample efficiency, first-correct dynamics, and robustness after controlling for pair budget and response length.

## 1. Research Objective

### 1.1 Primary Research Question

Can a 2B post-trained language model learn a stronger teacher-free mathematical reasoning policy when preference optimization uses every wrong attempt that precedes the first correct response, rather than only the initial failure?

### 1.2 Secondary Research Questions

1. Does multilevel feedback improve MATH Levels 4-5 exact answer accuracy over standard single-pair preference construction under an identical length-normalized objective?
2. Are gains retained when the multilevel dataset is matched to the standard dataset by pair count and optimizer-update budget?
3. Does privileged slight guidance improve collection yield without leaking answers into student prompts or final training records?
4. Does length normalization reduce the response-length bias of ordinary DPO without suppressing necessary mathematical derivations?
5. Does length-desensitized tail weighting provide an additional benefit, or does it over-penalize valid long reasoning?
6. How do offline preference methods compare with standard GRPO and on-policy distillation under the same base model, evaluator, LoRA coverage, and compute accounting?
7. Which failure categories are corrected by early hints, and which remain unresolved after three guidance rounds?

### 1.3 Falsifiable Hypotheses

H1: Multilevel LN-DPO produces higher teacher-free MATH Levels 4-5 accuracy than Standard LN-DPO at equal model, prompt, optimizer, and tuning budget.

H2: Multilevel Matched LN-DPO outperforms Standard LN-DPO when pair count and optimizer updates are equal. If it does not, any raw multilevel gain may be attributable to increased data volume rather than trajectory structure.

H3: Privileged slight guidance increases the fraction of initially wrong problems that reach a first correct response within three retries, while maintaining zero answer leakage under manual and model audit.

H4: Length-normalized DPO reduces median generated tokens and truncation relative to sequence-summed DPO without reducing exact accuracy. Sequence-summed DPO is not a primary method, but a small one-seed diagnostic may be retained if needed to measure this mechanism.

H5: Standard GRPO improves over the base model but is less data efficient than multilevel LN-DPO at the same measured GPU-hour budget.

H6: On-policy distillation improves answer accuracy but may copy teacher solution style more strongly than the slight-hint method; this difference should appear in response-style and teacher-similarity diagnostics.

### 1.4 Intended Contribution

The intended contribution is not a new answer generator or a rigid reasoning format. It is a data construction and preference optimization method for learning from ordered failed attempts under privileged but answer-safe guidance. The contribution has four parts:

- an iterative collection policy that stops at the first correct response;
- a multilevel preference group that preserves every failure before success;
- a fair matched-pair control that isolates trajectory diversity;
- a length-aware objective and evaluation protocol that keeps completion-length effects observable without making extended thinking a prerequisite.

## 2. Evolution of the Design

### 2.1 Original V1 Concept

The original July 2026 design proposed Qwen3-4B-Instruct as student, a stronger frozen teacher, Search-QA plus GSM8K, XML blocks such as `<plan>`, `<think>`, `<reflect>`, and `<final>`, teacher-written corrected rollouts, ordinary DPO, and a small format-SFT term. This was a reasonable controlled prototype, but it made output structure a central part of the method.

### 2.2 Why the Original Format Was Rejected

The project objective is to evaluate multilevel feedback, not XML compliance. Strict tags made parsing and format correction dominate the experiment and introduced failure modes unrelated to mathematical reasoning. The current design preserves raw model generations exactly and delegates answer extraction to a dedicated evaluator. No `<think>`, `<reflect>`, tool tag, or corrected-rollout format is required. The sole output contract for MATH is a clearly identifiable final `\boxed{}` answer followed by termination.

### 2.3 Why Teacher-Written Corrected Rollouts Were Rejected

If the teacher writes the chosen response, DPO may primarily distill teacher language and solution style. The current method requires the student itself to generate the first correct response after receiving slight hints. The resulting pair compares student failures with a student success, preserving on-policy response style and making the preference signal more directly about self-correction.

### 2.4 Why GSM8K Was Replaced

GSM8K contains useful grade-school word problems, but the pinned Qwen3.5-2B checkpoint already solved most audited samples. R1 showed 54 of 64 first attempts judged correct and yielded only one accepted preference pair. The later frozen baseline preflight reached 13 of 16 correct despite severe verbosity. A benchmark that generates few wrong-to-correct trajectories cannot adequately test multilevel preference construction.

MATH contains 12,500 competition-level problems with complete solutions. The official release provides 7,500 training examples and 5,000 test examples over seven subjects and five difficulty levels. Levels 4-5 are difficult enough to produce meaningful failed attempts and guidance trajectories while remaining automatically scoreable through final symbolic answers.

### 2.5 Why Ordinary DPO Was Replaced

Native reasoning responses vary substantially in length. Sequence-summed log probabilities scale with completion length and can create a preference signal that is partly a length artifact. The primary objective therefore normalizes policy-reference scores by completion-token count. A separate length-desensitized ablation downweights tokens beyond the shared chosen/rejected length. This design follows current TRL capabilities and keeps length treatment visible in every run manifest.

:::decision-flow

## 3. Exact Multilevel Feedback Method

### 3.1 Roles

| Role | Model | Privileged information | Thinking | Decoding | Maximum new tokens |
| --- | --- | --- | --- | --- | ---: |
| Student | Qwen3.5-2B post-trained | Problem, accumulated safe hints | Disabled | Sampled | 8192 |
| Teacher | Qwen3.5-9B post-trained | Problem, gold solution, failed response, evaluator result | Disabled | Greedy | 64 |
| Evaluator | Qwen3.5-9B post-trained | Problem, gold answer/solution, student response | Disabled | Greedy | 256 |
| Guidance guard | Qwen3.5-9B post-trained | Problem, gold answer, accumulated hints | Disabled | Greedy | 8 |
| Guidance critic | Qwen3.5-9B post-trained | Problem, gold solution, failed response, proposed hint | Disabled | Greedy | 8 |

The teacher, evaluator, guard, and critic may reuse the same pinned 9B weights, but they are separate logical roles with separate prompts, generation profiles, logs, and metrics. The teacher is post-trained/instruct, not the raw `Qwen3.5-9B-Base` checkpoint.

### 3.2 Student Generation Profile

The approved student profile is:

- `enable_thinking=false`;
- `do_sample=true`;
- `temperature=1.0`;
- `top_p=1.0`;
- `top_k=20`;
- `presence_penalty=2.0`;
- `max_new_tokens=8192`.

This is Qwen's recommended sampled non-thinking text profile, with a project-specific 8,192-token ceiling to preserve the fixed training sequence budget. The ceiling is not a target. Exact prompt tokens, generated tokens, EOS termination, length termination, latency, and finish reason must be recorded. A length-truncated response is incorrect even when an intermediate number matches the gold answer.

### 3.3 Attempt Zero

The student receives only the original problem and the frozen non-thinking boxed-answer-and-stop prompt. It does not receive the gold answer, reference solution, evaluator rubric, teacher hint, or information about future retries. The raw response is stored without rewriting.

### 3.4 Evaluation

The evaluator first extracts the final answer in a structured response. For MATH, deterministic verification then checks normalized symbolic or numeric equivalence against the gold final answer. Ambiguous cases are adjudicated by the model evaluator and preserved for manual audit. Evaluator serialization failures trigger bounded repair turns; exhaustion is a hard artifact failure.

### 3.5 Privileged Slight Guidance

When the student is wrong, the teacher sees the gold solution only to locate the earliest broad error. It returns one sentence of 5-25 words that identifies a relation, assumption, missing constraint, or verification to reconsider. The hint must not contain the exact answer, an equivalent expression, or sufficient information to copy the solution.

Examples of acceptable guidance:

- Recheck which quantity should be held fixed before applying the ratio.
- Verify whether the requested value is an angle, a length, or a count.
- Compare the boundary case with the expression you derived.

The guidance critic checks mathematical direction and relevance. The leakage guard checks the proposed hint together with all previous hints. A surface-invalid, semantically unsafe, or mathematically invalid hint is regenerated with the prior review included. Exhausted regeneration creates an explicit unresolved trajectory and no pair.

### 3.6 Retry Policy

The student retries from the original problem plus accumulated accepted guidance. Prior raw responses are not copied into the new student prompt. The student remains free to reason in its native style. Collection stops at the first correct answer or after three guidance rounds. Unresolved groups are stored, audited, and excluded from DPO pair creation.

### 3.7 Preference Group

For a successful trajectory:

```text
wrong_0 -> hint_0 -> wrong_1 -> hint_1 -> ... -> correct_k
```

the first correct response `correct_k` is the sole chosen completion. Every response before it is a rejected completion with its original attempt index retained.

Standard construction:

```text
(prompt, chosen=correct_k, rejected=wrong_0)
```

Multilevel construction:

```text
(prompt, chosen=correct_k, rejected=wrong_i) for every i < k
```

Matched multilevel construction samples from all multilevel pairs so that pair count, attempt-depth distribution, and optimizer updates match the standard condition according to a frozen deterministic rule.

### 3.8 Critical Prompt Invariant

Every DPO prompt is the original student-facing problem prompt. It excludes gold answers, reference solutions, teacher hints, evaluator results, attempt numbers, and future-correction information. This is what forces the student to internalize a better policy rather than depend on the teacher at inference time.

## 4. Length-Aware Preference Objectives

### 4.1 Standard DPO Reference

For prompt `x`, chosen response `y+`, rejected response `y-`, policy `pi_theta`, reference `pi_ref`, and regularization coefficient `beta`, ordinary DPO uses the sequence-level margin:

```text
Delta = [log pi_theta(y+|x) - log pi_ref(y+|x)]
      - [log pi_theta(y-|x) - log pi_ref(y-|x)]

L_DPO = -log sigmoid(beta * Delta)
```

Because each log probability is summed over response tokens, the magnitude depends on response length.

### 4.2 Primary Length-Normalized DPO

The primary objective uses the average policy-reference log ratio per completion token:

```text
s(y|x) = [log pi_theta(y|x) - log pi_ref(y|x)] / max(1, |y|)

Delta_LN = s(y+|x) - s(y-|x)

L_LN-DPO = -log sigmoid(beta * Delta_LN)
```

The locked TRL implementation exposes this as `loss_type=["sigmoid_norm"]`. It is used identically for Standard LN-DPO, Multilevel LN-DPO, and Multilevel Matched LN-DPO. This is the primary paper comparison.

### 4.3 Length-Desensitized DPO Ablation

TRL also exposes `ld_alpha`, which applies full weight to tokens in the shared chosen/rejected length and weight `alpha` to the verbose tail. The planned validation-only search is:

```text
ld_alpha in {0.25, 0.50, 0.75}
```

`ld_alpha=1.0` is ordinary sequence-summed DPO and is not a primary method. `ld_alpha=0.0` completely removes unmatched tails and is retained only as an optional stress point if the three planned values fail to reveal a trend.

Length normalization and length desensitization are not silently combined in the primary run. Combining them may over-penalize valid long solutions and obscure the source of a gain.

### 4.4 Response-SFT Anchor

No formatting loss is enabled by default. If prompt-only MATH preflight shows that native outputs cannot be reliably evaluated, a temporary response-SFT term may be applied to correct, evaluable, student-generated responses. Its initial weight must be prespecified and annealed to zero. It is a response-validity anchor, not XML-format training.

### 4.5 Length Metrics Required During Training

Every preference run records chosen and rejected token lengths, length difference, normalized and unnormalized reward margins, preference accuracy by length bin, truncation, response-SFT weight, policy-reference KL, and generated length at held-out checkpoints. A method is not credited with improvement if accuracy rises only because evaluation outputs become unevaluable or truncated.

## 5. Benchmark and Data Protocol

### 5.1 Primary Dataset: MATH

Use the official MATH competition dataset, pinned to an immutable repository revision. Materialize all seven subject configurations:

- algebra;
- counting and probability;
- geometry;
- intermediate algebra;
- number theory;
- prealgebra;
- precalculus.

The official release has 7,500 train examples and 5,000 test examples. Each row contains a problem, difficulty level, subject type, and full solution. The final answer is extracted from the reference solution and stored with extraction provenance.

### 5.2 Primary Difficulty Slice

Levels 4-5 form the primary study because they are expected to produce meaningful wrong-to-correct trajectories. The complete dataset is still downloaded, hashed, audited, and preserved. Lower levels are available for curriculum, prompt preflight, and secondary all-level reporting, but cannot dominate the primary training pool.

### 5.3 Split Construction

The official test split remains untouched. The official training split is deterministically divided within subject and difficulty strata:

- paper training: 90 percent of each Level 4-5 stratum;
- validation: 10 percent of each Level 4-5 stratum;
- validation tune: two thirds of validation;
- validation confirmation: one third of validation.

Exact counts are derived only after the pinned data snapshot is materialized. The manifest records every source key, source row index, content hash, subject, level, role, and selection hash. No normalized problem or content hash may cross roles.

### 5.4 Test Reporting

The primary final metric is accuracy on official MATH test Levels 4-5. A secondary metric reports the complete 5,000-example official MATH test. Adapted models remain test-blind until all prompts, evaluators, hyperparameters, checkpoint rules, and seeds are frozen. The base checkpoint may receive one descriptive pre-research test evaluation under an immutable protocol, but that result cannot alter research choices.

### 5.5 MATH Evaluation

The MATH evaluator must support boxed answer extraction, integer and decimal normalization, rational equivalence, simple algebraic equivalence, set and interval normalization where valid, units, and multiple-answer rows. Deterministic symbolic checks are preferred when reliable. The 9B evaluator handles ambiguous extraction and semantic equivalence. Every model judgment is manually audited on the preflight and on a stratified sample of full validation.

### 5.6 Secondary Dataset: SearchQA-8K

After MATH is complete, run the existing SearchQA-8K protocol using the original SearchQA release:

- 5,000 train examples sampled from official train;
- 1,000 validation examples sampled from official validation;
- 2,000 test examples sampled from official test;
- disjoint auxiliary hyperparameter pools from unused rows.

SearchQA remains secondary because its evidence packaging and answer-type evaluation introduce additional variables. The MATH study must complete first.

### 5.7 External Stress Tests

OlympiadBench English text-only open-ended math may be used as an evaluation-only transfer diagnostic. It is not a primary training benchmark because it lacks a clean official train/test structure for this experiment and mixes proof, multimodal, bilingual, math, and physics tasks. MATH-500 and recent AIME sets may also be used only as clearly labeled external tests.

## 6. Current Verified State

### 6.1 Repository and Implementation

The current implementation branch is `agent/qwen35-pretest`, pushed through commit `23108a63805cb28811df48f6c5cd17bad0fad083`. At that revision:

- 150 automated tests pass;
- Ruff lint passes;
- Python compilation passes;
- shell syntax checks pass;
- main and GRPO lockfiles validate;
- paper configs require exact role-specific generation profiles;
- baseline evaluation, strict shard merge, manual audit, and HTML reporting exist;
- standard, multilevel, and matched preference construction exist;
- LoRA coverage, deterministic hyperparameter ledgers, DPO, GRPO, and DAPO paths exist;
- one-second GPU telemetry and CPU-only merge accounting corrections are implemented.

MATH materialization, MATH symbolic evaluation, and length-aware DPO configuration are not yet implemented at this revision. They are the next code phase.

### 6.2 Immutable GSM8K R1 Diagnostic

R1 collected 64 GSM8K groups across jobs `12948_0`, `12952_0`, and `12964_0`. It used 56 minutes 40 seconds of allocated GPU time and peaked at 22,879 MiB of 46,068 MiB. Diagnostic metrics were:

| Metric | Value |
| --- | ---: |
| Examples | 64 |
| Student attempts | 66 |
| Initial evaluator-correct | 54 / 64 (84.375%) |
| First correct after guidance | 1 / 64 (1.5625%) |
| Unresolved | 9 / 64 (14.0625%) |
| Preference pairs | 1 |
| Teacher hint candidates | 32 |
| Surface-valid hints | 15 |
| Surface-invalid hints | 17 |
| Guard SAFE | 2 |
| Guard UNSAFE | 13 |

R1 is invalid for paper training. It mixed evaluator schemas, lacked authoritative generated-token and finish metadata, accepted incomplete responses containing intermediate gold numbers, used an over-restrictive surface policy, used an over-conservative leakage guard, lacked a separate correctness critic, included at least one directionally wrong hint, and yielded only one pair.

### 6.3 Teacher-Free GSM8K Micro

The one-example teacher-free baseline micro used commit `ea45bd7b1a47ea32c1a9dc3df330d593829da5ff`. It produced a correct EOS-terminated response of 3,058 tokens for a simple hamburger problem. Student latency was 65.334 seconds and evaluator latency was 7.207 seconds. The result proved that the generation and evaluation pipeline worked, but exposed excessive native thinking length.

### 6.4 Final 16-Example GSM8K Preflight

The corrected preflight used commit `23108a63805cb28811df48f6c5cd17bad0fad083` and Slurm job `12976_0`. It completed in 20 minutes 38 seconds.

:::baseline-chart

| Metric | Result | Gate | Status |
| --- | ---: | ---: | --- |
| Exact accuracy | 13 / 16 (81.25%) | Descriptive | Recorded |
| EOS termination | 14 / 16 (87.5%) | At least 95% implied by truncation gate | Failed |
| Length truncation | 2 / 16 (12.5%) | At most 5% | Failed |
| Manual/evaluator agreement | 16 / 16 (100%) | At least 95% | Passed |
| Metadata completeness | 100% | 100% | Passed |
| Teacher-free prompts | 100% | 100% | Passed |
| Runtime failure count | 0 | 0 unexplained | Passed |
| Average generated tokens | 3,523 | Descriptive | High |
| Average generation latency | 74.458 s | Descriptive | High |
| Sampled peak GPU memory | 24,767 MiB | Below 41,461 MiB | Passed |

Two responses hit exactly 8,192 tokens and were marked incorrect despite the evaluator extracting gold-equivalent numbers. A third EOS-terminated response answered 3.5 instead of 7 after misreading a per-person quantity. Manual review agreed with every evaluator decision. The formal audit failed only the truncation-rate gate.

### 6.5 Lessons from GSM8K

1. The base model is already too accurate for useful preference collection on many GSM8K rows.
2. Qwen3.5-2B thinking mode can produce very long meta-reasoning even for simple arithmetic.
3. A larger token ceiling prevents premature clipping but does not guarantee EOS termination.
4. Truncation override is necessary because a long unfinished trace can contain the correct number without delivering a valid answer.
5. Model-evaluator agreement is strong under the current structured evaluator, but the domain scorer must be upgraded for symbolic MATH answers.
6. One-second telemetry is needed because short evaluator memory peaks can be missed at ten-second intervals.
7. No full GSM8K baseline, collection, or training should continue. Existing GSM8K artifacts remain diagnostics only.

### 6.6 MATH Thinking-Mode Diagnostic and Mode Decision

The first immutable MATH Levels 4-5 diagnostic used source commit
`4c77ff6fac3eb43c89a9a742c7901b5427ab2b7a` and Slurm job `13021`. It evaluated
16 teacher-free examples with Qwen3.5-2B thinking mode under the former
`qwen-native-r2` protocol. The run completed with no runtime failures in 40 minutes
33 seconds, but only five responses terminated with EOS. Eleven hit exactly 8,192
tokens, yielding a 68.75% truncation rate and 31.25% protocol-valid accuracy. Nine of
the eleven truncated responses already contained a deterministically gold-equivalent
extracted answer before the mandatory truncation override.

This result is diagnostic-only. It demonstrates that thinking-mode termination is a
large nuisance variable that would dominate pair construction, training cost, and
held-out scoring. Because the research contribution is multilevel feedback rather
than extended reasoning, the primary protocol now uses explicit non-thinking mode.
The mode, prompt, and non-thinking sampling profile must be identical for baseline
evaluation, guidance collection, DPO pair prompts, adapted-checkpoint evaluation, and
the GRPO comparison. Thinking mode remains a prespecified secondary ablation and may
not replace the primary protocol after observing validation or test results.

### 6.7 Non-Thinking Comparison Result

The first `qwen-nonthinking-r1` comparison used pushed source commit
`15ec5fe41cc9010a2c63ce787a706b2389d787f8` and Slurm job `13038` on the exact same
16 examples as thinking-mode job `13021`. It completed without runtime failures in
31 minutes 48 seconds.

| Metric | Thinking `13021` | Non-thinking `13038` | Direction |
| --- | ---: | ---: | --- |
| Protocol-valid exact accuracy | 5/16 (31.25%) | 7/16 (43.75%) | improved |
| EOS termination | 5/16 (31.25%) | 8/16 (50.00%) | improved |
| Length truncation | 11/16 (68.75%) | 8/16 (50.00%) | improved but failed gate |
| Mean generated tokens | 6,752.625 | 5,299.875 | reduced 21.5% |
| Mean generation latency | 147.586 s | 115.278 s | reduced 21.9% |
| Total wall time | 40:33 | 31:48 | reduced 21.6% |
| Runtime failures | 0 | 0 | unchanged |

All eight EOS-terminated non-thinking responses contained a boxed answer. Two additional
responses emitted a box but continued until the 8,192-token ceiling, while six reached
the ceiling without a box. Non-thinking mode is therefore retained as the primary mode,
but `qwen-nonthinking-r1` remains a diagnostic profile because its 50% truncation rate
fails the 5% paper gate. DPO collection and training remain blocked. The next protocol
step is a fixed, disjoint train-only termination study within non-thinking mode; no
validation or official-test result may select its prompt or sampling profile.

## 7. Experimental Matrix

### 7.1 Required Main Methods

| ID | Method | Training signal | Pair/reward source | Primary purpose |
| --- | --- | --- | --- | --- |
| B0 | Frozen base | None | None | Teacher-free reference |
| B1 | Response SFT | Correct student responses | Student first-correct outputs | Supervised anchor |
| B2 | On-policy distillation | Teacher distribution/targets on student-visited problems | Student rollouts plus 9B correction | Distillation baseline |
| D1 | Standard LN-DPO | Length-normalized preference | First correct vs initial wrong | Standard preference baseline |
| D2 | Multilevel LN-DPO | Length-normalized preference | First correct vs every prior wrong | Proposed method |
| D3 | Matched Multilevel LN-DPO | Length-normalized preference | Deterministic matched subset | Pair-budget control |
| D4 | Multilevel LD-DPO | Tail-desensitized preference | Same multilevel pairs | Length regularization ablation |
| R1 | Standard GRPO | On-policy correctness reward | Four student generations per prompt | RL baseline |
| R2 | DAPO sensitivity | DAPO-style RL objective | Same reward/evaluator | Separately labeled sensitivity |

### 7.2 On-Policy Distillation Definition

The student generates on-policy responses on training prompts. The 9B teacher evaluates or supplies a corrected target on the student-visited state with access to the reference solution. The student is trained to match the teacher target or token distribution according to a frozen distillation objective. Teacher access is training-only. Distillation targets, prompts, and teacher tokens are logged separately from the main slight-hint trajectories.

This baseline is intentionally more direct than the proposed method. If it wins, the interpretation is that full teacher supervision is more effective than slight feedback. If multilevel LN-DPO wins or matches it with less teacher output, the result supports the efficiency of self-correction-based preferences.

### 7.3 Fairness Controls

- Every method starts from the same pinned Qwen3.5-2B revision.
- Every LoRA method uses the same architecture-audited target modules, rank, alpha, dropout, and precision.
- Standard, multilevel, and matched methods receive equal hyperparameter-search budgets.
- Standard and matched methods receive equal pair counts and optimizer updates.
- A shared-hyperparameter seed compares all DPO methods under one profile.
- Held-out prompts, evaluator, generation settings, and checkpoint-selection rules are identical.
- Teacher and evaluator GPU usage is included in total compute accounting.
- Official test data is not used for tuning, prompt changes, or stopping.

## 8. Ablation Plan

### 8.1 Preference Construction Ablations

1. Standard initial-failure pair versus full multilevel pairs.
2. Multilevel full pair budget versus matched pair budget.
3. Reject only attempt zero versus reject all prior attempts.
4. Attempt-depth stratification: compare pairs from early and late failures.
5. One pair per group sampled uniformly versus all pairs.

### 8.2 Teacher and Privilege Ablations

1. 9B privileged slight-hint teacher versus no hint.
2. 9B teacher versus same 2B student weights acting as privileged teacher.
3. Gold final answer only versus complete gold solution as teacher context.
4. One guidance round versus three rounds.
5. Accumulated hints versus latest hint only.
6. Teacher guidance with guard and critic versus guard-only diagnostic.

The unsafe no-guard condition must never feed unreviewed hints into a full run. It can be evaluated offline on already generated teacher candidates.

### 8.3 Length Ablations

1. Primary `sigmoid_norm` length-normalized DPO.
2. Length-desensitized DPO with `ld_alpha` in `{0.25, 0.50, 0.75}`.
3. Optional one-seed sequence-summed sigmoid DPO diagnostic.
4. Primary non-thinking protocol versus the frozen thinking-mode diagnostic profile.
5. No loop stopping versus token-level anomalous-loop stopping, only if the detector is frozen and all stopped responses are marked invalid.
6. Response-SFT anchor off versus annealed-to-zero anchor.

### 8.4 Model and Adaptation Ablations

1. LoRA rank 16 broad text-backbone coverage versus attention-only coverage preflight.
2. Optional one-seed full fine-tuning for Standard LN-DPO and Multilevel LN-DPO, only after LoRA results are complete and storage is available.
3. Qwen3.5-9B teacher versus 2B privileged self-teacher.

### 8.5 RL Ablations

1. Original GRPO versus DAPO sensitivity.
2. KL beta grid `{0.0, 0.001, 0.01, 0.04}`.
3. Correctness-only reward versus correctness plus a small conditional length cost, only as a separately labeled sensitivity.
4. Four generations versus a resource-limited two-generation diagnostic, not a substitute for the primary GRPO baseline.

## 9. Hyperparameters and Selection

### 9.1 LoRA Profile

| Field | Value |
| --- | --- |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Precision | BF16 |
| Quantization | None |
| Target policy | All audited text-backbone linear attention, full attention, and MLP projections |
| Exclusions | Vision encoder, multimodal projector, embeddings, output head |

### 9.2 Optimizer Foundation

| Field | Value |
| --- | --- |
| Optimizer | `adamw_torch_fused` |
| Adam beta1 / beta2 | 0.9 / 0.999 |
| Adam epsilon | 1e-8 |
| Weight decay | 0.01 initial |
| Maximum gradient norm | 1.0 |
| Scheduler | Cosine initial |
| Warmup | 5% integer optimizer steps |
| Effective DPO batch | 16 pairs |
| Maximum epochs | 1.0 |
| Combined DPO sequence ceiling | 10,240 tokens |

### 9.3 LN-DPO Search

Primary candidate matrix:

- learning rate: `{2e-6, 5e-6, 1e-5}`;
- beta: `{0.05, 0.1, 0.3, 0.5}`;
- loss type: `sigmoid_norm`;
- effective global batch: 16;
- maximum duration: one epoch.

Use deterministic successive halving. Screen all 12 learning-rate/beta candidates on a fixed pilot. Promote four, then test weight decay `{0.0, 0.01}`, warmup `{5%, 10%}`, and scheduler `{linear, cosine}` through a frozen fractional design. Promote two to full pilot data and two tuning seeds. Freeze one candidate before final seed runs.

### 9.4 LD-DPO Search

Use the selected or shared LN-DPO optimizer foundation and search `ld_alpha` in `{0.25, 0.50, 0.75}` with `loss_type=sigmoid`. This is a smaller ablation budget, not an independent unconstrained search.

### 9.5 GRPO Search

- learning rate: `{2e-6, 5e-6, 1e-5}`;
- KL beta: `{0.0, 0.001, 0.01, 0.04}`;
- clipping epsilon: 0.2;
- policy iterations: one;
- generations per prompt: four;
- reward scaling: within group;
- truncated completion masking: enabled;
- maximum completion: 8,192 tokens;
- vLLM presence penalty: 1.5.

### 9.6 Final Seeds

Final primary runs use seeds 17, 31, and 47. Tuning uses seeds 11 and 29 where a second seed is required. Any seed change must be frozen before test evaluation.

## 10. Evaluation and Statistics

### 10.1 Primary Outcome

Teacher-free exact answer accuracy on official MATH test Levels 4-5.

### 10.2 Secondary Outcomes

- complete official MATH test accuracy;
- accuracy by subject and difficulty;
- first-attempt accuracy during collection;
- cumulative success after each guidance step;
- unresolved rate;
- unsafe-guidance rate;
- average attempts to first correct;
- pair yield and pairs per successful group;
- median and mean response tokens;
- EOS and truncation rates;
- generation and evaluator latency;
- GPU-hours and peak memory.

### 10.3 Training Diagnostics

LN-DPO and LD-DPO record total loss, preference accuracy, chosen and rejected normalized scores, normalized reward margin, unnormalized diagnostic margin, policy-reference KL, gradient norm, clipping rate, learning rate, tokens per update, and checkpoint-level held-out accuracy.

GRPO records reward mean and standard deviation, zero-variance group rate, KL, clipping ratio, response length, truncation, reward by subject/level, and rollout throughput.

### 10.4 Statistical Analysis

- report mean and standard deviation over three final seeds;
- bootstrap 95% confidence intervals for accuracy and response length;
- paired McNemar tests for exact correctness on shared test examples;
- paired bootstrap for continuous metrics;
- Holm correction over prespecified primary method comparisons;
- report absolute accuracy difference, relative error reduction, and odds ratio where appropriate;
- include per-subject effect sizes and failure-category shifts.

Primary comparisons are:

1. Multilevel LN-DPO versus Standard LN-DPO.
2. Matched Multilevel LN-DPO versus Standard LN-DPO.
3. Multilevel LN-DPO versus frozen base.
4. Multilevel LN-DPO versus GRPO.
5. Multilevel LN-DPO versus on-policy distillation.

All other tests are secondary or exploratory and must be labeled accordingly.

## 11. Observability and Artifact Contract

### 11.1 Immutable Identity

Every run records:

- source commit;
- config hash;
- dataset manifest hash;
- split and row hashes;
- student, teacher, and evaluator revisions;
- package and lockfile versions;
- prompt protocol hash;
- generation profile;
- seed;
- Slurm job, node, partition, account, and GPU identity.

### 11.2 Per-Generation Records

Each model call records role, prompt hash, raw output, exact prompt tokens, exact generated tokens, EOS termination, length truncation, finish reason, latency, retry index, and failure status. Estimates are emitted only when exact tokenizer counts are unavailable and are never mixed with exact fields.

### 11.3 Collection Artifacts

- compressed trajectory groups;
- raw attempts;
- guidance prompts and outputs;
- surface-policy results;
- accumulated hint history;
- guard and critic decisions;
- evaluator raw outputs and repair turns;
- progress marker and completion marker;
- unresolved and failure ledgers;
- merged collection hash.

### 11.4 Training Artifacts

- immutable preference manifest;
- candidate and promotion ledger;
- TensorBoard event files;
- canonical JSONL step logs;
- adapter manifest and LoRA inventory;
- optimizer and scheduler state summary;
- validation predictions at prespecified checkpoints;
- selected-candidate freeze manifest;
- HTML report and paper-ready CSV tables.

### 11.5 Human-Readable Reports

Every phase generates an HTML report containing status, key metrics, plots, failures, fixes, and artifact paths. The final project generates paper-ready PDF figures and this comprehensive plan PDF. JSONL and immutable manifests remain the source of truth.

## 12. Turing GPU Execution Plan

### 12.1 Cluster Rules

- The login node is used only for Git, queue inspection, submission, and small-file reads.
- Dataset processing, model loading, inference, training, and evaluation run under Slurm.
- GPU jobs use the `u22` partition and account `priyesh.shukla`.
- Model caches and temporary environments live in node-local scratch.
- Persistent home storage contains manifests, metrics, reports, and final adapters only.
- No job starts when persistent storage has less than 8 GB free or exceeds 85% utilization.
- Failed shards remain explicit and cannot be silently skipped.

### 12.2 Resource Profile

Initial collection and evaluation use one RTX 6000 Ada 48 GB GPU per shard. Measured GSM8K peak memory was 24,767 MiB, leaving headroom for MATH preflight. The 8,192-token ceiling and longer MATH prompts may increase KV memory, so a 16-example MATH preflight must precede full arrays. Two GPUs may be used only when a measured single-GPU preflight fails memory or when vLLM server mode requires an isolated rollout device. The reason must be logged.

### 12.3 Storage Layout

```text
/home/aryama.murthy/multilevel-feedback-dpo/implementation
    synced source and scripts

/home/aryama.murthy/tfdpo-runs
    small persistent manifests, metrics, reports, adapters

/scratch/aryama.murthy/tfdpo-model-cache
    revision-keyed model cache on the allocated node

local workstation artifact archive
    compressed raw trajectories and completed run snapshots outside Git
```

### 12.4 Monitoring

GPU telemetry is sampled every second for paper jobs. Monitor utilization, memory, power, temperature, Slurm state, wall time, home usage, scratch usage, generated record count, and failure ledger. The agent reports accuracy, loss, reward, truncation, pair yield, memory, and wall time after every completed run.

## 13. Failure Gates

### 13.1 Baseline Gate

- 100% prediction coverage;
- complete exact generation metadata;
- teacher-free prompts and model calls;
- manual/evaluator agreement at least 95%;
- truncation at most 5%;
- nonempty responses at least 99%;
- peak GPU memory below 90%;
- zero unexplained failures.

### 13.2 Collection Gate

- evaluator parse success at least 99%;
- manual/evaluator agreement at least 95%;
- nonzero pair yield;
- zero answer leakage;
- guidance critic validity above the frozen threshold;
- truncation at most 5%;
- every expected shard complete and hash-valid;
- no mixed prompt, source, or evaluator protocol.

### 13.3 DPO Gate

- finite loss and gradients;
- at least one optimizer update;
- exact LoRA inventory match;
- no dataset or test leakage;
- checkpoint and ledger hashes valid;
- teacher-free held-out generation succeeds;
- memory below 90%;
- no silent sequence truncation inside training batches.

### 13.4 GRPO Gate

- reward/manual agreement at least 95%;
- zero-variance groups at most 50%;
- truncated completions at most 5%;
- finite KL, loss, gradients, and rewards;
- vLLM presence penalty active and verified;
- colocated memory below 90%, otherwise use the prespecified two-GPU server preflight.

## 14. Risks and Mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Accidental thinking-mode reactivation | High cost, invalid responses, and protocol mismatch | Hard config validation of `enable_thinking=false`, frozen protocol hash, EOS logging, and thinking mode only as a separately labeled ablation |
| MATH answer-equivalence errors | False reward or preference labels | Symbolic normalizer plus 9B adjudication plus manual audits |
| Teacher leaks answer | Invalid scientific claim | Slight-hint contract, accumulated semantic guard, manual leakage audit |
| Teacher gives wrong hint | Student is pushed away from solution | Separate privileged correctness critic and bounded regeneration |
| Too few successful trajectories | Low pair yield | MATH Levels 4-5 pilot, tune retry budget only before freeze, preserve unresolved groups |
| Too many unresolved trajectories | Weak DPO coverage | Report success curves; do not invent pairs; consider curriculum only as separate experiment |
| Length-aware loss over-penalizes proofs | Accuracy decreases | Keep LN-DPO primary, LD-DPO separate, report length/accuracy frontier |
| Pair-count confound | Multilevel gain reflects more examples | Matched multilevel control and equal optimizer updates |
| Benchmark contamination | Inflated base result | Use immutable held-out test, external Olympiad/AIME diagnostic, acknowledge limitation |
| Cluster storage exhaustion | Job failure or data loss | Scratch caches, compressed shards, 8 GB/85% gate, verified local archive |
| Mixed source revisions | Unreproducible run | Freeze commit/config/dataset/model hashes in every marker |
| Evaluator becomes teacher proxy | Inflated metrics | Deterministic checks where reliable, blinded manual audit, evaluator disagreement analysis |

## 15. End-to-End Execution Plan

:::timeline

### Phase 0: Freeze This Revised Design

1. Update living Markdown specifications from GSM8K to MATH Levels 4-5.
2. Mark all GSM8K data and results diagnostic-only.
3. Freeze the primary objective as `sigmoid_norm` LN-DPO.
4. Freeze LD-DPO as a separate `ld_alpha` ablation.
5. Commit and push the plan before code changes.

Exit criterion: design document, implementation plan, and decision log agree on dataset, methods, objectives, gates, and metrics.

### Phase 1: Implement MATH Data and Evaluation

1. Add a strict MATH paper config with pinned dataset revision and expected source counts.
2. Materialize all seven official subject configurations.
3. Parse level, subject, reference solution, boxed final answer, and source provenance.
4. Build stratified Level 4-5 train/tune/confirm roles without touching official test.
5. Implement symbolic answer normalization and evaluator-backed ambiguity routing.
6. Add fixture tests for rational, algebraic, set, interval, unit, and multiple-answer cases.
7. Extend reports to subject and difficulty breakdowns.

Exit criterion: complete manifest validates, no split overlap exists, and evaluator fixtures plus manual examples pass.

### Phase 2: Implement Length-Aware DPO

1. Require explicit DPO `loss_type` in paper config.
2. Add `sigmoid_norm` as the only primary DPO loss.
3. Add optional `ld_alpha` with strict range and method labeling.
4. Test exact TRL config propagation and refusal of silent ordinary sigmoid defaults.
5. Log chosen/rejected lengths and normalized/unnormalized margins.
6. Add candidate-ledger support for `ld_alpha` ablations.

Exit criterion: local unit tests verify objective selection, metrics, manifests, and failure behavior; locked TRL runtime accepts every field.

### Phase 3: Prompt and Baseline Preflight

1. Create a deterministic train-only prompt-development subset across subjects and Levels 4-5.
2. Implement the primary `qwen-nonthinking-r1` protocol using Qwen's non-thinking text sampling profile and a boxed-answer-and-stop prompt.
3. Preserve job `13038` as a diagnostic; do not treat its globally hash-selected validation subset as prompt-selection data.
4. On the disjoint train-only subset, compare a small prespecified non-thinking sampling grid around the official profile while keeping the boxed-answer-and-stop contract fixed. Thinking mode is not a candidate for promotion.
5. Verify locally that the chat template receives `enable_thinking=false` and that thinking-mode configs fail validation.
6. Select on protocol-valid correctness first, then truncation, median tokens, and GPU cost; manually inspect every truncated or post-box continuation.
7. Freeze the winning non-thinking prompt, sampling profile, config hash, and source commit under a new protocol identifier before validation.
8. Run one MATH example, then 16 genuinely stratified validation examples.
9. Record protocol-valid accuracy, raw answer attainment, EOS/length termination, post-answer continuation, median generated tokens, and GPU cost.
10. Manually audit every response and evaluator decision.

Exit criterion: at least 95% manual agreement, at most 5% truncation, complete metadata, no teacher context, and memory below 90%.

### Phase 4: Full Base Baseline

1. Evaluate complete MATH validation under deterministic shards.
2. Manually audit at least 64 stratified examples.
3. Freeze baseline report and hashes.
4. Run the base model once on official MATH test after protocol approval, with Levels 4-5 and all-level results separated.

Exit criterion: baseline artifacts, audits, plots, HTML report, and one-time test marker complete.

### Phase 5: Guidance Collection Preflight

1. Run 16 hard training examples through the complete slight-guidance loop.
2. Inspect every attempt, hint, guard decision, critic decision, and evaluator result.
3. Expand to a 64-example stratified preflight.
4. Measure first-attempt accuracy, success by step, unresolved rate, leakage, pair yield, tokens, latency, and memory.

Exit criterion: nonzero pair yield, zero leakage, at least 95% evaluator agreement, at most 5% truncation, and valid source/protocol hashes.

### Phase 6: Full MATH Collection

1. Estimate shard duration from preflight.
2. Submit one-GPU arrays at verified account concurrency.
3. Monitor queue, GPU telemetry, failures, and storage.
4. Merge only when every expected shard is complete.
5. Build standard, multilevel, and matched preference datasets.
6. Publish pair-depth and group-yield diagnostics.

Exit criterion: immutable preference manifests pass leakage, hash, count, and prompt-invariant audits.

### Phase 7: LN-DPO and LD-DPO Training

1. Run equal-budget successive halving for Standard LN-DPO, Multilevel LN-DPO, and Matched LN-DPO.
2. Freeze each selected candidate.
3. Run three final seeds for each primary method.
4. Run the LD-DPO `ld_alpha` ablation with the reduced frozen budget.
5. Generate teacher-free validation predictions at prespecified checkpoints.

Exit criterion: all adapters, logs, ledgers, checkpoints, validation predictions, and memory reports validate.

### Phase 8: SFT and On-Policy Distillation Baselines

1. Train the response-SFT baseline on first-correct student outputs.
2. Define and preflight the on-policy distillation target and loss.
3. Run equal LoRA coverage and final seeds.
4. Measure teacher-token cost and response-style similarity.

Exit criterion: baselines use the same splits and evaluation protocol and have complete compute accounting.

### Phase 9: GRPO and DAPO

1. Run a 32-prompt reward and memory preflight.
2. Enforce reward agreement, variance, truncation, and memory gates.
3. Run deterministic GRPO successive halving.
4. Run three final GRPO seeds.
5. Run one separately labeled DAPO sensitivity seed.

Exit criterion: reward, KL, clipping, truncation, adapters, and held-out predictions validate.

### Phase 10: Frozen Test and Statistics

1. Verify no adapted-method test markers exist.
2. Evaluate every selected method and seed on the official test.
3. Compute primary and corrected statistical comparisons.
4. Produce subject, level, length, guidance-step, and compute-efficiency analyses.
5. Generate paper-ready tables and figures.

Exit criterion: test artifacts are immutable and the statistical report reproduces from canonical predictions.

### Phase 11: SearchQA-8K

Repeat baseline, collection, LN-DPO, GRPO, evaluation, and statistics only after MATH completion. Use the frozen SearchQA evidence packaging and reward protocol. Cross-domain claims require both domains; the core MATH paper can be reported independently if SearchQA is delayed.

### Phase 12: Final Audit and Paper

1. Audit every requirement in this document.
2. Verify repository state, cluster queue, storage, local archives, and GitHub branch.
3. Produce final method diagrams, tables, failure analysis, and limitations.
4. Update the living decision log and reproduction commands.

## 16. Planned Paper Tables and Figures

### Main Tables

1. MATH Levels 4-5 primary accuracy across base, SFT, on-policy distillation, Standard LN-DPO, Multilevel LN-DPO, Matched LN-DPO, LD-DPO, and GRPO.
2. Complete MATH test accuracy by method and seed.
3. Subject and difficulty breakdown.
4. Collection dynamics: first-attempt accuracy, success by hint step, unresolved rate, pair yield, leakage rate.
5. Compute table: teacher tokens, student tokens, GPU-hours, peak memory, training updates.

### Main Figures

1. Method flow from wrong attempts to first correct and multilevel pairs.
2. Success-to-first-correct cumulative curve by guidance step.
3. Accuracy versus median response length.
4. Accuracy versus GPU-hours.
5. LN-DPO training curves: normalized margin, preference accuracy, KL, and validation accuracy.
6. Error-category transition matrix before and after training.

### Ablation Figures

1. `ld_alpha` sensitivity.
2. Pair depth and pair-budget matching.
3. 9B teacher versus privileged 2B self-teacher.
4. Prompt variant length/truncation frontier.
5. Standard GRPO versus DAPO sensitivity.

## 17. Decision Log

| Date | Decision | Rationale | Status |
| --- | --- | --- | --- |
| 2026-07-10 | Use Qwen3.5-2B student | Small post-trained model appropriate for controlled adaptation | Approved |
| 2026-07-10 | Use Qwen3.5-9B teacher/evaluator | Fully post-trained instruct model with stronger math capability | Approved |
| 2026-07-10 | Teacher receives privileged gold solution | Needed to locate earliest error | Approved with leakage guard |
| 2026-07-10 | Teacher returns slight hint, not corrected rollout | Keeps chosen response student-generated | Approved |
| 2026-07-10 | Use native Qwen output, not XML | Method should fit model style rather than force formatting | Approved |
| 2026-07-10 | Student ceiling 8,192 tokens | Avoids low artificial ceiling while retaining a hard bound | Approved |
| 2026-07-10 | Baseline before research | Establishes teacher-free reference and evaluator validity | Approved |
| 2026-07-10 | GSM8K is diagnostic only | Base accuracy and low pair yield make it weak for the main claim | Approved |
| 2026-07-10 | MATH Levels 4-5 is primary | Hard, public train/test, full solutions, competition reasoning | Approved |
| 2026-07-10 | Use LN-DPO instead of ordinary DPO | Controls sequence-length bias | Approved direction |
| 2026-07-10 | `sigmoid_norm` is primary | Existing locked TRL implementation and clean interpretation | Proposed exact freeze |
| 2026-07-10 | LD-DPO is separate ablation | Avoids silently mixing two length mechanisms | Proposed exact freeze |
| 2026-07-10 | GRPO and on-policy distillation remain baselines | Compare offline preferences with on-policy alternatives | Approved |

## 18. Immediate Next Actions

The next execution sequence is:

1. Commit this revised project plan and update the living design documents.
2. Add pinned MATH configuration and expected subject counts.
3. Implement deterministic MATH materialization and Level 4-5 stratified roles.
4. Implement symbolic MATH answer evaluation with 9B ambiguity adjudication.
5. Implement mandatory `sigmoid_norm` and optional `ld_alpha` configuration.
6. Add tests for MATH manifests, scorer equivalence, loss propagation, and observability.
7. Run the full local verification gate and push a new source revision.
8. Verify no Turing jobs are active and storage has at least 8 GB headroom.
9. Materialize MATH under a CPU Slurm job.
10. Run a train-only prompt and generation preflight.
11. Freeze and run the 16-example teacher-free MATH baseline gate.
12. Continue only if all baseline gates pass.

## 19. Reproducibility Checklist

- [ ] MATH source revision pinned.
- [ ] Seven subject counts verified against source.
- [ ] Full 7,500/5,000 train/test counts verified.
- [ ] Level 4-5 role counts frozen after materialization.
- [ ] No split overlap by source key, normalized problem, or content hash.
- [ ] Student, teacher, evaluator revisions pinned.
- [ ] Prompt and generation profile hashes frozen.
- [ ] `sigmoid_norm` explicitly configured for every primary DPO run.
- [ ] `ld_alpha` present only in labeled LD-DPO ablations.
- [ ] LoRA target inventory identical across methods.
- [ ] Hyperparameter candidate ledgers complete.
- [ ] Manual evaluator and leakage audits pass.
- [ ] Truncation at most 5% before full phases.
- [ ] All test evaluations use frozen manifests and one-time markers.
- [ ] Three final seeds complete for primary methods.
- [ ] GPU-hours, token counts, wall time, memory, and storage recorded.
- [ ] Raw artifacts archived locally and hashes verified.
- [ ] Paper tables regenerate from canonical predictions.
- [ ] Limitations and failed diagnostics are reported.

## 20. References and Source Log

[1] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023. https://arxiv.org/abs/2305.18290

[2] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, "Measuring Mathematical Problem Solving With the MATH Dataset," NeurIPS 2021. https://arxiv.org/abs/2103.03874

[3] Y. Meng, M. Xia, and D. Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward," NeurIPS 2024. The paper motivates average per-token sequence rewards; locked TRL exposes a length-normalized sigmoid preference objective as `sigmoid_norm`. https://arxiv.org/abs/2405.14734

[4] Z. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv preprint, 2024. Introduces GRPO for mathematical reasoning. https://arxiv.org/abs/2402.03300

[5] Q. Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," arXiv preprint, 2025. https://arxiv.org/abs/2503.14476

[6] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, M. Geist, and O. Bachem, "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes," ICLR 2024. https://arxiv.org/abs/2306.13649

[7] K. Cobbe et al., "Training Verifiers to Solve Math Word Problems," arXiv preprint, 2021. Introduces GSM8K. https://arxiv.org/abs/2110.14168

[8] M. Dunn, L. Sagun, M. Higgins, V. U. Guney, V. Cirik, and K. Cho, "SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine," arXiv preprint, 2017. https://arxiv.org/abs/1704.05179

[9] Qwen Team, "Qwen3.5-2B Model Card," official model documentation, accessed 2026-07-10. Documents the post-trained model, thinking-mode sampling, 262,144-token native context, and 2B thinking-loop warning. https://huggingface.co/Qwen/Qwen3.5-2B

[10] Qwen Team, "Qwen3.5-9B Model Card," official model documentation, accessed 2026-07-10. https://huggingface.co/Qwen/Qwen3.5-9B

[11] Hugging Face, "TRL DPO Trainer Documentation and Source," official software documentation, accessed 2026-07-10. Documents `sigmoid_norm`, `ld_alpha`, DPO metrics, and current configuration fields. https://github.com/huggingface/trl/blob/main/docs/source/dpo_trainer.md

[12] Hugging Face and EleutherAI, "hendrycks_math Dataset Snapshot," official mirror metadata, revision `21a5633873b6a120296cce3e2df9d5550074f4a3`, accessed 2026-07-10. https://huggingface.co/datasets/EleutherAI/hendrycks_math

[13] C. He et al., "OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems," ACL 2024. https://arxiv.org/abs/2402.14008

## Appendix A. Frozen Prompt Intent

### Student Base Prompt

```text
Solve the following mathematics problem.

Reason step by step using only as much detail as the problem needs. Put the final
answer in `\boxed{}` and stop immediately after the boxed answer.

Problem:
{problem}
```

This wording and the `qwen-nonthinking-r1` protocol identifier are frozen before validation. The prompt must not mention the gold solution, teacher, evaluator, retries, or preference training.

### Privileged Teacher Prompt Intent

```text
You are a privileged teacher reviewing a failed student attempt.

Use the gold solution only to locate the earliest broad error. Return one subtle,
directionally correct hint of 5-25 words. Do not reveal the final answer or solve the
problem for the student.

Problem: {problem}
Gold solution (teacher-only): {solution}
Student response: {response}
Evaluator result: {result}
Previous hint reviews: {reviews}

Return only the hint.
```

### Evaluator Prompt Intent

```text
Extract and judge the final answer to this MATH problem using the gold solution.
Return one strict JSON object containing answer, correct, confidence, and reason.
Do not reward an unfinished or length-truncated response.
```

## Appendix B. Collection Pseudocode

```text
for example in training_examples:
    prompt = build_student_prompt(example)
    attempts = []
    hints = []

    for attempt_index in range(0, max_guidance_steps + 1):
        response = student.generate(prompt_with(prompt, hints))
        generation = record_exact_generation_metadata(response)
        result = evaluator.evaluate(example, generation)
        attempts.append({response, generation, result})

        if result.correct and not generation.truncated:
            chosen = response
            rejected = all responses before chosen
            emit_trajectory_group(example, chosen, rejected, hints)
            break

        if attempt_index == max_guidance_steps:
            emit_unresolved_group(example, attempts, hints)
            break

        for regeneration in range(max_hint_regenerations + 1):
            hint = teacher.hint(example, response, result, previous_reviews)
            surface = validate_surface(hint)
            critic = guidance_critic.review(example, response, hint)
            guard = leakage_guard.review(example, hints + [hint])
            record_hint_and_reviews(hint, surface, critic, guard)

            if surface.valid and critic.valid and guard.safe:
                hints.append(hint)
                break
        else:
            emit_unresolved_group(example, attempts, hints)
            break
```

## Appendix C. Artifact Naming

```text
paper/math/
  data/<dataset-revision>/
  baseline/<source-commit>/<freeze-hash>/
  collection-preflight/<source-commit>/<protocol-hash>/
  collection/<source-commit>/<protocol-hash>/
  preferences/<collection-hash>/{standard,multilevel,matched}/
  tuning/{standard_ln,multilevel_ln,matched_ln,ld_dpo,grpo}/
  training/<method>/<candidate-freeze>/<seed>/
  evaluation/<method>/<seed>/<split>/
  reports/<report-hash>/
```

## Appendix D. Interpretation Rules

1. A diagnostic run is never upgraded into a paper result after the fact.
2. A failed gate blocks downstream phases; it is not converted into a warning.
3. A truncated response is incorrect even if the evaluator extracts the gold answer.
4. Missing shards, missing labels, or missing metrics are failures, not zeros.
5. Test results do not change prompts, rewards, hyperparameters, or stopping rules.
6. Multilevel gains must be reported together with the matched-pair control.
7. Length reductions count as improvements only when answer accuracy and evaluability are preserved.
8. Teacher-token and evaluator-token costs are included in method cost.
9. SearchQA results are not required to claim a MATH-only result, but cross-domain claims require both.
10. The final paper reports negative findings, failed preflights, and deviations from this plan.
