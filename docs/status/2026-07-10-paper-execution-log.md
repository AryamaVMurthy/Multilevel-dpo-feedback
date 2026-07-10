# Paper Experiment Execution Log

This is the living human-readable execution record for the paper-scale GSM8K and
SearchQA-8K study. Canonical machine-readable artifacts remain the source of truth.

## GSM8K R1 Collection Diagnostic

Status: completed as a diagnostic and failed the paper preflight gate on 2026-07-10.

### Identity And Artifacts

- Turing jobs: `12948_0`, `12952_0`, and `12964_0` on `node04`.
- Final continuation: `12964_0`, Slurm `COMPLETED`, exit `0:0`, elapsed `00:29:37`.
- Total allocated GPU time across the three attempts: `00:56:40`.
- Dataset manifest hash:
  `61a7a7f82f0ff75491b7b363504f85b0543628a860084cc0be66a01cf6f9eb6c`.
- Config hash:
  `ae44a81596f93e83fc607d0da82aa061f88ece5082e9f03eecf4c5f40dfbdb33`.
- Record artifact hash:
  `b403a733530d89145c46d2667fdcde5f65c90e2d7da4e6b525dc0a0d307073b9`.
- Durable local archive:
  `/home/aryamavmurthy/work/SLM-Research/multilevel-feedback-dpo-artifacts/gsm8k/r1/`.
- Peak observed GPU memory: 22,879 MiB of 46,068 MiB.
- Turing persistent storage after verified cache cleanup: 36 GB used, 15 GB free,
  71% utilization.

### Diagnostic Metrics

These values describe R1 behavior only. They are not paper results.

| Metric | Value |
| --- | ---: |
| examples | 64 |
| student attempts | 66 |
| first-attempt evaluator correctness | 54/64, 84.375% |
| first correct after guidance | 1/64, 1.5625% |
| unresolved groups | 9/64, 14.0625% |
| generated preference pairs | 1 |
| teacher hint candidates | 32 |
| surface-valid hints | 15 |
| surface-invalid hints | 17 |
| model guard SAFE | 2 |
| model guard UNSAFE | 13 |
| guard not run after surface rejection | 17 |
| evaluator calls with tracked zero repairs | 40 |
| evaluator calls with one repair | 3 |
| old-schema evaluator calls without repair metadata | 23 |

Surface-policy rejections were `explicit_operation=11`, `word_count=6`, and
`quantities=4`; one hint can contribute more than one reason.

### Gate Failures And Root Causes

1. **Mixed protocol:** records 0-22 were collected before evaluator repair metadata
   existed; record 23 onward used the repaired evaluator. The shard completion marker
   hashes config and dataset but not source commit, so R1 cannot prove one immutable
   collection policy.
2. **Unverifiable truncation:** paper collection logged word estimates rather than
   generated token counts and EOS/length finish reasons. Forty of 66 raw student
   outputs lack Qwen's closing thinking delimiter and several end mid-sentence. This is
   strong evidence of length termination, but it is not an authoritative truncation
   measurement.
3. **Evaluator accepts incomplete reasoning text:** some length-terminated responses
   contain the gold number inside unfinished reasoning. Without finish metadata, the
   evaluator can classify those responses as correct even though no final answer was
   completed.
4. **Over-restrictive surface policy:** prohibiting operation words and quantities
   rejected 17 of 32 hints before semantic review. This conflicts with the flexible,
   natural-Qwen method and prevents useful slight hints.
5. **Over-conservative leakage guard:** 13 of 15 surface-valid hints were marked
   `UNSAFE`, including broad relation-level hints that do not reveal a numeric answer.
6. **No guidance-correctness critic:** at least one privileged-teacher hint pointed in
   a mathematically wrong direction. Leakage safety alone is insufficient.
7. **Low pair yield:** only one pair was produced, so training from R1 is prohibited.

### Required R2 Corrections

- Start from example zero in a new artifact directory; never append to R1.
- Add source commit and a complete protocol fingerprint to progress, completion, merge,
  and run manifests.
- Log exact prompt/generated token counts, finish reason, EOS termination, and length
  truncation for every model call.
- Treat a length-truncated student response as incorrect regardless of extracted
  intermediate numbers.
- Use explicit role generation profiles: sampled native-thinking student; greedy
  non-thinking teacher, evaluator, leakage guard, and guidance critic.
- Increase the student completion ceiling to 8,192 tokens for R2, while retaining the
  user-approved Qwen sampling settings.
- Simplify the student prompt to reduce instruction-focused meta-reasoning without
  imposing a response format.
- Replace brittle operation/quantity prohibitions with a flexible short-hint surface
  contract and retain strict zero answer leakage through model review.
- Add a separate privileged guidance critic for mathematical correctness and relevance.
- Regenerate rejected greedy hints using explicit prior-review feedback.
- Run a small hard-example micro-preflight, then a fresh audited 64-example R2 gate.

## Mandatory Teacher-Free Baseline Gate

Status: one-example functional micro passed; the audited 16-example gate has not run
under the final observability commit.

Before corrected collection or any training, evaluate the pinned Qwen3.5-2B base
checkpoint without teacher guidance. The baseline protocol now includes:

- an immutable freeze over source commit, config, dataset manifest, student/evaluator
  revisions, prompt protocol, sampling profile, and generation seed;
- sampled native thinking at `temperature=1.0`, `top_p=0.95`, `top_k=20`, presence
  penalty `1.5`, and an 8,192-token completion ceiling;
- exact token counts, EOS/length finish reasons, truncation override, generation and
  evaluator latency, raw outputs, failure ledgers, GPU telemetry, and Slurm accounting;
- deterministic GPU shards, strict hash/ID-order merge, manual evaluator-agreement
  audit, HTML report, and one-time test markers.

The execution order is a one-example functional micro, a 16-example manually audited validation preflight, full
747-example validation, then one-time 1,319-example official-test baseline. The base
test result is descriptive and cannot affect prompts, collection, rewards,
hyperparameters, stopping, or model selection. Collection R2 remains blocked until the
baseline gate passes. The local suite passes 150 tests.

### Baseline Micro Execution

- Commit `ea45bd7b1a47ea32c1a9dc3df330d593829da5ff` was used for the first GPU micro.
- Job `12967` materialized the deterministic 16-example subset; job `12968` failed
  fast because an abbreviated source SHA was supplied; job `12969` created the freeze
  successfully with the full SHA.
- Job `12970_0` was cancelled after 9 minutes 54 seconds when the 16-example run was
  projected close to its wall-time limit. Its GPU telemetry is diagnostic only.
- Job `12971` materialized deterministic micro ID `gsm8k-5703`.
- Job `12972_0` completed the teacher-free one-example GPU micro in 1 minute 23 seconds.
- Job `12973` strictly merged the shard in 2 seconds; Turing automatically added GPU
  billing because the merge wrapper requested eight CPU cores. Both merge wrappers are
  now limited to two cores to prevent that waste.

Micro result: exact accuracy `1/1`; EOS termination `1/1`; truncation `0/1`; student
prompt/generated tokens `105/3,058`; student latency `65.334` seconds; evaluator tokens
`28`; evaluator latency `7.207` seconds. The raw response was correct but excessively
verbose for a simple arithmetic problem. Ten-second GPU telemetry missed the short 9B
evaluator peak, so paper GPU scripts now sample every second. Exact-token paper records
no longer emit an additional word-count estimate. These observability changes require a
new source commit and freeze before the 16-example baseline gate.
