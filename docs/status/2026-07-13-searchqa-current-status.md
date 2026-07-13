# SearchQA Minimal-Intervention Research Status

**Snapshot date:** 2026-07-13 (Asia/Kolkata)  
**Turing checkout:** `~/searchqa-dpo/fixed-retrieval-v1`  
**Last observed remote commit:** `d69ab13831bb7c2ff3510113b6d305383f77c10d`

**Current local implementation commit:** `f46913e712f510476824f4eaa0a1018190b8171c`

## Executive status

The fixed-retrieval SearchQA pipeline, strict cited-response evaluator, batched generation path, minimal-hint trajectory code, student-only SFT/DPO data gates, Turing launch contracts, and hardware probes exist and pass the local test suite. The pinned Qwen3-4B-Base student and Qwen3-32B 4-bit teacher both fit their assigned A100-40GB GPUs.

The raw student was not ready for preference-data collection at scale: the original audited validation preflight produced zero fully correct protocol responses. The controlled v2 scaffold plus explicit minimum-generation contract removed empty-output collapse and supplied enough canonically verified, student-generated, no-hint targets for an SFT overfit gate. A four-GPU Qwen3-4B full-parameter BF16 ZeRO-3 SFT run has now completed 20 optimizer steps with real save/resume. DPO, GRPO, DAPO, and full-scale SFT have not started.

The overfit checkpoint is a material protocol improvement on the untouched 32-row validation sample: all 32 queries and all 32 responses are nonempty, parse-valid, cited three-line outputs, with zero truncations. Canonical retrieval recall@8 is 25/32 and canonical exact-answer correctness is 18/32. Manual review found one additional semantically accepted alias (`Launch and Arrival` against gold `takeoff & landing (or launch & arrival)`), which is reported separately and is not silently converted into training supervision or a changed canonical score.

## Dataset state

Pinned source: `kyunghyuncho/search_qa` at revision `06907e45883b7cae435453b65d598447039fde79`.

| Split | Official rows | Materialized usable rows | Status |
|---|---:|---:|---|
| Train | 151,295 | 151,277 | Ready; 18 rows explicitly dropped for no usable evidence |
| Validation | 21,613 | 21,611 | Ready; 2 rows explicitly dropped for no usable evidence |
| Test | 43,228 | Not materialized | Intentionally untouched for final reporting |

Important identities:

- Train JSONL SHA-256: `6d5d679ffbca04051f63d802c8f709e3f3503a39201f6f219c712cb7af38da8c`
- Validation JSONL SHA-256: `7f33939cf10fd876a0e9aeb1f2536ff6a9e9f27a5720190bd08dd62ac452cb53`
- Audited 32-row sample SHA-256: `c3e59f4f5a0c551cb276f614553b88562f66d771118bd9442fcdaf6fdf48f75a`
- Retrieval identity SHA-256: `dbcbff5eeba529cb51361191378791b4e2afc371bdaee98ecac19140d83df97d`
- Source-schema identity SHA-256: `59fe2f48bbbc35395d689c16f5aabf41103e78a6511afeeeabfc5bde20a939dc`
- Active bounded prompt identity SHA-256: `5c52f50fb2122acd9c2fa3c334d7fe0cc276a92ffe6dac810510f0a3e0b94f27`
- Active config SHA-256: `992faecd7db375456ca71d2f3a1c5b0f01ccb80bd1a98fe84afd9e0060a7a501`

## Model and hardware evidence

- Student: `Qwen/Qwen3-4B-Base` revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Teacher: `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137`, 4-bit NF4 with BF16 compute and native teacher thinking.
- The 4B student fit probe passed. The 32B teacher fit probe passed; the 14B teacher fallback was not used.
- Generation baseline on node10: SDPA, batch 4, approximately 22.1 generated tokens/second, approximately 10.5 GiB framework peak memory, approximately 97.6% mean measured GPU utilization.
- Flash Attention 2 and Liger are absent in the pinned environment. The explicit current decision is `sdpa_baseline_selected`; no hidden kernel fallback is active.
- Two-GPU collection decision: teacher `cuda:0`, student `cuda:1`, both A100-SXM4-40GB.
- Job `13773` remeasured the sample-bound SDPA baseline at commit `909d9f3`: 22.126 tokens/second, 97.56% mean GPU utilization, and 10.47 GiB framework peak memory. Its decision SHA-256 was `73bfd5404097e5da42b3164642bed56f218ad0d6a9c5ba5094b7df48953ffd84`.
- That decision is now intentionally stale because the bounded teacher prompt changes the prompt identity and config. A fresh commit-bound decision is required before rerunning collection.

## Storage evidence

- Turing home quota: 50 GiB total, 42 GiB used, 8.6 GiB available (83% used at this snapshot).
- Node10 scratch: 14 TiB total, 5.8 TiB available.
- Active fixed-retrieval checkout: 6.5 GiB, of which 6.4 GiB is materialized data.
- Full 4B checkpoints with optimizer state and full trajectory dumps cannot safely be written to home. Run data, optimizer checkpoints, model caches, and trajectory shards must live under a manifest-bound node10 scratch root; home should retain source, small logs, metrics, manifests, and final research summaries.
- The SFT overfit output occupies about 165 GiB on node10 scratch because it contains three retained full optimizer checkpoints plus a final model. It is not in home quota. Retention will be reduced only after the required resume and checkpoint identities are persisted and verified.
- Other large home paths (`slm-research-qwen`, 22 GiB; `browser-agent-run`, 7.6 GiB; obsolete plain-answer checkout, 5.3 GiB) were measured but not deleted. Cleanup requires an artifact/lineage audit so unrelated work is not removed.

## Exhaustive 32-row student audit

Artifacts audited:

- `outputs/preflight-v5/direct-predictions.jsonl`
- `outputs/preflight-v5/direct-metrics.json`
- `outputs/preflight-v5/two_pass-predictions.jsonl`
- `outputs/preflight-v5/two_pass-metrics.json`
- Slurm job `13760` logs

### Direct mode

| Metric | Value |
|---|---:|
| Fully correct responses | 0 / 32 |
| Valid three-line format | 1 / 32 (3.125%) |
| Nonempty scored response | 2 / 32 (6.25%) |
| Valid-citation rate | 28.125% |
| Retrieval recall@8 | 46.875% |
| Empty raw query | 10 / 32 |
| Empty raw response | 13 / 32 |
| Query-invalid-format failures | 12 / 32 |
| Line-count failures | 13 / 32 |
| Label-order failures | 5 / 32 |
| Markup-forbidden failures | 1 / 32 |

The direct model has useful latent behavior, but not a usable protocol policy. Examples:

- Gold `Hemingway`; query retrieved an answer-bearing result at rank 1; raw response was only `Hemingway`. Semantically right, but no reasoning or citation and therefore correctly rejected.
- Gold `ATMs`; raw response contained `Automated teller machines`, cited reasoning, and a source list, but omitted the required `Answer:` label. It was correctly rejected as `label_order`.
- Gold `Bucharest`; raw response answered `Moscow`. This is a substantive answer error, not just formatting.
- The only parse-valid three-line response answered `Porkchop plot` when the question asked for the two plotted flight events. It was structurally valid but answer-incorrect and unsupported, so it was correctly excluded.
- Ten rows generated an empty query; thirteen generated no response.

The zero answer metric is therefore not solely a parser artifact. Several raw responses reveal correct answer knowledge, but the audited set contains no fully correct, supported, protocol-valid student continuation.

### Two-pass mode

| Metric | Value |
|---|---:|
| Fully correct responses | 0 / 32 |
| Valid format | 0 / 32 |
| Nonempty scored response | 0 / 32 |
| Retrieval recall@8 | 21.875% |
| Empty raw query | 17 / 32 |
| Empty raw response | 28 / 32 |
| Query truncation | 5 / 32 |

The custom two-pass path leaks prompt instructions into the generated query and response. Examples include `Do not explain your answer...`, `This text will not be scored`, and copied response-policy instructions. It is not a viable bootstrap mode and must remain disabled unless a later trained checkpoint passes a fresh controlled comparison.

## Teacher evidence

The Qwen3-32B 4-bit teacher produced a schema-valid, answer-free hint in the fit probe:

`Review the top source's snippet for the correct name.`

This proves model fit and basic hint parsing only. It does not prove that 32 real examples can be resolved, that hints remain answer-free at all escalation levels, or that no-hint siblings become successful. Those properties still require the collection smoke test.

## Trajectory and training status

- No complete teacher-guided trajectory JSONL exists yet.
- The first smoke launch, job `13761`, stopped before model collection because its optimization decision was bound to the full validation hash while the input was the 32-row sample hash. The fail-fast identity check worked as intended.
- Job `13764` then failed before model loading because `POLICY_HASH` was a human-readable label instead of a SHA-256. The cache contract now computes and verifies a canonical student-policy identity from model, revision/checkpoint identity, and policy version; malformed or mismatched hashes fail before model loading.
- Job `13774` passed all provenance and hardware gates, loaded the Qwen3-4B student on GPU 1 and the Qwen3-32B 4-bit teacher on GPU 0 without OOM or fallback, and generated the first student batch. It then failed safely before teacher generation because teacher prompt 0 contained 18,837 tokens while only 3,584 input tokens fit the 4,096 total-token budget with a 512-token teacher reserve.
- The root cause was unbounded duplication of every materialized SearchQA source inside the private teacher prompt. The audited sample has 6–99 complete source records per row and up to about 61 KB of source JSON. The teacher already receives the retrieved top records, private gold answer, raw attempt, deterministic diagnostics, and escalation history; duplicating all nonretrieved records was unnecessary.
- Commit `d9e909e` removes complete-source duplication and retains only compact retrieved `source_id`/title/snippet records plus `available_source_count`. This is deterministic context selection, not hidden truncation.
- Job `13776` confirmed the unbounded prompt was gone, then exposed a separate valid base-model state before teacher inference: an empty/invalid student query produced zero retrieved records, and the compact-record validator rejected the empty list. Empty retrieval is now explicitly accepted only when deterministic diagnostics identify `query/retrieval` as the repair region. No fake source is inserted; non-query repair regions still reject empty retrieval.
- Job `13778` reached real 32B teacher generation with native thinking and no OOM, using about 27.4 GiB on the teacher GPU and about 9.9 GiB on the resident student GPU. A 96-token teacher cap ended inside an unterminated Qwen thinking block, so the strict final JSON extractor correctly rejected it. The cap was then restored to the fit-verified 512 tokens, and collection began logging every rendered teacher prompt token count plus its 4,096-token budget decision before inference.
- Job `13780` proved all 32 rendered prompts fit with the 512-token reserve (395–1,685 prompt tokens; maximum allowed input 3,584), but at least one output still exhausted 512 tokens before closing native thinking. The next controlled cap is 1,024 tokens. Collection now logs every teacher output token count and all malformed-thinking indices without logging private reasoning text; a malformed block remains a hard failure with `fallback_reason=teacher_thinking_budget_exhausted`.
- Job `13782` completed its first teacher round and then failed safely at 20:54 IST. All 32 prompts fit (395–1,685 input tokens against a 3,072-token allowance), and 31/32 native-thinking outputs closed within 147–543 tokens. Only zero-based batch index 25 reached exactly 1,024 tokens and remained inside its thinking block. This is a single budget outlier, not OOM or a batch-wide teacher failure.
- The next implementation uses an explicit bounded per-row retry: preserve 1,024 as the primary cap, retry only outputs that are both malformed and exactly budget-exhausted at 2,048, verify the retry prompt remains within 4,096 before inference, log original indices/reason/token counts, and fail if the retry remains malformed. Short malformed outputs are not retried. This is a declared control policy, not a hidden fallback.
- Local commit `3bb0267` adds a canonical trajectory auditor producing JSON, JSONL, CSV, and HTML with exact question/gold/query/retrieval/response/hint/retry/sibling/eligibility evidence. Runtime latency and per-row teacher token counts remain explicit unavailable fields when older logs do not contain correlatable telemetry; they are never estimated.
- Local commit `c52962c` adds batched multi-seed no-hint bootstrap collection and a one-GPU Turing launcher. The model is loaded once, each seed is generated across the active example batch, every artifact is canonically revalidated, and duplicate/tampered/incomplete output fails explicitly.
- Job `13783` remeasured the commit/config-bound SDPA baseline after adding explicit teacher retry: 22.132 generated tokens/second, 97.6% mean measured utilization, 10.47 GiB framework peak memory. Frozen decision SHA-256: `4095b1d0eee507c10d33765b9e8e50f3a6a04630dd4c21ce155d3435d5405d30`.
- Job `13786` selected the deterministic 128-example train-only bootstrap pool in one streaming pass over the pinned train file. Pool SHA-256: `f2c3df4c0ef3272183e07d26e8546516391972722b4bc64f69d697f135959836`; selected-ID SHA-256: `bbaba67ed0d9e4cc824469cfcd9a60623bacdf756527cac3f6f3dd3b9c324a05`.
- Job `13789` terminated after 11:21. The bounded retry mechanism worked as designed: only exact-cap malformed row 25 was retried at 2,048 tokens and its retry closed after 270 tokens. Collection then stopped at the independent strict hint-leakage gate for validation row 7 (`the Tour de France`), reported as `hint contains a normalized gold answer token`. The rejected final hint was not persisted, so the detector must first report the exact overlapping normalized token before any leakage rule can be safely changed.
- Local commit `f0f154f` splits bootstrap SFT selection by task. A no-hint query is eligible when canonical BM25 retrieves answer-bearing evidence even if the response fails; a response remains eligible only when it is parse-valid, answer-correct, supported, cited, untruncated, student-generated, and no-hint.
- Job `13790` completed in 5:06 with exact cardinality (128 examples, 256 candidates). Query eligibility is useful: 147 candidates across 90 unique examples retrieved answer-bearing evidence. Response eligibility is insufficient: only 8/256 outputs parsed, 5/256 had the correct answer, and 2/256 were fully correct/supported/cited across two unique examples. There were 50 empty queries and 59 empty responses; dominant response errors were line count (138) and label order (47).
- Because unchanged scaling would miss the response-SFT gate, commit `a078c6e` adds the controlled v2 response scaffold. The model-visible prompt ends with `Response:` then `Answer:`; generation supplies only the continuation, while canonical evaluation explicitly composes the fixed `Answer: ` prefix. The prefix and raw generated continuation are persisted and revalidated, so this is not hidden parser repair. Query prompting remains unchanged.
- Root-cause inspection found that the generation runtime allowed EOS at the first generated token. The explicit-minimum change passes `min_new_tokens` into model generation, rejects invalid min/max combinations, requires both bootstrap minima at the launcher boundary, and records them in the artifact manifest. The controlled rerun freezes query minimum 2 and response minimum 8; it does not post-process or fabricate output.
- Commit `cf7e53a` deployed the explicit minimum-token contract. Job `13792` reran the identical 128 examples and seeds 11/12 with only the v2 scaffold and query/response minima changed. It completed with exact cardinality in 5:22. Empty queries fell from 50 to zero; parse-valid responses rose from 8 to 144; answer-correct responses rose from 5 to 83; and fully protocol-correct responses rose from 2 to 44 across 33 unique examples. Answer-bearing retrieval rose from 147 candidates/90 examples to 186 candidates/100 examples. No response truncated.
- The strict SFT selector further requires lexical cited-answer support. It produced 100 unique query targets and 26 unique response targets (126 total), all nonempty, student-generated, no-hint, and within 4,096 tokens. SFT artifact SHA-256: `144e4e41c43f2fef60077cb73ead1b1880e0a514864205b3d9feae088ce3048a`; report SHA-256: `4b4983fcf6faf27481ee09d3a51ab3f09958e8c66b2d4a8dc636b12648993d88`.
- Remaining v2 failures are no longer dominated by empty output: 58 candidates use forbidden grouped-citation markup, 39 violate the exact three-line count, 8 have a nonempty but invalid query format, one query truncates, four duplicate a citation, one has an empty answer, and one exceeds the reasoning limit. These outputs remain rejected; none are repaired into supervision.
- Job `13797` completed the same train-only pilot with seeds 11–14: 512 candidates, zero empty queries, 19 absent downstream responses caused by invalid nonempty query format, 290 parse-valid responses across 116 examples, 175 answer-correct responses across 65 examples, and 89 fully correct responses across 47 examples. Strict lexical-support filtering produced 102 unique query targets and 35 unique response targets. Raw bootstrap artifact SHA-256: `8b377e862ab4d801c4ab63c8b3089601fdd17b915ec34e18856b3d2e3aa12b10`.
- Job `13799` built 137 verified SFT rows (102 query, 35 response), all student-generated, no-hint, nonempty, and at most 4,096 tokens. SFT artifact SHA-256: `d2b6c4d5cd28e49e7659a72b9b57949741b9a89483849d01fc1dbcecb8f42e90`. Deterministic balancing selected 32 query plus 32 response targets without replacement; balanced artifact SHA-256: `ac49f26c8eb5d7ec56ec1c9e4b4d75c155a4d981634db2a2b7611ce154ea3f5f`.
- Job `13801` ran actual full-parameter Qwen3-4B BF16 SFT on four A100 GPUs with ZeRO-3, max length 4,096, and a real resume from step 5 through step 20. The 64-row overfit evaluation moved from loss `0.2633` / token accuracy `0.9140` at step 5 to loss `0.1116` / token accuracy `0.9700` at step 20. Checkpoint 20 model SHA-256: `919dff1363d5827728497479e30dc8a767aa67f50ef2bbec938978a559b2355e`.
- Slurm marked job `13801` failed only after training. `save_total_limit=3` correctly retired checkpoint 5 after checkpoints 10, 15, and 20 existed, but post-training manifest code then attempted to hash the retired directory and emitted `checkpoint has no files: .../checkpoint-5`. Resume and all 20 optimizer steps had already succeeded. The launcher now records the initial checkpoint hash before resume and records whether the directory remains retained after training.
- Job `13806` generated one deterministic no-hint candidate for each row of the untouched 32-row validation sample from checkpoint 20. All 32 queries and 32 responses are nonempty; all responses parse into the exact cited three-line schema; no query or response truncates. Retrieval recall@8 is 25/32, canonical answer correctness is 18/32, and 10/32 also satisfy the stricter lexical cited-answer-support SFT rule. Every output was inspected. Errors are substantive answer/target-selection or retrieval errors, not empty-output or protocol collapse.
- Job `13808` deterministically regenerated all 64 balanced training prompts from checkpoint 20, with hashes bound to both the model and SFT data. Exact decoded-text reproduction is 33/64: 18/32 query and 15/32 response. There are zero empty outputs, zero response truncations, and one query truncation. Exact-text reproduction is intentionally not treated as the capability score because correct responses may paraphrase reasoning or cite another valid retrieved source.
- Job `13810` canonically revalidated every generated continuation against its original example, seed-selected bootstrap artifact, fixed retrieval context, and hashes with no repair. Generated queries retrieve answer-bearing evidence for 29/32 targets; one query repeats `cotton` to the 32-token cap and two valid queries miss answer-bearing evidence. All 32 generated responses parse; 31/32 have the exact canonical answer and 30/32 also cite a source with lexical answer support. The canonical answer miss is `Madeleine K. Albright` versus `Madeleine Albright` (F1 `0.8`), which remains a visible strict-metric miss rather than being normalized away.
- Turing automatically added one GPU to job `13810` because its CPU-only launcher requested four CPU cores. The audit finished in three seconds and did not execute CUDA work. The launcher is reduced to two cores so future capability audits remain CPU-only under the observed cluster billing rule.
- Job `13811` selected 4,096 train-only examples at seed `20260713` and deterministically assigned them to eight shards. Pool SHA-256: `a924bd068e0b1c1bda7d6b0c515db8778b41cd4e4ada8ef1762db0d35b23f6ff`; selected-ID SHA-256: `c24a0aa4885bd064f94a0a945811e2c57ee84e01b5ae649d6bf6107dbfa66b65`. Shards contain 479–551 examples and their individual hashes are sealed in the preparation manifest.
- Jobs `13812`–`13819` launched all eight checkpoint-20 rollout shards concurrently. Seven were cancelled after shard 3 failed, because mixing pre-fix and post-fix evaluator semantics would invalidate lineage. The exact root cause was train row `train-88271`, whose legitimate gold answer is standalone `A` (vitamin A). SQuAD-style article removal normalized `A` to an empty string, and retrieval scoring raised `gold answer must contain searchable normalized tokens` after model loading.
- Commit `d69ab13` fixes the evaluator at the normalization source: a standalone answer `A`, `An`, or `The` remains searchable, while articles are still removed from multiword answers. Retrieval and cited lexical-support checks use the same article-aware phrase matcher. Evaluator identity is bumped to `cited-response-evaluator-v2-standalone-article-answer`; old and new rollout shards cannot be merged. The exact vitamin-A regression and cited-response support are covered by tests.
- Job `13822` reran only the previously failing vitamin-A row against checkpoint 20. It completed in 56 seconds, retrieved answer-bearing evidence at rank 1, and produced a nonempty parse-valid cited response (`Vitamin A`) with no exception or truncation. The strict exact-answer metric still records `Vitamin A` versus gold `A` as a visible miss; no alias repair is applied.
- Jobs `13825`–`13832` are the clean evaluator-v2 restart: eight concurrent one-A100 workers, two seeds per example, query/response batch size 4, explicit 2/32 and 8/256 token bounds, temperature `0.7`, top-p `0.9`, checkpoint model SHA-256 verification, and no-hint direct generation.
- No optimizer step has run for DPO, GRPO, or DAPO. Full-scale SFT has not started; larger checkpoint-20 student-only rollout collection is now the active data gate.

## Verification state

- Local verification: 267 unit tests passed. Ruff, Python compile checks, Slurm shell syntax checks, and Git whitespace checks pass. One expected upstream Torch deprecation warning remains.
- Ruff: passed.
- Python compile checks: passed.
- All Slurm shell launchers: `bash -n` passed.
- Git whitespace checks: passed.

## Planning conclusions from the logs

1. Keep the student in direct mode for bootstrap. Preserve native thinking for the instruct teacher. Re-evaluate a student thinking mode only after SFT, using the official model/template behavior of the exact trained checkpoint.
2. Run the corrected 32-row teacher-guided smoke as a diagnostic, not as authorization for full collection. Inspect every attempt, hint, retry, sibling, teacher-leakage check, and eligibility decision.
3. Split query and response bootstrap eligibility. A student-generated no-hint query can be a valid query-SFT target when it is well formed and retrieves answer-bearing evidence, even if the downstream response fails. Response-SFT targets must still be fully parse-valid, answer-correct, supported, cited, no-hint student outputs on an identical canonical retrieval context.
4. If the smoke confirms near-zero response eligibility, perform broad batched no-hint student oversampling on a train-derived bootstrap pool and retain only canonically verified successes. Do not repair malformed strings, prepend missing labels after generation, copy teacher answers, or synthesize hidden targets.
5. Add a prompt/scaffold experiment only as a measured candidate: compare the current free-form completion prompt with a fixed non-answer prefix such as `Answer:` and stronger base-model continuation examples. Promote it only if raw model outputs, strict metrics, and output identity gates improve. The scaffold must be explicit in the prompt and training context; it cannot be a parser recovery path.
6. Start SFT before primary teacher collection if broad no-hint response coverage is sufficient. The observed failure is largely instruction/format adherence, so a small student-only SFT bootstrap is causally justified and likely necessary before collecting high-yield minimal interventions.
7. Require manual response audits throughout training: fixed examples, random examples, correct/incorrect cases, retrieval failures, citation failures, hint leakage, and no-hint sibling behavior at every promotion checkpoint.

## Immediate next evidence gate

The SFT overfit promotion gate has passed: the manifest fix, deterministic reproduction, canonical capability scoring, held-out generation, exact response inspection, and save/resume evidence all exist. The next full-SFT data gate requires:

- Select a larger train-only bootstrap pool at a frozen seed and hash; validation and official test remain excluded.
- Shard it deterministically and run checkpoint-20 student-only, no-hint rollouts in parallel across available GPUs with batched query/response generation and explicit per-shard manifests.
- Merge only exact shard IDs with duplicate/missing detection, then canonically revalidate every candidate before SFT selection.
- Require nonempty, untruncated, answer-bearing query targets and fully parse-valid, answer-correct, cited, lexically supported response targets. Do not repair or synthesize rejected outputs.
- Freeze full-SFT train/eval splits by trajectory ID, learning rate, checkpoint interval, generation-validation interval, and stop/promotion criteria before launching the longer four-GPU full-parameter run.
- Preserve checkpoint 20 and the recorded checkpoint-5 resume hash; remove redundant scratch checkpoints only after the retention manifest is written and verified.

The independent teacher gate still requires:

- A corrected 32-row teacher-guided trajectory file with exact row parity.
- A tokenizer-measured teacher-prompt budget report proving every rendered smoke prompt fits the 4,096 total-token contract before scaling.
- A per-attempt table showing question, gold answer (audit-only), query, top retrieved records, raw response, parser/score diagnostics, teacher hint, retry, and sibling outcomes.
- Zero teacher-answer leakage and zero hidden parser repair.
- Measured resolution rate, hint-level curve, no-hint sibling success rate, SFT eligibility, preference eligibility, latency, throughput, and GPU utilization.
- A written decision choosing teacher-first collection, student-only bootstrap SFT, or an explicit prompt-scaffold experiment based on those numbers.
