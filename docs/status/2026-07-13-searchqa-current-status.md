# SearchQA Minimal-Intervention Research Status

**Snapshot date:** 2026-07-13 (Asia/Kolkata)  
**Turing checkout:** `~/searchqa-dpo/fixed-retrieval-v1`  
**Verified remote commit:** `bf9bac061dbf42e41cbf5aeac7658ff56e012534`

## Executive status

The fixed-retrieval SearchQA pipeline, strict cited-response evaluator, batched generation path, minimal-hint trajectory code, student-only SFT/DPO data gates, Turing launch contracts, and hardware probes exist and pass the local test suite. The pinned Qwen3-4B-Base student and Qwen3-32B 4-bit teacher both fit their assigned A100-40GB GPUs.

The raw student is not ready for training-data collection at scale. On the audited 32-row validation preflight, neither direct nor private two-pass generation passed the structural gate, and no rollout was a fully correct, supported, protocol-valid response. No SFT, DPO, GRPO, or DAPO optimizer step has run.

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
- Active prompt identity SHA-256: `7b17e0c27641ea46f57afe38044c78c660a7be051e850c5b5f0eed09e34a90e8`

## Model and hardware evidence

- Student: `Qwen/Qwen3-4B-Base` revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Teacher: `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137`, 4-bit NF4 with BF16 compute and native teacher thinking.
- The 4B student fit probe passed. The 32B teacher fit probe passed; the 14B teacher fallback was not used.
- Generation baseline on node10: SDPA, batch 4, approximately 22.1 generated tokens/second, approximately 10.5 GiB framework peak memory, approximately 97.6% mean measured GPU utilization.
- Flash Attention 2 and Liger are absent in the pinned environment. The explicit current decision is `sdpa_baseline_selected`; no hidden kernel fallback is active.
- Two-GPU collection decision: teacher `cuda:0`, student `cuda:1`, both A100-SXM4-40GB.
- Sample-bound generation decision SHA-256: `866369e355585c0a691aafcdb308847dcd04512fda5ce176c5314e4e539b6bdf`.

## Storage evidence

- Turing home quota: 50 GiB total, 42 GiB used, 8.6 GiB available (83% used at this snapshot).
- Node10 scratch: 14 TiB total, 5.8 TiB available.
- Active fixed-retrieval checkout: 6.5 GiB, of which 6.4 GiB is materialized data.
- Full 4B checkpoints with optimizer state and full trajectory dumps cannot safely be written to home. Run data, optimizer checkpoints, model caches, and trajectory shards must live under a manifest-bound node10 scratch root; home should retain source, small logs, metrics, manifests, and final research summaries.
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
- A measured sample-bound generation decision now exists and is ready for a corrected smoke launch.
- Because the audited base set has zero fully correct continuations, the existing SFT builder would currently produce zero rows for this sample.
- No optimizer step has run for SFT, DPO, GRPO, or DAPO.

## Planning conclusions from the logs

1. Keep the student in direct mode for bootstrap. Preserve native thinking for the instruct teacher. Re-evaluate a student thinking mode only after SFT, using the official model/template behavior of the exact trained checkpoint.
2. Run the corrected 32-row teacher-guided smoke as a diagnostic, not as authorization for full collection. Inspect every attempt, hint, retry, sibling, teacher-leakage check, and eligibility decision.
3. Split query and response bootstrap eligibility. A student-generated no-hint query can be a valid query-SFT target when it is well formed and retrieves answer-bearing evidence, even if the downstream response fails. Response-SFT targets must still be fully parse-valid, answer-correct, supported, cited, no-hint student outputs on an identical canonical retrieval context.
4. If the smoke confirms near-zero response eligibility, perform broad batched no-hint student oversampling on a train-derived bootstrap pool and retain only canonically verified successes. Do not repair malformed strings, prepend missing labels after generation, copy teacher answers, or synthesize hidden targets.
5. Add a prompt/scaffold experiment only as a measured candidate: compare the current free-form completion prompt with a fixed non-answer prefix such as `Answer:` and stronger base-model continuation examples. Promote it only if raw model outputs, strict metrics, and output identity gates improve. The scaffold must be explicit in the prompt and training context; it cannot be a parser recovery path.
6. Start SFT before primary teacher collection if broad no-hint response coverage is sufficient. The observed failure is largely instruction/format adherence, so a small student-only SFT bootstrap is causally justified and likely necessary before collecting high-yield minimal interventions.
7. Require manual response audits throughout training: fixed examples, random examples, correct/incorrect cases, retrieval failures, citation failures, hint leakage, and no-hint sibling behavior at every promotion checkpoint.

## Immediate next evidence gate

The next stage is complete only when all of the following exist:

- A corrected 32-row teacher-guided trajectory file with exact row parity.
- A per-attempt table showing question, gold answer (audit-only), query, top retrieved records, raw response, parser/score diagnostics, teacher hint, retry, and sibling outcomes.
- Zero teacher-answer leakage and zero hidden parser repair.
- Measured resolution rate, hint-level curve, no-hint sibling success rate, SFT eligibility, preference eligibility, latency, throughput, and GPU utilization.
- A written decision choosing teacher-first collection, student-only bootstrap SFT, or an explicit prompt-scaffold experiment based on those numbers.
