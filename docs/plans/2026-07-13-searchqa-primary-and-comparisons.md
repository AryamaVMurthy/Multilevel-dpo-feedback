# SearchQA Primary DPO and Comparison Arms

**Goal:** Train Qwen3 base models with full-parameter minimal-intervention DPO on SearchQA, then compare frozen SFT, GRPO, and DAPO arms under the same teacher-free evaluation protocol.

**Primary method:** One student XML response, one compact XML feedback intervention, repeated with accumulated hints until the student produces the first correct XML answer. The chosen DPO completion is always student-generated. Teacher answers and hints never appear in the DPO prompt.

**Models:** `Qwen/Qwen3-4B-Base` student; quantized `Qwen/Qwen3-32B` teacher when the one-GPU fit probe passes; explicit `Qwen/Qwen3-14B` fallback otherwise. The student may use `Qwen/Qwen3-1.7B-Base` only after a recorded 4B full-finetuning OOM.

**Training:** BF16 full-parameter updates, DeepSpeed ZeRO-3, gradient checkpointing, explicit attention implementation, and total sequence length 4096. DPO uses TRL `sigmoid_norm`; GRPO and DAPO use deterministic SearchQA reward functions. No adapters or teacher-written completions.

## Frozen execution order

1. Materialize pinned SearchQA train/validation/test artifacts and evidence packs.
2. Evaluate the raw base checkpoint teacher-free.
3. Run the explicit SFT necessity gate; train SFT only if required.
4. Collect batched iterative trajectories with one teacher GPU and student generation workers.
5. Reuse cached trajectories whenever policy and protocol hashes match.
6. Build standard, multilevel, matched, and verified repair-region preference rows.
7. Run full-parameter DPO rounds with checkpoint evaluation and resume support.
8. Freeze the best DPO protocol/checkpoint.
9. Train and evaluate SFT-only, GRPO, and DAPO comparison arms.
10. Run final seeds, statistical comparisons, and the untouched test set.
11. Generate the HTML report with raw artifact links and GPU/job accounting.

## Failure policy

Missing data, malformed XML, teacher leakage, cache hash mismatch, OOM, unsupported trainer APIs, invalid CUDA, incomplete checkpoints, or Slurm/authentication failures stop the run with explicit context. Only the configured model-size fallbacks are permitted, and each records `fallback_reason`.

## Current status

- Local SearchQA/XML/trajectory/preference/offline-reuse tests pass.
- TRL 1.8.0 SFT, DPO, GRPO, and DAPO configurations have been API-probed.
- Turing scripts pass shell validation.
- The local RTX 3050 has only 6 GB and is not a valid full-training environment.
- Turing access is verified as `aryama.murthy@turing.iiit.ac.in` under Slurm account `priyesh.shukla`.
- Obsolete legacy checkouts, failed MATH/Qwen3 runs, and their 7.4G of artifacts were removed; the active checkout and raw SearchQA artifacts are preserved.
- Turing preflight passed on an RTX 6000 Ada with BF16, bitsandbytes, DeepSpeed, Transformers, and TRL. FlashAttention-2 is absent, so SDPA is selected explicitly and the reason is logged.
- Official pinned SearchQA train/validation/test materialization completed with per-split manifests; rows with no usable official snippet are counted as `no_usable_evidence` drops.
- The raw Qwen3-4B baseline is currently generating validation and test predictions before SFT/DPO proceeds.
- Baseline shard merging now requires exact input/prediction line-count parity and removes duplicate shard inputs only after atomic merge publication, reducing home-quota pressure before full-parameter SFT.
- SFT target construction was tightened to a short answer-centered excerpt of real evidence and made streaming; queued rebuild jobs 13644/13645 feed SFT job 13630 before full-parameter training.
- Empty student generations now have an explicit `__EMPTY_RESPONSE__` XML error/rejection sentinel, so the iterative teacher contract and DPO rows remain truthful and fail-fast without silently inventing a response; local regression suite: 29 passed.
- SFT job 13630 was dependency-optimized to wait for raw validation evaluation and both compact SFT-data rebuilds, while raw test generation/evaluation proceeds independently; raw test metrics remain mandatory for the final report.
