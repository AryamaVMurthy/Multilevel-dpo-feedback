# SearchQA Minimal-Intervention DPO

This repository trains Qwen3 base students on SearchQA with full-parameter BF16 updates.

The active production protocol is fixed retrieval with cited reasoning:

```text
raw structured SearchQA row -> one strict search query -> frozen per-row BM25 retrieval
                            -> three-line cited answer/reasoning/sources response
                            -> canonical scoring and URL rendering
```

`generate-searchqa` rebuilds versioned prompts from `question` and dataset-owned `sources`; it never consumes a stored formatted prompt. Retrieval is frozen at BM25 `top_k=8`, `k1=1.2`, and `b=0.75`, with effective top-k reduced only when a row has fewer than eight sources. Model output is normal text. The renderer alone appends canonical dataset-owned titles and URLs. Active evaluation and preflight recompute retrieval and cited scoring and reject prediction-owned retrieval, metadata, metrics, or rendering that does not match.

The older plain short-answer and minimal-intervention workflow (`generate`, `collect`, and its training-data commands) is archival and must be selected explicitly. In that archival protocol, the student emits a short answer such as `Hemingway`; the teacher emits an internal answer-free hint.

The primary student is `Qwen/Qwen3-4B-Base`. Full finetuning uses DeepSpeed ZeRO-3, gradient checkpointing, BF16, and a strict 4096-token total limit. The 1.7B base student is allowed only after a recorded 4B OOM under the minimum viable ZeRO-3 configuration. The primary teacher is `Qwen/Qwen3-32B` in 4-bit NF4 on one GPU; 14B is allowed only after an explicit, archived 32B failure and a new run with the fallback model named in its manifest.

Raw base, SFT, minimal-intervention DPO, GRPO, and DAPO are evaluated teacher-free. Direct and private two-pass student thinking are compared on train-derived development data before the official validation protocol is frozen. Test data is untouched until selection is complete.

Run from `implementation/` with `PYTHONPATH=src`. Active generation uses independently configurable query and cited-response batches and completion budgets, with a strict total context limit of 4096 tokens and explicit truncation metadata. Every rollout cache is keyed by model revisions, dataset revision, prompt version, thinking modes, decoding, intervention policy, seed, and checkpoint hash; any mismatch fails.

Local verification:

```bash
PYTHONPATH=src uv run --frozen pytest -q
uv run --frozen ruff check src tests
python3 -m compileall -q src tests
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

Real model work runs only in Turing Slurm allocations. See `docs/plans/2026-07-13-plain-answer-thinking-design.md` and `docs/plans/2026-07-13-plain-answer-thinking.md`.
