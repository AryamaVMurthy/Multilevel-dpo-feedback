# SearchQA Minimal-Intervention DPO

This repository trains Qwen3 base students on SearchQA with full-parameter BF16 updates.

The active protocol is:

```text
plain student answer -> one slight answer-free hint -> plain student retry
                     -> accumulate and minimally escalate hints until exact-correct or capped
```

The student emits only a short answer such as `Hemingway`. The teacher is a post-trained Qwen3 instruct model using native private thinking and emits one internal JSON hint. The teacher never writes the chosen answer. DPO chosen answers are successful student generations; DPO prompts are the original hint-free question and evidence.

The primary student is `Qwen/Qwen3-4B-Base`. Full finetuning uses DeepSpeed ZeRO-3, gradient checkpointing, BF16, and a strict 4096-token total limit. The 1.7B base student is allowed only after a recorded 4B OOM under the minimum viable ZeRO-3 configuration. The primary teacher is `Qwen/Qwen3-32B` in 4-bit NF4 on one GPU; 14B is allowed only after an explicit, archived 32B failure and a new run with the fallback model named in its manifest.

Raw base, SFT, minimal-intervention DPO, GRPO, and DAPO are evaluated teacher-free. Direct and private two-pass student thinking are compared on train-derived development data before the official validation protocol is frozen. Test data is untouched until selection is complete.

Run from `implementation/` with `PYTHONPATH=src`. Every rollout cache is keyed by model revisions, dataset revision, prompt version, thinking modes, decoding, intervention policy, seed, and checkpoint hash; any mismatch fails.

Local verification:

```bash
PYTHONPATH=src uv run --frozen pytest -q
uv run --frozen ruff check src tests
python3 -m compileall -q src tests
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

Real model work runs only in Turing Slurm allocations. See `docs/plans/2026-07-13-plain-answer-thinking-design.md` and `docs/plans/2026-07-13-plain-answer-thinking.md`.
