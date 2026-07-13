# SearchQA Minimal-Intervention DPO

This repository trains Qwen3 base models on SearchQA only.

The primary method is full-parameter DPO over student-generated XML responses:

```text
student answer → one minimal XML hint → student retry → first correct answer
```

The teacher never writes the chosen answer. Hints accumulate until the student succeeds or the trajectory is explicitly unresolved. DPO prompts contain only the original question and evidence, so inference is teacher-free.

All model I/O uses XML:

```xml
<response><answer>...</answer><evidence>...</evidence></response>
```

Training is full-parameter BF16 with a fixed total sequence limit of 4096 tokens and DeepSpeed ZeRO-3. Adapter training, auxiliary domains, private-reasoning scaffolds, teacher-written answers, and fake data are not part of the active project.

SFT, GRPO, and DAPO are comparison arms after primary DPO is frozen. SFT may also be used as a warm-start only when the explicit SFT gate requires it.

Run from `implementation/` with `PYTHONPATH=src`:

```bash
uv run tfdpo prepare-searchqa ...
uv run tfdpo collect ...
uv run tfdpo build-preferences ...
uv run tfdpo train-sft ...
uv run tfdpo train-dpo ...
uv run tfdpo train-grpo ...
uv run tfdpo train-dapo ...
uv run tfdpo evaluate ...
uv run tfdpo generate ...
```

Local verification:

```bash
PYTHONPATH=src uv run --frozen pytest -q
PYTHONPATH=src uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen python -m compileall -q src tests
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

Real model work must run inside Turing Slurm allocations. Login nodes are only for source synchronization, submission, and log inspection.
