# Local Concise Generation Sweep Implementation Plan

**Goal:** Implement and execute the reproducible train-only Qwen3-4B presence-penalty sweep on Turing.

**Architecture:** Extend the locked generation schema with explicit Qwen sampling fields and balanced-box stopping, then run deterministic profile/example tasks through one observable CLI. Artifacts are append-safe and summarized into Markdown, HTML, JSON, CSV, and plots.

**Tech Stack:** Python 3.12, Transformers, PyTorch CUDA, Qwen chat template, SymPy evaluator, pytest, Matplotlib, NVIDIA SMI.

## Tasks

1. Add failing unit tests for nested balanced boxes, incomplete/empty boxes, prompt-token exclusion, stopping metadata, `min_p`, and `repetition_penalty` propagation.
2. Implement the minimal Transformers stopping criterion and strict generation schema fields.
3. Add failing tests for deterministic train-only stratification, immutable profile manifests, resumable records, promotion, and report metrics.
4. Implement the sweep CLI, GPU telemetry, exact artifact schema, promotion rule, plots, HTML, and Markdown report.
5. Run `PYTHONPATH=src uv run --frozen --extra dev python -m pytest -q` and config validation.
6. Download only the pinned model revision and official MATH training source; record hashes and disk usage.
7. Run one candidate/example smoke and inspect raw output, termination, correctness, tokens, latency, and VRAM.
8. Run Stage A, promote three candidates without changing the rule, and inspect all failures/truncations.
9. Run Stage B only if the 8,192-token smoke remains within VRAM; otherwise record an explicit feasibility failure.
10. Generate the final comparison report, update the living design/status documents, commit scoped files, and push the canonical branch.
