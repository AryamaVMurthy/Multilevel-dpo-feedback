# Multilevel Feedback DPO

Paper-scale Qwen3 research comparing Base, Standard LN-DPO, Multilevel LN-DPO,
pair-budget-matched LN-DPO, GRPO, and DAPO on official MATH Levels 4-5, followed
by SearchQA-8K.

Active protocol:

- Student: pinned post-trained `Qwen/Qwen3-4B`.
- Privileged roles: pinned post-trained `Qwen/Qwen3-8B`.
- Non-thinking generation for every role.
- Strict tagged evaluator and teacher outputs; no JSON repair or silent fallback.
- Frozen teacher-free baseline before collection or training.
- Three paired answer-free feedback policies, with one selected on validation.
- MATH completion before SearchQA begins.

The only active plan is
[`docs/plans/2026-07-12-canonical-math-searchqa-execution.md`](docs/plans/2026-07-12-canonical-math-searchqa-execution.md).
Historical job evidence remains in `docs/status/turing-usage-log.md`; it is not
an active instruction source.

Implementation and verification commands are in `implementation/README.md`.
