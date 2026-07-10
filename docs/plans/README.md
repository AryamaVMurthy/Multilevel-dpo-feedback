# Active Experiment Plans

The paper-scale experiment has one active design and one active execution plan:

1. `2026-07-10-paper-scale-experiment-design.md` freezes the research question,
   datasets, methods, evaluation protocol, and completion gates.
2. `2026-07-10-paper-scale-experiment-implementation.md` defines the complete
   26-task implementation and Turing execution sequence.

Supporting canonical specifications:

- `../design/native_iterative_guidance_dpo.md`: living method specification and
  decision log.
- `../design/training_hyperparameter_protocol.md`: optimizer, Qwen3.5 LoRA coverage,
  deterministic hyperparameter search, model selection, and freeze rules.
- `../status/2026-07-10-paper-execution-log.md`: living run-by-run findings, failures,
  fixes, metrics, and artifact references.
The removed 2026-07-09 planning and status documents described superseded XML
formatting, fixture-only pipelines, pretest settings, and an obsolete SSH blocker.
Their relevant findings remain in the living method specification and Git history.
Historical smoke constants must not be used for paper training.
