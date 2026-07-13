# SearchQA training runtime

This package contains the plain-answer SearchQA collection, strict cache reuse, preference construction, quality preflight, evaluation, reporting, and full-finetuning runtime.

Students produce only short plain answers. Teacher-native thinking and optional student scratchpads remain private metadata and never enter scored answers or DPO chosen/rejected completions. Teacher feedback is exactly one validated answer-free JSON hint.

Every run requires explicit model and dataset revisions, 4096 total length, thinking and decoding settings, seed, Slurm resources, and output paths. Missing settings, malformed feedback, answer leakage, cache mismatch, CUDA failure, or incomplete artifacts fail explicitly.

Use `probe-model` for real model-fit checks, `preflight-quality` for 32-example train-dev response review, and `select-thinking-mode` before launching the full raw baseline.
