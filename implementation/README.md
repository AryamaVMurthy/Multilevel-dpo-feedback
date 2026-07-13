# SearchQA training runtime

This package contains the SearchQA-only XML collection, offline-reuse, preference, evaluation, and full-finetuning runtime.

Every run must provide explicit model revisions, dataset source/split, 4096 maximum length, Slurm resources, output paths, and a DeepSpeed configuration. Missing or incompatible settings fail before model loading.
