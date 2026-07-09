# Turing Smoke Attempt Status

Date: 2026-07-09

## Scope

Attempted Task 9 from `docs/plans/2026-07-09-basic-model-backed-pipeline.md` after completing the local implementation and local verification.

## Local Verification Completed

Commands:

```bash
cd implementation
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
PYTHONPATH=src python3 -m text_feedback_dpo.cli generate-pipeline \
  --config configs/basic_smoke.yaml \
  --output-dir runs/local-fake-smoke \
  --fake-smoke
```

Observed local fake-smoke metrics:

```json
{
  "accepted_pairs": 1,
  "examples_total": 5,
  "rejected_examples": 4,
  "run_id": "qwen35-basic-smoke",
  "student_model": "Qwen/Qwen3.5-2B",
  "teacher_mode": "stronger_model",
  "teacher_model": "Qwen/Qwen3.5-9B",
  "verification_missing_rejections": 0
}
```

## Turing SSH Attempt

Commands attempted:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
  aryama.murthy@turing.iiit.ac.in 'hostname && whoami'

ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
  aryamavmurthy@turing.iiit.ac.in 'hostname && whoami'
```

Observed result for both usernames:

```text
Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password,hostbased).
```

## Status

Turing model-load and pair-generation Slurm jobs were not submitted because SSH authentication failed before reaching the login node.

This is not a silent fallback. The remaining required remediation is to authenticate to Turing, then run:

```bash
cd ~/Multilevel-dpo-feedback/implementation
sbatch -A priyesh.shukla --job-name=tfdpo-model-load \
  --export=ALL,TURING_ACCOUNT=priyesh.shukla \
  scripts/turing_model_load_smoke.sh

sbatch -A priyesh.shukla --job-name=tfdpo-basic-pairs \
  --export=ALL,TURING_ACCOUNT=priyesh.shukla,CONFIG=configs/basic_smoke.yaml \
  scripts/turing_basic_pair_generation.sh
```
