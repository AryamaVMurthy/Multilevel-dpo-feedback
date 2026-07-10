# Turing Smoke Attempt Status

Date: 2026-07-09

## Scope

Attempted the model-backed pipeline smoke task from the superseded 2026-07-09 plan
after completing local implementation and verification. That plan was removed after
its findings were incorporated into `docs/design/native_iterative_guidance_dpo.md`;
paper execution now follows
`docs/plans/2026-07-10-paper-scale-experiment-implementation.md`.

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

ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
  -l 'aryama.murthy@research.iiit.ac.in' turing.iiit.ac.in 'hostname && whoami'

ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
  -l 'aryama.murthy@students.iiit.ac.in' turing.iiit.ac.in 'hostname && whoami'

ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
  -l 'aryamavmurthy@research.iiit.ac.in' turing.iiit.ac.in 'hostname && whoami'

ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
  -l 'aryamavmurthy@students.iiit.ac.in' turing.iiit.ac.in 'hostname && whoami'
```

Observed result for all usernames:

```text
Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password,hostbased).
```

Additional auth diagnostics:

```bash
ssh-add -l
ssh-keygen -y -f ~/.ssh/id_ed25519
ssh -vvv -o BatchMode=yes -o ConnectTimeout=8 \
  -o StrictHostKeyChecking=accept-new \
  -i ~/.ssh/id_ed25519 aryama.murthy@turing.iiit.ac.in 'hostname'
```

Observed:

```text
The agent has no identities.
ssh-keygen -y -f ~/.ssh/id_ed25519 exited 0 and printed the local public key.
OpenSSH offered /home/aryamavmurthy/.ssh/id_ed25519 to turing.iiit.ac.in.
The server rejected the offered key and returned Permission denied.
No Kerberos credentials were available.
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
