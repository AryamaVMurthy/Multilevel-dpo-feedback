# Turing Usage Log

Append-only record for authorized Turing actions in this repository.

Do not record passwords, tokens, private keys, or any other credential material.

Historical note: entries below may refer to the former 10 requested GPU-hour daily reservation policy.

No actions have been logged under this control policy yet.

## 2026-07-11T13:09:30+05:30 - cluster state audit

- Approval reference: user requested end-to-end non-thinking experiment execution; date-scoped god switch is active for 2026-07-11.
- Bounded command set: one non-interactive SSH command running `hostname`, `date`, `git status`/`git rev-parse` for the remote project, `squeue`, `sacct` for jobs since 2026-07-10T18:30:00 UTC, `df`, and bounded artifact/cache directory listings.
- Purpose: establish remote revision, active and historical Slurm state, July 11 requested GPU-hour reservations, storage headroom, and existing immutable artifacts before any mutation or submission.
- Target paths: `/home/aryama.murthy/tfdpo-runs`, remote project checkout, and node-visible scratch metadata only.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; no Turing state change.

### Result - 2026-07-11T13:16:00+05:30

- Exit status: 0.
- Provenance: jobs 13003-13008 ran from `/home/yashas.kotre/src/Multilevel-dpo-feedback-qwen35-pretest`; job 13021 ran from that checkout's `implementation` directory. The collaborator's MATH source and logs are not artifacts owned by the current Turing account.
- Local GitHub verification: canonical branch advanced to commit `f93a9632d9a4ff8dd235c75356b6d2a745b4c4c0`, which records completed non-thinking job 13038.
- Gate result from the pushed immutable report: 7/16 protocol-valid correct, 8/16 EOS terminated, 8/16 length-truncated, mean 5,299.875 generated tokens, mean 115.278 seconds latency. The 50% truncation rate fails the frozen 5% maximum.
- Progression decision: preference collection and all DPO/GRPO training remain blocked. Required next experiment is a prespecified disjoint train-only non-thinking termination study; no validation/test result may select that protocol.

### Result - 2026-07-11T13:14:30+05:30

- Exit status: 0.
- Findings: documented persistent storage contains GSM8K artifacts only; the broken checkout contains no MATH dataset; no matching job logs were present under its `implementation/logs` directory.
- Failure: official MATH source and job 13021 outputs remain unlocated, so materialization cannot safely proceed from guessed paths.

## 2026-07-11T13:15:00+05:30 - Slurm path provenance audit

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one non-interactive SSH command running read-only `sacct` and `scontrol show job` for jobs 13003-13008 and 13021, followed by bounded listing of the recorded work/output paths.
- Purpose: recover authoritative Slurm working-directory and stdout/stderr provenance for the MATH source and baseline diagnostic.
- Target paths: paths returned by Slurm accounting for jobs 13003-13008 and 13021.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; no Turing state change.

### Result - 2026-07-11T13:13:00+05:30

- Exit status: 0.
- Checkout finding: `.git` is a 100-byte pointer to a nonexistent workstation worktree metadata path; source files and prior run directories remain present.
- Slurm finding: jobs 13003-13008 were CPU-only; 13003 downloaded MATH successfully and 13004-13008 failed fast. Job 13021 completed with 1 GPU and a 03:00:00 requested walltime, adding 3 requested GPU-hours beyond the earlier 10-hour reservation set.
- Budget status: at least 13 requested GPU-hours are visible on July 11; no new GPU submission is permitted.
- Artifact finding: no model cache manifest was found in the bounded node01 search and the broad home-name search did not locate MATH artifacts; exact repository run paths require a narrower follow-up.

## 2026-07-11T13:13:30+05:30 - exact source and prior MATH run audit

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one non-interactive SSH command using read-only `find`, `ls`, `du`, and selected non-secret Slurm stdout/stderr tails for jobs 13003-13008 and 13021.
- Purpose: identify the downloaded official MATH source, failed materialization destination, prior baseline artifacts, model cache path/node, and concrete failure context.
- Target paths: remote checkout `runs/` and `implementation/logs/`, `/home/aryama.murthy/tfdpo-runs`, and node01 scratch.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; no Turing state change.

### Result - 2026-07-11T13:12:00+05:30

- Exit status: 128 before accounting/artifact inspection.
- Failure: `/home/aryama.murthy/multilevel-feedback-dpo/.git` resolves to a workstation-only worktree metadata path, so the Turing checkout is not a valid standalone Git repository.
- Safety response: no mutation was attempted; remote synchronization is blocked pending an explicit checkout-layout audit and repair.

## 2026-07-11T13:12:30+05:30 - broken checkout and artifact layout audit

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one non-interactive SSH command using read-only `ls`, `sed` on the non-secret `.git` pointer, targeted `find`, `du`, and `sacct` for jobs 13003-13021.
- Purpose: establish the broken checkout structure and exact MATH/model artifact paths before a fail-fast standalone-clone repair.
- Target paths: `/home/aryama.murthy/multilevel-feedback-dpo`, `/home/aryama.murthy/tfdpo-runs`, and `/scratch/node01/aryama.murthy/text-feedback-dpo`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; no Turing state change.

## 2026-07-11 - policy update

- The repository-enforced daily requested GPU-hour limit was removed by explicit user instruction.
- Historical entries retain the former 10 requested GPU-hour policy for audit accuracy.
- Future Turing jobs must continue to record requested and accounted GPU-hours, but usage is no longer a repository submission gate.

### Result - 2026-07-11T13:09:46+05:30

- Exit status: 0; host `turing.iiit.ac.in`; no active jobs.
- Storage: home 36/50 GiB used (15 GiB available); node01 scratch 5.5/14 TiB used (8.4 TiB available).
- Reserved GPU-hours found since 2026-07-11T00:00:00+05:30: job 12970 = 3, job 12972 = 3, job 12973 = 1, and job 12976 = 3; cumulative reservation = 10 GPU-hours.
- Budget consequence: the 10 requested GPU-hour daily limit is exhausted; no additional GPU job may be submitted on 2026-07-11 without an explicit limit change.
- Failure: the bounded command omitted the intended remote `git` inspection and artifact listing was dominated by environment cache entries; a second zero-GPU bounded audit is required.

## 2026-07-11T13:11:00+05:30 - remote revision and MATH artifact audit

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one non-interactive SSH command running read-only `git status`, `git remote`, `git rev-parse`, targeted `sacct` for jobs 13003-13021, and bounded `find`/`du` over MATH run, source, model-cache-manifest, and Slurm-log paths.
- Purpose: determine whether commit `15ec5fe` and official MATH artifacts are present and identify exact inputs for zero-GPU materialization.
- Target paths: `/home/aryama.murthy/multilevel-feedback-dpo`, `/home/aryama.murthy/tfdpo-runs`, and `/scratch/node01/aryama.murthy/text-feedback-dpo`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; no Turing state change.

## 2026-07-11T21:09:03+05:30 - Qwen3 pre-submission cluster audit

- Approval reference: user explicitly requested complete Qwen3 MATH-then-SearchQA execution and training on Turing; the repository records the active 2026-07-11 Asia/Kolkata god switch.
- Bounded command set: one non-interactive BatchMode SSH command running `hostname`, `date`, `id`, `df`, `squeue`, `sacct`, `sacctmgr`, `git ls-remote`, and bounded `ls` checks for the proposed standalone clone and node01 scratch roots.
- Purpose: verify SSH, Slurm account, queue, recent accounting, home capacity, pushed source commit, and absence or state of the standalone Qwen3 paths before any remote mutation or submission.
- Target paths: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3` and `/scratch/node01/aryama.murthy/tfdpo-qwen3`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; read-only remote action.

### Result - 2026-07-11T21:09:32+05:30

- Exit status: 0; host `turing.iiit.ac.in`; no active jobs were listed.
- Slurm association: account `priyesh.shukla`, QOS `high`.
- Home storage: 36/50 GiB used, 15 GiB available.
- GitHub branch: `agent/qwen3-math-searchqa` resolves to `7ffd5bd547b6ecba1d0adc8d959341e1fb93cea3`.
- Both proposed standalone Qwen3 paths were absent. No remote state changed.

## 2026-07-11T21:09:43+05:30 - create standalone Qwen3 source clone

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command that requires the destination to be absent, clones only `agent/qwen3-math-searchqa` from GitHub, verifies exact commit `7ffd5bd547b6ecba1d0adc8d959341e1fb93cea3`, checks clean status, and creates the compact persistent artifact/log root.
- Purpose: replace the broken workstation worktree pointer with a clean standalone Turing repository before any Slurm work.
- Target paths: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3` and `/home/aryama.murthy/tfdpo-qwen3-artifacts`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; creates source and artifact directories in Turing home.

### Result - 2026-07-11T21:09:49+05:30

- Exit status: 128 during `git clone`; no Slurm resources allocated.
- Failure: Git `index-pack` could not create a thread and exited with `Resource temporarily unavailable` on the login node.
- Safety response: exact-commit verification and artifact-directory creation did not run. No retry or alternate clone method was attempted before diagnosis.

## 2026-07-11T21:10:20+05:30 - diagnose Turing clone thread failure

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one read-only BatchMode SSH command inspecting user process/thread counts, `ulimit`, memory, filesystem capacity, and the bounded partial clone layout/status.
- Purpose: distinguish user process exhaustion, memory pressure, storage pressure, and partial-repository state before selecting a corrective action.
- Target path: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; read-only remote action.

### Result - 2026-07-11T21:10:40+05:30

- Exit status: 0.
- Evidence: user limit 200 processes, only 7 live user threads, 275 GiB available RAM, 15 GiB available home storage, and no partial destination remained.
- Finding: persistent user-process exhaustion, memory pressure, disk pressure, and dirty partial-clone state are contradicted by current evidence. The failure occurred specifically while Git selected `index-pack` threading on the login host.

## 2026-07-11T21:10:52+05:30 - inspect Git clone concurrency inputs

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one read-only BatchMode SSH command printing visible CPU count and system/global Git `pack.threads` configuration.
- Purpose: test the hypothesis that Git's automatic pack threading can exceed the 200-process login-node limit despite a low steady-state user thread count.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; read-only remote action.

### Result - 2026-07-11T21:11:12+05:30

- Exit status: 0.
- Evidence: the login node exposes 144 CPUs and neither global nor system Git config caps `pack.threads`.
- Hypothesis: automatic high-core-count `index-pack` concurrency exhausted the effective login-node thread allowance during clone.

## 2026-07-11T21:11:27+05:30 - retry standalone clone with one pack thread

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: repeat the exact standalone single-branch clone with only `git -c pack.threads=1`, then verify exact commit, clean status, and create compact artifact/log directories.
- Purpose: test and correct the diagnosed login-node clone-concurrency failure without altering repository content or history.
- Target paths: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3` and `/home/aryama.murthy/tfdpo-qwen3-artifacts`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; creates source and artifact directories in Turing home.

### Result - 2026-07-11T21:11:49+05:30

- Exit status: 0.
- Root cause confirmed: capping Git pack work to one thread allowed the same clone to complete.
- Standalone clone is clean on `agent/qwen3-math-searchqa` at `7ffd5bd547b6ecba1d0adc8d959341e1fb93cea3`.
- Storage: source clone 970 KiB, artifact root 1.5 KiB, and home remains at 36/50 GiB used.

## 2026-07-11T21:11:59+05:30 - synchronize logged Qwen3 source revision

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command fetching the Qwen3 branch with `pack.threads=1`, fast-forwarding only, and verifying exact expected commit and clean status.
- Purpose: bring the standalone clone to the new source commit that includes this append-only cluster audit before Slurm setup.
- Target path: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; fast-forward source update only.

### Result - 2026-07-11T21:12:22+05:30

- Exit status: 0.
- Standalone clone fast-forwarded cleanly to `062f6d93c4d3af26e1a611e9ca6094a954a0261b`.

## 2026-07-11T21:12:53+05:30 - submit locked Qwen3 environment setup

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command rechecking clean source, exact commit, queue, and home capacity; creating only the Slurm log directory; then one `sbatch --parsable` submission constrained to node01 using `turing_setup_environment.sh`.
- Purpose: install the exact `uv.lock` environment and cache on node-local scratch before model or dataset work.
- Source: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3/implementation` at `062f6d93c4d3af26e1a611e9ca6094a954a0261b`.
- Scratch targets: `/scratch/aryama.murthy/tfdpo-qwen3/runtime/uv_cache` and `/scratch/aryama.murthy/tfdpo-qwen3/runtime/project_venv` on node01.
- Requested resources: account `priyesh.shukla`; partition `u22`; node01; 2 tasks; 16 GiB RAM; 01:00:00 walltime; 0 GPUs; 0 requested GPU-hours.

### Submission result - 2026-07-11T21:13:16+05:30

- Exit status: 0; Slurm job ID `13111`.
- Pre-submit queue was empty and home storage remained 36/50 GiB used.

## 2026-07-11T21:13:29+05:30 - monitor environment setup job 13111

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue` and `sacct` for job 13111 plus bounded stdout/stderr and environment-verification inspection after terminal state.
- Purpose: verify exit status, node, elapsed resources, locked Python environment, and explicit failure context before any model or dataset job.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:15:42+05:30

- Slurm state: `COMPLETED`, exit code `0:0`, node01, elapsed 00:02:04, MaxRSS 6,830.50 MiB.
- Installed locked runtime: Python 3.12.13, Torch 2.13.0+cu126, Transformers 5.13.0, TRL 1.8.0, PEFT 0.19.1, Datasets 5.0.0, and all other `uv.lock` packages.
- GPU allocation and accounted GPU-hours: 0.
- The verification file is node-local and not directly visible on the login host; successful job exit proves the script completed the verification command. The next node01 job must still check the runtime before use.

## 2026-07-11T21:18:20+05:30 - synchronize dataset-staging fix

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command fetching with one pack thread, fast-forwarding only, and verifying commit `fa0b00f9c672f91f82ea4a9b7fb61f66b6bcf400` and clean status.
- Purpose: deploy the tested node-local-runtime reuse fix before MATH source download.
- Target path: `/home/aryama.murthy/multilevel-feedback-dpo-qwen3`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours; fast-forward source update only.

### Result - 2026-07-11T21:18:40+05:30

- Exit status: 0; standalone clone fast-forwarded cleanly to `fa0b00f9c672f91f82ea4a9b7fb61f66b6bcf400`.
