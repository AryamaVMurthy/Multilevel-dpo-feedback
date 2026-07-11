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

## 2026-07-11T21:20:40+05:30 - synchronize verified-runtime gate

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync with one pack thread, exact commit verification, and clean-status check.
- Purpose: deploy the tested verified-runtime requirement before model and dataset staging.
- Expected commit: `a126bde37394ec4384fc6deb73496f9b48963b28`.
- Requested resources: control-plane SSH only; 0 GPUs; 0 requested GPU-hours.

### Result - 2026-07-11T21:20:55+05:30

- Exit status: 0; standalone clone fast-forwarded cleanly to `a126bde37394ec4384fc6deb73496f9b48963b28`.

## 2026-07-11T21:21:06+05:30 - submit Qwen3 model and full MATH source staging

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command rechecking source commit/status, queue, and home capacity, then submitting exactly two node01 CPU-only jobs: `turing_stage_model_cache.sh` and `turing_download_math_source.sh`.
- Source: standalone clone at `a126bde37394ec4384fc6deb73496f9b48963b28`; config `implementation/configs/paper/math.yaml`.
- Model target: `/scratch/aryama.murthy/tfdpo-qwen3/models`; runtime root `/scratch/aryama.murthy/tfdpo-qwen3/runtime`.
- Dataset target: `/scratch/aryama.murthy/tfdpo-qwen3/datasets/math-source`; exact official counts 7,500 train and 5,000 test across seven subjects.
- Model-staging resources: account `priyesh.shukla`, u22, node01, 2 tasks, 32 GiB, 02:00:00, 0 GPUs.
- MATH-download resources: account `priyesh.shukla`, u22, node01, 2 tasks, 32 GiB, 01:00:00, 0 GPUs.
- Total requested GPU-hours: 0.

### Submission result - 2026-07-11T21:21:27+05:30

- Exit status: 0.
- Model staging job: `13113`.
- Full MATH source job: `13114`.
- Pre-submit queue was empty and home storage remained 36/50 GiB used.

## 2026-07-11T21:21:39+05:30 - monitor staging jobs 13113 and 13114

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, and bounded stdout/stderr for both jobs; after terminal state, a short node01 Slurm verification job may inspect node-local model/data manifests only after a separate logged submission.
- Purpose: capture exact state, exit code, node, elapsed resources, download errors, and artifact readiness without assuming login-node visibility into node-local scratch.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:29:18+05:30

- Job 13113: `COMPLETED`, exit `0:0`, node01, elapsed 00:07:36, MaxRSS 24,376,104 KiB, 0 GPUs.
- Model manifest schema `tfdpo-model-cache-v1`, config SHA-256 `58f9ec23ad48f5abb6f2dee8df9d9f8bc8cdc5da183791ac05b8de54551a9ee3`, source commit `a126bde37394ec4384fc6deb73496f9b48963b28`.
- Frozen snapshots: Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c` and Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Job 13114: `COMPLETED`, exit `0:0`, node01, elapsed 00:01:23, MaxRSS 179,692 KiB, 0 GPUs.
- MATH job generated all seven official subject train/test splits and passed the aggregate 7,500 train / 5,000 test count check.
- Total accounted GPU-hours: 0.

## 2026-07-11T21:31:16+05:30 - sync and submit full MATH materialization

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command fast-forwarding only to `0544a4674ed16ce9704e92d1fee28cb494202eb6`, verifying clean status and empty queue, then submitting one node01 CPU-only `turing_materialize_dataset.sh` job.
- Purpose: normalize all official rows, preserve provenance and reviewed errata, quarantine source duplicates and train-test overlaps, derive deterministic Level 4-5 train/validation and 2:1 tune/confirmation roles, hash artifacts, and retain the untouched 5,000-row test split.
- Source path: `/scratch/aryama.murthy/tfdpo-qwen3/datasets/math-source`.
- Output path: `/scratch/aryama.murthy/tfdpo-qwen3/datasets/math-materialized-v1`.
- Requested resources: account `priyesh.shukla`, u22, node01, 2 tasks, 32 GiB, 01:00:00, 0 GPUs, 0 requested GPU-hours.

### Submission result - 2026-07-11T21:31:56+05:30

- Source fast-forward and exact commit verification succeeded.
- Slurm job ID: `13116`; pre-submit queue was empty.

## 2026-07-11T21:32:16+05:30 - monitor MATH materialization job 13116

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, and bounded job stdout/stderr until terminal state.
- Purpose: capture materialization failure context and resource accounting before a separate node-local manifest audit.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:33:01+05:30

- Job 13116: `COMPLETED`, exit `0:0`, node01, elapsed `00:00:02`, MaxRSS `8,712 KiB`, 0 GPUs.
- The materializer retained the official 5,000-row test role, created 402 Level 4-5 validation rows partitioned into 270 tune and 132 confirmation rows, and quarantined four reviewed official-train/official-test overlaps with row-level provenance.
- The job emitted a full manifest to stdout; the next tested revision replaces that noisy output with a bounded summary. No dataset is accepted from this result until the separate node-local integrity and protocol audit passes.
- Total accounted GPU-hours: 0.

## 2026-07-11T21:38:15+05:30 - deploy and audit frozen MATH dataset

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with a one-thread Git fetch and fast-forward-only update, exact source/status/queue/storage checks, then submission of exactly one node01 CPU-only `turing_audit_dataset.sh` job.
- Purpose: independently recompute manifest integrity, compressed role counts and hashes, role disjointness, Level 4-5 policy, validation partitioning, official 5,000-row test preservation, and the reviewed four-row overlap quarantine.
- Dataset: `/scratch/aryama.murthy/tfdpo-qwen3/datasets/math-materialized-v1`.
- Audit artifact: `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/math-dataset-audit.json`.
- Requested resources: account `priyesh.shukla`, u22, node01, 2 tasks, 16 GiB, `00:30:00`, 0 GPUs, 0 requested GPU-hours.

### Submission result - 2026-07-11T21:39:09+05:30

- Standalone clone fast-forwarded cleanly to `97eba67e37522232a8c22f1c26d8d050c4979f95`; the pre-submit queue was empty and home storage remained 36/50 GiB used.
- Slurm job ID: `13117`.

## 2026-07-11T21:39:09+05:30 - monitor MATH dataset audit job 13117

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, and bounded stdout/stderr, followed by reading the compact audit JSON only if the job completes successfully.
- Purpose: capture exact audit evidence and explicit failure context before permitting any GPU preflight.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:39:42+05:30

- Job 13117: `FAILED`, exit `1:0`, node01, elapsed `00:00:01`, MaxRSS `500 KiB`, 0 GPUs.
- Explicit failure: `MATH validation fraction mismatch for math:algebra:level5: expected 43, found 44`.
- Root cause is in the new auditor: binary floating-point evaluation of `1 - 0.9` yielded a value just below 0.1, while the materializer uses the exact protocol literal `0.10`. The dataset was not modified and no audit artifact was written.
- Progression remains stopped. A regression-tested auditor correction and a fresh immutable audit job are required before GPU work.
- Total accounted GPU-hours: 0.

## 2026-07-11T21:40:17+05:30 - deploy auditor correction and retry MATH audit

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with one-thread Git fetch and fast-forward-only update, exact clean/source/queue checks, confirmation that the failed attempt wrote no audit artifact, then submission of exactly one node01 CPU-only dataset audit job.
- Correction evidence: the 435-row half-rounding regression now passes; the complete local suite passes 191 tests, Ruff, compileall, shell parsing, and diff checks.
- Dataset and output remain `/scratch/aryama.murthy/tfdpo-qwen3/datasets/math-materialized-v1` and `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/math-dataset-audit.json`.
- Requested resources: account `priyesh.shukla`, u22, node01, 2 tasks, 16 GiB, `00:30:00`, 0 GPUs, 0 requested GPU-hours.

### Submission result - 2026-07-11T21:40:45+05:30

- Standalone clone fast-forwarded cleanly to `331b032b25af62d897a8991e818f97696f45e5dc`; the queue was empty and the failed attempt had left no audit JSON.
- Slurm job ID: `13118`.

## 2026-07-11T21:40:45+05:30 - monitor corrected MATH dataset audit job 13118

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, and the compact audit JSON only after successful completion.
- Purpose: obtain the independent acceptance evidence required before any GPU preflight.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:41:01+05:30

- Job 13118: `COMPLETED`, exit `0:0`, node01, elapsed `00:00:01`, MaxRSS `612 KiB`, 0 GPUs.
- Audit status: `passed`; roles are 3,589 train, 402 validation, and 5,000 official test; nested validation is 270 tune and 132 confirmation; overlap quarantine count is four.
- Manifest content SHA-256: `6986f3e9a68540117cc51ca3045feb7dc797a4cf965919b925b4bb6270d76858`; manifest file SHA-256: `2c72bb66f5a742e4caf3c04b396265314ff38cc3c5ce68323063e39cdddbb3e7`; source artifact SHA-256: `20a10a84deaf42018c4634c4fcc3eaf0fcd013fcba4db7a19b93ac0a807f33bf`.
- Immutable compact evidence: `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/math-dataset-audit.json`.
- Total accounted GPU-hours: 0.

## 2026-07-11T21:43:41+05:30 - deploy and run exact Qwen3 one-GPU preflight

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with one-thread Git fetch and fast-forward-only update, source/status/queue/storage checks, then submission of exactly one node01 one-GPU `turing_model_load_smoke.sh` job; no training or evaluation dataset access.
- Purpose: verify the staged exact Qwen3-4B/8B snapshots offline, CUDA and BF16 without fallback, non-thinking chat generation for both models, and the complete 252-projection rank-16 student LoRA inventory.
- Model cache: `/scratch/aryama.murthy/tfdpo-qwen3/models`; immutable report: `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/math-model-preflight.json`.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 CPUs, 64 GiB, `00:30:00`; 0.5 requested GPU-hours.

### Submission result - 2026-07-11T21:44:19+05:30

- Standalone clone fast-forwarded cleanly to `a96983c9619be7b7805eb8323339525bc505766a`; queue was empty, home usage remained 36/50 GiB, and no prior preflight report existed.
- Slurm job ID: `13119`.

## 2026-07-11T21:44:19+05:30 - monitor exact Qwen3 model preflight job 13119

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, and the compact model report only after successful completion.
- Purpose: capture model identity, non-thinking behavior, LoRA coverage, peak memory, latency, GPU, and failure evidence before decoding experiments.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:44:48+05:30

- Job 13119: `COMPLETED`, exit `0:0`, node01, elapsed `00:00:18`, MaxRSS `1,001,208 KiB`, one RTX 6000 Ada GPU; accounted GPU time `0.005` GPU-hours.
- Both exact frozen snapshots loaded offline in BF16 and generated non-thinking completions. Qwen3-4B peak allocated memory was `8,068,640,768` bytes; Qwen3-8B was `16,397,877,760` bytes.
- Qwen3-4B exposed exactly 36 layers and 252 audited text projections; coverage SHA-256 `4a36eb07f8abc7ddfc0d44cd39c8fe3571f6daddca19f7b63bf55f2af8931644`; estimated rank-16 LoRA parameters `33,030,144`.
- Immutable evidence: `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/math-model-preflight.json`; config SHA-256 `58f9ec23ad48f5abb6f2dee8df9d9f8bc8cdc5da183791ac05b8de54551a9ee3`.

## 2026-07-11T21:48:23+05:30 - deploy and run MATH train-only decoding screening

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with one-thread Git fetch and fast-forward-only update, clean source/queue/storage and artifact-absence checks, then submission of exactly one node01 one-GPU `turing_decoding_sweep.sh` screening job.
- Purpose: evaluate all five frozen presence-penalty candidates on the same 12 deterministic stratified Level 4-5 training examples with a 4,096-token ceiling; rank by protocol-valid accuracy, truncation, correct answers per million tokens, median length, and latency.
- Inputs are bound to the passed dataset audit and offline model-cache manifest. Validation and test roles are not read.
- Output: `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/decoding/screening-v1` plus sibling GPU telemetry; any existing output is a hard error.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 CPUs, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T21:49:02+05:30

- Standalone clone fast-forwarded cleanly to `71985c480a77ca073f4ebe3f3139ab9593228d42`; queue was empty, home usage remained 36/50 GiB, and the screening output did not exist.
- Slurm job ID: `13120`.

## 2026-07-11T21:49:02+05:30 - monitor MATH decoding screening job 13120

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, and compact manifest/selection files only after terminal state.
- Purpose: capture per-profile progress, failures, resource use, and the exact three-profile promotion without reading validation or test artifacts.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

## 2026-07-11T21:55:47+05:30 - cancel protocol-invalid decoding screening job 13120

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH `scancel 13120`, followed only by read-only `sacct` and bounded log inspection.
- Stop reason: code review found that the sweep constructed an ad hoc MATH prompt instead of the frozen `build_native_student_prompt` used by baseline, collection, and inference. Selecting presence penalty under a different prompt would violate the freeze protocol.
- The partial output contains nine diagnostic records and must never be promoted or resumed into a paper run. It remains at `screening-v1` with no completed selection manifest.
- Requested resources: cancellation only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T21:56:04+05:30

- Job 13120: `CANCELLED`, node01, elapsed `00:07:02`, one GPU; accounted partial diagnostic time `0.1172` GPU-hours.
- Twelve presence-0 records completed before cancellation; no selection manifest was created. The entire `screening-v1` directory is permanently diagnostic and excluded from paper promotion.
- Progression remains stopped until exact prompt identity is regression-tested and a new output directory is used.

## 2026-07-11T21:56:51+05:30 - deploy prompt-identity correction and rerun decoding screening

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with one-thread Git fetch and fast-forward-only update, exact clean/queue/artifact checks, then submission of exactly one node01 one-GPU decoding screening job into a new immutable output directory.
- Correction evidence: sweep prompts are now constructed only by `build_native_student_prompt`; exact byte identity with baseline is regression-tested; every prompt hash and frozen prompt-protocol ID is written into the sweep manifest and records. Full local gate passes 196 tests, Ruff, compileall, shell parsing, and diff checks.
- Output: `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/decoding/screening-v2`; `screening-v1` remains excluded diagnostics.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 CPUs, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T21:57:24+05:30

- Standalone clone fast-forwarded cleanly to `a90cb006c7236f8d4ff49ca27a61caea322e3508`; queue was empty and `screening-v2` did not exist.
- Slurm job ID: `13121`.

## 2026-07-11T21:57:24+05:30 - monitor prompt-valid decoding screening job 13121

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, and compact manifest/selection artifacts after terminal state.
- Purpose: verify all 60 exact prompt/profile records and obtain the three-profile train-only promotion.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:10:03+05:30

- Job 13121: `COMPLETED`, exit `0:0`, node01, elapsed `00:12:35`, MaxRSS `1,280,700 KiB`, one GPU; accounted time `0.2097` GPU-hours.
- All 60 required records completed with zero truncations. Promotion order: `presence-1`, `presence-1.5`, `presence-0`.
- Screening accuracies were 8/12 for presence 1, 8/12 for presence 1.5, 7/12 for presence 0, 7/12 for presence 2, and 5/12 for presence 0.5. Presence 1 outranked 1.5 by correct answers per million tokens, median length, and latency after the accuracy/truncation tie.
- Selection SHA-256 bindings: records `f2b09a11bdd0434b95f296f27bc78d14c19c849a164b18a32114b939331bc8e1`; sweep manifest `481451273798110c9f4ed701a901b6e248485d987ad20ae19f73d09d24f4da73`.

## 2026-07-11T22:10:13+05:30 - deploy and run disjoint decoding confirmation

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with one-thread Git fetch and fast-forward-only update, clean source/queue/output checks, then submission of exactly one node01 one-GPU confirmation job.
- Purpose: evaluate exactly the three promoted profiles on 32 deterministic, screening-disjoint Level 4-5 training examples at the full 8,192-token ceiling; select exactly one profile with the frozen ranking order.
- Confirmation code must verify the screening selection and sweep hashes plus exact config, dataset, audit, model-cache, and model identity bindings before loading the GPU model.
- Output: `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/decoding/confirmation-v1`.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 CPUs, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T22:10:57+05:30

- Standalone clone fast-forwarded cleanly to `93de73c6d7e20205c9bbba4adae40c8146a9fe48`; queue was empty and the confirmation output did not exist.
- Slurm job ID: `13123`.

## 2026-07-11T22:10:57+05:30 - monitor disjoint decoding confirmation job 13123

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, and compact manifest/selection artifacts after terminal state.
- Purpose: capture all 96 confirmation records, disjointness and context-binding failures, resource use, and the single selected profile.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:28:51+05:30

- Job 13123: `COMPLETED`, exit `0:0`, node01, elapsed `00:17:54`, MaxRSS `1,283,424 KiB`, one GPU; accounted time `0.2983` GPU-hours.
- Selected `presence-1.5`: 17/32 protocol-valid correct, zero truncations, 1,348.89 correct answers per million generated tokens, median 337 tokens, mean latency 8.91 seconds.
- `presence-1`: 16/32 correct, zero truncations. `presence-0`: 12/32 correct with one 8,192-token truncation.
- Confirmation records SHA-256 `33f9cea16882a7ac09df9b50a45da6c92f40567f1c54214ebacef80784a63c96`; sweep manifest SHA-256 `769809d1c14f48b42293cf4b96d9ef17cb3e474ed394523eb75c0a8efbd3e940`.

## 2026-07-11T22:30:31+05:30 - freeze selected decoding and refresh model-cache binding

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH command with one-thread Git fetch and fast-forward-only update; exact source/status checks; preserve the selection-time model-cache manifest as a compact immutable artifact; run the tested CPU-only `freeze-decoding` command; then submit exactly one node01 CPU-only model-cache staging job for the newly frozen config hash.
- Frozen source commit: `5f31c5010a47db0b35603a99356c1c9c6ffa227e`; MATH config SHA-256 `d2fa3e60c02aae8d87018f4550e99bc593d2bb4610b5461d7e13db78523b01ae`; selected presence penalty `1.5`.
- Freeze output: `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/math-decoding-freeze.json`; preserved selection cache manifest: `/home/aryama.murthy/tfdpo-qwen3-artifacts/manifests/model-cache-selection.json`.
- Requested staging resources: account `priyesh.shukla`, u22, node01, 2 tasks, 32 GiB, `02:00:00`, 0 GPUs, 0 requested GPU-hours.

### Failed control-plane attempt - 2026-07-11T22:31:36+05:30

- Exit status: 1 before any freeze write or Slurm submission.
- Explicit failure: the login host could not see node01-local `/scratch/aryama.murthy/tfdpo-qwen3/models/tfdpo-model-cache-manifest.json`.
- No fallback was used; neither compact output existed afterward and no cache-refresh job was submitted.
- Corrective action: run cache verification, preservation, and decoding freeze inside a tested node01 CPU job, then submit cache refresh with an `afterok` dependency.

## 2026-07-11T22:32:40+05:30 - submit node-local decoding freeze and dependent cache refresh

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync and clean/output/queue checks, then exactly two node01 CPU-only submissions: `turing_freeze_decoding.sh` and `turing_stage_model_cache.sh` with `afterok` dependency on the freeze job.
- Freeze wrapper verifies the live selection-time cache-manifest hash on node01 before preserving it and refuses any pre-existing output. Cache refresh cannot start if freeze fails.
- Local evidence: full suite passes 200 tests, Ruff, compileall, all shell parsing, and diff checks.
- Freeze resources: account `priyesh.shukla`, u22, node01, 2 tasks, 8 GiB, `00:30:00`, 0 GPUs. Cache-refresh resources: 2 tasks, 32 GiB, `02:00:00`, 0 GPUs. Total requested GPU-hours: 0.

### Submission result - 2026-07-11T22:33:18+05:30

- Standalone clone fast-forwarded cleanly to `d3c284720c8bda2db89b55b2fb67fd1be0c9941e`; queue was empty and both output files were absent.
- Freeze job: `13124`. Dependent cache-refresh job: `13125` with `afterok:13124`.

## 2026-07-11T22:33:18+05:30 - monitor decoding freeze and cache refresh jobs 13124-13125

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, and compact freeze/cache manifests only after successful terminal states.
- Purpose: verify selection-time cache preservation, final config binding, dependency enforcement, and exact refreshed model revisions before baseline work.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:33:35+05:30

- Jobs 13124 and 13125 both `COMPLETED`, exit `0:0`, node01, elapsed `00:00:02` each, 0 GPUs.
- Freeze binds selected profile `presence-1.5`, frozen config SHA-256 `d2fa3e60c02aae8d87018f4550e99bc593d2bb4610b5461d7e13db78523b01ae`, exact model revision, prompt protocol, dataset/audit hashes, and disjoint 12/32 IDs.
- Selection-time cache-manifest SHA-256 `5c5f270b3e2a33d1163d56761cdffe0dfca1642cde6d401f1de19a94fca42b06` was verified on node01 and preserved before refresh.
- Refreshed cache manifest retains exact Qwen3-4B/8B revisions and now binds the final config hash. Total accounted GPU-hours: 0.

## 2026-07-11T22:34:09+05:30 - materialize one-example MATH validation micro and freeze baseline identity

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH clean/queue/output check, then exactly two node01 CPU-only submissions: a one-example `math_subject_level` validation subset materialization and the immutable teacher-free baseline freeze.
- Source validation artifact: audited `/scratch/aryama.murthy/tfdpo-qwen3/datasets/math-materialized-v1/validation.jsonl.zst`; subset seed `20260713`, count 1.
- Outputs: `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/baseline/micro-one/validation.jsonl.zst` and `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/baseline/baseline-freeze.json`.
- Frozen baseline source commit: `d3c284720c8bda2db89b55b2fb67fd1be0c9941e`.
- Requested resources: each account `priyesh.shukla`, u22, node01, 2 tasks, at most 4 GiB, at most `00:15:00`, 0 GPUs, 0 requested GPU-hours.

### Submission result - 2026-07-11T22:35:05+05:30

- Standalone clone fast-forwarded cleanly to `5b410ef881d1b9f0ce857e155ce92320bb1a39cc`; queue and both target paths were empty.
- Micro materialization job: `13126`. Baseline-freeze job: `13127`.

## 2026-07-11T22:35:05+05:30 - monitor baseline micro materialization and freeze jobs 13126-13127

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, and compact manifests only after successful terminal states.
- Purpose: verify the one-example stratum selection, copied dataset identity, and final teacher-free baseline binding before the first GPU evaluation.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:35:26+05:30

- Jobs 13126 and 13127 both `COMPLETED`, exit `0:0`, node01, elapsed `00:00:01`, 0 GPUs.
- Micro example: `math-geometry-train-201`; source validation SHA-256 `b2b93cab808a73617fff62f1db023a2d526dcc7387158cba24c0b5e72fc26372`; output SHA-256 `7b447b2d019d2b1b0b2777643d5da2e03403ca212966dda9840b4f5fed1a45c0`.
- Baseline freeze binds source commit `d3c284720c8bda2db89b55b2fb67fd1be0c9941e`, final config hash, exact 4B checkpoint, exact 8B evaluator, prompt protocol, dataset manifest, seeds, and teacher-free status.

## 2026-07-11T22:35:44+05:30 - run one-example teacher-free MATH baseline micro

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH clean/queue/output check and exactly one node01 one-GPU base-checkpoint evaluation submission with array index 0 only.
- Input: immutable one-example validation subset and baseline freeze above. No adapter, teacher guidance, validation expansion, or test access.
- Output: `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/baseline/micro-one/evaluation/shard-0000` plus GPU telemetry.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T22:36:29+05:30

- Standalone clone fast-forwarded cleanly to `3137ce9214b131c76e693ac100490c5c27c48805`; queue and evaluation output were empty.
- Slurm array job: `13128`, index 0 only.

## 2026-07-11T22:36:29+05:30 - monitor one-example baseline micro job 13128

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, and the one prediction/metrics/completion marker only after terminal state.
- Purpose: capture exact raw response, extraction/evaluator judgment, termination, token counts, latency, GPU memory, and any failure before manual inspection.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:37:23+05:30

- Job 13128: `FAILED`, exit `1:0`, node01, elapsed `00:00:06`, MaxRSS `596 KiB`, one GPU; accounted time `0.0017` GPU-hours.
- Explicit failure occurred before model loading: Hugging Face offline lookup searched the default `$HF_HOME/hub` path, while the verified snapshots were staged directly under `MODEL_CACHE_DIR`.
- Offline mode prevented a network fallback as intended. No prediction, metrics, failure ledger, or evaluation-complete marker was written; only GPU telemetry exists in the failed `evaluation` directory.
- Progression remains stopped. The cache lookup path must be explicitly bound and the micro rerun into a fresh directory.

## 2026-07-11T22:40:38+05:30 - verify offline model lookup and retry one-example baseline micro

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync plus clean/queue/output checks; submit one CPU-only offline model-ID lookup verification on node01, then one array-index-0 GPU evaluation with an `afterok` dependency on that verifier.
- Corrective change: every model-dependent paper wrapper now binds `HF_HUB_CACHE` to the exact node-local staged cache. The verifier resolves both pinned model IDs and revisions with `local_files_only=True`, checks the cache-manifest/config binding, and writes a new immutable lookup artifact.
- The failed `evaluation` directory from job 13128 remains diagnostic and will not be resumed or overwritten. Retry output is the fresh `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/baseline/micro-one/evaluation-v2` directory.
- Local evidence: 200 tests pass with `PYTHONPATH=src`, Ruff passes, compileall passes, all shell scripts parse, and the diff check passes.
- Lookup resources: account `priyesh.shukla`, u22, node01, 2 tasks, 16 GiB, `00:30:00`, 0 GPUs. Retry resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T22:42:06+05:30

- Standalone clone fast-forwarded cleanly to `834562eee80ec6b3a6eb5d98780d1a1732cd73bd`; queue was empty and both new output paths were absent.
- CPU offline-lookup job: `13129`. Dependent one-example GPU evaluation job: `13130`, array index 0 only, with `afterok:13129`.

## 2026-07-11T22:42:06+05:30 - monitor offline lookup and baseline micro retry jobs 13129-13130

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded stdout/stderr, then compact lookup/prediction/metrics/completion artifacts only after successful terminal states.
- Purpose: prove ID-plus-revision resolution from node-local offline cache, enforce dependency behavior, and manually inspect the only generated response before expanding the preflight.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:42:54+05:30

- Job 13129 `COMPLETED` in 5 seconds and proved offline ID-plus-revision lookup for exact Qwen3-4B and Qwen3-8B with config SHA-256 `d2fa3e60c02aae8d87018f4550e99bc593d2bb4610b5461d7e13db78523b01ae`; lookup artifact SHA-256 is `3b651ceb7f4f1538f70fb79e7cee2dd6ac77194f6ed3202647414af10cd7cb21`.
- Dependent job 13130 index 0 `COMPLETED` in 18 seconds, exit `0:0`; accounted GPU time `0.005` GPU-hours and observed peak telemetry memory was 24,038 MiB.
- The only response ended exactly with `FINAL: \\boxed{6}`, contained no thinking markup, terminated through the balanced final-answer stopper, was not truncated, and matched the reference. The deterministic evaluator and pinned 8B evaluator both marked it correct with confidence 1.0 and no parse failures or regenerations.
- Completion marker and hashes are present, failures are empty, teacher-free is true, and the frozen Qwen3-4B revision is exact. Manual one-example inspection gate passes.

## 2026-07-11T22:42:54+05:30 - run stratified 16-example MATH validation preflight

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH clean/queue/output check; submit one CPU-only `math_subject_level` 16-example validation materialization on node01, then one array-index-0 GPU evaluation with an `afterok` dependency.
- Input remains the audited validation split; subset seed `20260713`, count 16. The official test split remains untouched.
- Outputs are new immutable paths under `/home/aryama.murthy/tfdpo-qwen3-artifacts/math/baseline/preflight-16`; no prior artifact will be resumed or overwritten.
- CPU resources: account `priyesh.shukla`, u22, node01, 2 tasks, 4 GiB, `00:15:00`, 0 GPUs. GPU resources: 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T22:43:41+05:30

- Standalone clone fast-forwarded cleanly to `a3a304afcb7129dbe32a4872a011fd1106f54510`; queue and preflight output path were empty.
- Stratified subset materialization job: `13131`. Dependent one-GPU evaluation job: `13132`, array index 0 only, with `afterok:13131`.

## 2026-07-11T22:43:41+05:30 - monitor MATH validation preflight jobs 13131-13132

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, subset manifest, predictions, failures, metrics, completion marker, and GPU telemetry after terminal states.
- Purpose: verify stratification, exact answer extraction, malformed-template count, truncation, evaluator behavior, and all 16 raw responses before the full validation baseline.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:50:09+05:30

- Job 13131 `COMPLETED` in 1 second and materialized 16 deterministic subject-level strata with subset SHA-256 `f0d1dcbf10517e14f1246abf290a1ad899ace8583f9dcdda547c1599e9b4bf62`.
- Job 13132 index 0 `COMPLETED` in 3 minutes 7 seconds, exit `0:0`; accounted GPU time `0.0519` GPU-hours and observed peak telemetry memory was 24,506 MiB.
- All 16 responses were non-empty and manually inspected; truncation was 0%, failure ledger was empty, generation metadata was complete, teacher-free was true, and no thinking markup or malformed chat-template output was present. Eleven responses used the balanced `FINAL:` stopper and five valid boxed responses ended by EOS.
- Raw evaluator accuracy was 8/16. Manual review identified one explicit evaluator disagreement: `math-number_theory-train-649` answered option D, whose displayed value is 13 and matches the gold answer, but the deterministic scorer compared the letter to the numeric value.
- Progression is stopped at the evaluator-audit gate until official MATH multiple-choice labels are mapped to their displayed values and the immutable predictions are rescored. This is an explicit root-cause correction, not an accuracy override.

## 2026-07-11T22:50:09+05:30 - rescore and audit MATH validation preflight

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: after test-first implementation and full local verification, one BatchMode SSH fast-forward-only sync plus clean/output checks and exactly one CPU-only node01 rescore-and-audit submission.
- Rescore rule: only an exact A-E final answer in an official MATH problem with explicit displayed choices is mapped to that choice value; malformed or duplicate choice structures fail explicitly. Original predictions remain immutable.
- Manual audit labels cover all 16 responses with per-example notes. New outputs: `preflight-16/rescore` and `preflight-16/audit`; existing evaluation artifacts are unchanged.
- Requested resources: account `priyesh.shukla`, u22, node01, 2 tasks, 4 GiB, `00:15:00`, 0 GPUs; 0 requested GPU-hours.

### Submission result - 2026-07-11T22:51:04+05:30

- Standalone clone fast-forwarded cleanly to `1f551d1981fa4188788e11e650b551bd21c80294`; queue and both output paths were empty.
- CPU rescore-and-audit job: `13133`.

## 2026-07-11T22:51:04+05:30 - monitor MATH preflight rescore and audit job 13133

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, rescore manifest, audit JSON, disagreement ledger, and report/hash checks after terminal state.
- Purpose: require at least 95% agreement across all 16 manual labels, complete metadata, teacher-free status, and at most 5% truncation before full validation.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T22:52:19+05:30

- Job 13133 `COMPLETED` in 1 second, exit `0:0`, with 0 GPUs.
- Immutable rescore changed exactly one decision, `math-number_theory-train-649`, from incorrect to correct; accuracy is now 9/16 and four cases explicitly required model judgment. Rescored predictions SHA-256 is `e8fe408c9cd2962b3a10b38a46362337b0f96fe9d671a5579bf2138a53084dfb`.
- Audit passes every gate: 16/16 manual agreement (100% versus required 95%), complete generation metadata, teacher-free true, 0% truncation versus maximum 5%, zero disagreements, and a non-empty HTML report.
- The stratified validation preflight gate passes. The original predictions and raw metrics remain preserved alongside the explicit rescore provenance.

## 2026-07-11T22:52:19+05:30 - re-freeze corrected baseline and run full MATH validation

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync plus clean/queue/output checks; submit one CPU-only immutable baseline re-freeze bound to the corrected evaluator commit, then one array-index-0 one-GPU evaluation over all 402 audited validation examples with an `afterok` dependency.
- The original baseline freeze remains preserved. Corrected freeze output is `math/baseline/baseline-freeze-v2.json`; evaluation output is the fresh `math/baseline/full-validation/evaluation` directory.
- Input is only the audited Level 4-5 validation split. The official test split remains untouched and no model selection uses test data.
- Freeze resources: account `priyesh.shukla`, u22, node01, 2 tasks, 4 GiB, `00:15:00`, 0 GPUs. Evaluation resources: 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T22:53:06+05:30

- Standalone clone fast-forwarded cleanly to `f21d8a82baa3b93c2a1dcfbf9a3ccaf811f0bf2d`; queue and both corrected output paths were empty.
- Corrected baseline freeze job: `13134`. Dependent full-validation one-GPU job: `13135`, array index 0 only, with `afterok:13134`.

## 2026-07-11T22:53:06+05:30 - monitor corrected freeze and full MATH validation jobs 13134-13135

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs and progress artifacts; inspect the freeze, predictions, failures, metrics, completion marker, and telemetry only after successful terminal states.
- Purpose: complete and freeze the teacher-free validation baseline before any official-test access or guidance collection.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Failure result - 2026-07-11T22:58:58+05:30

- Job 13134 `COMPLETED` in 1 second and wrote the corrected freeze bound to commit `f21d8a82baa3b93c2a1dcfbf9a3ccaf811f0bf2d`.
- Job 13135 index 0 `FAILED` explicitly after 2 minutes 42 seconds on `math-algebra-train-1218`: the pinned 8B evaluator exhausted its configured serialization-repair attempts without producing a valid JSON object. The failure ledger records stage `evaluation` and `ModelOutputParseError`.
- No prediction file, metrics, or completion marker was written. The failed shard is preserved and will never be resumed into a paper run. Peak observed telemetry memory was 24,150 MiB, so this was not an OOM; accounted GPU time was `0.045` GPU-hours.
- Existing failure observability retained only the final error message, despite the evaluator exception carrying every raw attempt. Progression remains stopped until the raw evaluator attempts and corresponding student response are durably captured and inspected.

## 2026-07-11T22:58:58+05:30 - reproduce full-validation evaluator serialization failure with complete evidence

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: after test-first observability correction and full local verification, one BatchMode SSH fast-forward-only sync; submit one CPU-only exact-index lookup for `math-algebra-train-1218`, inspect its result, re-freeze the baseline to the observability commit, and run only that deterministic one-row shard on one GPU in a fresh diagnostic directory.
- The failure ledger now preserves all raw evaluator attempts, parse errors, the student response, and its hash. No recovery or semantic default is introduced; the diagnostic is expected to fail if the defect reproduces.
- Requested lookup/freeze resources: account `priyesh.shukla`, u22, node01, 2 tasks, at most 4 GiB, at most `00:15:00`, 0 GPUs. Diagnostic resources: 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Lookup submission result - 2026-07-11T23:01:12+05:30

- Standalone clone fast-forwarded cleanly to `71d3bb448303b96b5469763f1cf22e32033204d4`; queue and lookup output were empty.
- Exact dataset-row lookup job: `13136`.

## 2026-07-11T23:01:12+05:30 - monitor exact diagnostic row lookup job 13136

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, and the exact lookup artifact after terminal state.
- Purpose: obtain the canonical zero-based validation index and prove dataset identity before submitting a one-row diagnostic shard.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result and diagnostic submission gate - 2026-07-11T23:01:37+05:30

- Job 13136 `COMPLETED` in 1 second, exit `0:0`, with 0 GPUs.
- The exact row is canonical zero-based index 23 of 402, stratum `math:algebra:level4`, row hash `85bdde528a309c56dfe6c78851f8a6304771b6c95e4acc7fdf24d6225aedfdf5`, in validation file SHA-256 `b2b93cab808a73617fff62f1db023a2d526dcc7387158cba24c0b5e72fc26372`.
- Next bounded submission under the already logged diagnostic action: create `baseline-freeze-v3.json` bound to the complete-observability commit, then run array index 23 with `NUM_SHARDS=402` into a fresh diagnostic directory. This deterministically selects only the failed example while preserving its original per-ID generation seed.

### Diagnostic submission result - 2026-07-11T23:02:17+05:30

- Standalone clone fast-forwarded cleanly to `814166ec7232e5611f4309d93ca9e099d2b438e8`; queue, freeze, and diagnostic output were empty.
- Baseline freeze-v3 job: `13137`. Dependent exact one-row diagnostic job: `13138`, array index 23 only with `NUM_SHARDS=402` and `afterok:13137`.

## 2026-07-11T23:02:17+05:30 - monitor evaluator failure reproduction jobs 13137-13138

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, and either the complete failure ledger or success artifacts after terminal state.
- Purpose: capture every evaluator serialization attempt and the exact student response for root-cause analysis without advancing the baseline.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Reproduction result - 2026-07-11T23:06:06+05:30

- Job 13137 `COMPLETED` in 1 second. Exact one-row job 13138 index 23 reproduced the failure and exited `1:0` after 20 seconds; accounted GPU time `0.0056` GPU-hours and peak observed telemetry memory was 24,086 MiB.
- The student response is mathematically correct, ends with `FINAL: \\boxed{\\left( -\\frac{1}{8}, \\frac{1}{2} \\right)}`, and has SHA-256 `3e57526229cc67bbe9f06007c30d8ca7a88ed0aa3a26ad96a8f948cc928772b3`.
- All three 8B evaluator attempts were identical in judgment but invalid JSON because the answer string contained unescaped LaTeX backslashes. The generic repair prompt repeated the same invalid serialization three times. The complete raw attempts and parse failures are now preserved in the diagnostic ledger.
- Root cause is evaluator output serialization, not student generation, mathematical evaluation, memory, or model-cache lookup.

## 2026-07-11T23:06:06+05:30 - verify plain-ASCII evaluator serialization repair

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: after test-first prompt correction and full local verification, one BatchMode SSH fast-forward-only sync and clean/output checks; create `baseline-freeze-v4.json`, then rerun only validation array index 23 of 402 in a fresh `evaluation-v2` diagnostic directory.
- Correction stays agentic: the evaluator prompt and its explicit repair-feedback loop require plain-ASCII answers and explain JSON backslash escaping. No parser fallback, string substitution, fabricated judgment, or silent default is added.
- Freeze resources: account `priyesh.shukla`, u22, node01, 2 tasks, 4 GiB, `00:15:00`, 0 GPUs. Diagnostic resources: 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T23:06:53+05:30

- Standalone clone fast-forwarded cleanly to `18f40b9989bcab823caf38e633358d7aa9731724`; queue, freeze-v4, and diagnostic-v2 paths were empty.
- Baseline freeze-v4 job: `13139`. Dependent exact one-row diagnostic-v2 job: `13140`, array index 23 only with `NUM_SHARDS=402` and `afterok:13139`.

## 2026-07-11T23:06:53+05:30 - monitor evaluator serialization repair jobs 13139-13140

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, then the one prediction/evaluator record or complete failure ledger after terminal state.
- Purpose: require a valid, auditable evaluator object for the exact former failure before retrying full validation.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T23:07:38+05:30

- Job 13139 `COMPLETED` in 1 second. Exact diagnostic job 13140 index 23 `COMPLETED` in 19 seconds, exit `0:0`; accounted GPU time `0.0053` GPU-hours and peak observed telemetry memory was 24,054 MiB.
- The student response reproduced byte-for-byte. The 8B evaluator returned valid JSON on its first attempt with plain-ASCII answer `(-1/8, 1/2)`, no parse failures or regenerations, and both deterministic and model judgments marked it correct.
- Failure ledger is empty, completion marker and hashes are present, and prediction SHA-256 is `eb762948f1144a4b1c0a21ec70fc5ec74d00d805deffe8cd505ea2a46fe95bd2`. The evaluator-serialization repair gate passes.

## 2026-07-11T23:07:38+05:30 - retry complete MATH validation baseline after serialization repair

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync plus clean/queue/output checks and exactly one node01 array-index-0 one-GPU evaluation over all 402 audited validation examples.
- Uses immutable `baseline-freeze-v4.json` and source commit `18f40b9989bcab823caf38e633358d7aa9731724`. Failed `full-validation/evaluation` remains diagnostic; retry output is the fresh `full-validation/evaluation-v2` directory.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T23:08:27+05:30

- Standalone clone fast-forwarded cleanly to `b4a34c10b1eca7b4e41fe6e5e8e6b14f479ffb6c`; queue and retry output were empty.
- Full-validation retry job: `13141`, array index 0 only.

## 2026-07-11T23:08:27+05:30 - monitor full MATH validation retry job 13141

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs and telemetry; inspect predictions, failures, metrics, and completion marker only after terminal success.
- Purpose: freeze the complete teacher-free validation baseline before official-test access and collection.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Failure result and authorization stop - 2026-07-12T00:01:25+05:30

- Job 13149 index 0 `FAILED` after 21 minutes 41 seconds on `math-geometry-train-398`; accounted GPU time `0.3614` GPU-hours and peak observed telemetry memory was 24,496 MiB.
- No prediction, metrics, or completion marker was written. `evaluation-v3` remains an immutable failed diagnostic with the full student response and all evaluator attempts.
- All three evaluator attempts used a quoted answer but reintroduced an unescaped LaTeX backslash in `12/\\sqrt{3}`, despite the plain-ASCII instruction and repair feedback. This confirms that prompt-only enforcement of JSON escaping is not sufficiently reliable for the full paper protocol.
- The July 11, 2026 Turing god-switch window expired at midnight Asia/Kolkata while job 13149 was running. No further remote mutation or allocation will be submitted without renewed authorization.
- Required remediation after renewal: replace the evaluator's JSON answer serialization with a strict escape-free structured protocol or grammar-constrained decoding, re-run unit tests, exact diagnostics, the 16-example manual audit, and then full validation in a fresh directory. No fallback was used; `fallback_reason` is not applicable.

### Failure result - 2026-07-11T23:29:02+05:30

- Job 13141 index 0 `FAILED` after 19 minutes 16 seconds on later example `math-geometry-train-15`; accounted GPU time `0.3211` GPU-hours and peak observed telemetry memory was 24,462 MiB.
- No prediction, metrics, or completion marker was written. The complete student response and all three raw evaluator attempts are preserved; failed `evaluation-v2` will not be resumed.
- This is a distinct JSON contract failure: every attempt emitted the correct tuple `(-5, 6)` as an unquoted JSON value. The evaluator prompt required an answer string, but the generic repair loop repeated the unquoted tuple.
- Progression remains stopped. The repair feedback must explicitly require double quotation marks around answer values that look like tuples, numbers, lists, or intervals.

## 2026-07-11T23:29:02+05:30 - verify quoted evaluator answer contract on exact later failure

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: after test-first prompt strengthening and full local verification, one BatchMode SSH fast-forward-only sync; submit one CPU exact-index lookup for `math-geometry-train-15`, then re-freeze and rerun only its deterministic one-row shard in a fresh diagnostic directory after inspecting the lookup.
- Correction remains prompt-and-control-loop based. The parser stays strict; no unquoted-value coercion or hidden recovery is introduced.
- Lookup/freeze resources: account `priyesh.shukla`, u22, node01, 2 tasks, at most 4 GiB, at most `00:15:00`, 0 GPUs. Diagnostic resources: 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Lookup submission result - 2026-07-11T23:29:47+05:30

- Standalone clone fast-forwarded cleanly to `a884276701974842b040c40be4598546ee8dec63`; queue and lookup output were empty.
- Exact dataset-row lookup job: `13143`.

## 2026-07-11T23:29:47+05:30 - monitor geometry failure row lookup job 13143

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, and exact lookup artifact after terminal state.
- Purpose: bind a one-row diagnostic to canonical validation index and dataset hash.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result and diagnostic gate - 2026-07-11T23:30:12+05:30

- Job 13143 `COMPLETED` in 1 second, exit `0:0`, with 0 GPUs.
- `math-geometry-train-15` is canonical zero-based index 148 of 402, stratum `math:geometry:level5`, row hash `bb0672d14e3f15256955f5b9fc75dd9add3b2dcb32ccf1f659f831c17cb3c911`, in the same audited validation file.
- Next bounded submission under the logged diagnostic action: create `baseline-freeze-v5.json` bound to the quoted-answer prompt commit, then run array index 148 with `NUM_SHARDS=402` into a fresh diagnostic directory.

### Diagnostic submission result - 2026-07-11T23:30:56+05:30

- Standalone clone fast-forwarded cleanly to `4c07747f00013cadb4c8884d09c640c42c28e3d8`; queue, freeze-v5, and diagnostic paths were empty.
- Baseline freeze-v5 job: `13144`. Dependent exact one-row diagnostic job: `13145`, array index 148 only with `NUM_SHARDS=402` and `afterok:13144`.

## 2026-07-11T23:30:56+05:30 - monitor quoted-answer diagnostic jobs 13144-13145

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, then the one prediction/evaluator record or complete failure ledger after terminal state.
- Purpose: require a quoted JSON answer for the exact former tuple failure before any broader rerun.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T23:31:31+05:30

- Job 13144 `COMPLETED` in 1 second. Exact diagnostic job 13145 index 148 `COMPLETED` in 17 seconds, exit `0:0`; accounted GPU time `0.0047` GPU-hours and peak observed telemetry memory was 24,032 MiB.
- The student response reproduced byte-for-byte. The evaluator returned valid JSON on its first attempt with quoted answer string `(-5, 6)`, no parse failures or regenerations, and both judgments correct.
- Failure ledger is empty and prediction SHA-256 is `3e3a8051bf7255703ded17a30e220c89308501128cdd14c8dc6c64175fa51ae7`. The quoted-answer diagnostic gate passes.

## 2026-07-11T23:31:31+05:30 - repeat stratified 16-example audit under final evaluator protocol

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync plus clean/queue/output checks and exactly one node01 array-index-0 one-GPU evaluation of the already immutable 16-example subset into fresh `preflight-16/evaluation-v2`.
- Uses `baseline-freeze-v5.json`; prior preflight predictions, rescore, labels, and audit remain immutable. Manual labels will only be applied after confirming the deterministic student responses and hashes.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T23:32:19+05:30

- Standalone clone fast-forwarded cleanly to `4046fb83806c65d050a75180d9b0cfb4842bea5b`; queue and preflight-v2 output were empty.
- Final-protocol preflight job: `13146`, array index 0 only.

## 2026-07-11T23:32:19+05:30 - monitor final-protocol MATH preflight job 13146

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, predictions, failures, metrics, completion marker, and response-hash comparison after terminal success.
- Purpose: re-establish the manually audited 16-example gate under the final evaluator serialization protocol.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T23:35:53+05:30

- Job 13146 index 0 `COMPLETED` in 3 minutes 6 seconds, exit `0:0`; accounted GPU time `0.0517` GPU-hours and peak observed telemetry memory was 24,550 MiB.
- All 16 student response hashes and canonical ID order exactly match the manually reviewed preflight. The final evaluator protocol produced zero parse failures and zero regenerations.
- Accuracy is 9/16, failure ledger is empty, truncation is 0%, completion marker is valid, and predictions SHA-256 is `52bfb3dfbfc944f7c0f6786d62a99040f87cb4d6d1304b1b948aa3ba677c793a`.

## 2026-07-11T23:35:53+05:30 - audit final-protocol preflight and gate full-validation retry

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH clean/output check and exactly one CPU-only node01 immutable rescore-and-audit job using all 16 existing human labels; inspect the audit before any GPU resubmission.
- Outputs are fresh `preflight-16/rescore-v2` and `preflight-16/audit-v2`; final-protocol predictions remain immutable.
- Requested resources: account `priyesh.shukla`, u22, node01, 2 tasks, 4 GiB, `00:15:00`, 0 GPUs; 0 requested GPU-hours.

### Submission result - 2026-07-11T23:36:34+05:30

- Standalone clone fast-forwarded cleanly to `db894d6fcf932f547f97149c6db50e33ed07f4c6`; queue and both audit outputs were empty.
- Final-protocol preflight audit job: `13148`.

## 2026-07-11T23:36:34+05:30 - monitor final-protocol preflight audit job 13148

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs, rescore manifest, audit JSON, disagreements, and report hashes after terminal state.
- Purpose: enforce 95% evaluator agreement, 5% truncation maximum, metadata, and teacher-free gates under the final protocol.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.

### Result - 2026-07-11T23:37:24+05:30

- Job 13148 `COMPLETED` in 2 seconds, exit `0:0`, with 0 GPUs.
- Rescore changed zero decisions: original and rescored correctness are both 9/16, with three explicit model-judgment cases.
- Audit passes all gates with 16/16 manual agreement, zero disagreements, complete generation metadata, teacher-free true, and 0% truncation. The final-protocol preflight gate passes.

## 2026-07-11T23:37:24+05:30 - run full MATH validation under fully audited evaluator protocol

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: one BatchMode SSH fast-forward-only sync plus clean/queue/output checks and exactly one node01 array-index-0 one-GPU evaluation of all 402 audited validation examples into fresh `full-validation/evaluation-v3`.
- Uses immutable `baseline-freeze-v5.json` and source commit `4c07747f00013cadb4c8884d09c640c42c28e3d8`. Both earlier failed full-validation directories remain immutable diagnostics.
- Requested resources: account `priyesh.shukla`, u22, node01, 1 GPU, 16 tasks, 64 GiB, `03:00:00`; 3.0 requested GPU-hours.

### Submission result - 2026-07-11T23:38:28+05:30

- Standalone clone fast-forwarded cleanly to `3c87bb8eaae8aef301c4cfa3112da04ecb9892cc`; queue and full-validation-v3 output were empty.
- Fully audited full-validation job: `13149`, array index 0 only.

## 2026-07-11T23:38:28+05:30 - monitor fully audited MATH validation job 13149

- Approval reference: same end-to-end request and active 2026-07-11 god switch.
- Bounded command set: read-only BatchMode SSH polling of `squeue`, `sacct`, bounded logs and telemetry; inspect predictions, failures, metrics, and completion marker only after terminal success.
- Purpose: freeze the complete teacher-free validation baseline before official-test access and collection.
- Requested resources: monitoring only; no new allocation; 0 additional requested GPU-hours.
