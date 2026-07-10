# Turing access, credentials, and usage controls

These rules are mandatory for every agent working in this repository.

## Credentials

- Treat `creds.txt` and every credential value as secret.
- Never print, quote, copy, commit, log, diff, or paste a password into any document, chat response, source file, shell history, command line, environment variable, or tool output.
- A password may be entered only into an interactive SSH password prompt in the shell after the user has explicitly authorized that exact login. Do not use `sshpass`, scripted password injection, or any equivalent mechanism.
- Do not open `creds.txt` unless the user has explicitly authorized the specific Turing action that needs it.

## Explicit per-action approval

- Do not run any Turing-related command without the user's explicit approval in the current conversation.
- This includes SSH, SCP, rsync, Slurm inspection commands, `sinteractive`, `sbatch`, `squeue`, `sacct`, `scontrol`, `scancel`, file operations on Turing, and commands executed after logging in.
- Before requesting approval, state the exact command or bounded command set, purpose, expected resources, and whether it changes Turing state. Approval for one action does not authorize later actions.
- Never submit, cancel, modify, or retry a job on the agent's own initiative.

## Daily hard limit

- The daily Turing budget is a user-specified maximum of cumulative **requested GPU-hours** in Asia/Kolkata time. A job reserves `requested GPUs x requested wall time` against that budget before submission.
- The authorized daily limit is **10 requested GPU-hours**. Do not perform a Turing action that would exceed it unless the user explicitly changes this limit.
- Before every proposed Turing job, calculate the already reserved GPU-hours for the day plus the new reservation. If the total would exceed the limit, stop immediately and report the limit, current total, and proposed job; do not submit it.
- Never treat a lower observed usage as permission to exceed the reserved-budget limit unless the user explicitly changes the limit.

## Mandatory append-only usage log

- Maintain `docs/status/turing-usage-log.md` as an append-only log.
- Before an authorized Turing action, record the timestamp, user approval reference, exact command or bounded command set, purpose, target paths, and requested resources.
- After it finishes, append the result: job ID when applicable, Slurm state or command exit status, elapsed time, accounted GPU-hours when available, output/artifact paths, and any failure.
- Never place passwords, tokens, private keys, full connection strings, or other credential material in the log.
