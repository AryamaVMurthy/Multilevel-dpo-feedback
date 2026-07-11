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

## Date-scoped god switch

- The user may temporarily suspend only the per-action approval requirement by saying the exact phrase `god privileges blessed` and specifying the Asia/Kolkata calendar day for which it applies.
- The switch expires automatically at midnight Asia/Kolkata on that day and does not carry into later dates or conversations without a new explicit activation.
- While active, the agent may run, inspect, submit, monitor, fix, and retry Turing work without asking again, but every action must still be recorded in the append-only usage log.
- The credential rules and GPU-accounting requirements are never suspended by this switch.
- Active exception: the user activated the switch for **2026-07-11 Asia/Kolkata only** in the current conversation.

## GPU accounting

- There is no repository-enforced daily requested GPU-hour limit.
- Before each GPU job, record requested GPUs, wall time, and the resulting requested GPU-hours in the usage log.
- Report accounted GPU-hours when Slurm makes them available. Resource accounting is for observability and experiment-cost analysis, not a submission gate.

## Mandatory append-only usage log

- Maintain `docs/status/turing-usage-log.md` as an append-only log.
- Before an authorized Turing action, record the timestamp, user approval reference, exact command or bounded command set, purpose, target paths, and requested resources.
- After it finishes, append the result: job ID when applicable, Slurm state or command exit status, elapsed time, accounted GPU-hours when available, output/artifact paths, and any failure.
- Never place passwords, tokens, private keys, full connection strings, or other credential material in the log.
