# Superseded SearchQA protocols

The earlier markup-based response protocol and its July 13 job graph are invalid and retired. They must not be used as a baseline or resumed.

The plain-short-answer protocol was subsequently superseded when the required output changed to active fixed retrieval with cited reasoning. Its runs may be retained only as explicitly labeled archival baselines.

The active research design is [2026-07-13-fixed-retrieval-cited-reasoning-design.md](2026-07-13-fixed-retrieval-cited-reasoning-design.md), and its executable plan is [2026-07-13-fixed-retrieval-cited-reasoning.md](2026-07-13-fixed-retrieval-cited-reasoning.md).

Current invariants are fixed SearchQA source retrieval, canonical dataset citations, concise visible reasoning, no XML student output, private scratchpads, one strict internal JSON hint, student-generated verified no-hint training targets, same-context DPO pairs, 4096 total tokens, full BF16 student finetuning, a 4-bit post-trained instruct teacher, complete cache identities, regular resumable checkpoints, and untouched-test final reporting.
