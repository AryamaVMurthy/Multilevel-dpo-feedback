# Superseded SearchQA protocol

The earlier markup-based response protocol and its July 13 job graph are invalid and retired. They must not be used as a baseline or resumed.

The active research design is [2026-07-13-plain-answer-thinking-design.md](2026-07-13-plain-answer-thinking-design.md), and its executable plan is [2026-07-13-plain-answer-thinking.md](2026-07-13-plain-answer-thinking.md).

Current invariants are plain short student answers, private thinking only, one strict internal JSON hint, student-generated exact-correct DPO choices, 4096 total tokens, full BF16 student finetuning, a 4-bit post-trained instruct teacher, complete cache identities, train-dev prompt selection, regular resumable checkpoints, and untouched-test final reporting.
