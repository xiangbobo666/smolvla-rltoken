# Project instructions

## RLT knowledge base

The repository-owned RLT knowledge base is located in `docs/knowledge-base/`.
It is an independent copy; do not assume that it is synchronized with the
author's Obsidian vault.

Before making or reviewing changes related to system design, algorithms,
training objectives, actor-critic behavior, or data pipelines, read the
relevant notes in `docs/knowledge-base/`. In particular:

- Use `SmolVLA_RLT_plan.md` for the overall design and implementation plan.
- Use `算法实现.md` and the notes under `详解/` for algorithm and loss details.
- Use `数据管线.md` for dataset and data-pipeline behavior.

Treat these notes as project context rather than unquestionable truth. Resolve
disagreements in favor of verified code behavior, tests, experiment results,
and explicit user decisions, and record important discrepancies in the
relevant note.

When an implementation change makes the repository knowledge base inaccurate,
update the corresponding note in the same change. Keep relative links and
files under `附件/` intact when moving or renaming notes.
