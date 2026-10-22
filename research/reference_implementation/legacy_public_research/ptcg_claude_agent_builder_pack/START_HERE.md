# PTCG AI Battle — Claude Agent Builder Pack v1

## Goal

Use this pack with Claude as the research/training context for building our next Pokémon TCG AI Battle agent.

The immediate engineering target is **~1000+ ladder rating**, with the explicit understanding that no rating is guaranteed. Our current visible baseline is around 781.4 and the visible leader in the captured screenshot is 1135.7.

## What's inside

- `CLAUDE_MASTER_PROMPT.md` — paste this into Claude as the primary instruction.
- `research/` — public research sources, inventory, model audits, benchmark evidence and the working bootstrap notebook.
- `competition_sdk/cg/` — the extracted CABT SDK package from the competition materials.
- `repo_template/` — our research repo skeleton and scripts used to clone/inventory the public agents.
- `current/` — current baseline notes and leaderboard screenshot.
- `notes/` — selected source excerpts from our Scio inspection.

## Public repositories to clone

The exact source list is in `research/public_research_sources.csv` and can also be cloned by the included public-research bootstrap tooling.

## Important limitation

This pack contains the official SDK bundle and the reproducible scripts/manifests for obtaining the public repositories. The ephemeral full cloned working trees from the Kaggle notebook were not persisted as Library files, so Claude should clone the listed public repositories when network access is available rather than assuming their complete trees are embedded here.

Likewise, the original competition ZIP's private/input card PDFs and CSVs are not redistributed in this pack. `research/competition_inputs.csv` records exactly which competition inputs were identified and where they are expected.

## Non-negotiable methodology

- Reuse public work as research, not as an unexamined copy.
- Preserve legal-action masking and submission compatibility.
- Never fabricate training results, win rates or Kaggle scores.
- Every improvement must have a controlled before/after experiment.
- Keep a reproducible path from training checkpoint -> inference -> submission archive -> smoke test.
