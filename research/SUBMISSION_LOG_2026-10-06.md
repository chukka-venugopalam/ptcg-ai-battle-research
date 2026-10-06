# Submission log — 2026-10-06

## B-v2 / symmetry-averaged submission

Submitted artifact: `submissions/2026-10-06/submission_Bv2_sym8.tar.gz`

This is the exact `.tar.gz` used for today's submission. It is B-v1 plus the
test-time symmetry setting `sym_k=8`.

### Exact change

The deployed `main.py` adds the `sym_k=8` agent argument. The policy weights,
deck, and SDK/runtime assets are otherwise unchanged from B-v1.

### Validation recorded

- 60-card deck.
- Policy network loads successfully.
- 0 fallbacks.
- 0 missing-network events.
- `main.py` and `deck.csv` are at the root of the submission archive.
- Functional validation completed successfully.
- The larger seat-balanced local battery did not establish superiority over B-v1.

### Interpretation

This was selected as a low-risk, decision-different variant for the day's
ladder run. It should not be described as a demonstrated 1200-level upgrade.

Full evidence: `research/B_V2_REPORT.md`

### Artifact checksum

SHA-256:

`169e1ac85e9edba28ccf19ff99e244c2016712096e7cdb0aab69cf9e0574318c`

## Suggested Git commit message

`Add B-v2 sym8 submission and validation record`
