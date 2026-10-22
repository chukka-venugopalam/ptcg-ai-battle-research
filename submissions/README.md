# Submission archive policy

- `current/submission.tar.gz` is the operational copy of the currently preserved reference agent.
- `releases/YYYY-MM-DD_name/` is immutable release metadata, report and exact archive.
- `archive/` stores historical candidate archives by provenance; never replace an archive in place.
- `registry.csv` is the searchable inventory. Scores are kept only when reported; any missing archive-to-score mapping is explicitly marked.
- Today's archive is identical to the 2026-10-15 R2 compatibility archive. Don't upload it again if that hash is already active.
- Before a candidate is promoted to `current/`, it must pass the testing protocol in `docs/EXPERIMENT_PROTOCOL.md` and the runtime/strategic gates documented in its release report.
