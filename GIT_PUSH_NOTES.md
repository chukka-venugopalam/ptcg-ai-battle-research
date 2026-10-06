# Git push notes — 2026-10-06

Recommended commit message:

`Add B-v2 sym8 submission and validation record`

After extracting this package into the existing repository, review the diff,
then commit the new submission artifact and research record.

Suggested commands:

```bash
git status
git add README.md submission/README.md research/B_V2_REPORT.md research/SUBMISSION_LOG_2026-10-06.md submissions/2026-10-06/submission_Bv2_sym8.tar.gz
git commit -m "Add B-v2 sym8 submission and validation record"
git push
```

The historical `.tar.gz` is intentionally outside `submission/` so the runtime
folder continues to contain only deployment assets.
