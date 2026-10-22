# Build Info — 2026-10-22

- Repository: `ptcg-ai-battle-lab`
- Branch: `main`
- Initial commit: `7d2de2abd50ea985c007b33754f408cac8af38e3`
- Git remote: none configured; this is a portable local repository prepared for a user-selected private remote.
- Local/private inputs: excluded by `.gitignore`; tracked research reports, mission evidence, archive manifests, raw mission archives, and submission archives remain in Git.
- Current submission SHA-256: `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`
- Submission structural validation: PASS (`tools/validate_submission.py`).
- Repository tests: `python -m unittest discover -s tests -v` — 5 passed.
- Latest native smoke: PASS, 1 full game, 158 action selections, 4.070 seconds, zero fallbacks and zero missing-network calls on both sides.
- Out-of-range probes: PASS for card IDs 1300/1431 and attack ID 1755.
- Strategic win-rate improvement: NOT ESTABLISHED; Mission 6.2 comparisons remain inconclusive.
