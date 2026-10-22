# Submission Workflow

1. Create candidate code from a frozen baseline.
2. Record hypothesis and source hash in `research/registries/experiments.csv` before testing.
3. Build one archive with root `main.py` and `deck.csv`.
4. Run `python tools/validate_submission.py <archive>`.
5. Run native engine smoke and OOV probes when the SDK is available.
6. Run fixed-size A/B screening with balanced seats. Do not label it confirmation unless its plan was fixed before examining results.
7. Add the candidate via `python tools/register_submission.py` and keep the original unmodified archive.
8. Read the current active submission list before uploading. The competition tracks only the latest two active submissions (per the competition rules documented during this research); an upload can displace a previously active candidate.
9. After a live upload, record exact archive SHA, timestamp, displayed score, episode count and screenshot/replay references.

Do not upload a same-hash duplicate just to make a new dated file.
