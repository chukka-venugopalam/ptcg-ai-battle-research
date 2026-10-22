# Project Status — 2026-10-22

## What is deliverable today

- `submission.tar.gz`: byte-identical copy of the previously validated R2 compatibility candidate.
- The archive has valid root entries (`main.py`, `deck.csv`), 39 tar members, no `__pycache__`/`.pyc`, a 60-card deck, and the exact R2 native library SHA-256 that matches the official R2 package.
- Local validation completed on 2026-10-22: the latest run of `tools/smoke_test_submission.py` completed one native-engine self-play game in 158 selections (4.070 seconds). The 60-card deck was accepted; both sides reported zero fallbacks, zero missing-network calls, and no errors. Synthetic embedding boundary checks passed for card IDs 1300 and 1431 and attack ID 1755. Earlier smoke runs of the same archive completed in 181, 182 and 239 selections. The reusable test command is `python tools/smoke_test_submission.py submissions/current/submission.tar.gz`.

## What it is not

- It is not a new trained policy.
- It is not a new deck or strategic improvement over the October 15 candidate.
- The Mission 6.2 screening did not prove any of the tested variants better.
- A 1500+ live rating is not established or guaranteed.

## Important use note

The today's dated tarball SHA-256 is `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`, the same hash as `submission_R2_compat_2026-10-15.tar.gz`. **Do not re-upload this duplicate if that exact archive is already active.** Use it as the preserved safe baseline/reference for the next proper deck/policy experiment.
