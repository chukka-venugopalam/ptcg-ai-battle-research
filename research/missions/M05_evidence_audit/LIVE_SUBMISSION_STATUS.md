# LIVE_SUBMISSION_STATUS — Mission 5, Phase A
Generated 2026-10-16 from the Claude sandbox.

## Result: UNKNOWN (blocked). No live Kaggle evidence was collected.
| Check | Result | Evidence |
|---|---|---|
| `kaggle` CLI present | NOT FOUND (`which kaggle` empty) | VERIFIED |
| Credentials (`~/.kaggle`, `KAGGLE_*` env) | NONE | VERIFIED |
| Egress to kaggle.com | `curl: (6) Could not resolve host: www.kaggle.com` | VERIFIED |
| Sandbox allow-list | pypi/github/npm only; kaggle.com not allowed | VERIFIED (network config) |
| Playground slug, rules, deadline, latest-N semantics | UNKNOWN | not reachable |
| Submission list/IDs/timestamps/episode counts | UNKNOWN | not reachable |
| Which submission a new upload would evict | UNKNOWN | see below |

## What is only REPORTED (memory of user screenshots, 2026-10-09/10; not refreshed)
Bv1 backup 813.5, R12 528.2 (both reported active); submission.tar.gz 826.0, Bv2_sym8 781.6, B2_counter_source_grimmsnarl 789.9 (reported inactive). Different ages/episode counts/uncertainty, so no code-change attribution.
Eviction: if "latest 2 submissions are active" holds, an upload evicts the OLDER of the two active ones. Which is older is UNKNOWN until timestamps are read. **Do not upload.** Bv1 stays safe.

## Run on an authenticated machine (read-only; nothing here uploads)
```bash
kaggle --version; kaggle competitions -h          # record exact subcommand names first
kaggle competitions submissions -c <SLUG> -v      # ids, filenames, dates, status, score
kaggle competitions leaderboard -c <SLUG> --show
kaggle competitions episodes <SUBMISSION_ID> -v   # per active submission; episode counts
```
Open `https://www.kaggle.com/competitions/<SLUG>/rules`; save text with access date (deadline, "latest N" wording, data-sharing limits).
Mission 3's `tools/fetch_kaggle_evidence.sh` exists but its subcommands are INFERRED (self-declared), has never run, and must be checked against `kaggle competitions -h` before use. Save every raw output with timestamp + exit code and `sha256sum` it.
