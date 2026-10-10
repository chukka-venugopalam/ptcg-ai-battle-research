# repro/ — reproducibility package

What this is: manifests, hashes and small tools. It does **not** contain third-party code or weights.

| File | Purpose |
|---|---|
| `SOURCE_MANIFEST.csv` | 45 inspected repos with pinned commit SHAs, dates, license-file guess, tracked engine binaries |
| `ARTIFACT_HASHES.csv` | SHA-256 of weight-like files, evidence figures/spreadsheets and leaderboard snapshot zips (kinds noted; pickles never loaded) |
| `claims_ledger.csv` | Key claims with tier, source, and how to falsify |
| `fetch_sources.sh` | `bash fetch_sources.sh` shallow-fetches each repo at its pinned SHA into `third_party/` (skips the UNSAFE repo; nothing is executed). Run it outside `/mnt/user-data` if git reports "dubious ownership" |
| `verify_manifest.py` | Re-checks SHAs and hashes after fetching |
| `inspect_npz.py` | Lists `.npz` array shapes / `.zip` members; refuses pickle formats |
| `leaderboard_snapshots.py` | Re-computes the Simulation rating-distribution table from Scio's CSV snapshots |
| `stats_gates.py` + `tests/` | Wilson CI, two-proportion test, required games, acceptance-gate function; `python3 -m unittest discover -s tests` |

Setup: Python 3.10+, `pip install numpy pandas`. Tested here: unit tests pass (6/6); snapshot script reproduces the report table; fetch + verify run on 44 repos with 0 mismatches over 179 checks (the UNSAFE repo is skipped by design).

Not tested: any third-party agent, engine, or training code (none was executed).
