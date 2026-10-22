# ARCHIVE_DIFFS — BLOCKED (0 archives received)
No source, deck, weight, flag or entrypoint comparison was possible. Nothing is inferred about whether the R2 compat candidate, the 826.0/780.4 `submission.tar.gz` rows, or the Bv1 backup share bytes.
Ready: `repro/run_phase_AB.sh BUNDLE OUT 20` hashes the bundle against `input_manifest.json`, audits each archive (tar safety, deck vs engine card table, cold start without `__file__` + SDK last-callable rule, N real BO3, safety gate) and writes `ARCHIVE_FORENSICS.csv`, `AGENT_SAFETY_AUDITS.csv`, `ARCHIVE_DIFFS_raw.txt`.
Dry run on two FIXTURE archives (not your agents): `raw/fixture_dryrun/` — manifest check 2/2 verified, audit exit 0 for both, diff shows `main.py` changed only. That proves the script runs, nothing about Bv1/R2/B2/Bv2.
