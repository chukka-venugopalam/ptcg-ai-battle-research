# Repository Tools

All tools use the Python standard library and should be run from the repository root. The native smoke test uses the archive's runtime/SDK and may be platform-dependent.

- `validate_submission.py ARCHIVE.tar.gz`: inspect gzip/tar, path safety, required root members, 60-card deck size and Python syntax. It does not estimate win rate.
- `smoke_test_submission.py ARCHIVE.tar.gz`: run the exec-style deck handshake, one native-engine self-play smoke game, legal-action checks, health counters and available embedding boundary probes. It is an operational check, not a strategy comparison.
- `register_submission.py`: validate, copy and register a submission archive, including SHA-256, weights/deck fingerprints and a manifest. It refuses duplicate hashes and never replaces a previous archive.
- `new_mission.py`: generate a mission folder (`REPORT.md`, `results.csv`, `run_manifest.json`) and add a row to `mission_registry.csv`. It rejects duplicate IDs before creating files.
- `new_experiment.py`: pre-register an experiment with baseline/candidate identifiers, a fixed sample plan, opponent pool, seat policy and stopping rule. It marks the result `NOT RUN`, refuses duplicate experiment IDs, and preserves prior records when upgrading older CSV schema.

Useful commands:

```bash
python -m unittest discover -s tests -v
python tools/validate_submission.py submissions/current/submission.tar.gz
python tools/smoke_test_submission.py submissions/current/submission.tar.gz
python tools/new_mission.py --id 8 --title "Replay-Derived Policy Training" --status planned
```

See `docs/EXPERIMENT_PROTOCOL.md` before interpreting results. Keep original outputs and hashes; never overwrite the baseline archive.
