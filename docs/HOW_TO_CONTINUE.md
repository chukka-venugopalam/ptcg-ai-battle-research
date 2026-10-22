# How to Continue the Project

## Next research item

Mission 7 is prepared but not executed in this workspace. The prompt is `research/missions/M07_meta_and_deck_upgrade/MISSION_PROMPT.md`. It should develop a current-meta dataset, investigate genuinely different deck/policy candidates, and evaluate them against a preserved baseline. Mission 7 input bundle artifacts are tracked in the source archive inventory / Mission 7 input bundle reference; original large official SDK and input packages are deliberately git-ignored under `local_data/private_inputs/`.

## Add a mission

```bash
python tools/new_mission.py --id 8 --title "Replay-Derived Policy Training" --status planned
```

This creates a new folder with `REPORT.md`, `results.csv` and `run_manifest.json`, and adds the mission to `research/registries/mission_registry.csv`. The tool rejects registered IDs before creating the folder. Fill the report from actual commands/results only.

## Add an experiment

Pre-register the hypothesis and evaluation protocol before running the candidate:

```bash
python tools/new_experiment.py \
  --id E004 \
  --baseline submissions/current/submission.tar.gz \
  --candidate candidate-archive-or-hash \
  --question "Does the new policy improve held-out game win rate?" \
  --protocol "Single-game outcomes; draws separate; fixed sample" \
  --opponent-pool "frozen pool manifest path" \
  --seat-policy "50/50 seats for each pairing" \
  --planned-games 1000 \
  --stopping-rule "Run exactly 1000 games; report confidence interval; apply predeclared gate"
```

The row is entered as `pre_registered` / `NOT RUN`. After the run, update the row with completed game count, actual result, status and decision, and keep the complete raw results CSV under the mission folder. Do not alter the stopping rule after inspecting interim results without identifying the work as exploratory.

## Add a submission archive

```bash
python tools/register_submission.py --archive /path/to/submission.tar.gz \
  --date 2026-10-23 --name candidate-01 \
  --notes "Exact change and experiment ID"
```

The tool validates and stores a new archive under `submissions/archive/`, calculates its SHA-256, deck hash and model hash, writes a manifest and updates `submissions/registry.csv`. It refuses to register an identical hash by default. It does not automatically promote a candidate to `submissions/current/`.

## Promotion procedure

1. Preserve the current archive and record its SHA-256.
2. Register the experiment before evaluation.
3. Validate the candidate structure, run smoke tests, then run a seat-balanced tournament against a frozen opponent pool.
4. Review errors, illegal actions, fallbacks, confidence intervals and matchup regressions.
5. Only after a predeclared performance gate passes, create a dated release under `submissions/releases/`, update its manifest/report and record it in the submission registry.
6. Before Kaggle upload, inspect the active submissions and avoid uploading a duplicate hash. Record the exact hash and live evaluation evidence afterward.

## Historical artifact availability

A `poffin_force=True` candidate was described in older notes, but its original archive bytes and raw row-level results were not present in the available build inputs. The report preserves its recorded SHA-256 and reported 102–97 screen over 199 decided games as a historical note. Do not recreate an archive from its hash alone or treat that screen as a confirmed win.

## Data and licensing

The repo keeps extracted findings and original research ZIPs needed for provenance. Large or license-sensitive packages and the official SDK belong in `local_data/private_inputs/`, ignored by Git. Keep the repository private by default; review upstream licences and competition terms before publication.
