# PTCG AI Battle Lab

A version-controlled research and release repository for the **Pokémon TCG AI Battle Challenge Playground**. It keeps the available research cycles, mission evidence, archived submission packages, current reference artifact, evaluation scripts, hashes, and repeatable workflows in one place.

> **Status as of 2026-10-22:** The dated submission artifact is a byte-for-byte copy of the R2 compatibility candidate last validated on 2026-10-15. Its SHA-256 is `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`. Mission 6.2 did not establish a statistically decisive strategy improvement, so no speculative modification was promoted. **Do not upload this duplicate if the same hash is already active.** This archive is a safe compatibility reference, not a new policy or a proven 1500+ agent.

## Start here

1. [`docs/PROJECT_STATUS_2026-10-22.md`](docs/PROJECT_STATUS_2026-10-22.md) — what was packaged and validated today.
2. [`docs/RESEARCH_SUMMARY_SIX_CYCLES.md`](docs/RESEARCH_SUMMARY_SIX_CYCLES.md) — consolidated results, what was learned, and remaining gaps.
3. [`submissions/current/VALIDATION_REPORT.md`](submissions/current/VALIDATION_REPORT.md) — exact archive identity and tests run.
4. [`submissions/registry.csv`](submissions/registry.csv) — submission hashes, provenance, history and promotion state.
5. [`research/registries/mission_registry.csv`](research/registries/mission_registry.csv) and [`research/registries/experiments.csv`](research/registries/experiments.csv) — mission status and experimental record.
6. [`docs/HOW_TO_CONTINUE.md`](docs/HOW_TO_CONTINUE.md) — next steps for adding missions, experiments and releases.

## Repository layout

```text
.
├── README.md, CHANGELOG.md, LICENSES.md, BUILD_INFO.md
├── docs/                         status, research synthesis, protocols and workflows
├── research/
│   ├── missions/M01..M07/        extracted reports, source files, CSVs, scripts and evidence
│   ├── raw_archives/             original research/mission ZIPs retained for provenance
│   ├── reference_implementation/ reusable legacy research scripts and notebook
│   └── registries/               mission, experiment, score, archive and source records
├── submissions/
│   ├── current/                  dated working reference, manifest and validation report
│   ├── releases/                 dated release packages
│   ├── archive/                  historical submission tarballs
│   └── registry.csv              central hash/provenance/status index
├── templates/                    report, experiment and release templates
├── tools/                        submission validation, smoke tests and scaffolding scripts
├── tests/                        repository-tool tests
└── local_data/                   optional large/private inputs; deliberately git-ignored
```

Mission 7 has a prompt and input bundle, but no executed Mission 7 output is recorded in this repository as of the date above. The repository preserves what was actually inspected or run; it does not turn plans into results.

## Today's submission artifact

- File: [`submissions/releases/2026-10-22_R2_compat/submission.tar.gz`](submissions/releases/2026-10-22_R2_compat/submission.tar.gz)
- SHA-256: `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`
- Size: 4,839,805 bytes; 39 tar members; `main.py` and `deck.csv` at the archive root; 60 cards.
- Latest native-engine smoke: one full self-play game, 158 action selections in 4.070 seconds, with zero fallback and zero missing-network calls for both sides. Boundary probes for card IDs 1300/1431 and attack ID 1755 passed.
- **Strategic strength: not established.** Smoke validation tests execution, not win rate.

The `submissions/current/submission.tar.gz` copy has the same bytes/hash. Keep this artifact as a known reference; do not re-submit a duplicate just to make a newer filename.

## Quick commands

Use Python 3.10+; repository maintenance scripts use only the Python standard library. Agent execution depends on the bundled runtime/SDK.

Validate the archive structure and Python syntax:

```bash
python tools/validate_submission.py submissions/current/submission.tar.gz
```

Run the local native-engine smoke test (requires the packaged SDK/runtime to work on your OS):

```bash
python tools/smoke_test_submission.py submissions/current/submission.tar.gz
```

Run tests:

```bash
python -m unittest discover -s tests -v
# or, if pytest is installed:
python -m pytest -q
```

Create a new mission scaffold and registry row:

```bash
python tools/new_mission.py --id 8 --title "Replay-Derived Policy Training" --status planned
```

Pre-register a fixed experiment before looking at its results:

```bash
python tools/new_experiment.py \
  --id E004 \
  --baseline submissions/current/submission.tar.gz \
  --candidate candidate-name-or-hash \
  --question "Does the new policy improve held-out game win rate?" \
  --protocol "Single-game outcomes; fixed sample; retain draws as a separate outcome" \
  --opponent-pool "research/opponents/frozen_pool_v1.csv" \
  --seat-policy "50/50 seats for each pairing" \
  --planned-games 1000 \
  --stopping-rule "Run exactly 1000 games; report CI; promote only if predeclared gate passes"
```

Register a candidate without overwriting prior artifacts:

```bash
python tools/register_submission.py --archive path/to/submission.tar.gz \
  --date 2026-10-23 --name candidate-01 \
  --notes "Describe the exact change and reference the experiment ID"
```

After each run, update the mission report and experiment row with the actual results, full CSV paths and archive SHA-256. A candidate is not promoted just because it packages or runs successfully.

## Evidence and release rules

- Never confuse a fixture-agent test, smoke game or synthetic ID probe with a competitive win-rate test.
- Keep game-level and match-level counts distinct; balance seats and freeze the opponent pool before confirmation testing.
- Preserve raw outputs and hashes. Do not fill in missing results from memory or inference.
- Use `status=NOT RUN` until there are complete, inspectable result files.
- Do not claim a live score without a current submission record and screenshot/log that maps to an exact archive hash.
- Do not promote a candidate until a pre-registered local gate passes; inspect active Kaggle submissions before any upload.

## Git and input-data policy

The checkout has a local Git history but no remote configured. Review the files and license/privacy boundaries, then create a **private** GitHub repository and connect it if desired:

```bash
git status
git log --oneline -5
git remote add origin <your-private-github-repository-url>
git push -u origin main
```

`local_data/` is intentionally ignored by Git to keep large input bundles and the official competition package out of normal commits. Some research outputs reference these local files; their inventories and SHA-256 records are retained in the tracked research registry. Copy the Mission 7 input bundle into `local_data/private_inputs/` from the separately provided bundle when you need to rerun that mission. Review all upstream competition terms and licenses before making anything public.
