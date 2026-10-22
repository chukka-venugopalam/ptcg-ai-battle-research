# Changelog

## 2026-10-22 — Research archive and release workflow

- Consolidated Missions 1–6.2 and available report/evidence files under `research/missions/`.
- Added current R2 compatibility reference release and its validation manifest/report.
- Recorded latest validation: structural pass, 5 repository unit tests after tooling extension, one native-engine smoke game (latest: 158 actions in 4.070 seconds), no fallbacks/missing-network calls, and OOV boundary checks passed.
- Added `tools/new_experiment.py` to pre-register an experiment and extend older experiment registry schemas safely.
- Hardened `tools/new_mission.py` so duplicate mission IDs are rejected before any orphan directory is created.
- Documented that Mission 6.2 showed no statistically decisive strategy improvement, and Mission 7 remains planned/not executed.

## 2026-10-06 to 2026-10-20 — Research history

See `research/registries/mission_registry.csv`, `research/registries/experiments.csv`, the six-cycle synthesis, and each mission's own report for the timeline. Do not treat proposed experiments as completed experiments.
