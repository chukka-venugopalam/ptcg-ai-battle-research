# Mission 6.2 actual tournament evidence bundle

This bundle contains the final write-once per-game CSVs and the analysis scripts used in the local R2 native-engine comparison. The scripts reference `/mnt/data/m61_work/agents` and `/mnt/data/m61_work/engine`; they are included for audit, not guaranteed to run without rebuilding that extracted workspace.

## Summary

- 128-game screen, 100-game `counter_source` confirmation, 100-game Shadowy Darkness Energy confirmation and 2 smoke games.
- All logged games had zero engine errors, invalid actions, action errors, fallbacks and tracebacks.
- No tested policy/deck change provided statistically decisive improvement. Mission 6.2 decision: NO-GO for another upload.
- Full analysis and exact hashes are in `PTCG_MISSION_6_2_ACTUAL_TOURNAMENT_REPORT_2026-10-20.md`.

## Statistical caveat

These are local single-game outcomes against the specified archives, not the Kaggle rating system itself. The original agent set mostly shares the same learned weights and deck; the screen is not a broad meta evaluation.
