# R2_ENGINE_FINGERPRINTS — Phase C: UNKNOWN
- `libcg` / official CABT SDK / card table: NOT in uploads; `find / -name "libcg*"` found only unrelated Graphviz `libcgraph` (VERIFIED, not the engine).
- `kaggle-environments`: NOT installed here, so Mission 3's engine-backed tests were skipped (VERIFIED, 22 skips).
- Mission 3's claimed environment (`kaggle-environments==1.33.0`, `bo=3`, 1,431 card IDs) is REPORTED; engine equivalence to the live ladder remains INFERRED, not proven. Matching card counts are not equivalence.
**Needed:** SHA-256 of `libcg` (.so/.dll), the card table file, and the rules/config JSON from (a) the user's official SDK/submission commit and (b) Mission 3's environment; plus the exact package version/commit. Until both fingerprints match, local harness results are not portable to the ladder.
