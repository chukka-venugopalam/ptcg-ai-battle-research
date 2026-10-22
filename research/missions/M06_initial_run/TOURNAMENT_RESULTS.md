# TOURNAMENT_RESULTS — Mission 6
**Real-agent results: NONE.** R2-compat, Bv1, B2 and Bv2 played 0 BO3 matches (archives not received). `TOURNAMENT_MATCHES.csv` and `TOURNAMENT_GAMES.csv` are header-only by design. No strength statement about any agent is made or implied.

## What did run on the REAL engine (public kaggle-environments 1.33.0; fixture agents; harness validation only)
| Run | Matches | Result |
|---|---|---|
| In-process probe (`tools/local_probe.py`, random/first fixtures) | 38 BO3 (≈96 games), all DONE | canonical BO3 rule = engine reward in 38/38; `yourIndex` == agent seat on every call; IS_FIRST: seat 0 chooses in rounds 1 and 3, seat 1 in round 2; YES ⇒ chooser goes first (10/10), NO ⇒ second (0/11); 0 game-level draws |
| Harness smoke via `tools/tournament.py` (subprocess agent hosts; `raw/smoke/`) | 40 BO3 (cand 16, ctrl 16, h2h 8) | 0 failures, 0 BO3-rule mismatches, exports OK, `--stage screen` correctly refused (4 matches/cell < 10). Fixture scores are plumbing output, NOT strength evidence |
| Mission 3 test suite on the real engine | 77 tests | **77 passed, 0 skipped** (130 s). Mission 5's sandbox had no engine: 55 passed / 22 skipped |
| This package's tests | 32 | canonical stats 21 · tournament driver incl. non-skippable engine-required tests 7 · mirror check on real observations 4 |
| Hard-failure demonstration | system python without the engine | `tournament.py run` → exit 5, no output file |
Five local replays were opened; schema in `REPLAY_SCHEMA.md` (LOCAL fixture replays, not official).

## Failure modes / uncertainty
Fixture agents exercise plumbing only. Equivalence of this engine to the live R2 package is UNKNOWN. No game-level draw occurred in 78 real matches, so the 1W-0L-2D match-win branch is verified by source reading, not by an observed case. The mirror check cannot test agents that call `search_begin` (opaque `search_begin_input`).
