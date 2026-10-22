# Experiment Protocol

## Before a run

1. Pin baseline and candidate SHA-256 hashes.
2. State one hypothesis and one primary outcome.
3. Freeze the opponent pool, decklists, seeds and number of games (or the valid sequential stopping rule).
4. Balance seats and define how draws, crashes, illegal actions and timeouts are counted.
5. Record the correct evaluation unit for the competition: game-level outcomes when ladder episodes are individual games; match-level/BO3 summaries only as a separate diagnostic if those are locally used.

## During a run

- Write each game exactly once to a CSV/JSONL with candidate hash, baseline hash, opponent, seed, seat, result, action count, fallback/errors and elapsed time.
- Do not stop early because an interim score looks favorable.
- Do not discard failed games; report them separately and apply the predeclared failure policy.
- Keep exploratory screening separate from confirmatory tests.

## After a run

- Verify row count against the runner's final count.
- Recalculate the primary metric directly from the write-once records.
- Report absolute counts and confidence intervals.
- Compare all opponent/seat cells; don't pool away a decisive matchup regression.
- Make the promotion decision before a Kaggle upload.

## Runtime pass != strategic pass

A package that loads, passes `BattleStart`, returns legal actions, and has no fallback only clears the operational gate. Promotion needs credible performance gains against the frozen test pool. Live rating changes require the exact active submission/hash and episode history; filenames alone are not enough.
