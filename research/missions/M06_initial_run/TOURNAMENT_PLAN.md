# TOURNAMENT_PLAN — pre-registered BEFORE any real-agent match (Mission 6, Phase D)
Status: **plan only for real agents (0 real-agent matches exist).** Driver, statistics and tests are built and validated on FIXTURE agents with the real engine.

## Estimand and metric (fixed now)
- Unit: one completed BO3 match. Games are diagnostics (`TOURNAMENT_GAMES.csv`), never inference units.
- Metric: field-weighted mean match score (W=1, D=0.5, L=0); candidate-minus-control difference on the same panel, independent arms, equal seat weights. Match win probability is reported separately (`win_prob_ci`). Draw semantics = the local-engine rule in `cabt.py` (VERIFIED by source and 78 real matches); the live-ladder rule is UNVERIFIED.
- Strata: opponent × agent seat (player index, constant through a BO3 in cabt). The harness alternates seat by match index, so each cell is balanced when its N is even. Game-level first-player is recorded, not stratified.
- Panel weights: from the official census if obtainable, otherwise a panel and weights declared in writing before running, plus an equal-weight sensitivity report. Never weights derived from how often an opponent happened to be sampled.
- Seeds: the native engine RNG is not seedable (Mission 3 finding), so paired-seed designs are invalid. Swapped-side play (both seat assignments of every pairing) is used instead. The harness job seed is recorded in the CSV and is NOT an engine seed.

## Stages (every N is in BO3 MATCHES)
| Stage | Purpose | N | Decision use |
|---|---|---|---|
| 0 smoke / safety | cold start, deck legality, illegal/fallback/latency per archive | ≥20 per compared pair (10 per seat); `submission_audit.py --matches 20` | safety only, never strength |
| 1 screen | exclude clear regressions, safety, rough effect | ≥200 per pairing (100 per seat); MDES ≈ 14 pp at 80% power | exclusion only (`ScreenRule`: field diff ≥ −3 pp, no catastrophic cell); passing proves nothing |
| 2 confirm | the only stage that can promote | **1,600 matches per arm** (≥ `n_per_arm(0.05)` = 1570), one-sided α 0.025, power 0.80 for +5 pp, FRESH matches, analysed once via `confirm(..., fresh_matches=True)` | promotion |
Early rejection only: SPRT on decisive head-to-head matches (H0 p=.50 vs H1 p=.45, α=β=.05) may stop a failing candidate; it can never stop a candidate for success. `confirm()` raises `DesignError` unless the analysable N equals the registered N and the data are flagged fresh.
Detectable effect by budget (worst-case SD 0.5; with draws SD≈0.42 it is about 15% smaller): 200/arm → 14.0 pp · 400 → 9.9 · 1,000 → 6.3 · 1,600 → 5.0 · 3,000 → 3.6. If 1,600 per arm is unaffordable, run the largest affordable fixed N, report the CI, and do **not** claim a small improvement.
Budget: wall time ≈ N_total × t_match ÷ workers. Fixture agents on 1 core ≈ 3 s/match; real agents are UNKNOWN. Measure t_match in Stage 0 and write the achievable N into this file before Stage 1.

## Failure treatment (canonical)
Candidate-caused failure (exception, illegal output, hang, non-DONE status of the candidate seat) = loss, flagged. Opponent/infrastructure failures are excluded but counted; more than 2% excluded fails the gate. An empty positively weighted cell raises.

## Order of play (exact agents first)
(1) R2 compat candidate vs Bv1, vs B2, vs Bv2 (both seats) → (2) choose the best-performing available control from stage 1 → (3) one controlled candidate vs that control. One change per candidate.

## Promotion gate (predeclared)
All of: safety gate PASS with zero unexplained candidate faults across all stages; stage 1 not rejected; stage-2 lower bound of the field-score difference > 0 at the registered N; no confirmed regression (Newcombe upper bound < −0.05) against any opponent with weight ≥5% and ≥50 matches per arm; seat-stratified deltas reported (diagnostic); exact archive SHA-256 recorded; baseline archive untouched. Otherwise keep the baseline; a large stage-1 result at insufficient N is a labelled screening lead only.

## Commands (real engine required; the driver exits 5 without it)
```bash
python tools/tournament.py run --cand DIR_A --control DIR_B --h2h 100 --opp label=DIR:weight --matches 100 \
       --workers 1 --deadline 1500 --out raw/run.jsonl --outdir results/ --stage screen
python tools/tournament.py export --results raw/run.jsonl --weights label=w --outdir results/ --stage confirm --fresh   # once, at the registered N
python tools/mirror_check.py capture --main DIR/main.py --out obs.json --games 6 && python tools/mirror_check.py check --main DIR/main.py --obs obs.json
```
