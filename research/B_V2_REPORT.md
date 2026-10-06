# B-v2 report: what was tried against frozen B-v1, what was kept, and what that does and does not mean

**Bottom line.** No candidate beat B-v1 by the superiority bar. The shipped archive (`submission_Bv2_sym8.tar.gz`) is B-v1 plus
one pinned setting, test-time symmetry averaging (`sym_k=8`). It is a measured *non-inferior, decision-different* variant,
**not** a demonstrated upgrade. Pooled over three matchups, 1,500 games per arm: delta = +0.008, 95% CI [-0.024, +0.040].

## 1. Exact change from B-v1
One line in `main.py`: `AGENT_KWARGS` gains `'sym_k': 8`. Net (`sa/policy_net.npz`, md5 `4790c469...`, repo name `policy_v5_s2`),
`deck.csv` (md5 `f66d170b...`), `cg/` and the file list are byte-identical to B-v1. At each decision the net's option probabilities are
averaged over 8 relabellings of bench-slot numbers (`sa/symavg.py`), because the net demonstrably reads slot numbers that carry no game
meaning (repo EVIDENCE 8bt: 16.9% of decisions unstable under a relabelling). Measured here: B-v1 and B-v2 choose differently on
**8.21%** of 2,667 selects on identical observations (repo reports 8.36%). Cost: mean decision latency 0.5 ms -> 3.3 ms, worst single decision 198 ms,
thinking pool never below 599.8 s.

## 2. Benchmark (same session as the baseline, seat-alternating, Grimmsnarl piloted by the candidate; rule-agent opponents)
Confirmation, 500 games per arm per cell. P0 = goes first (seat 0 is always asked IS_FIRST and both agents choose to go first).

| cell | arm | win rate [Wilson 95%] | P0 | P1 | turns | selects | lat ms | fallbacks / net-missing | delta vs B-v1 [95%] |
|---|---|---|---|---|---|---|---|---|---|
| Abomasnow (target) | B-v1 | 0.438 [0.40, 0.48] | 0.52 | 0.36 | 11.2 | 116 | 0.46 | 0 / 0 | |
| | B-v2 | 0.472 [0.43, 0.52] | 0.55 | 0.40 | 10.9 | 113 | 3.27 | 0 / 0 | +0.034 [-0.028, +0.096] |
| Dragapult (control, ~38% of ladder) | B-v1 | 0.819 [0.78, 0.85] | 0.84 | 0.80 | 11.8 | 169 | 0.50 | 0 / 0 | |
| | B-v2 | 0.825 [0.79, 0.86] | 0.86 | 0.79 | 11.5 | 168 | 3.34 | 0 / 0 | +0.006 [-0.041, +0.053] |
| v10 (control, public LB-950 rule agent, no MCTS) | B-v1 | 0.615 [0.57, 0.66] | 0.63 | 0.60 | 11.6 | 157 | 0.48 | 0 / 0 | |
| | B-v2 | 0.600 [0.56, 0.64] | 0.65 | 0.55 | 11.4 | 154 | 3.31 | 0 / 0 | -0.015 [-0.076, +0.046] |
| **pooled (inverse variance)** | | | | | | | | | **+0.008 [-0.024, +0.040]** |

Every delta interval contains 0. Superiority is **not** shown.

## 3. Why it was shipped anyway (and the limits of that reason)
* Its pooled estimate is positive both here (+0.008) and in the repo's own 2,000-game mirror (0.513 [0.492, 0.535], EVIDENCE 8bu, also a null).
  The worst single cell here is -0.015 (v10). Every other candidate I tested had a negative point estimate in its target cell or a clearly worse cell.
* No cell is worse than -0.015; pooled lower bound -0.024. A non-inferiority margin of -0.03 is met, **but I chose that margin after seeing
  the numbers, so treat it as descriptive, not pre-registered.**
* The repo's own ladder record (HANDOFF, EVIDENCE 8ak) says decision-identical agents read 63-87 points apart and the shown score is the
  better of two active submissions. A decision-different, non-inferior agent therefore buys an independent ladder draw at ~zero idea risk.
  That is a variance hedge. Expect the same strength as B-v1, not more.

## 4. Candidates rejected (all vs B-v1, same session, Abomasnow, n=300 per arm unless noted)
| candidate | win rate | delta vs B-v1 [95%] | verdict |
|---|---|---|---|
| all three targeting rules on (chip + spread + counter-source) | 0.440 | -0.047 [-0.126, +0.033] | rejected |
| chip-target rule only | 0.420 | -0.067 [-0.146, +0.013] | rejected |
| Boss's Orders gust + target ranking | 0.423 | -0.063 [-0.143, +0.016] | rejected |
| "attack beats drag" Boss veto (lethal-dominance rule) | 0.457 | -0.020 [-0.100, +0.060] | rejected |
| Boss veto when nothing is KO-able | 0.427 | -0.050 [-0.130, +0.030] | rejected |
| drag ranking only | 0.450 | -0.027 [-0.106, +0.053] | rejected |
| route to lean net A | 0.347 (v10: 0.440 vs B-v1 0.615) | far worse | rejected |
| B+A ensemble (earlier run, n=200) | 0.435 / v10 0.620 / head-to-head 0.517 [0.45, 0.59] | within noise | rejected |
| 7 Grimmsnarl deck variants under the B-v1 net (no paired baseline; B-v1 in-session 0.477-0.487) | 0.373-0.467 | none above | rejected |

Pattern: every override that moves the policy off its learned choice has a negative point estimate. This matches the repo's own 35-day record,
which I did **not** re-run: one-ply value lookahead 0/10 and 0.158 (E20, E22), turn sequencer about -89 Elo, policy iteration 0.501 -> 0.481 (E27),
and "deviating from the clone costs -0.389; the best evaluator recovers 12%" (EVIDENCE 8cf).

## 5. What the data says about the target matchup
* **B-v1 vs Mega Abomasnow is about 0.45, not 0.405**: pooled over 1,660 games across six runs, 95% CI about 0.43-0.48. The 0.405 read was a low draw.
  Single-run readings swung 0.40-0.49.
* Mechanism (300-game diagnostic, `bench/abomasnow_diag.py`): their attack discards 6 cards and does 100 damage per Water Energy among them
  (34 of 60 cards), so by my own arithmetic (hypergeometric, fresh 60-card deck) about 47% of hits do 400+ damage, more than the 320 HP of our largest Pokemon.
  In losses they attack 5.3 times per game (3.2 of them with that attack) over 12.3 turns; in wins 3.4 times over 9.3 turns. In losses the Mega is in
  play by turn 3.7 in 163 of 163 games; in wins it appears in only 96 of 137. Terminal-state classification (200 games): 54 of 94 wins end with
  their board empty (their deck holds only 10 Pokemon); deck-out is negligible (3 wins, 1 loss). The matchup is a high-variance race that rule overrides did not fix.
* This deck is the starter-kit sample; it is **not** in the Aug-17 ladder meta list quoted in the Fusic writeup (Dragapult 38.2%, Slowking 20.8%,
  Ogerpon 18.3%, ...), so even a real gain here would carry little ladder weight.
* **P1 is not an abnormal weakness.** B-v1's overall first-vs-second gap is 0.720 vs 0.652 (6.8 points), against about +8.3 points reported for a different agent in the Fusic writeup.

## 6. Verification of the shipped archive (extracted, loaded the way Kaggle loads `main.py`)
agent PolicyAgent | net live True | sym_k 8 | chip/spread/src False/False/False | wall True (inert, chip off) | deck 60 cards, 19 distinct ids |
40 games vs a trivial-legal opponent: 40/40 wins | decisions 2,667 | STATS calls 2,667, fallbacks 0, net_missing 0, first_error None |
latency mean 3.62 ms, max 66.7 ms | pool left min 599.8 s | size 4.6 MiB (builder cap 197.7 MiB) | Kaggle syntax check passed under Python 3.11.15.
The 40-game test shows the bundle loads and plays; it says nothing about strength.

## 7. Caveats that bear on how to read any ladder score
* **Your 781 is not this net's documented history.** Repo HANDOFF shows this exact net (`policy_v5_s2`, rules off) reading 1004.5 -> 1010.1 -> 1044.3 on the
  competition board, unsettled, and the repo warns that another agent drifted to 905 after reading 979, that identical tarballs read 958 vs 834 on boards of different size,
  and that identical agents read 63-87 apart. Either your build differs from the one the repo shipped, or the Playground board is on a different scale or has settled, or your read is a low draw.
  I cannot tell which from here. Diff your 781 bundle against this one before trusting it.
* Policy-vs-policy cells in my earlier grid varied more than their CIs (Ogerpon cell 0.53, 0.485, 0.49, 0.38). Cause unknown. None of the evidence above relies on them.
* Engine is unseeded; per-cell numbers drift between runs, which is why every comparison above is paired in one session.
* Not run: any new training (no GPU, no torch, no corpus here), vlook/sequencer/oracle (closed in the repo), Slowking and strong-Dragapult/Ogerpon opponents (none available).

## 8. The path that could move the ladder (needs Kaggle GPU)
`pipeline/kaggle_train_candidate.ipynb`: rating-cut sweep on rating-filtered demonstrators, which is the repo's own documented next step after its unfiltered
292,008-episode corpus lost to v5_s2 (0.440). Train at least 3 seeds per cut (the repo's seed-null is about 0.482, and v5_s2 was a best-of-seeds pick),
then gate end-to-end with `bench/ab.py` at 1,000 games per arm over Abomasnow plus two controls. Held-out loss or top-k is not evidence of strength.
Notebook was never executed; its training arguments come from the scripts' argparse, not a run.
