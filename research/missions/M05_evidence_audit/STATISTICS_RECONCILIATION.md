# STATISTICS_RECONCILIATION — Phase E (the only phase executable without live data)

## Verified here (Python 3.12.3; 3.13 not available, so the brief's "3.13" claim is UNTESTED here)
- Mission 4 `test_eval_stats.py`: 9 passed. Mission 3 suite: **55 passed, 22 skipped** (all 22 engine-backed: `kaggle-environments not installed`); Mission 3's reported "77 passing" is therefore not reproduced without the engine. Mission 3 `--selftest`: OK.
- New canonical module `tools/ptcg_eval_canonical.py` + `tests/test_canonical.py`: **21 passed**; `tools/mc_coverage.py` output in `MC_COVERAGE_OUTPUT.txt`. VERIFIED, synthetic data only; says nothing about agent strength.

## Conflicts found between the two modules
| Issue | Mission 3 | Mission 4 | Canonical |
|---|---|---|---|
| Inferential unit | game | game | **BO3 match** (games = diagnostics only) |
| "300/500/1000" | 1000 = matches per arm, total | 300/500 = games per opponent | **always MATCHES; `n_total_per_arm` (exact) and `min_per_cell` (opponent×seat) are separate parameters** |
| Seat | cell key, summed away in gate | recorded, only balance-checked | stratum with fixed 0.5 weights; empty seat cell raises |
| Alpha | one-sided 0.025 stated | 1.96 labelled "97.5% one-sided" | one-sided 0.025 ≡ lower end of 95% two-sided CI; power 0.80 |
| Draws | binary counts only | Wilson on W+½D (binomial variance) | Wilson on points: conservative (Var ≤ p(1−p)); P(win) CI separate (`win_prob_ci`) |
| Failures | missing cell raises | missing cell KeyError | `cand_fail`=loss (flagged); `opp_fail`/`infra_fail` excluded and counted; >2% excluded fails gate; empty cell raises |
| SPRT | sign bug fixed | none | Mission 3 fix kept, regression test added; early-rejection only |
| Peeking | not enforced | not enforced | `confirm()` raises `DesignError` unless data are fresh and N equals the pre-registered N |
| Python | ≥3.9 | ≥3.10; `binomialvariate` needs 3.12 | stdlib ≥3.9 (tested 3.12 only) |
Mission 4's per-arm pooled sample-size function treated games as independent trials and is not retained. Legacy modules are in `legacy/` for reference only; the canonical module is the single gate.

## Key numbers (matches; normal approx., one-sided α=0.025, power 0.80)
Per arm to detect a difference in mean match score: +3pp ≈ 4,361; +4pp ≈ 2,453; +5pp ≈ 1,570; +8pp ≈ 614 (worst-case SD 0.5; with draws SD≈0.42, +5pp ≈ 1,108).
MDES at 300 / 500 / 1,000 / 3,000 matches per arm: ±11.4 / 8.9 / 6.3 / 3.6 pp.
**Consequence:** Mission 3's confirm gate of 1,000 BO3 matches per arm has only ~60% power for a true +5pp (MC: 0.60–0.61) and ~0.78–0.79 only at 1,500. A +3pp improvement is not confirmable at any plausible budget.

## Monte Carlo (3,000 trials, multinomial W/D/L, 3 opponents × 2 seats)
Coverage of the diff interval: 0.94–0.97 across n=600–3,000 and draw prob 0/0.15 (nominal 0.95; slightly conservative with draws). False-positive rate at true Δ=0: 0.014–0.028 (nominal 0.025). Weights are treated as fixed; uncertainty in the opponent panel is not included, and official BO3 draw/tiebreak semantics are UNKNOWN (the game→match rule in the code is an labelled ASSUMPTION).

## Still open
Opponent weights cannot be set without a real census (CURRENT_META is REPORTED_NOT_CURRENT). The ladder's own rating uncertainty cannot be tied to match counts without the official rated unit.
