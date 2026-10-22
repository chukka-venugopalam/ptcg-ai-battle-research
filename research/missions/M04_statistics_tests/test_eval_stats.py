"""Tests for eval_stats.py. Run:  python3 test_eval_stats.py   (exit code 0 == all passed).

These tests exercise statistics code only. They do NOT touch the CABT engine, which is not
available in the sandbox that produced this file; engine-level tests are listed as PROPOSED in
INDEPENDENT_METHODS_RESEARCH.md and SELECTED_EXPERIMENT_PLAN.md.
"""
from __future__ import annotations

import math
import random
import sys

import eval_stats as es


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def test_wilson_known_values() -> None:
    lo, hi = es.wilson_interval(50, 100)
    assert approx(lo, 0.4038, 5e-4) and approx(hi, 0.5962, 5e-4), (lo, hi)
    lo, hi = es.wilson_interval(0, 10)
    assert lo == 0.0 and approx(hi, 0.2775, 5e-4), (lo, hi)
    lo, hi = es.wilson_interval(10, 10)
    assert approx(lo, 0.7225, 5e-4) and hi == 1.0, (lo, hi)
    # fractional successes (draws as half points) must be accepted
    lo, hi = es.wilson_interval(50.5, 100)
    assert lo < 0.505 < hi


def test_wilson_coverage_monte_carlo() -> None:
    rng = random.Random(123)
    for p in (0.45, 0.55, 0.65):
        n, trials, covered = 300, 6000, 0
        for _ in range(trials):
            k = rng.binomialvariate(n, p)
            lo, hi = es.wilson_interval(k, n)
            covered += lo <= p <= hi
        cov = covered / trials
        assert 0.93 <= cov <= 0.97, (p, cov)


def _records(opp: str, seat0: str, seat1: str):
    out = []
    for seat, spec in ((0, seat0), (1, seat1)):
        for ch in spec:
            out.append(es.GameRecord(opp, seat, ch))
    return out


def test_summarize_counts_and_seats() -> None:
    seat0 = "W" * 30 + "L" * 18 + "D" * 2
    seat1 = "W" * 25 + "L" * 25
    s = es.summarize(_records("A", seat0, seat1))["A"]
    assert (s.n, s.wins, s.losses, s.draws) == (100, 55, 43, 2)
    assert approx(s.score, 0.56, 1e-12)
    assert s.n_seat0 == 50 and s.n_seat1 == 50 and s.seat_balanced
    assert approx(s.score_seat0, 0.62, 1e-12) and approx(s.score_seat1, 0.50, 1e-12)
    unbalanced = es.summarize(_records("B", "W" * 10, "L" * 14))["B"]
    assert not unbalanced.seat_balanced


def test_bad_records_rejected() -> None:
    for bad in (("x", 2, "W"), ("x", 0, "Z")):
        try:
            es.GameRecord(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad record {bad}")


def test_pooled_uses_declared_weights_not_sample_counts() -> None:
    a = es.summarize(_records("A", "W" * 40 + "L" * 10, "W" * 40 + "L" * 10))["A"]      # 0.80, n=100
    b = es.summarize(_records("B", "W" * 10 + "L" * 40, "W" * 10 + "L" * 40))["B"]      # 0.20, n=100
    # B sampled 4x as often must NOT change the pooled score when weights are declared
    b_many = es.summarize(_records("B", "W" * 40 + "L" * 160, "W" * 40 + "L" * 160))["B"]  # 0.20, n=400
    p1 = es.pooled_score({"A": a, "B": b}, {"A": 0.5, "B": 0.5})
    p2 = es.pooled_score({"A": a, "B": b_many}, {"A": 0.5, "B": 0.5})
    assert approx(p1.score, 0.5, 1e-12) and approx(p2.score, 0.5, 1e-12)
    assert p2.se < p1.se  # more games on B -> tighter pooled interval
    try:
        es.pooled_score({"A": a}, {"A": 0.5, "B": 0.5})
    except KeyError:
        pass
    else:
        raise AssertionError("missing weighted opponent was not rejected")


def test_required_games_reference_values() -> None:
    n = es.required_games_head_to_head(0.04)
    assert 1200 <= n <= 1250, n
    assert es.required_games_head_to_head(0.02) > es.required_games_head_to_head(0.04) * 3
    m = es.required_games_per_arm(0.55, 0.04)
    assert 2350 <= m <= 2450, m


def _synth(rng, opp_names, n_each, p_by_opp):
    recs = []
    for opp in opp_names:
        for i in range(n_each):
            seat = i % 2
            win = rng.random() < p_by_opp[opp]
            recs.append(es.GameRecord(opp, seat, "W" if win else "L"))
    return es.summarize(recs)


def test_screen_and_confirm_logic() -> None:
    rng = random.Random(7)
    opps = ["o1", "o2", "o3", "o4", "o5"]
    w = {o: 1.0 for o in opps}
    base = {o: 0.52 for o in opps}
    better = {o: 0.58 for o in opps}  # true +6pp everywhere
    ctrl_s = _synth(rng, opps, 300, base)
    cand_s = _synth(rng, opps, 300, better)
    sd = es.screen(cand_s, ctrl_s, w)
    assert sd.passed, sd.reasons
    ctrl_c = _synth(rng, opps, 500, base)
    cand_c = _synth(rng, opps, 500, better)
    cd = es.confirm(cand_c, ctrl_c, w)
    assert cd.passed, cd.reasons
    # a null candidate must fail confirmation
    null_c = _synth(random.Random(8), opps, 500, base)
    nd = es.confirm(null_c, ctrl_c, w)
    assert not nd.passed
    # too few games must be flagged even if the point estimate looks great
    small = _synth(rng, opps, 40, better)
    sd2 = es.screen(small, ctrl_s, w)
    assert not sd2.passed and any("games" in r for r in sd2.reasons)
    # a catastrophic regression against one opponent blocks confirmation
    mixed = dict(better)
    mixed["o3"] = 0.30
    cand_m = _synth(random.Random(9), opps, 500, mixed)
    md = es.confirm(cand_m, ctrl_c, w)
    assert not md.passed and any("o3" in r for r in md.reasons)


def test_winners_curse_and_confirmation_error_rate() -> None:
    res = es.simulate_screen_false_positive(num_variants=10, n_per_opponent=300, num_opponents=5,
                                            threshold=0.02, trials=1500, seed=11)
    # with ten identical (null) variants the "best" one clears +2pp about half the time
    assert 0.35 <= res["p_best_ge_threshold"] <= 0.65, res
    assert 0.014 <= res["mean_best_observed_delta"] <= 0.026, res
    single = es.simulate_screen_false_positive(1, 300, 5, threshold=0.02, trials=1500, seed=12)
    assert single["p_best_ge_threshold"] < res["p_best_ge_threshold"]
    fp = es.simulate_confirm_false_positive(n_per_opponent=500, num_opponents=5, trials=4000, seed=13)
    assert 0.010 <= fp <= 0.045, fp  # nominal one-sided 2.5%


def test_holm() -> None:
    assert es.holm([0.001, 0.04, 0.03]) == [True, False, False]
    assert es.holm([0.2, 0.3]) == [False, False]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
