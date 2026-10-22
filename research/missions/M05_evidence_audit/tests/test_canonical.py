import math, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import ptcg_eval_canonical as e
from ptcg_eval_canonical import Cell, MatchRecord, tally, DesignError, MissingCellError

W = {"A": 0.6, "B": 0.4}

def mk(spec, prefix="m", status="ok"):
    """spec: {(opp,seat): (w,d,l)} -> MatchRecords"""
    out, i = [], 0
    for (o, s), (w, d, l) in spec.items():
        for r, c in (("W", w), ("D", d), ("L", l)):
            for _ in range(c):
                out.append(MatchRecord(f"{prefix}{i}", o, s, r, status)); i += 1
    return out

def full(w, d, l, prefix):
    return mk({(o, s): (w, d, l) for o in W for s in (0, 1)}, prefix)

# ---- units / definitions
def test_match_from_games():
    assert e.match_result_from_games(["W", "W"]) == "W"
    assert e.match_result_from_games(["L", "W", "L"]) == "L"
    assert e.match_result_from_games(["W", "L", "D"]) == "D"
    with pytest.raises(ValueError): e.match_result_from_games(["W", "X"])

def test_duplicate_match_id_rejected():
    r = MatchRecord("x", "A", 0, "W")
    with pytest.raises(ValueError): tally([r, r])

def test_games_never_enter_inference():
    g = e.GameRecord("m1", 0, 0, 0, "W")
    with pytest.raises(Exception): tally([g])  # GameRecord is not a MatchRecord

def test_draw_points_and_win_prob_differ():
    c = Cell(w=40, d=20, l=40)
    assert c.score == 0.5 and c.points == 50
    p, lo, hi = e.win_prob_ci(c)
    assert p == 0.4 and hi < 0.5 < e.wilson(c.points, c.n)[1]  # P(win) != mean score

def test_wilson_edges():
    assert e.wilson(0, 10)[0] == 0 and e.wilson(10, 10)[1] == 1
    with pytest.raises(ValueError): e.wilson(11, 10)
    with pytest.raises(ValueError): e.wilson(0, 0)

# ---- failures / missing
def test_cand_fail_counts_as_loss_and_excluded_reported():
    t = tally([MatchRecord("1", "A", 0, "W", "cand_fail"), MatchRecord("2", "A", 0, "W", "opp_fail"),
               MatchRecord("3", "A", 0, "W", "infra_fail"), MatchRecord("4", "A", 0, "W")])
    assert t.cells[("A", 0)] == Cell(1, 0, 1) and t.n_cand_fail == 1 and t.n_excluded == 2
    assert abs(t.excluded_frac - 0.5) < 1e-12

def test_missing_cell_raises_not_renormalises():
    t = tally(mk({("A", 0): (5, 0, 5)}))
    with pytest.raises(MissingCellError): e.field_score_ci(t.cells, W)

def test_seat_missing_raises():
    t = tally(mk({("A", 0): (5, 0, 5), ("B", 0): (5, 0, 5)}))
    with pytest.raises(MissingCellError): e.field_score_ci(t.cells, W)

def test_weights_predeclared_not_sampling_driven():
    a = tally(mk({("A", 0): (90, 0, 10), ("A", 1): (90, 0, 10), ("B", 0): (1, 0, 1), ("B", 1): (1, 0, 1)}))
    r = e.field_score_ci(a.cells, W)
    assert abs(r["score"] - (0.6 * 0.9 + 0.4 * 0.5)) < 1e-9

# ---- intervals
def test_field_ci_contains_estimate_and_narrows():
    small = e.field_score_ci(tally(full(6, 2, 4, "s")).cells, W)
    big = e.field_score_ci(tally(full(60, 20, 40, "b")).cells, W)
    assert small["lo"] < small["score"] < small["hi"]
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])

def test_two_seat_strata_match_pooled_newcombe_approximately():
    # seats are independent halves of the same data: stratified MOVER must be close to (not identical to) pooled Newcombe
    a, b = Cell(60, 0, 40), Cell(45, 0, 55)
    ca = {("A", 0): Cell(30, 0, 20), ("A", 1): Cell(30, 0, 20)}
    cb = {("A", 0): Cell(22, 0, 28), ("A", 1): Cell(23, 0, 27)}
    r = e.diff_ci(ca, cb, {"A": 1.0})
    d, lo, hi = e.newcombe_cell_diff(a, b)
    assert abs(r["diff"] - d) < 1e-12 and abs(r["lo"] - lo) < 0.01 and abs(r["hi"] - hi) < 0.01

def test_diff_sign():
    r = e.diff_ci(tally(full(70, 0, 30, "c")).cells, tally(full(40, 0, 60, "k")).cells, W)
    assert r["diff"] > 0.29 and r["lo"] > 0

# ---- planning (all in MATCHES)
def test_planning_numbers():
    assert e.n_per_arm(0.05) == 1570 and e.n_per_arm(0.03) == 4361
    assert abs(e.mdes(1000) - 0.0626) < 5e-4
    assert e.n_per_arm(0.05, sd=0.42) < e.n_per_arm(0.05)   # draws shrink SD
    assert e.mdes(e.n_per_arm(0.05)) <= 0.05 + 1e-3

def test_bo3_game_transform():
    assert abs(e.bo3_win_from_game_win(0.5) - 0.5) < 1e-12
    assert e.bo3_win_from_game_win(0.6) == pytest.approx(0.648)

# ---- SPRT regression (legacy sign bug)
def test_sprt_sign_regression():
    llr, lo, up, dec = e.sprt(10, 100, 0.5, 0.45)   # 10/100 strongly favours LOWER p
    assert llr > 0 and dec == "accept H1"           # legacy returned accept H0
    assert e.sprt(70, 100, 0.5, 0.45)[3] == "accept H0"  # legacy returned accept H1
    assert e.sprt(70, 100, 0.5, 0.55)[3] == "accept H1"
    assert e.sprt(50, 100, 0.5, 0.55)[3] == "continue"
    with pytest.raises(ValueError): e.sprt(1, 2, 0.5, 0.5)
    with pytest.raises(ValueError): e.sprt(3, 2, 0.5, 0.6)

# ---- gates
def test_confirm_requires_fresh_and_exact_n():
    c, k = tally(full(70, 0, 30, "c")), tally(full(40, 0, 60, "k"))
    with pytest.raises(DesignError): e.confirm(c, k, W, e.ConfirmRule(n_total_per_arm=400))
    with pytest.raises(DesignError): e.confirm(c, k, W, e.ConfirmRule(n_total_per_arm=401), fresh_matches=True)

def test_confirm_passes_clear_win_and_fails_null():
    r = e.ConfirmRule(n_total_per_arm=400, min_per_cell=20, min_opp_n=50)
    assert e.confirm(tally(full(70, 0, 30, "c")), tally(full(40, 0, 60, "k")), W, r, True).passed
    assert not e.confirm(tally(full(50, 0, 50, "c")), tally(full(50, 0, 50, "k")), W, r, True).passed

def test_confirm_flags_excluded_and_regression():
    r = e.ConfirmRule(n_total_per_arm=400)
    cs = full(70, 0, 30, "c")[:]
    spec = {(o, s): (70, 0, 30) for o in W for s in (0, 1)}
    spec[("B", 0)] = (5, 0, 95); spec[("B", 1)] = (5, 0, 95)
    cand = tally(mk(spec, "c"))
    d = e.confirm(cand, tally(full(40, 0, 60, "k")), W, r, True)
    assert any("regression vs B" in x for x in d.reasons)
    ex = mk({("A", 0): (1, 0, 0)}, "x", "infra_fail") * 0 + [MatchRecord(f"e{i}", "A", 0, "W", "infra_fail") for i in range(20)]
    cand2 = tally(full(70, 0, 30, "d") + ex)
    d2 = e.confirm(cand2, tally(full(40, 0, 60, "k2")), W, r, True)
    assert not d2.passed and any("excluded fraction" in x for x in d2.reasons)

def test_screen_is_exclusion_only():
    c, k = tally(full(30, 0, 30, "c")), tally(full(30, 0, 30, "k"))
    assert e.screen(c, k, W).passed                      # equal arms pass the exclusion screen (proves nothing)
    assert not e.screen(tally(full(10, 0, 50, "c")), k, W).passed
    assert not e.screen(tally(full(2, 0, 3, "c")), k, W).passed  # too few per cell

# ---- Monte Carlo (small, deterministic; full run in tools/mc_coverage.py)
def _draw(rnd, n, pw, pd):
    w = d = 0
    for _ in range(n):
        u = rnd.random(); w += u < pw; d += pw <= u < pw + pd
    return Cell(w, d, n - w - d)

def test_mc_field_coverage_with_draws():
    import random
    rnd = random.Random(7); hits = trials = 0
    pw, pd = 0.42, 0.16; truth = pw + 0.5 * pd
    for _ in range(300):
        cells = {(o, s): _draw(rnd, 60, pw, pd) for o in W for s in (0, 1)}
        r = e.field_score_ci(cells, W); hits += r["lo"] <= truth <= r["hi"]; trials += 1
    assert hits / trials >= 0.93   # conservative interval; nominal 0.95

def test_mc_null_false_positive_rate():
    import random
    rnd = random.Random(11); fp = 0; trials = 400
    for _ in range(trials):
        c = {(o, s): _draw(rnd, 50, 0.45, 0.10) for o in W for s in (0, 1)}
        k = {(o, s): _draw(rnd, 50, 0.45, 0.10) for o in W for s in (0, 1)}
        fp += e.diff_ci(c, k, W)["lo"] > 0
    assert fp / trials <= 0.04     # nominal one-sided 0.025
