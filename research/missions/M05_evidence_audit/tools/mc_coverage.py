#!/usr/bin/env python3
"""Monte Carlo operating characteristics of ptcg_eval_canonical under the BO3-MATCH unit (needs numpy).
Matches in a cell ~ multinomial(win, draw, loss).  Prints coverage, false-positive and power for fixed-N confirm."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ptcg_eval_canonical as e
from ptcg_eval_canonical import Cell

W = {"A": 0.4, "B": 0.3, "C": 0.3}
rng = np.random.default_rng(20261016)

def arm(n_total, pw, pd, shift=0.0):
    per = n_total // (len(W) * 2)
    cells = {}
    for o in W:
        for s in (0, 1):
            pwin = min(max(pw + shift, 0.01), 0.98 - pd)
            w, d, l = rng.multinomial(per, [pwin, pd, 1 - pwin - pd])
            cells[(o, s)] = Cell(int(w), int(d), int(l))
    return cells

def run(n, pd, delta, trials=3000, pw0=None):
    pw0 = (0.5 - 0.5 * pd) if pw0 is None else pw0    # true control score 0.5
    cover = fp = hit = 0
    truth_c = pw0 + 0.5 * pd
    for _ in range(trials):
        c, k = arm(n, pw0, pd, delta), arm(n, pw0, pd, 0.0)
        r = e.diff_ci(c, k, W)
        true_d = (min(max(pw0 + delta, .01), .98 - pd) - pw0)
        cover += r["lo"] <= true_d <= r["hi"]
        hit += r["lo"] > 0
    return cover / trials, hit / trials

print("n_per_arm(matches), draw_prob, true_delta_in_win_prob -> coverage95, P(lower>0)  [delta=0 row = false-positive rate]")
for n in (600, 1000, 1500, 3000):
    for pd in (0.0, 0.15):
        for delta in (0.0, 0.05):
            cov, pos = run(n, pd, delta)
            print(f"n={n:5d} pd={pd:.2f} delta={delta:.2f}  coverage={cov:.3f}  P(lo>0)={pos:.3f}")
