#!/usr/bin/env python3
"""ptcg_eval_stats.py v2 -- statistics for PTCG-ladder candidate evaluation (stdlib only, Python >= 3.9).

Supersedes BOTH duplicate modules from the research packages:
  * reserch2.zip  files.zip/ptcg_eval_stats.py   (two-sided sample sizes; sprt() sign bug)
  * pokemon_reserch1.zip repro_package/stats_gates.py (one-sided required_games)
See HARNESS_AUDIT_AND_TESTS.md for the defect list.  Changes vs the legacy modules:

  1. sprt(): the legacy code negated the log-likelihood ratio when p1 < p0.  A Wald SPRT statistic
     is log L(p1)/L(p0) for EITHER direction; only the data decide the sign.  Fixed + tested.
  2. Sidedness is explicit (`sides=`).  Pre-registered convention (PREREG below): total alpha = 0.05 split
     over two tails  <=>  "lower end of a 95% two-sided interval must exceed 0"  <=>  one-sided alpha = 0.025.
     The legacy stats_gates.required_games used one-sided alpha = 0.05 (617 games for +5pp) while its own gate
     used a 95% Wilson lower bound (one-sided 0.025), so the gate was under-powered vs its advertised 80%.
  3. weighted_rate() no longer silently renormalises weights around archetypes with no completed games.
     A missing positive-weight cell raises MissingCellError (or is reported as missing_mass if allow_missing=True).
  4. New: weighted_rate_ci / weighted_diff_ci -- stratified MOVER (Zou & Donner 2008; Newcombe's hybrid-score
     interval generalised to a weighted sum of independent proportions).  The CI for ONE candidate's field rate and
     the CI for the DIFFERENCE candidate-minus-control are different functions on purpose.
  5. New: gate_screen / gate_confirm -- the pre-registered gates as executable code.

It does NOT know anything about the cabt engine and never touches Kaggle.
    python ptcg_eval_stats.py --selftest     # legacy cross-checks + new regression checks
    python ptcg_eval_stats.py --tables       # reproduces the power / noise tables quoted in the research reports
"""
from __future__ import annotations

import argparse
import math
import random
from statistics import NormalDist
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

_N = NormalDist()
Z95 = _N.inv_cdf(0.975)

# Pre-registered convention (change only with a written statistical justification).
PREREG = dict(alpha_total=0.05, sides=2, power=0.80, ci_level=0.95,
              note="alpha_total=0.05 over two tails == one-sided 0.025 == 'lower end of 95% two-sided CI > 0'")


class MissingCellError(ValueError):
    """A positively-weighted stratum has no completed games; refusing to renormalise around it."""


# --------------------------------------------------------------------------- intervals / tests
def wilson(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or not (0 <= k <= n):
        raise ValueError("need n > 0 and 0 <= k <= n")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float, float]:
    """Pooled two-proportion z-test. Returns (diff, z, two-sided p)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be > 0")
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return p1 - p2, z, 2 * (1 - _N.cdf(abs(z)))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> Tuple[float, float]:
    """Newcombe (1998) hybrid-score CI for p1 - p2 (INDEPENDENT samples)."""
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def binom_two_sided_p(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of pmf <= observed pmf)."""
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0,1)")

    def logpmf(i: int) -> float:
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * math.log(p0) + (n - i) * math.log(1 - p0))
    lo = logpmf(k) + 1e-12
    return min(1.0, sum(math.exp(logpmf(i)) for i in range(n + 1) if logpmf(i) <= lo))


# --------------------------------------------------------------------------- sample size / MDES
def _za(alpha: float, sides: int) -> float:
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    return _N.inv_cdf(1 - alpha / sides)


def n_one_sample(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80, sides: int = 2) -> int:
    """Games needed to detect true win rate p1 vs a KNOWN p0.  `alpha` is the TOTAL type-I error
    (sides=2 splits it over two tails; sides=1 puts it all in one)."""
    za, zb = _za(alpha, sides), _N.inv_cdf(power)
    num = za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))
    return math.ceil((num / (p1 - p0)) ** 2)


def n_two_sample(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80, sides: int = 2) -> int:
    """Games PER ARM to detect p1 vs p2 with two INDEPENDENT arms (unpooled alternative variance)."""
    za, zb = _za(alpha, sides), _N.inv_cdf(power)
    pb = (p1 + p2) / 2
    num = za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / (p1 - p2)) ** 2)


def _mdes(n: int, fn, p0: float) -> float:
    lo, hi = p0 + 1e-6, 0.999
    for _ in range(80):
        mid = (lo + hi) / 2
        if fn(mid) <= n:
            hi = mid
        else:
            lo = mid
    return hi


def mdes_one_sample(n: int, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80, sides: int = 2) -> float:
    return _mdes(n, lambda p1: n_one_sample(p1, p0, alpha, power, sides), p0)


def mdes_two_sample(n_per_arm: int, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80, sides: int = 2) -> float:
    return _mdes(n_per_arm, lambda p1: n_two_sample(p1, p0, alpha, power, sides), p0)


def gate_power(n: int, p_true: float, min_rate: float = 0.55, p0: float = 0.5, z: float = Z95) -> float:
    """EXACT probability that the legacy stats_gates.gate_decision rule ("pooled rate >= min_rate AND Wilson lower
    bound > p0", n games) passes when the true win rate is p_true.  Exposes the legacy gate's real power."""
    total = 0.0
    for k in range(n + 1):
        if k / n >= min_rate and wilson(k, n, z)[0] > p0:
            logpmf = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                      + k * math.log(p_true) + (n - k) * math.log(1 - p_true))
            total += math.exp(logpmf)
    return total


# --------------------------------------------------------------------------- BO3 and rating scales
def bo3_from_game(p: float) -> float:
    """P(win best-of-3) if games are i.i.d. with win prob p (ignores first-player effects)."""
    return p * p * (3 - 2 * p)


def game_from_bo3(m: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bo3_from_game(mid) < m else (lo, mid)
    return (lo + hi) / 2


def elo_p(delta: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10 ** (-delta / scale))


def probit_p(delta: float, beta: float = 100.0) -> float:
    """TrueSkill-style: P(win) = Phi(delta / (sqrt(2) * beta))."""
    return _N.cdf(delta / (math.sqrt(2) * beta))


def rating_se(n: int, model: str = "elo400") -> float:
    """SE (rating points) of a performance rating from n games vs ~equal-rated opponents (p ~ 0.5). ASSUMED scales."""
    if model == "elo400":
        return 2 * (400 / math.log(10)) / math.sqrt(n)
    if model == "probit100":
        slope = _N.pdf(0) / (math.sqrt(2) * 100)
        return 0.5 / math.sqrt(n) / slope
    raise ValueError(model)


def episodes_to_resolve(gap: float, model: str = "elo400", z: float = Z95) -> int:
    """Episodes EACH of two submissions needs before a rating gap of `gap` points is ~z-sigma significant."""
    se1 = rating_se(1, model)
    return math.ceil((z * math.sqrt(2) * se1 / gap) ** 2)


# --------------------------------------------------------------------------- sequential testing
def _check_sprt_args(p0: float, p1: float, alpha: float, beta: float) -> None:
    if not (0 < p0 < 1 and 0 < p1 < 1) or p0 == p1:
        raise ValueError("need 0 < p0, p1 < 1 and p0 != p1")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha, beta must be in (0,1)")


def sprt(k: int, n: int, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.05):
    """Wald SPRT of H0: p = p0 against H1: p = p1, for p1 > p0 (improvement) OR p1 < p0 (regression).

    LLR = log L(p1)/L(p0).  Accept H1 when LLR >= log((1-beta)/alpha); accept H0 when LLR <= log(beta/(1-alpha)).
    The statistic is direction-agnostic: the LEGACY version negated it when p1 < p0, which made 10/100 wins
    'accept H0' for a 45% regression hypothesis (wrong) and 70/100 'accept H1' (wrong).  Returns
    (llr, lower_boundary, upper_boundary, decision)."""
    _check_sprt_args(p0, p1, alpha, beta)
    if not (0 <= k <= n):
        raise ValueError("need 0 <= k <= n")
    llr = k * math.log(p1 / p0) + (n - k) * math.log((1 - p1) / (1 - p0))
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    dec = "accept H1" if llr >= upper else "accept H0" if llr <= lower else "continue"
    return llr, lower, upper, dec


def sprt_expected_n(p0: float, p1: float, alpha: float = 0.05, beta: float = 0.05, under: str = "H1") -> float:
    """Wald's approximation of E[N] (ignores overshoot, so slightly optimistic)."""
    _check_sprt_args(p0, p1, alpha, beta)
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    p = p1 if under == "H1" else p0
    drift = p * math.log(p1 / p0) + (1 - p) * math.log((1 - p1) / (1 - p0))
    p_acc_h1 = (1 - beta) if under == "H1" else alpha
    return (p_acc_h1 * upper + (1 - p_acc_h1) * lower) / drift


def sprt_simulate(p_true: float, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.05,
                  n_max: int = 5000, n_sims: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo operating characteristics of the (truncated) SPRT."""
    _check_sprt_args(p0, p1, alpha, beta)
    rnd = random.Random(seed)
    up, lo = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    w, l = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    h1 = h0 = und = 0
    tot = 0
    for _ in range(n_sims):
        s = 0.0
        n = 0
        while n < n_max:
            n += 1
            s += w if rnd.random() < p_true else l
            if s >= up:
                h1 += 1
                break
            if s <= lo:
                h0 += 1
                break
        else:
            und += 1
        tot += n
    return dict(p_accept_h1=h1 / n_sims, p_accept_h0=h0 / n_sims, p_undecided=und / n_sims, mean_n=tot / n_sims)


# --------------------------------------------------------------------------- Bradley-Terry
def bradley_terry(wins: Dict[Tuple[str, str], float], pseudo: float = 0.5,
                  iters: int = 2000, tol: float = 1e-10) -> Dict[str, float]:
    """MM algorithm (Hunter 2004). wins[(i, j)] = number of times i beat j.
    `pseudo` adds that many wins to BOTH directions of every observed pair to keep the MLE finite.
    Returns Elo-scaled ratings (400*log10(strength)), centred at 0."""
    players = sorted({p for pair in wins for p in pair})
    w: Dict[Tuple[str, str], float] = {}
    for (i, j), v in wins.items():
        w[(i, j)] = w.get((i, j), 0.0) + v
    for (i, j) in list(w):
        w.setdefault((j, i), 0.0)
    for k in list(w):
        w[k] += pseudo
    s = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for i in players:
            wi = sum(v for (a, _b), v in w.items() if a == i)
            den = 0.0
            for j in players:
                if j == i:
                    continue
                nij = w.get((i, j), 0.0) + w.get((j, i), 0.0)
                if nij:
                    den += nij / (s[i] + s[j])
            new[i] = wi / den if den else s[i]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / g for p, v in new.items()}
        if max(abs(new[p] - s[p]) for p in players) < tol:
            s = new
            break
        s = new
    return {p: 400 * math.log10(v) for p, v in s.items()}


# --------------------------------------------------------------------------- stratified comparison (MH odds ratio)
def mantel_haenszel(strata: Sequence[Tuple[int, int, int, int]]) -> Dict[str, float]:
    """Stratified comparison of agent X vs agent Y across strata (e.g. opponent archetype x seat).
    Each stratum = (a, b, c, d): X wins, X losses, Y wins, Y losses.
    Returns MH common odds ratio (X vs Y), Robins-Breslow-Greenland 95% CI, CMH chi-square and p."""
    R = S = 0.0
    sPR = sPSQR = sQS = 0.0
    sa = sE = sV = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        Rk, Sk = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        R += Rk
        S += Sk
        sPR += P * Rk
        sPSQR += P * Sk + Q * Rk
        sQS += Q * Sk
        sa += a
        sE += (a + b) * (a + c) / n
        sV += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if R == 0 or S == 0:
        return {"or": float("nan"), "lo": float("nan"), "hi": float("nan"), "chi2": float("nan"), "p": float("nan")}
    or_mh = R / S
    var = sPR / (2 * R * R) + sPSQR / (2 * R * S) + sQS / (2 * S * S)
    se = math.sqrt(var)
    lo, hi = math.exp(math.log(or_mh) - Z95 * se), math.exp(math.log(or_mh) + Z95 * se)
    chi2 = (abs(sa - sE) - 0.5) ** 2 / sV if sV > 0 else float("nan")
    p = 2 * (1 - _N.cdf(math.sqrt(chi2))) if chi2 == chi2 else float("nan")
    return {"or": or_mh, "lo": lo, "hi": hi, "chi2": chi2, "p": p}


# --------------------------------------------------------------------------- field-weighted rates (strict)
Cell = Tuple[int, int]  # (wins, completed matches)


def _norm_weights(weights: Mapping[Hashable, float]) -> Dict[Hashable, float]:
    pos = {k: float(v) for k, v in weights.items() if v and v > 0}
    tw = sum(pos.values())
    if tw <= 0:
        raise ValueError("weights must contain at least one positive entry")
    return {k: v / tw for k, v in pos.items()}


def _check_cells(cells: Mapping[Hashable, Cell], w: Mapping[Hashable, float], allow_missing: bool, who: str):
    """Return (usable_keys, missing_keys).  Raise MissingCellError unless allow_missing."""
    missing = [k for k in w if k not in cells or cells[k][1] <= 0]
    if missing and not allow_missing:
        raise MissingCellError(f"{who}: no completed matches in positively-weighted cell(s) {missing!r}; "
                               "refusing to renormalise around them (failed cells must fail the gate)")
    for k in w:
        if k in cells:
            kk, nn = cells[k]
            if not (0 <= kk <= nn):
                raise ValueError(f"{who}: bad cell {k!r}: {cells[k]!r}")
    return [k for k in w if k not in missing], missing


def weighted_rate_ci(cells: Mapping[Hashable, Cell], weights: Mapping[Hashable, float],
                     z: float = Z95, allow_missing: bool = False) -> Dict[str, object]:
    """Field-weighted win rate  sum_a w_a p_a  with a stratified MOVER interval:
         L = est - sqrt(sum w_a^2 (p_a - l_a)^2),  U = est + sqrt(sum w_a^2 (u_a - p_a)^2)
    where (l_a, u_a) is the Wilson interval of cell a.  Cells are assumed independent.  Weights are treated as FIXED
    (conditional on the frozen opponent panel); uncertainty in the panel itself is NOT included."""
    w = _norm_weights(weights)
    keys, missing = _check_cells(cells, w, allow_missing, "weighted_rate_ci")
    mass = sum(w[k] for k in keys)
    est = lo_sq = hi_sq = 0.0
    n_total = 0
    for k in keys:
        kk, nn = cells[k]
        p, (l, u) = kk / nn, wilson(kk, nn, z)
        wk = w[k] / mass
        est += wk * p
        lo_sq += (wk * (p - l)) ** 2
        hi_sq += (wk * (u - p)) ** 2
        n_total += nn
    return dict(rate=est, lo=est - math.sqrt(lo_sq), hi=est + math.sqrt(hi_sq), n=n_total,
                missing=missing, missing_mass=1.0 - mass, incomplete=bool(missing))


def weighted_diff_ci(cand: Mapping[Hashable, Cell], ctrl: Mapping[Hashable, Cell],
                     weights: Mapping[Hashable, float], z: float = Z95, allow_missing: bool = False) -> Dict[str, object]:
    """Field-weighted DIFFERENCE  candidate - control  over INDEPENDENT samples (no seed pairing is possible: the
    native engine RNG is not seedable).  Stratified MOVER: with one stratum it equals Newcombe's hybrid-score CI.
        L = d - sqrt(sum w^2[(pC-lC)^2 + (uK-pK)^2]),  U = d + sqrt(sum w^2[(uC-pC)^2 + (pK-lK)^2])
    A stratum must be complete in BOTH arms or the call raises MissingCellError (unless allow_missing)."""
    w = _norm_weights(weights)
    kc, mc = _check_cells(cand, w, allow_missing, "candidate")
    kk_, mk = _check_cells(ctrl, w, allow_missing, "control")
    keys = [k for k in kc if k in kk_]
    missing = sorted(set(mc) | set(mk), key=str)
    mass = sum(w[k] for k in keys)
    pc = pk = lo_sq = hi_sq = 0.0
    for k in keys:
        a, na = cand[k]
        b, nb = ctrl[k]
        p1, p2 = a / na, b / nb
        l1, u1 = wilson(a, na, z)
        l2, u2 = wilson(b, nb, z)
        wk = w[k] / mass
        pc += wk * p1
        pk += wk * p2
        lo_sq += (wk ** 2) * ((p1 - l1) ** 2 + (u2 - p2) ** 2)
        hi_sq += (wk ** 2) * ((u1 - p1) ** 2 + (p2 - l2) ** 2)
    d = pc - pk
    return dict(diff=d, lo=d - math.sqrt(lo_sq), hi=d + math.sqrt(hi_sq), cand_rate=pc, ctrl_rate=pk,
                n_cand=sum(cand[k][1] for k in keys), n_ctrl=sum(ctrl[k][1] for k in keys),
                missing=missing, missing_mass=1.0 - mass, incomplete=bool(missing))


def weighted_diff_bootstrap(cand: Mapping[Hashable, Cell], ctrl: Mapping[Hashable, Cell],
                            weights: Mapping[Hashable, float], n_boot: int = 4000, seed: int = 0,
                            allow_missing: bool = False) -> Tuple[float, float, float]:
    """Alternative to MOVER: parametric Jeffreys-Beta bootstrap of both arms (independent). Returns (diff, lo, hi)."""
    w = _norm_weights(weights)
    kc, _ = _check_cells(cand, w, allow_missing, "candidate")
    kk_, _ = _check_cells(ctrl, w, allow_missing, "control")
    keys = [k for k in kc if k in kk_]
    mass = sum(w[k] for k in keys)
    rnd = random.Random(seed)
    pt = sum(w[k] / mass * (cand[k][0] / cand[k][1] - ctrl[k][0] / ctrl[k][1]) for k in keys)
    boots: List[float] = []
    for _ in range(n_boot):
        v = 0.0
        for k in keys:
            a, na = cand[k]
            b, nb = ctrl[k]
            v += w[k] / mass * (rnd.betavariate(a + .5, na - a + .5) - rnd.betavariate(b + .5, nb - b + .5))
        boots.append(v)
    boots.sort()
    return pt, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def weighted_rate(per_arch: Dict[str, Tuple[int, int]], weights: Dict[str, float],
                  n_boot: int = 4000, seed: int = 0, allow_missing: bool = False) -> Tuple[float, float, float]:
    """LEGACY-COMPATIBLE wrapper returning (rate, lo, hi) -- now STRICT about missing cells and using MOVER
    (n_boot/seed accepted for API compatibility)."""
    r = weighted_rate_ci(per_arch, weights, allow_missing=allow_missing)
    return r["rate"], r["lo"], r["hi"]


# --------------------------------------------------------------------------- pre-registered gates as code
def _by_arch(cells: Mapping[Tuple[str, int], Cell]) -> Dict[str, Cell]:
    out: Dict[str, List[int]] = {}
    for (a, _seat), (k, n) in cells.items():
        o = out.setdefault(a, [0, 0])
        o[0] += k
        o[1] += n
    return {a: (v[0], v[1]) for a, v in out.items()}


def gate_screen(cand: Mapping[str, Cell], ctrl: Mapping[str, Cell], weights: Mapping[str, float],
                field_margin: float = 0.03, arch_margin: float = 0.15, min_arch_n: int = 25) -> Dict[str, object]:
    """G1 screen = EXCLUSION ONLY.  Reject if field-weighted rate < control - 3pp, or any archetype with
    >= min_arch_n matches in both arms is <= -15pp.  Passing proves nothing."""
    d = weighted_diff_ci(cand, ctrl, weights)
    reasons = []
    if d["diff"] < -field_margin:
        reasons.append(f"field-weighted diff {d['diff']:+.3f} < -{field_margin}")
    for a in cand:
        if a in ctrl and min(cand[a][1], ctrl[a][1]) >= min_arch_n:
            da = cand[a][0] / cand[a][1] - ctrl[a][0] / ctrl[a][1]
            if da <= -arch_margin:
                reasons.append(f"archetype {a}: {da:+.3f} <= -{arch_margin}")
    return dict(passed=not reasons, rejected_reasons=reasons, diff=d)


def gate_confirm(cand: Mapping[str, Cell], ctrl: Mapping[str, Cell], weights: Mapping[str, float],
                 min_matches_per_arm: int = 1000, min_share: float = 0.05, min_arch_n: int = 50,
                 regress: float = 0.05, z: float = Z95) -> Dict[str, object]:
    """G2 confirmation.  PASS only if ALL hold:
       (a) >= min_matches_per_arm completed matches in each arm;
       (b) lower end of the 95% stratified-MOVER interval of the field-weighted difference > 0;
       (c) no archetype with share >= min_share and >= min_arch_n matches in both arms has a Newcombe 95% UPPER bound
           below -regress (a confirmed regression)."""
    w = _norm_weights(weights)
    d = weighted_diff_ci(cand, ctrl, w, z=z)
    checks = []
    nc, nk = d["n_cand"], d["n_ctrl"]
    checks.append(("N per arm", nc >= min_matches_per_arm and nk >= min_matches_per_arm, f"cand={nc} ctrl={nk}"))
    checks.append(("lower CI of field-weighted diff > 0", d["lo"] > 0, f"diff={d['diff']:+.4f} CI=[{d['lo']:+.4f},{d['hi']:+.4f}]"))
    bad = []
    for a, share in w.items():
        if share >= min_share and a in cand and a in ctrl and min(cand[a][1], ctrl[a][1]) >= min_arch_n:
            lo, hi = newcombe_diff(cand[a][0], cand[a][1], ctrl[a][0], ctrl[a][1], z)
            if hi < -regress:
                bad.append(f"{a}: CI=[{lo:+.3f},{hi:+.3f}]")
    checks.append(("no confirmed archetype regression", not bad, "; ".join(bad) or "none"))
    return dict(passed=all(c[1] for c in checks), checks=checks, diff=d)


# --------------------------------------------------------------------------- self-test and tables
def _selftest() -> None:
    def close(a: float, b: float, tol: float = 6e-4) -> bool:
        return abs(a - b) <= tol

    # legacy cross-checks (published Wilson intervals, 4dalols/pokemon-agent PRs #3, #12, #14)
    cases = [(47, 100, .375, .567), (32, 100, .237, .417), (45, 100, .356, .548),
             (216, 300, .667, .768), (193, 300, .588, .695), (117, 150, .707, .839), (80, 100, .711, .867)]
    for k, n, lo, hi in cases:
        a, b = wilson(k, n)
        assert close(a, lo, 1.2e-3) and close(b, hi, 1.2e-3), (k, n, a, b, lo, hi)
    d, z, p = two_prop_z(216, 300, 193, 300)
    assert abs(z - 2.0) < 0.05 and abs(p - 0.04) < 0.01, (z, p)
    assert n_two_sample(0.55, 0.50) == 1565 and n_one_sample(0.55) == 783
    assert n_one_sample(0.55, sides=1) == 617          # == legacy stats_gates.required_games(0.05)
    assert abs(bo3_from_game(0.5) - 0.5) < 1e-12 and abs(game_from_bo3(bo3_from_game(0.6)) - 0.6) < 1e-9
    assert abs(binom_two_sided_p(60, 100) - 0.0569) < 1e-3
    rnd = random.Random(1)
    true = {"A": 200.0, "B": 0.0, "C": -100.0, "D": -100.0}
    wins: Dict[Tuple[str, str], float] = {}
    names = list(true)
    for _ in range(40000):
        i, j = rnd.sample(names, 2)
        pij = elo_p(true[i] - true[j])
        w_, l_ = (i, j) if rnd.random() < pij else (j, i)
        wins[(w_, l_)] = wins.get((w_, l_), 0) + 1
    est = bradley_terry(wins, pseudo=0.0)
    off = sum(est.values()) / 4 - sum(true.values()) / 4
    for k in true:
        assert abs(est[k] - off - true[k]) < 12, (k, est[k], true[k])
    mh = mantel_haenszel([(60, 40, 50, 50)] * 6)
    assert abs(mh["or"] - (60 * 50) / (40 * 50)) < 1e-9
    # NEW: the SPRT sign regression test required by Mission 3
    assert sprt(10, 100, 0.50, 0.45)[3] == "accept H1", sprt(10, 100, 0.50, 0.45)   # 10/100 supports a 45% regression
    assert sprt(70, 100, 0.50, 0.45)[3] == "accept H0", sprt(70, 100, 0.50, 0.45)   # 70/100 contradicts it
    assert sprt(70, 100, 0.50, 0.55)[3] == "accept H1" and sprt(10, 100, 0.50, 0.55)[3] == "accept H0"
    # NEW: strict missing cells
    try:
        weighted_rate({"a": (5, 10)}, {"a": .5, "b": .5})
        raise AssertionError("missing cell was silently renormalised")
    except MissingCellError:
        pass
    # NEW: one stratum == Newcombe
    r = weighted_diff_ci({"a": (60, 100)}, {"a": (50, 100)}, {"a": 1.0})
    nl, nh = newcombe_diff(60, 100, 50, 100)
    assert abs(r["lo"] - nl) < 1e-12 and abs(r["hi"] - nh) < 1e-12
    print("selftest OK")


def _tables() -> None:
    print("== MDES (true win rate detectable at 80% power, total alpha=.05, TWO-sided == one-sided .025) ==")
    print(f"{'games':>6} {'vs fixed 50%':>14} {'two arms (each n)':>20}")
    for n in (100, 150, 300, 600, 1000, 1500, 3000):
        print(f"{n:>6} {mdes_one_sample(n):>14.3f} {mdes_two_sample(n):>20.3f}")
    print("\n== games needed for a target true win rate (two-sided | one-sided alpha=.05) ==")
    for p1 in (0.53, 0.55, 0.575, 0.60, 0.65):
        print(f"p1={p1:.3f}: vs fixed 50% n={n_one_sample(p1):>5} | {n_one_sample(p1, sides=1):>5};"
              f"  two independent arms n/arm={n_two_sample(p1, 0.5):>5} | {n_two_sample(p1, 0.5, sides=1):>5}")
    print("\n== Wilson 95% half-width at p=0.5 ==")
    for n in (35, 100, 150, 300, 1000, 3000):
        lo, hi = wilson(n // 2, n)
        print(f"n={n:>5}: +/-{(hi - lo) / 2 * 100:4.1f} pp")
    print("\n== BO3 amplification of a single-game win rate ==")
    for p in (0.50, 0.52, 0.55, 0.60, 0.65, 0.70):
        print(f"game {p:.2f} -> BO3 {bo3_from_game(p):.3f}")
    print("\n== SE of a ladder rating estimate after n episodes (opponents ~ equal rating) [ASSUMED scales] ==")
    print(f"{'n':>6} {'Elo-400':>9} {'probit100':>10}")
    for n in (35, 100, 300, 1000):
        print(f"{n:>6} {rating_se(n):>9.1f} {rating_se(n, 'probit100'):>10.1f}")
    print("\n== SPRT H0 p=0.50 vs H1 p=0.55 (alpha=beta=0.05) ==")
    print(f"Wald E[N|H1]={sprt_expected_n(.5, .55):.0f}  E[N|H0]={sprt_expected_n(.5, .55, under='H0'):.0f}")
    oc = sprt_simulate(0.55, 0.5, 0.55, n_sims=2000, seed=3)
    print(f"simulated (p_true=.55): {oc}")
    print("\n== exact power of the LEGACY gate (rate>=.55 AND Wilson lower>.5) ==")
    for n in (300, 500, 800, 1000, 1500):
        print(f"n={n:>5}: P(pass | true .55)={gate_power(n, .55):.3f}   P(pass | true .60)={gate_power(n, .60):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    if a.selftest or not (a.selftest or a.tables):
        _selftest()
    if a.tables:
        _tables()
