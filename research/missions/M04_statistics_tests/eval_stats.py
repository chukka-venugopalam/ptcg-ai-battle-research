"""Seat-balanced evaluation statistics for PTCG agent screening and confirmation.

Pure standard library (Python >= 3.10; tested on 3.12). No engine dependency, so every
function here can be executed and tested without the CABT engine. It does NOT run games;
it consumes per-game records produced by whatever harness plays them.

Conventions
-----------
* A game record is (opponent, agent_seat, result) with result in {"W", "L", "D"} from the
  evaluated agent's point of view. A draw counts as half a point.
* Per-opponent uncertainty uses Wilson score intervals on (wins + 0.5 * draws) / games.
* Pooled scores use PREDECLARED opponent weights, never weights derived from how often an
  opponent happened to be sampled.
* Screening is cheap and allowed to admit false positives; confirmation uses fresh games
  and a one-sided rule that is fixed before the games are played.

Usage sketch
------------
    recs = [GameRecord("dragapult_pilot", 0, "W"), ...]
    summ = summarize(recs)
    weights = {"dragapult_pilot": 0.4, "lucario_pilot": 0.3, "mirror": 0.3}
    print(pooled_score(summ, weights))
    print(confirm(candidate_summ, control_summ, weights, ConfirmRule()))
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Z95 = NormalDist().inv_cdf(0.975)  # 1.95996...
_VALID = {"W", "L", "D"}


# --------------------------------------------------------------------------- basics
def wilson_interval(successes: float, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a proportion; `successes` may be fractional (draws = 0.5)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


@dataclass(frozen=True)
class GameRecord:
    opponent: str
    agent_seat: int  # 0 or 1
    result: str      # 'W' | 'L' | 'D' from the evaluated agent's perspective

    def __post_init__(self) -> None:
        if self.agent_seat not in (0, 1):
            raise ValueError(f"agent_seat must be 0 or 1, got {self.agent_seat!r}")
        if self.result not in _VALID:
            raise ValueError(f"result must be one of {_VALID}, got {self.result!r}")


@dataclass(frozen=True)
class MatchupSummary:
    opponent: str
    n: int
    wins: int
    losses: int
    draws: int
    score: float
    lo: float
    hi: float
    n_seat0: int
    n_seat1: int
    score_seat0: Optional[float]
    score_seat1: Optional[float]

    @property
    def seat_balanced(self) -> bool:
        """True when the two seats differ by at most one game."""
        return abs(self.n_seat0 - self.n_seat1) <= 1

    @property
    def var(self) -> float:
        """Plug-in variance of the estimated score (binomial approximation)."""
        return self.score * (1.0 - self.score) / self.n


def _points(result: str) -> float:
    return 1.0 if result == "W" else 0.5 if result == "D" else 0.0


def summarize(records: Iterable[GameRecord]) -> Dict[str, MatchupSummary]:
    """Aggregate raw game records into per-opponent summaries."""
    buckets: Dict[str, List[GameRecord]] = {}
    for r in records:
        buckets.setdefault(r.opponent, []).append(r)
    out: Dict[str, MatchupSummary] = {}
    for opp, rs in buckets.items():
        n = len(rs)
        w = sum(r.result == "W" for r in rs)
        l = sum(r.result == "L" for r in rs)
        d = sum(r.result == "D" for r in rs)
        pts = sum(_points(r.result) for r in rs)
        lo, hi = wilson_interval(pts, n)
        s0 = [r for r in rs if r.agent_seat == 0]
        s1 = [r for r in rs if r.agent_seat == 1]
        out[opp] = MatchupSummary(
            opponent=opp, n=n, wins=w, losses=l, draws=d, score=pts / n, lo=lo, hi=hi,
            n_seat0=len(s0), n_seat1=len(s1),
            score_seat0=(sum(_points(r.result) for r in s0) / len(s0)) if s0 else None,
            score_seat1=(sum(_points(r.result) for r in s1) / len(s1)) if s1 else None,
        )
    return out


def _normalise(weights: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0 or any(v < 0 for v in weights.values()):
        raise ValueError("weights must be non-negative and sum to a positive number")
    return {k: v / total for k, v in weights.items()}


@dataclass(frozen=True)
class Pooled:
    score: float
    se: float
    lo: float
    hi: float


def pooled_score(summ: Mapping[str, MatchupSummary], weights: Mapping[str, float],
                 z: float = Z95) -> Pooled:
    """Weighted mean of per-opponent scores using PREDECLARED weights (normal approximation)."""
    w = _normalise(weights)
    missing = [k for k in w if k not in summ]
    if missing:
        raise KeyError(f"no games recorded for weighted opponents: {missing}")
    score = sum(w[k] * summ[k].score for k in w)
    var = sum((w[k] ** 2) * summ[k].var for k in w)
    se = math.sqrt(var)
    return Pooled(score, se, score - z * se, score + z * se)


@dataclass(frozen=True)
class Delta:
    delta: float
    se: float
    lo: float
    hi: float
    per_opponent: Dict[str, float]


def pooled_delta(candidate: Mapping[str, MatchupSummary], control: Mapping[str, MatchupSummary],
                 weights: Mapping[str, float], z: float = Z95) -> Delta:
    """Candidate minus control on the same predeclared opponent panel (independent arms)."""
    w = _normalise(weights)
    for k in w:
        if k not in candidate or k not in control:
            raise KeyError(f"opponent {k!r} missing from candidate or control summaries")
    d = sum(w[k] * (candidate[k].score - control[k].score) for k in w)
    var = sum((w[k] ** 2) * (candidate[k].var + control[k].var) for k in w)
    se = math.sqrt(var)
    per = {k: candidate[k].score - control[k].score for k in w}
    return Delta(d, se, d - z * se, d + z * se, per)


# --------------------------------------------------------------------------- decision rules
@dataclass(frozen=True)
class ScreenRule:
    """Cheap filter. Allowed to pass false positives (see `simulate_screen_false_positive`)."""
    min_games_per_opponent: int = 300
    min_pooled_delta: float = 0.02
    worst_opponent_floor: float = -0.06  # point-estimate floor per opponent


@dataclass(frozen=True)
class ConfirmRule:
    """Fixed before the confirmation games are played; run on FRESH games only."""
    min_games_per_opponent: int = 500
    z_one_sided: float = Z95          # 97.5% one-sided == 95% two-sided lower bound
    min_pooled_delta_lower: float = 0.0
    worst_opponent_floor: float = -0.05
    require_seat_balance: bool = True


@dataclass(frozen=True)
class Decision:
    passed: bool
    reasons: Tuple[str, ...]
    delta: Delta


def _common_checks(candidate, control, weights, min_games, require_balance) -> List[str]:
    reasons: List[str] = []
    for k in weights:
        for name, arm in (("candidate", candidate), ("control", control)):
            if arm[k].n < min_games:
                reasons.append(f"{name}:{k} has {arm[k].n} < {min_games} games")
            if require_balance and not arm[k].seat_balanced:
                reasons.append(f"{name}:{k} seats unbalanced ({arm[k].n_seat0}/{arm[k].n_seat1})")
    return reasons


def screen(candidate: Mapping[str, MatchupSummary], control: Mapping[str, MatchupSummary],
           weights: Mapping[str, float], rule: ScreenRule = ScreenRule()) -> Decision:
    reasons = _common_checks(candidate, control, weights, rule.min_games_per_opponent, True)
    d = pooled_delta(candidate, control, weights)
    if d.delta < rule.min_pooled_delta:
        reasons.append(f"pooled delta {d.delta:+.4f} < {rule.min_pooled_delta:+.4f}")
    for k, v in d.per_opponent.items():
        if v < rule.worst_opponent_floor:
            reasons.append(f"opponent {k}: delta {v:+.4f} < floor {rule.worst_opponent_floor:+.4f}")
    return Decision(not reasons, tuple(reasons), d)


def confirm(candidate: Mapping[str, MatchupSummary], control: Mapping[str, MatchupSummary],
            weights: Mapping[str, float], rule: ConfirmRule = ConfirmRule()) -> Decision:
    reasons = _common_checks(candidate, control, weights, rule.min_games_per_opponent,
                             rule.require_seat_balance)
    d = pooled_delta(candidate, control, weights, z=rule.z_one_sided)
    if d.lo <= rule.min_pooled_delta_lower:
        reasons.append(f"lower bound {d.lo:+.4f} <= {rule.min_pooled_delta_lower:+.4f}")
    for k, v in d.per_opponent.items():
        if v < rule.worst_opponent_floor:
            reasons.append(f"opponent {k}: delta {v:+.4f} < floor {rule.worst_opponent_floor:+.4f}")
    return Decision(not reasons, tuple(reasons), d)


# --------------------------------------------------------------------------- planning
def required_games_head_to_head(delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Games for ONE candidate-vs-control head-to-head series to detect true win rate 0.5+delta.

    Two-sided test of H0: p = 0.5 using the normal approximation with unequal variances.
    """
    if not 0 < delta < 0.5:
        raise ValueError("delta must be in (0, 0.5)")
    nd = NormalDist()
    za, zb = nd.inv_cdf(1 - alpha / 2), nd.inv_cdf(power)
    p1 = 0.5 + delta
    n = ((za * 0.5 + zb * math.sqrt(p1 * (1 - p1))) / delta) ** 2
    return math.ceil(n)


def required_games_per_arm(p0: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Games per ARM to detect an absolute difference `delta` between two independent arms
    that each play the same opponent panel (pooled over the panel)."""
    if not (0 < p0 < 1 and 0 < p0 + delta < 1 and delta != 0):
        raise ValueError("invalid p0/delta")
    nd = NormalDist()
    za, zb = nd.inv_cdf(1 - alpha / 2), nd.inv_cdf(power)
    p1 = p0 + delta
    pbar = (p0 + p1) / 2
    n = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1)))
         / delta) ** 2
    return math.ceil(n)


# --------------------------------------------------------------------------- simulation helpers
def _binom(rng: random.Random, n: int, p: float) -> int:
    return rng.binomialvariate(n, p) if hasattr(rng, "binomialvariate") else sum(
        rng.random() < p for _ in range(n))


def simulate_screen_false_positive(num_variants: int, n_per_opponent: int, num_opponents: int,
                                   threshold: float = 0.02, p_control: float = 0.5,
                                   true_delta: float = 0.0, trials: int = 2000, seed: int = 0
                                   ) -> Dict[str, float]:
    """Winner's-curse demo: screen `num_variants` candidates (all with the same TRUE delta) against one
    control on `num_opponents` equal-weight opponents. Returns how often the best-looking candidate
    clears `threshold`, and the mean observed delta of the best-looking candidate."""
    rng = random.Random(seed)
    n = n_per_opponent * num_opponents
    hits, best_sum = 0, 0.0
    for _ in range(trials):
        ctrl = _binom(rng, n, p_control) / n
        best = max(_binom(rng, n, p_control + true_delta) / n - ctrl for _ in range(num_variants))
        best_sum += best
        hits += best >= threshold
    return {"p_best_ge_threshold": hits / trials, "mean_best_observed_delta": best_sum / trials}


def simulate_confirm_false_positive(n_per_opponent: int, num_opponents: int, p_control: float = 0.5,
                                    trials: int = 4000, seed: int = 1) -> float:
    """False-positive rate of the one-sided confirmation rule when the true delta is zero
    (pooled independent-arm z-test on equal-weight opponents)."""
    rng = random.Random(seed)
    nd = NormalDist()
    z = nd.inv_cdf(0.975)
    fp = 0
    for _ in range(trials):
        c1 = [_binom(rng, n_per_opponent, p_control) / n_per_opponent for _ in range(num_opponents)]
        c0 = [_binom(rng, n_per_opponent, p_control) / n_per_opponent for _ in range(num_opponents)]
        w = 1.0 / num_opponents
        d = sum(w * (a - b) for a, b in zip(c1, c0))
        var = sum((w ** 2) * ((a * (1 - a)) + (b * (1 - b))) / n_per_opponent for a, b in zip(c1, c0))
        fp += (d - z * math.sqrt(var)) > 0
    return fp / trials


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down; returns reject flags in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    m = len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


if __name__ == "__main__":  # tiny CLI: print the planning tables used in SELECTED_EXPERIMENT_PLAN.md
    print("head-to-head games for +delta over 50% (alpha=.05, power=.8):")
    for d in (0.02, 0.03, 0.04, 0.05, 0.08):
        print(f"  +{d:.2f}: {required_games_head_to_head(d)}")
    print("games per arm vs a shared panel at p0=0.55:")
    for d in (0.03, 0.04, 0.05):
        print(f"  +{d:.2f}: {required_games_per_arm(0.55, d)}")
