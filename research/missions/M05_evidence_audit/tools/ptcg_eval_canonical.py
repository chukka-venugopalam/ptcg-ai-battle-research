#!/usr/bin/env python3
"""ptcg_eval_canonical.py -- THE single canonical evaluation-statistics module (stdlib only, Python >= 3.9).

Supersedes Mission 3 `ptcg_eval_stats.py` (game-level cells) and Mission 4 `eval_stats.py` (game-level records).
Both treated the single GAME as the inferential unit.  This module uses the BO3 MATCH.

DEFINITIONS (pre-registered; change only with written justification in STATISTICS_RECONCILIATION.md)
  unit        one completed BO3 match = one observation.  Games inside a match are diagnostics (GameRecord), never
              entered in any CI, test, or sample-size count.
  point metric  match score s in {1 (W), 0.5 (D), 0 (L)}.  Field score = sum_cell w_cell * mean(s_cell).
  cell        (opponent, first_seat) where first_seat = the seat the agent occupied in game 1 of the match.
              Cell weight = opponent_weight * seat_weight (seat_weight 0.5/0.5).  Seat is a stratum, not a replicate.
  N           ALL sample sizes are MATCHES.  `*_total_per_arm` = across all opponents and seats of one arm.
              `min_per_cell` = per (opponent, seat) cell.  Nothing in this module is "games per opponent".
  alpha       gate = lower end of a 95% two-sided interval > 0  <=>  one-sided alpha 0.025.  Power target 0.80.
  interval    Wilson on (points, n).  Because s is in [0,1] with mean p, Var(s) <= p(1-p) (Bhatia-Davis), so a Wilson
              interval on points is conservative when draws exist; it is NOT an exact binomial interval for P(win).
              Coverage is checked by Monte Carlo in tools/mc_coverage.py.  P(win) has its own interval (win_prob_ci).
  failures    status 'ok' | 'cand_fail' (counted as a LOSS, flagged) | 'opp_fail' | 'infra_fail' (excluded, counted).
              Excluded fraction above `max_excluded_frac` FAILS the gate; a positively weighted empty cell raises.
  sequential  `sprt()` is for head-to-head EARLY REJECTION only (decisive matches, draws dropped).  Confirmation is
              fixed-N: `confirm()` requires exactly the pre-planned N per arm and refuses early-stopped data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

_N = NormalDist()
Z95 = _N.inv_cdf(0.975)
ALPHA_ONE_SIDED = 0.025
POWER = 0.80
SEAT_WEIGHTS = {0: 0.5, 1: 0.5}
_PTS = {"W": 1.0, "D": 0.5, "L": 0.0}
_STATUS = {"ok", "cand_fail", "opp_fail", "infra_fail"}


class MissingCellError(ValueError):
    """A positively weighted cell has no completed matches; renormalising around it is forbidden."""


class DesignError(ValueError):
    """Data do not match the pre-registered fixed-N design (early stop, over-collection, no fresh flag)."""


# ----------------------------------------------------------------------------- records
@dataclass(frozen=True)
class GameRecord:
    """DIAGNOSTIC ONLY.  Never used for inference."""
    match_id: str
    game_index: int
    agent_seat: int
    first_player_seat: Optional[int]
    result: str  # W/L/D from agent perspective

    def __post_init__(self):
        if self.agent_seat not in (0, 1) or self.result not in _PTS:
            raise ValueError(f"bad GameRecord {self!r}")


@dataclass(frozen=True)
class MatchRecord:
    """One BO3 match = one observation."""
    match_id: str
    opponent: str
    first_seat: int          # agent seat in game 1
    result: str              # W/L/D from agent perspective (ignored when status is opp_fail/infra_fail)
    status: str = "ok"

    def __post_init__(self):
        if self.first_seat not in (0, 1):
            raise ValueError(f"first_seat must be 0/1: {self.first_seat!r}")
        if self.status not in _STATUS:
            raise ValueError(f"status must be in {_STATUS}: {self.status!r}")
        if self.result not in _PTS:
            raise ValueError(f"result must be W/L/D: {self.result!r}")


def match_result_from_games(games: Sequence[str]) -> str:
    """ASSUMPTION (UNVERIFIED against official rules): BO3 winner = first to 2 game wins; if neither side reaches 2
    (e.g. W,L,D or D,D,D) the match is a draw.  Official draw/tiebreak semantics are UNKNOWN until rules are read."""
    if any(g not in _PTS for g in games) or not 1 <= len(games) <= 3:
        raise ValueError(f"bad games {games!r}")
    w, l = games.count("W"), games.count("L")
    return "W" if w >= 2 else "L" if l >= 2 else "D"


@dataclass(frozen=True)
class Cell:
    w: int = 0
    d: int = 0
    l: int = 0

    @property
    def n(self) -> int:
        return self.w + self.d + self.l

    @property
    def points(self) -> float:
        return self.w + 0.5 * self.d

    @property
    def score(self) -> float:
        return self.points / self.n


Cells = Mapping[Tuple[str, int], Cell]


@dataclass(frozen=True)
class Tally:
    cells: Dict[Tuple[str, int], Cell]
    n_ok: int
    n_cand_fail: int
    n_excluded: int

    @property
    def excluded_frac(self) -> float:
        tot = self.n_ok + self.n_cand_fail + self.n_excluded
        return self.n_excluded / tot if tot else 0.0


def tally(matches: Iterable[MatchRecord]) -> Tally:
    acc: Dict[Tuple[str, int], List[int]] = {}
    ok = cf = ex = 0
    seen = set()
    for m in matches:
        if m.match_id in seen:
            raise ValueError(f"duplicate match_id {m.match_id!r}: matches must be unique observations")
        seen.add(m.match_id)
        if m.status in ("opp_fail", "infra_fail"):
            ex += 1
            continue
        res = "L" if m.status == "cand_fail" else m.result
        cf += m.status == "cand_fail"
        ok += m.status == "ok"
        a = acc.setdefault((m.opponent, m.first_seat), [0, 0, 0])
        a[{"W": 0, "D": 1, "L": 2}[res]] += 1
    return Tally({k: Cell(*v) for k, v in acc.items()}, ok, cf, ex)


# ----------------------------------------------------------------------------- intervals
def wilson(points: float, n: int, z: float = Z95) -> Tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= points <= n:
        raise ValueError("points must lie in [0, n]")
    p = points / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (0.0 if points == 0 else max(0.0, c - h)), (1.0 if points == n else min(1.0, c + h))


def win_prob_ci(cell_or_tuple, z: float = Z95) -> Tuple[float, float, float]:
    """CI for P(win a BO3 match) = wins/n with draws counted as NON-wins: a genuine binomial event.
    Not the same estimand as the average match score."""
    c = cell_or_tuple
    lo, hi = wilson(c.w, c.n, z)
    return c.w / c.n, lo, hi


def _cell_weights(opp_weights: Mapping[str, float], seat_weights: Mapping[int, float] = SEAT_WEIGHTS) -> Dict[Tuple[str, int], float]:
    pos = {k: float(v) for k, v in opp_weights.items() if v and v > 0}
    if any(v < 0 for v in opp_weights.values()) or not pos:
        raise ValueError("opponent weights must be >= 0 with a positive total")
    tw = sum(pos.values())
    sw = sum(seat_weights.values())
    return {(o, s): (v / tw) * (sw_ / sw) for o, v in pos.items() for s, sw_ in seat_weights.items() if sw_ > 0}


def _require(cells: Cells, cw: Mapping[Tuple[str, int], float], who: str) -> None:
    missing = [k for k in cw if k not in cells or cells[k].n <= 0]
    if missing:
        raise MissingCellError(f"{who}: no completed matches in positively weighted cell(s) {sorted(missing)!r}")


def field_score_ci(cells: Cells, opp_weights: Mapping[str, float], z: float = Z95) -> Dict[str, object]:
    """Field score (average match score) with stratified MOVER interval over opponent x seat cells."""
    cw = _cell_weights(opp_weights)
    _require(cells, cw, "field_score_ci")
    est = lo2 = hi2 = 0.0
    for k, w in cw.items():
        c = cells[k]
        p, (l, u) = c.score, wilson(c.points, c.n, z)
        est += w * p
        lo2 += (w * (p - l)) ** 2
        hi2 += (w * (u - p)) ** 2
    return dict(score=est, lo=est - math.sqrt(lo2), hi=est + math.sqrt(hi2), n=sum(cells[k].n for k in cw))


def diff_ci(cand: Cells, ctrl: Cells, opp_weights: Mapping[str, float], z: float = Z95) -> Dict[str, object]:
    """Candidate-minus-control field-score difference, INDEPENDENT arms (native engine RNG is not seedable).
    Stratified MOVER / Newcombe hybrid-score; a cell must be complete in BOTH arms."""
    cw = _cell_weights(opp_weights)
    _require(cand, cw, "candidate")
    _require(ctrl, cw, "control")
    d = lo2 = hi2 = 0.0
    per: Dict[Tuple[str, int], float] = {}
    for k, w in cw.items():
        a, b = cand[k], ctrl[k]
        p1, p2 = a.score, b.score
        l1, u1 = wilson(a.points, a.n, z)
        l2, u2 = wilson(b.points, b.n, z)
        d += w * (p1 - p2)
        per[k] = p1 - p2
        lo2 += w * w * ((p1 - l1) ** 2 + (u2 - p2) ** 2)
        hi2 += w * w * ((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return dict(diff=d, lo=d - math.sqrt(lo2), hi=d + math.sqrt(hi2), per_cell=per,
                n_cand=sum(cand[k].n for k in cw), n_ctrl=sum(ctrl[k].n for k in cw))


def newcombe_cell_diff(a: Cell, b: Cell, z: float = Z95) -> Tuple[float, float, float]:
    l1, u1 = wilson(a.points, a.n, z)
    l2, u2 = wilson(b.points, b.n, z)
    d = a.score - b.score
    return d, d - math.sqrt((a.score - l1) ** 2 + (u2 - b.score) ** 2), d + math.sqrt((u1 - a.score) ** 2 + (b.score - l2) ** 2)


# ----------------------------------------------------------------------------- power / planning (MATCHES)
def n_per_arm(delta: float, sd: float = 0.5, alpha_one_sided: float = ALPHA_ONE_SIDED, power: float = POWER) -> int:
    """Total MATCHES per arm for a two-arm difference of mean match scores.  sd = SD of one match score.
    sd=0.5 is the worst case (no draws, p=0.5); with draws SD is smaller.  Normal approximation, equal arms."""
    if delta <= 0 or sd <= 0:
        raise ValueError("delta, sd must be > 0")
    return math.ceil(2 * ((_N.inv_cdf(1 - alpha_one_sided) + _N.inv_cdf(power)) * sd / delta) ** 2)


def mdes(n: int, sd: float = 0.5, alpha_one_sided: float = ALPHA_ONE_SIDED, power: float = POWER) -> float:
    """Minimum detectable difference in match score at `n` MATCHES per arm."""
    return (_N.inv_cdf(1 - alpha_one_sided) + _N.inv_cdf(power)) * sd * math.sqrt(2.0 / n)


def bo3_win_from_game_win(p: float) -> float:
    """P(win BO3) from per-game win prob, no draws, independent games (ASSUMPTION; not a CI input)."""
    return p * p * (3 - 2 * p)


# ----------------------------------------------------------------------------- sequential early rejection
def sprt(k: int, n: int, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.05):
    """Wald SPRT for decisive matches (k wins of n decisive; DRAWS DROPPED by the caller).  H0:p=p0 vs H1:p=p1,
    either direction.  LLR = log L(p1)/L(p0) (the legacy code negated it when p1<p0).  Early-rejection use only."""
    if not (0 < p0 < 1 and 0 < p1 < 1) or p0 == p1 or not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("bad SPRT parameters")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    llr = k * math.log(p1 / p0) + (n - k) * math.log((1 - p1) / (1 - p0))
    up, lo = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    return llr, lo, up, ("accept H1" if llr >= up else "accept H0" if llr <= lo else "continue")


# ----------------------------------------------------------------------------- gates
@dataclass(frozen=True)
class ScreenRule:
    """EXCLUSION ONLY.  Passing proves nothing."""
    min_per_cell: int = 10
    field_margin: float = 0.03
    cell_margin: float = 0.25   # opponent x seat cells are small: only reject catastrophic cells


@dataclass(frozen=True)
class ConfirmRule:
    n_total_per_arm: int = 1000     # MATCHES per arm, all opponents and seats combined; EXACT (fixed-N)
    min_per_cell: int = 20          # MATCHES per opponent x seat cell
    z: float = Z95                  # lower end of 95% two-sided == one-sided alpha 0.025
    max_excluded_frac: float = 0.02
    min_share: float = 0.05
    regress: float = 0.05
    min_opp_n: int = 50


@dataclass(frozen=True)
class Decision:
    passed: bool
    reasons: Tuple[str, ...]
    stats: Dict[str, object] = field(default_factory=dict)


def screen(cand: Tally, ctrl: Tally, opp_weights: Mapping[str, float], rule: ScreenRule = ScreenRule()) -> Decision:
    reasons: List[str] = []
    cw = _cell_weights(opp_weights)
    for name, t in (("candidate", cand), ("control", ctrl)):
        for k in cw:
            if k not in t.cells or t.cells[k].n < rule.min_per_cell:
                reasons.append(f"{name} cell {k}: < {rule.min_per_cell} matches")
    if reasons:
        return Decision(False, tuple(reasons))
    d = diff_ci(cand.cells, ctrl.cells, opp_weights)
    if d["diff"] < -rule.field_margin:
        reasons.append(f"field diff {d['diff']:+.3f} < -{rule.field_margin}")
    for k, v in d["per_cell"].items():
        if v <= -rule.cell_margin:
            reasons.append(f"cell {k}: {v:+.3f} <= -{rule.cell_margin}")
    return Decision(not reasons, tuple(reasons), d)


def confirm(cand: Tally, ctrl: Tally, opp_weights: Mapping[str, float], rule: ConfirmRule = ConfirmRule(),
            fresh_matches: bool = False) -> Decision:
    """Fixed-N confirmation on FRESH matches.  Raises DesignError on early stop / over-collection / non-fresh data
    (to prevent peeking-and-stopping).  Returns PASS only if every check holds."""
    if not fresh_matches:
        raise DesignError("confirm() requires fresh_matches=True (matches not used for screening/selection)")
    for name, t in (("candidate", cand), ("control", ctrl)):
        got = sum(c.n for c in t.cells.values())
        if got != rule.n_total_per_arm:
            raise DesignError(f"{name}: {got} analysable matches != pre-registered N {rule.n_total_per_arm}")
    cw = _cell_weights(opp_weights)
    reasons: List[str] = []
    for name, t in (("candidate", cand), ("control", ctrl)):
        if t.excluded_frac > rule.max_excluded_frac:
            reasons.append(f"{name}: excluded fraction {t.excluded_frac:.3f} > {rule.max_excluded_frac}")
        for k in cw:
            if k not in t.cells or t.cells[k].n < rule.min_per_cell:
                reasons.append(f"{name} cell {k}: < {rule.min_per_cell} matches")
    if any("cell" in r for r in reasons):
        return Decision(False, tuple(reasons))
    d = diff_ci(cand.cells, ctrl.cells, opp_weights, rule.z)
    if not d["lo"] > 0:
        reasons.append(f"lower bound {d['lo']:+.4f} <= 0 (diff {d['diff']:+.4f})")
    tw = sum(v for v in opp_weights.values() if v > 0)
    for opp, wv in opp_weights.items():
        if wv / tw >= rule.min_share:
            a = _merge(cand.cells, opp)
            b = _merge(ctrl.cells, opp)
            if a.n >= rule.min_opp_n and b.n >= rule.min_opp_n:
                _, lo, hi = newcombe_cell_diff(a, b, rule.z)
                if hi < -rule.regress:
                    reasons.append(f"confirmed regression vs {opp}: CI [{lo:+.3f},{hi:+.3f}]")
    return Decision(not reasons, tuple(reasons), d)


def _merge(cells: Cells, opp: str) -> Cell:
    cs = [c for (o, _), c in cells.items() if o == opp]
    return Cell(sum(c.w for c in cs), sum(c.d for c in cs), sum(c.l for c in cs))
