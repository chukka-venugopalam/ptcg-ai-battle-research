"""Dev → mid policy handoff readiness from observable game progress.

Two BC nets trained on the same recipe with complementary episode spans
(`--episode-span 0:0.5` = development, `0.5:1` = mid) need a live switch:
use the mid net once the position looks past setup.

Scale
-----
`readiness_score` ∈ [0, 1]. Higher means further into the contested game.
Suggested default threshold: ``DEFAULT_THRESHOLD`` (0.56) — use mid iff
``score >= threshold``, else stay on the development net.

Semantics
---------
Training splits by decision-row fraction within each episode. That fraction is
not observable at inference (remaining select count is unknown). This score is
a transparent PROXY built from the same state fields ``features.featurize``
already exposes: turn, prizes taken, and board development (evolutions /
energy / occupied slots). It is NOT a learned classifier.

``DEFAULT_THRESHOLD`` is calibrated on the original v5_s2 train cut
(pds_v4 replay dirs 2026-07-26..29): it minimizes mean |first-crossing
index − n//2| against the train-time mid-half label. See
``scripts/p99_calibrate_readiness.py``. Median readiness at the cut row
was 0.54; per-row F1 also peaks at 0.54. Frozen value is the timing-error
τ (0.56). PhaseHandoff stays per-select, not sticky.

Hook
----
``prefer_mid(obs)`` for a boolean gate, or ``PhaseHandoff`` / ``net=dev|mid``
in ``bcagent`` (parallel to ``net=a+b`` ensembles) to switch per decision.
"""
from __future__ import annotations

import numpy as np

# Turn at which the turn-component alone reaches 1.0. p91's "mid" band starts
# at turn 5; with weight 0.5 that alone contributes 0.25 at turn 5, so prizes
# and board must also look developed before the calibrated 0.56 threshold fires.
TURN_REF = 10

# Suggested handoff: mid agent when readiness_score >= this.
# Calibrated by p99_calibrate_readiness.py (timing-error min vs n//2).
DEFAULT_THRESHOLD = 0.56

# Weights for the three [0, 1] components (must sum to 1).
W_TURN = 0.50
W_PRIZE = 0.30
W_BOARD = 0.20


def _occupied(pl: dict) -> list[dict]:
    pks: list[dict] = []
    act = pl.get("active") or []
    if act and act[0] is not None:
        pks.append(act[0])
    for pk in pl.get("bench") or []:
        if pk is not None:
            pks.append(pk)
    return pks


def _board_dev(pl: dict) -> float:
    """How developed one side's board looks, in [0, 1]."""
    pks = _occupied(pl)
    if not pks:
        return 0.0
    from . import cards as cdb  # lazy: empty-board paths need no engine

    evolved = 0
    energized = 0
    for pk in pks:
        c = cdb.card(pk["id"])
        if c.get("stage1") or c.get("stage2"):
            evolved += 1
        if pk.get("energies"):
            energized += 1
    n = len(pks)
    # Occupied slots / 6 (active + 5 bench): empty board stays near 0.
    fill = min(n, 6) / 6.0
    return (0.40 * (evolved / n)
            + 0.40 * (energized / n)
            + 0.20 * fill)


def readiness_from_state(state: dict, me: int | None = None) -> float:
    """Map a live ``current`` state dict to readiness in [0, 1].

    Formula (clipped weighted sum of [0, 1] terms)::

        turn_c  = min(turn, 10) / 10
        prize_c = min(my_prizes_taken + opp_prizes_taken, 6) / 6
        board_c = 0.5 * (_board_dev(me) + _board_dev(opp))
        score   = 0.50 * turn_c + 0.30 * prize_c + 0.20 * board_c
    """
    if me is None:
        me = int(state.get("yourIndex", 0))
    players = state.get("players") or []
    if me not in (0, 1) or len(players) < 2:
        return 0.0
    mypl = players[me] or {}
    oppl = players[1 - me] or {}

    turn = int(state.get("turn") or 0)
    turn_c = min(turn, TURN_REF) / float(TURN_REF)

    my_taken = 6 - len(mypl.get("prize") or [])
    opp_taken = 6 - len(oppl.get("prize") or [])
    prize_c = min(max(my_taken, 0) + max(opp_taken, 0), 6) / 6.0

    board_c = 0.5 * (_board_dev(mypl) + _board_dev(oppl))

    score = W_TURN * turn_c + W_PRIZE * prize_c + W_BOARD * board_c
    return float(max(0.0, min(1.0, score)))


def readiness_from_obs(obs: dict) -> float:
    """Convenience: readiness of ``obs['current']``."""
    state = obs.get("current") or {}
    return readiness_from_state(state)


# Public alias — the name callers should prefer at the decision site.
readiness_score = readiness_from_obs


def prefer_mid(obs: dict, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True → hand off to the midgame net; False → keep the development net."""
    return readiness_score(obs) >= threshold


class PhaseHandoff:
    """Two nets, one decision: pick mid when readiness crosses ``threshold``.

    Duck-types enough of ``policynet.Net`` / ``Ensemble`` for ``PolicyAgent``
    (``choose``, ``scores``, ``opt_in`` / ``state_in`` passthroughs). Count
    rules and auxiliary heads come from the active member for that select.

    Logs the first ``dev → mid`` crossing each game (turn drop resets). Set
    ``log_handoff=False`` to silence.
    """

    def __init__(self, early, mid, threshold: float = DEFAULT_THRESHOLD,
                 log_handoff: bool = True):
        if early is None or mid is None:
            raise ValueError("PhaseHandoff needs both early and mid nets")
        self.early = early
        self.mid = mid
        self.threshold = float(threshold)
        self.log_handoff = bool(log_handoff)
        self.primary = early  # shape introspection defaults to the mid-safe early net
        self._phase: str | None = None
        self._last_turn = -1

    def active(self, obs: dict):
        state = obs.get("current") or {}
        turn = int(state.get("turn") or 0)
        # New game: harness reuses the agent; turn resets to the opening.
        if turn < self._last_turn:
            self._phase = None
        self._last_turn = turn

        use_mid = prefer_mid(obs, self.threshold)
        phase = "mid" if use_mid else "dev"
        if phase != self._phase:
            if (self.log_handoff and self._phase == "dev" and phase == "mid"):
                score = readiness_score(obs)
                me = int(state.get("yourIndex", 0))
                print(f"[handoff] seat={me} turn={turn} "
                      f"readiness={score:.3f} >= {self.threshold:g} "
                      f"dev→mid", flush=True)
            self._phase = phase
        return self.mid if use_mid else self.early

    @property
    def opt_in(self) -> int:
        return self.primary.opt_in

    @property
    def state_in(self) -> int:
        return self.primary.state_in

    def scores(self, obs: dict) -> np.ndarray:
        return self.active(obs).scores(obs)

    def choose(self, obs: dict) -> list[int]:
        return self.active(obs).choose(obs)


def load_phase_handoff(early_path: str, mid_path: str,
                       threshold: float = DEFAULT_THRESHOLD,
                       log_handoff: bool = True):
    """Load ``early|mid`` nets. Returns None if either member fails to load."""
    from . import policynet

    early = policynet.load(early_path)
    mid = policynet.load(mid_path)
    if early is None or mid is None:
        return None
    return PhaseHandoff(early, mid, threshold=threshold,
                        log_handoff=log_handoff)


def _self_check() -> None:
    """Tiny synthetic sanity check — run from repo: ``python -m sa.readiness``
    with ``PYTHONPATH=agents`` (no card DB needed: empty boards)."""
    def empty_pl(prizes: int = 6) -> dict:
        return {"active": [], "bench": [], "prize": list(range(prizes)),
                "hand": [], "discard": [], "deckCount": 40, "handCount": 5,
                "poisoned": False, "burned": False, "asleep": False,
                "paralyzed": False, "confused": False}

    # Setup: turn 2, full prizes, empty board → well below any reasonable τ.
    early = {"turn": 2, "yourIndex": 0,
             "players": [empty_pl(), empty_pl()]}
    s0 = readiness_from_state(early, 0)
    # 0.50*(2/10) + 0.30*0 + 0.20*0 = 0.10
    assert abs(s0 - 0.10) < 1e-6, s0
    assert not prefer_mid({"current": early})

    # Contested: turn 8 + 2 prizes taken (empty board) → score 0.50.
    # 0.50*(8/10) + 0.30*(2/6) + 0 = 0.40 + 0.10 = 0.50
    # Pin the formula, not DEFAULT_THRESHOLD (calibrated separately).
    mid = {"turn": 8, "yourIndex": 0,
           "players": [empty_pl(prizes=5), empty_pl(prizes=5)]}
    s1 = readiness_from_state(mid, 0)
    assert abs(s1 - 0.50) < 1e-6, s1
    assert prefer_mid({"current": mid}, threshold=0.5)
    print(f"readiness self-check ok: early={s0:.3f} mid={s1:.3f} "
          f"threshold={DEFAULT_THRESHOLD}")


if __name__ == "__main__":
    _self_check()
