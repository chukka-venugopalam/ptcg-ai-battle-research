"""Budget the 600s per-game overage pool across our decisions."""
from __future__ import annotations

import os

RESERVE_S = 45.0        # never plan to dip into the last chunk
EMERGENCY_S = 0.03      # per-decision spend once the pool is nearly gone
MAIN_CAP_S = float(os.environ.get("SA_MAIN_CAP", "4.5"))
MINOR_CAP_S = float(os.environ.get("SA_MINOR_CAP", "2.0"))
SPEND_MULT = float(os.environ.get("SA_SPEND_MULT", "1.0"))

MAIN = 0  # SelectType


class TimeManager:
    def __init__(self, main_cap: float | None = None,
                 minor_cap: float | None = None,
                 spend_mult: float | None = None):
        # per-instance overrides so two configs can be A/B'd in one process
        # (the SA_* env vars are module-level and apply to both sides)
        self.main_cap = MAIN_CAP_S if main_cap is None else main_cap
        self.minor_cap = MINOR_CAP_S if minor_cap is None else minor_cap
        self.spend_mult = SPEND_MULT if spend_mult is None else spend_mult

    def budget(self, obs: dict, n_combos: int = 4) -> float:
        rem = float(obs.get("remainingOverageTime", 600.0))
        avail = rem - RESERVE_S
        if avail <= 5.0:
            return EMERGENCY_S
        turn = obs["current"]["turn"]
        # crude forecast of remaining *searchable* decisions this game
        est_left = max(18.0, 75.0 - 2.0 * turn)
        per = (avail / est_left) * self.spend_mult
        sel = obs["select"]
        # branchier decisions deserve more of the pool
        width = 0.6 + 0.08 * min(n_combos, 20)
        if sel["type"] == MAIN:
            return max(0.05, min(per * 1.6 * width, self.main_cap))
        return max(0.04, min(per * width, self.minor_cap))
