"""Observable matchup routing for E2 residual adapters.

Routes are inferred only from cards the acting seat can see on the opponent:
active, bench, and discard. Prize, hand, and deck contents are never used, and
post-game census labels are never used at inference.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Stable integer ids used by the trainer and the exported checkpoint.
ROUTE_GENERAL = 0
ROUTE_MIRROR = 1
ROUTE_ALAKAZAM = 2

ROUTE_NAMES = {
    ROUTE_GENERAL: "general",
    ROUTE_MIRROR: "mirror",
    ROUTE_ALAKAZAM: "alakazam",
}
NAME_TO_ROUTE = {name: rid for rid, name in ROUTE_NAMES.items()}

# Visible evolution lines. Alakazam is checked first so a board that somehow
# shows both lines (never observed in pds_v4) prefers the rarer specialist.
ALAKAZAM_IDS = frozenset({741, 742, 743})  # Abra / Kadabra / Alakazam
MIRROR_IDS = frozenset({646, 647, 648})    # Impidimp / Morgrem / Grimmsnarl


def route_from_ids(ids: Iterable[int]) -> int:
    """Map a set of visible opponent card ids to a route id."""
    seen = {int(x) for x in ids if int(x)}
    if seen & ALAKAZAM_IDS:
        return ROUTE_ALAKAZAM
    if seen & MIRROR_IDS:
        return ROUTE_MIRROR
    return ROUTE_GENERAL


def visible_opponent_ids(obs: dict) -> set[int]:
    """Collect opponent active/bench/discard card ids from an observation."""
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    players = state.get("players") or []
    if me not in (0, 1) or len(players) < 2:
        return set()
    op = players[1 - int(me)] or {}
    ids: set[int] = set()

    def add_card(card) -> None:
        if not card:
            return
        if isinstance(card, dict):
            cid = card.get("id")
            if cid:
                ids.add(int(cid))
            for attached in card.get("cards") or []:
                add_card(attached)
        else:
            ids.add(int(card))

    active = op.get("active") or []
    if active:
        add_card(active[0] if isinstance(active, list) else active)
    for pk in op.get("bench") or []:
        add_card(pk)
    for card in op.get("discard") or []:
        add_card(card)
    return ids


def route_from_obs(obs: dict) -> int:
    """Hard route for a live observation."""
    return route_from_ids(visible_opponent_ids(obs))


def routes_from_corpus(slots: np.ndarray, opp_discard_flat: np.ndarray,
                       opp_discard_off: np.ndarray) -> np.ndarray:
    """Vector of route ids for corpus rows.

    `slots` columns 6..11 are the opponent active and five bench card ids,
    matching `features.featurize`. Opponent discard is the only other bag that
    is both observable and stored in the shards.
    """
    n = len(slots)
    routes = np.zeros(n, dtype=np.int64)
    opp_slots = slots[:, 6:].astype(np.int64, copy=False)
    ala = np.array(sorted(ALAKAZAM_IDS), dtype=np.int64)
    mir = np.array(sorted(MIRROR_IDS), dtype=np.int64)
    # Slot hits are cheap to vectorize; discard needs a per-row scan.
    slot_ala = np.isin(opp_slots, ala).any(axis=1)
    slot_mir = np.isin(opp_slots, mir).any(axis=1)
    routes[slot_ala] = ROUTE_ALAKAZAM
    routes[~slot_ala & slot_mir] = ROUTE_MIRROR
    # Fill remaining rows that only reveal the line through discard.
    need = np.where(~slot_ala & ~slot_mir)[0]
    if len(need) and len(opp_discard_flat):
        flat = opp_discard_flat.astype(np.int64, copy=False)
        for i in need:
            a, b = int(opp_discard_off[i]), int(opp_discard_off[i + 1])
            if b <= a:
                continue
            chunk = flat[a:b]
            if np.isin(chunk, ala).any():
                routes[i] = ROUTE_ALAKAZAM
            elif np.isin(chunk, mir).any():
                routes[i] = ROUTE_MIRROR
    return routes
