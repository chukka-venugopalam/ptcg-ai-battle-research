"""Determinization: fill in hidden zones to produce search_begin arguments.

Our own hidden zones (deck order, prizes) are sampled from the known 60-card
list minus everything visible. The opponent's are sampled from a predicted
decklist chosen from a library of top-meta decks by best overlap with what
they have revealed.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from . import cards as cdb

_LIB_PATH = Path(__file__).resolve().parent / "deck_library.json"
_FALLBACK_ENERGY = 3  # Basic {W} Energy: safe pad card


def _load_library() -> list[tuple[Counter, float]]:
    if _LIB_PATH.exists():
        data = json.loads(_LIB_PATH.read_text(encoding="utf-8"))
        return [(Counter({int(k): v for k, v in d["cards"].items()}),
                 float(d.get("weight", 1.0))) for d in data]
    return []


_library: list[tuple[Counter, float]] | None = None


def library() -> list[tuple[Counter, float]]:
    global _library
    if _library is None:
        _library = _load_library()
    return _library


def public_cards(state: dict, p: int, include_hand: bool) -> list[int]:
    """Every card of player p whose identity is visible to us."""
    pl = state["players"][p]
    out: list[int] = []
    for c in pl["discard"]:
        out.append(c["id"])
    for pk in list(pl["active"]) + list(pl["bench"]):
        if pk is None:
            continue
        out.append(pk["id"])
        for c in pk["energyCards"]:
            out.append(c["id"])
        for c in pk["tools"]:
            out.append(c["id"])
        for c in pk["preEvolution"]:
            out.append(c["id"])
    for c in state.get("stadium") or []:
        if c["playerIndex"] == p:
            out.append(c["id"])
    for c in pl["prize"]:
        if c is not None:
            out.append(c["id"])
    if include_hand and pl["hand"] is not None:
        for c in pl["hand"]:
            out.append(c["id"])
    return out


def predict_opp_deck(observed: Counter) -> Counter:
    """Best-matching library decklist for the observed opponent cards."""
    best: Counter | None = None
    best_score = None
    for dl, weight in library():
        overflow = sum((observed - dl).values())  # observed beyond decklist
        score = (-overflow, weight)
        if best_score is None or score > best_score:
            best_score = score
            best = dl
    if best is not None:
        return best
    # no library: fabricate observed + energy padding
    fab = Counter(observed)
    for cid, cnt in list(observed.items()):
        if cdb.is_basic_pokemon(cid):
            fab[cid] = max(cnt, 3)
    total = sum(fab.values())
    if total < 60:
        fab[_FALLBACK_ENERGY] += 60 - total
    return fab


class World:
    """search_begin argument bundle."""
    __slots__ = ("my_deck", "my_prize", "opp_deck", "opp_prize", "opp_hand",
                 "opp_active")

    def __init__(self, my_deck, my_prize, opp_deck, opp_prize, opp_hand,
                 opp_active):
        self.my_deck = my_deck
        self.my_prize = my_prize
        self.opp_deck = opp_deck
        self.opp_prize = opp_prize
        self.opp_hand = opp_hand
        self.opp_active = opp_active


def _fill_positions(prize_slots: list, pool: list[int],
                    rng: random.Random) -> list[int]:
    """Prize list honoring revealed positions, drawing unknowns from pool."""
    out = []
    for slot in prize_slots:
        if slot is not None:
            out.append(slot["id"])
        else:
            out.append(pool.pop() if pool else _FALLBACK_ENERGY)
    return out


def determinize(obs: dict, my_decklist: list[int], known_opp_hand: list[int],
                rng: random.Random) -> World:
    state = obs["current"]
    me = state["yourIndex"]
    opp = 1 - me
    mypl = state["players"][me]
    oppl = state["players"][opp]

    # ---- my hidden zones -----------------------------------------------------
    my_pool = list(my_decklist)
    for cid in public_cards(state, me, include_hand=True):
        if cid in my_pool:
            my_pool.remove(cid)
    # if the select is showing our own deck, those cards are known to be in
    # the deck -- the prize pool is exactly the remainder
    sel_deck = (obs.get("select") or {}).get("deck")
    if sel_deck:
        for c in sel_deck:
            if c and c.get("playerIndex") == me and c["id"] in my_pool:
                my_pool.remove(c["id"])
    rng.shuffle(my_pool)
    # revealed prizes were already excluded from the pool
    my_unrevealed = [c for c in mypl["prize"] if c is None]
    n_deck = mypl["deckCount"]
    # pool = deck + unrevealed prizes; sample prizes from it
    my_prize = _fill_positions(mypl["prize"], my_pool, rng)
    my_deck = my_pool[:]  # remainder
    while len(my_deck) < n_deck:
        my_deck.append(_FALLBACK_ENERGY)
    del my_unrevealed

    # ---- opponent hidden zones ----------------------------------------------
    observed = Counter(public_cards(state, opp, include_hand=False))
    known_hand = list(known_opp_hand)
    predicted = predict_opp_deck(observed + Counter(known_hand))
    fill = list((predicted - observed - Counter(known_hand)).elements())
    rng.shuffle(fill)

    n_opp_hand = oppl["handCount"]
    n_opp_deck = oppl["deckCount"]
    needed = (n_opp_hand - len(known_hand)) + n_opp_deck + \
        sum(1 for c in oppl["prize"] if c is None)
    while len(fill) < needed:
        fill.append(_FALLBACK_ENERGY)

    opp_hand = known_hand[: n_opp_hand]
    while len(opp_hand) < n_opp_hand:
        opp_hand.append(fill.pop())
    opp_prize = _fill_positions(oppl["prize"], fill, rng)
    opp_deck = fill[:]
    while len(opp_deck) < n_opp_deck:
        opp_deck.append(_FALLBACK_ENERGY)

    # ---- face-down opponent active ------------------------------------------
    opp_active: list[int] = []
    active = oppl["active"]
    if len(active) > 0 and active[0] is None:
        basics = [cid for cid in (opp_hand + opp_deck)
                  if cdb.is_basic_pokemon(cid)]
        if not basics:
            basics = [cid for cid in predicted if cdb.is_basic_pokemon(cid)]
        opp_active = [rng.choice(basics)] if basics else [305]  # Dunsparce

    return World(my_deck, my_prize, opp_deck, opp_prize, opp_hand, opp_active)
