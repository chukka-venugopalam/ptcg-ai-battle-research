"""State featurization shared by the value-net trainer and the agent.

`featurize(state, me)` -> (dense float32 vector, id-bag int32 arrays).
The id bags are embedded by the net (sum of embedding rows), so the feature
layout here and the net architecture must move together. Bump VERSION when
changing either.
"""
from __future__ import annotations

import numpy as np

from . import cards as cdb
from .textdmg import best_estimated_damage

VERSION = 3
N_CARD_IDS = 1300          # card id space (ids are 1..1267 today)
N_SLOTS = 12               # my active, my bench x5, opp active, opp bench x5
PER_SLOT = 18
N_GLOBAL = 26
DENSE_DIM = N_GLOBAL + N_SLOTS * PER_SLOT

# --- the v4 block (day 12), APPENDED at the very end of the state vector ----
# `scripts/p18_missing_state_audit.py` enumerated every field of the
# observation that `featurize()` never reads and measured how much each varies
# at a real decision point. It killed three candidates that had been on the
# plan for two days -- opponent hand size, prizes remaining and turn number are
# ALL already encoded above (lines "put(...)" below) -- and two it found
# itself: `remainDamageCounter` is 0 at 100% of decisions and `remainEnergyCost`
# at 99.1%, so neither can explain a single miss.
#
# What survived, with the miss mass it targets (v3 net, 12,939 held-out rows):
#   * turnActionCount -- 20 distinct values, modal share 17% in MAIN.
#     MAIN is 2,629 of 3,902 misses. The net re-scores a barely-changed board
#     several times per turn with no idea how deep into the turn it is.
#   * the select's EFFECT card -- which card caused this select. Modal share
#     26% in TO_HAND (674 misses): the same context means "tutor a Trainer"
#     (Petrel), "take a Supporter" (Poke Pad), "recover from discard" (Night
#     Stretcher) or "search anything" (Ultra Ball), and the net scores each
#     option INDEPENDENTLY, so it never sees the option set that would reveal
#     which. This is the one input that tells it what kind of choice it is.
#   * the stadium -- 7 distinct, 61% Spikemuth Gym, and Area Zero Underdepths
#     changes the bench size. Absent entirely, including from every id bag.
#   * `retreated` / `stadiumPlayed` -- the two missing members of the
#     once-per-turn quartet whose other two (`supporterPlayed`,
#     `energyAttached`) have been encoded since v1. `retreated` is 43%
#     non-modal in SWITCH.
#
# ⚠ APPENDED, NEVER INSERTED -- and the append lands after `seld`, which is the
# LAST block of the state vector (see policynet.scores / train_policy.forward).
# A v3 net simply slices to its own `state_in` and reads byte-identical input,
# which is what lets v3 and v4 run head-to-head in one process (rule 4).
N_EXTRA = 8                # dense scalars
N_XSLOT = 2                # card ids embedded through the existing slot table:
#                            (stadium in play, the select's effect card)

# --- ablating the v4 block (day 13) -----------------------------------------
# The block shipped whole, so nothing said WHICH member bought the 37 Elo. Each
# name below is a drop-one arm; the trainer zeroes those columns of the corpus
# and records the surviving mask in the npz, so `policynet` reproduces exactly
# the input the net was trained on WITHOUT rebuilding a corpus or changing a
# single layer width. Same arch, same init, same rows -- only the content of a
# few columns differs, which is a tighter control than removing the dimensions.
# Indices are (xdense col, ...) then (N_EXTRA + xslot col, ...).
X_GROUPS: dict[str, tuple[int, ...]] = {
    "turnAction": (0,),
    "retreated": (1,),
    "stadiumPlayed": (2,),
    "stadium": (3, N_EXTRA + 0),      # the in-play flag AND the card id
    "benchMax": (4,),
    "tools": (5, 6),
    "poolSize": (7,),
    "effect": (N_EXTRA + 1,),         # which card caused this select
}


def extra_feats(state: dict, sel: dict,
                me: int) -> tuple[np.ndarray, np.ndarray]:
    """The v4 block: (dense scalars, card ids to embed). See the note above."""
    x = np.zeros(N_EXTRA, dtype=np.float32)
    mypl, oppl = state["players"][me], state["players"][1 - me]
    stad = state.get("stadium") or []

    def tools(pl) -> int:
        n = 0
        for pk in ([pl["active"][0] if pl["active"] else None]
                   + list(pl["bench"])):
            if pk:
                n += len(pk.get("tools") or [])
        return n

    x[0] = min(int(state.get("turnActionCount") or 0), 24) / 24.0
    x[1] = 1.0 if state.get("retreated") else 0.0
    x[2] = 1.0 if state.get("stadiumPlayed") else 0.0
    x[3] = 1.0 if stad else 0.0
    x[4] = (mypl.get("benchMax") or 5) / 8.0
    x[5] = min(tools(mypl), 4) / 4.0
    x[6] = min(tools(oppl), 4) / 4.0
    x[7] = min(len(sel.get("deck") or []), 60) / 60.0

    eff = sel.get("effect")
    ids = np.zeros(N_XSLOT, dtype=np.int32)
    ids[0] = stad[0]["id"] if stad else 0
    ids[1] = (eff or {}).get("id", 0) if isinstance(eff, dict) else 0
    ids[ids >= N_CARD_IDS] = 0
    return x, ids

# --- the v6 card-attribute block (day 20), APPENDED after the v4 block -------
# E6 (docs/experiments/embeddings/E6-identity-channel.md) priced the identity
# channel by permuting embedding rows on the frozen v5 net: scrambling only the
# OPPONENT's card ids costs 0.838 -> 0.587 against rule:crustle, whose four
# Pokemon are all in vocabulary, and 0.625 -> 0.607 against rule:v10, whose six
# are all OUT of it. We do not read Mega Lucario badly; we cannot read it at
# all, because `slot_emb` has no trained row for any of its Pokemon.
#
# A per-card embedding row can only ever describe cards the corpus contained.
# These attributes come from the card DB, which covers all 1,267 cards, so an
# unseen Pokemon arrives as "Fighting, weak to Psychic, has an ability" instead
# of an untrained N(0, 1) vector. That is the only channel here that transfers.
#
# Sized BEFORE building (rule 14, scripts/p55_attr_sizing.py, at the decision):
#   energyType   10 distinct  modal 0.438  H/Hmax 0.720   at the opp active
#   weakness      9 distinct  modal 0.443  H/Hmax 0.732
#   hasAbility    2 distinct  modal 0.717  H/Hmax 0.860
#   weak-to-our-active's-type fires on 12.1% of decisions
# and the gate KILLED two candidates before they cost anything: `aceSpec` is a
# single value across the whole corpus, and `pokemonType`/`evolutionType` are
# fully redundant -- the six flags at slot +4..+9 give 12 distinct signatures
# and none maps to more than one value of either. Shipping those would have
# repeated EVIDENCE 8ab, where five leftover columns measured -22 Elo against
# having no block at all.
#
# `resistance` is marginal (3 distinct, modal 0.835) so it gets one flag, not
# a one-hot.
N_ATTR_ETYPE = 11          # energyType, 0..10
N_ATTR_WEAK = 9            # weakness: index 0 = none, else the type 1..8
PER_SLOT_ATTR = N_ATTR_ETYPE + N_ATTR_WEAK + 3   # + ability, resist, weakHit
N_ATTR = N_SLOTS * PER_SLOT_ATTR

# Same drop-one machinery as X_GROUPS, over columns of the attr vector. The
# block ships whole, so without these nothing would say WHICH member paid.
_A_ETYPE, _A_WEAK = 0, N_ATTR_ETYPE
_A_ABILITY = _A_WEAK + N_ATTR_WEAK
_A_RESIST, _A_WEAKHIT = _A_ABILITY + 1, _A_ABILITY + 2


def _a_group(lo: int, hi: int) -> tuple[int, ...]:
    return tuple(s * PER_SLOT_ATTR + i
                 for s in range(N_SLOTS) for i in range(lo, hi))


A_GROUPS: dict[str, tuple[int, ...]] = {
    "attrEnergyType": _a_group(_A_ETYPE, _A_ETYPE + N_ATTR_ETYPE),
    "attrWeakness": _a_group(_A_WEAK, _A_WEAK + N_ATTR_WEAK),
    "attrAbility": _a_group(_A_ABILITY, _A_ABILITY + 1),
    "attrResist": _a_group(_A_RESIST, _A_RESIST + 1),
    "attrWeakHit": _a_group(_A_WEAKHIT, _A_WEAKHIT + 1),
}


def attr_feats(state: dict, me: int) -> np.ndarray:
    """The v6 block: card attributes for all 12 slots, same slot order as
    `featurize` (my active, my bench x5, opp active, opp bench x5)."""
    a = np.zeros(N_ATTR, dtype=np.float32)
    mypl, oppl = state["players"][me], state["players"][1 - me]

    def active_type(pl) -> int | None:
        act = pl["active"][0] if pl["active"] else None
        return cdb.card(act["id"]).get("energyType") if act else None

    # the type that would be ATTACKING each side's slots
    facing = (active_type(oppl), active_type(mypl))

    slots = ([mypl["active"][0] if mypl["active"] else None]
             + [(mypl["bench"][i] if i < len(mypl["bench"]) else None)
                for i in range(5)]
             + [oppl["active"][0] if oppl["active"] else None]
             + [(oppl["bench"][i] if i < len(oppl["bench"]) else None)
                for i in range(5)])

    for si, pk in enumerate(slots):
        if pk is None:
            continue
        c = cdb.card(pk["id"])
        base = si * PER_SLOT_ATTR

        et = c.get("energyType") or 0
        if 0 <= et < N_ATTR_ETYPE:
            a[base + _A_ETYPE + et] = 1.0

        weak = c.get("weakness") or 0
        if 0 <= weak < N_ATTR_WEAK:
            a[base + _A_WEAK + weak] = 1.0

        a[base + _A_ABILITY] = 1.0 if c.get("skills") else 0.0
        a[base + _A_RESIST] = 1.0 if c.get("resistance") else 0.0
        # does whatever is facing this slot hit it for weakness?
        att = facing[0 if si < 6 else 1]
        a[base + _A_WEAKHIT] = 1.0 if (weak and att == weak) else 0.0

    return a


# id bags: per-slot card id (12), my hand, my discard, opp discard, opp known
BAG_NAMES = ("slots", "my_hand", "my_discard", "opp_discard")


def _slot_feats(pk: dict | None, mypl: dict, oppl: dict, out: np.ndarray,
                base: int) -> int:
    """Write PER_SLOT features for one pokemon slot; return its card id."""
    if pk is None:
        return 0
    c = cdb.card(pk["id"])
    hp = pk["hp"]
    max_hp = pk["maxHp"] or 1
    out[base + 0] = 1.0
    out[base + 1] = hp / 300.0
    out[base + 2] = max_hp / 300.0
    out[base + 3] = 1.0 - hp / max_hp
    out[base + 4] = 1.0 if c.get("basic") else 0.0
    out[base + 5] = 1.0 if c.get("stage1") else 0.0
    out[base + 6] = 1.0 if c.get("stage2") else 0.0
    out[base + 7] = 1.0 if c.get("ex") else 0.0
    out[base + 8] = 1.0 if c.get("megaEx") else 0.0
    out[base + 9] = 1.0 if c.get("tera") else 0.0
    out[base + 10] = min(len(pk["energies"]), 6) / 6.0
    out[base + 11] = (c.get("retreatCost") or 0) / 4.0
    out[base + 12] = 1.0 if pk["appearThisTurn"] else 0.0
    out[base + 13] = min(best_estimated_damage(pk, mypl, oppl), 400) / 400.0
    out[base + 14] = len(pk["tools"]) / 1.0 if pk["tools"] else 0.0
    out[base + 15] = cdb.prize_value(pk["id"]) / 3.0
    out[base + 16] = _cost_satisfaction(pk)
    own = c.get("energyType")
    out[base + 17] = min(sum(1 for e in pk["energies"]
                             if e == own or e >= 10), 4) / 4.0
    return pk["id"]


def _cost_satisfaction(pk: dict) -> float:
    """How close the attached energy comes to paying this Pokemon's cheapest
    attack (1.0 = can attack now)."""
    best = 0.0
    have = pk["energies"]
    for aid in cdb.card(pk["id"]).get("attacks") or []:
        a = cdb.attacks().get(aid)
        if not a:
            continue
        cost = a["energies"]
        if not cost:
            return 1.0
        if cdb.energy_satisfied(cost, have):
            return 1.0
        best = max(best, min(len(have), len(cost)) / len(cost))
    return best


def featurize(state: dict, me: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    opp = 1 - me
    mypl = state["players"][me]
    oppl = state["players"][opp]

    dense = np.zeros(DENSE_DIM, dtype=np.float32)
    g = 0

    def put(v):
        nonlocal g
        dense[g] = v
        g += 1

    put(min(state["turn"], 40) / 40.0)
    put(1.0 if state["firstPlayer"] == me else 0.0)
    put((6 - len(mypl["prize"])) / 6.0)      # my prizes taken
    put((6 - len(oppl["prize"])) / 6.0)
    put(len(mypl["prize"]) / 6.0)
    put(len(oppl["prize"]) / 6.0)
    put(min(mypl["deckCount"], 60) / 60.0)
    put(min(oppl["deckCount"], 60) / 60.0)
    put(1.0 if mypl["deckCount"] == 0 else 0.0)
    put(1.0 if oppl["deckCount"] == 0 else 0.0)
    put(min(mypl["handCount"], 12) / 12.0)
    put(min(oppl["handCount"], 12) / 12.0)
    put(1.0 if state["supporterPlayed"] else 0.0)
    put(1.0 if state["energyAttached"] else 0.0)
    for pl in (mypl, oppl):
        put(1.0 if pl["poisoned"] else 0.0)
        put(1.0 if pl["burned"] else 0.0)
        put(1.0 if pl["asleep"] else 0.0)
        put(1.0 if pl["paralyzed"] else 0.0)
        put(1.0 if pl["confused"] else 0.0)
    put(sum(1 for pk in mypl["bench"] if pk is not None) / 5.0)
    put(sum(1 for pk in oppl["bench"] if pk is not None) / 5.0)
    assert g == N_GLOBAL, g

    slot_ids = np.zeros(N_SLOTS, dtype=np.int32)
    slots = ([mypl["active"][0] if mypl["active"] else None]
             + [(mypl["bench"][i] if i < len(mypl["bench"]) else None)
                for i in range(5)]
             + [oppl["active"][0] if oppl["active"] else None]
             + [(oppl["bench"][i] if i < len(oppl["bench"]) else None)
                for i in range(5)])
    for si, pk in enumerate(slots):
        side_my, side_opp = (mypl, oppl) if si < 6 else (oppl, mypl)
        cid = _slot_feats(pk, side_my, side_opp,
                          dense, N_GLOBAL + si * PER_SLOT)
        slot_ids[si] = cid if cid < N_CARD_IDS else 0

    def bag(cards_list) -> np.ndarray:
        return np.asarray([c["id"] for c in cards_list
                           if c is not None and c["id"] < N_CARD_IDS],
                          dtype=np.int32)

    bags = {
        "slots": slot_ids,
        "my_hand": bag(mypl["hand"] or []),
        "my_discard": bag(mypl["discard"]),
        "opp_discard": bag(oppl["discard"]),
    }
    return dense, bags
