"""Per-option features for the policy net.

Resolves what card/attack an option refers to and produces:
  * dense per-option vector (type one-hot + misc scalars + TARGET STATE)
  * card id (for embedding), attack id (for embedding)
Layout must stay in sync with policy trainer/inference. Bump VERSION on change.

## The v3 block, and the diagnosis behind it (ROADMAP B1, 2026-07-30)

`features.py` has ALWAYS given the net per-slot HP, damage fraction, attached
energy, prize value and best-estimated damage, for all 12 slots. So the standing
description of the blind spot -- "the net cannot see HP" -- was **wrong**, or at
least imprecise in the way that matters. The net can see the whole board's HP.

What it could not see is **which option points at which slot**. The v2 per-option
vector was a type one-hot plus eight scalars, of which the only positional
information was *area* flags (`active` / `bench` / `hand`) -- **`opt["index"]` and
`opt["inPlayIndex"]` were never encoded at all.** Consequences, and they explain
every rule in `targeting.py`:

  * Two options pointing at two different Pokemon on the bench got **identical**
    dense vectors, distinguishable only by the card-id embedding.
  * So two copies of the same card -- two Munkidori, two Dwebble -- were
    **exactly indistinguishable**. That is `energy_spread` (bare vs loaded
    Munkidori, measured 143-to-94 *against* the right answer, i.e. worse than a
    coin flip) and `chip_target` (which of their benched Pokemon dies to 30).

**So the gap was never "no HP features"; it was no BINDING between an option and
its target's state.** The rules work by re-deriving that binding by hand. This
block gives it to the net directly, which is the actual B1 experiment.

⚠ **Appended, never inserted.** Indices 0..24 are byte-identical to v2, so a v2
net still reads exactly what it was trained on. `policynet.Net` derives its own
option width from `head_in` and slices; that is what lets the shipped net and a
candidate net run **in the same process** for a head-to-head A/B (HANDOFF rule 4)
across a feature-layout change. **Never insert into the middle of this vector.**
"""
from __future__ import annotations

import numpy as np

VERSION = 3

N_OPTION_TYPES = 17
OPT_DENSE_V2 = N_OPTION_TYPES + 8      # 25 -- the shipped `policy_lw2` layout
N_TARGET_FEATS = 12                    # the v3 block, appended
OPT_DENSE_V3 = OPT_DENSE_V2 + N_TARGET_FEATS   # 37 -- the shipped v5 layout

# --- the v6 card-attribute block (day 20), appended after v3 ----------------
# `cardType` is the strongest thing the rule-14 gate found anywhere (7 distinct,
# modal 0.416, H/Hmax 0.780 at `opt_card`) and it is absent from BOTH vectors
# today. That is a textbook binding failure of the same family as B1: the state
# vector has carried `supporterPlayed` since v1, but every Trainer -- Item,
# Tool, Supporter, Stadium -- shares option type 7, so the net cannot tell
# which options that flag forbids. It has to infer "is this a Supporter" from
# a card-id embedding row, which is exactly the channel that fails on cards the
# corpus never contained.
N_CARD_TYPES = 7                       # 0 Pokemon .. 6 Special Energy
N_OPT_ATTR = N_CARD_TYPES + 2          # + target has ability, target weak to us
OPT_DENSE = OPT_DENSE_V3 + N_OPT_ATTR
# Widths a net may legitimately have been trained at. The dim guard accepts these
# and nothing else -- an unknown width is a stale net, not a new one.
KNOWN_OPT_DENSE = (OPT_DENSE_V2, OPT_DENSE_V3, OPT_DENSE)
N_ATTACK_IDS = 1600  # option_features returns (dense, card_id, attack_id, target_id)

# --- the v5 pooled option-set block (day 13) --------------------------------
# Every option is scored INDEPENDENTLY against one shared state vector, so the
# net has never been able to see the option SET -- it cannot tell whether it is
# choosing among 3 Trainers in hand or 40 cards in the deck, nor how the option
# in front of it compares to its alternatives. That is the same class of defect
# as B1 (no binding between an option and its target) and §8y (no `effect` card
# saying what kind of choice this is), and it is the last one of the class that
# is cheap: a deep-sets encoder in its minimal form.
#
# phi = the per-option encoding the head already builds
#       [opt_dense[:opt_cols], card_emb, atk_emb, tgt_emb]
# pool = elementwise mean and max over the select's options, plus two count
#        scalars (the count alone answers "3 Trainers or 40 deck cards")
# rho  = the existing state MLP, which now takes the pool as input
#
# ⚠ APPENDED to the STATE vector, after the v4 block, never inserted. A v3/v4
# net slices to its own `state_in` and reads byte-identical input, which is what
# keeps two feature generations runnable in one process (HANDOFF rule 4).
N_POOL_SCALARS = 2


def pool_width(opt_cols: int, emb: int) -> int:
    """Width of the v5 pooled block for a net trained at `opt_cols` and `emb`."""
    return 2 * (opt_cols + 3 * emb) + N_POOL_SCALARS


def pool_scalars(n: int) -> np.ndarray:
    """The two option-count scalars. Linear saturates at 40 (deck searches run
    to 60); the log keeps 2-vs-4 legible, which is where most selects live."""
    v = np.zeros(N_POOL_SCALARS, dtype=np.float32)
    v[0] = min(n, 40) / 40.0
    v[1] = float(np.log1p(n) / np.log(41.0))
    return v

# AreaType ints
_DECK, _HAND, _DISCARD, _ACTIVE, _BENCH, _PRIZE, _STADIUM = 1, 2, 3, 4, 5, 6, 7
_LOOKING = 12

CHIP_DAMAGE = 30      # Shadow Bullet's snipe and Adrena-Brain's 3 counters

# Imported lazily and cached: `targeting` pulls in `cards` + `textdmg`, and this
# module is imported by the dataset builder as well as the agent. No cycle exists
# today (targeting does not import optfeat) -- the laziness is to keep the import
# graph one-directional if that ever changes.
_CDB = None
_BEST_DAMAGE = None


def _cdb():
    global _CDB
    if _CDB is None:
        from . import cards
        _CDB = cards
    return _CDB


def _best_damage():
    """`targeting.best_damage` -- weakness- and payability-aware expected damage.
    Exact for this deck (every payable attack is flat damage); under-reads
    elsewhere, which only makes the feature conservative."""
    global _BEST_DAMAGE
    if _BEST_DAMAGE is None:
        from .targeting import best_damage
        _BEST_DAMAGE = best_damage
    return _BEST_DAMAGE


def _card_at(state: dict, sel: dict, player: int, area: int,
             index: int) -> int:
    """Best-effort card id at (player, area, index); 0 if unknown."""
    try:
        if area == _LOOKING:
            look = state.get("looking")
            if look and index < len(look) and look[index]:
                return look[index]["id"]
            return 0
        if area == _DECK:
            deck = sel.get("deck")
            if deck and index < len(deck) and deck[index]:
                return deck[index]["id"]
            return 0
        pl = state["players"][player]
        if area == _HAND:
            hand = pl.get("hand")
            if hand and index < len(hand) and hand[index]:
                return hand[index]["id"]
            return 0
        if area == _DISCARD:
            if index < len(pl["discard"]):
                return pl["discard"][index]["id"]
            return 0
        if area == _ACTIVE:
            act = pl["active"]
            if act and act[0] is not None:
                return act[0]["id"]
            return 0
        if area == _BENCH:
            if index < len(pl["bench"]) and pl["bench"][index] is not None:
                return pl["bench"][index]["id"]
            return 0
        if area == _PRIZE:
            pr = pl["prize"]
            if index < len(pr) and pr[index] is not None:
                return pr[index]["id"]
            return 0
        if area == _STADIUM:
            st = state.get("stadium") or []
            if st:
                return st[0]["id"]
            return 0
    except (KeyError, IndexError, TypeError):
        return 0
    return 0


def _pokemon_at(state: dict, player: int, area: int, index: int) -> dict | None:
    """The Pokemon dict an option points at, or None. Unlike `_card_at` this
    returns the live object, because the v3 block needs its HP and energy."""
    try:
        pl = state["players"][player]
        if area == _ACTIVE:
            act = pl["active"]
            return act[0] if act and act[0] is not None else None
        if area == _BENCH:
            bench = pl["bench"]
            if 0 <= index < len(bench):
                return bench[index]
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _target_pokemon(state: dict, opt: dict, t: int, me: int,
                    player: int) -> tuple[dict | None, int]:
    """Resolve (the Pokemon this option acts on, its owner).

    Three shapes, matching how the engine words each option type:
      * ATTACH / EVOLVE (8, 9) -> the in-play Pokemon at inPlayArea/inPlayIndex,
        always ours.
      * ATTACK (13)            -> their Active, the thing we are about to hit.
      * everything else        -> (playerIndex, area, index), which is how every
        CARD/ABILITY/damage-counter select names a Pokemon.
    """
    if t in (8, 9):
        return _pokemon_at(state, me, opt.get("inPlayArea") or 0,
                           opt.get("inPlayIndex") or 0), me
    if t == 13:
        return _pokemon_at(state, 1 - me, _ACTIVE, 0), 1 - me
    return _pokemon_at(state, player, opt.get("area") or 0,
                       opt.get("index") or 0), player


def option_features(obs: dict, opt: dict) -> tuple[np.ndarray, int, int, int]:
    """-> (dense vector, card_id, attack_id, target_id) for one option.
    target_id = the in-play Pokemon an ATTACH/EVOLVE points at."""
    state = obs["current"]
    sel = obs["select"]
    me = state["yourIndex"]

    dense = np.zeros(OPT_DENSE, dtype=np.float32)
    t = opt.get("type") or 0
    if t < N_OPTION_TYPES:
        dense[t] = 1.0
    x = N_OPTION_TYPES

    card_id = 0
    attack_id = 0
    target_id = 0

    area = opt.get("area")
    index = opt.get("index") or 0
    player = opt.get("playerIndex")
    player = me if player is None else player

    if t == 7:  # PLAY: index into my hand
        card_id = _card_at(state, sel, me, _HAND, opt.get("index") or 0)
    elif t in (8, 9):  # ATTACH / EVOLVE: card at (area,index) onto target
        card_id = _card_at(state, sel, me, area or _HAND, index)
        target_id = _card_at(state, sel, me, opt.get("inPlayArea") or 0,
                             opt.get("inPlayIndex") or 0)
    elif t in (3, 10, 11):  # CARD / ABILITY / DISCARD
        card_id = _card_at(state, sel, player, area or 0, index)
    elif t == 13:  # ATTACK
        attack_id = opt.get("attackId") or 0
        act = state["players"][me]["active"]
        if act and act[0] is not None:
            card_id = act[0]["id"]
        opp_act = state["players"][1 - me]["active"]
        if opp_act and opp_act[0] is not None:
            target_id = opp_act[0]["id"]
    elif t == 15:  # SKILL
        card_id = opt.get("cardId") or 0
    elif t in (4, 5, 6):  # TOOL_CARD / ENERGY_CARD / ENERGY on a pokemon
        card_id = _card_at(state, sel, player, area or 0, index)

    dense[x + 0] = 1.0 if player == me else 0.0
    dense[x + 1] = (opt.get("number") or 0) / 10.0
    dense[x + 2] = 1.0 if area == _ACTIVE else 0.0
    dense[x + 3] = 1.0 if area == _BENCH else 0.0
    dense[x + 4] = 1.0 if area == _HAND else 0.0
    dense[x + 5] = (opt.get("energyIndex") or 0) / 5.0
    ipa = opt.get("inPlayArea")
    dense[x + 6] = 1.0 if ipa == _ACTIVE else 0.0
    dense[x + 7] = 1.0 if ipa == _BENCH else 0.0

    # --- v3: the option's TARGET state (indices 25..36) -------------------
    # Everything above this line is v2 and must not move.
    v = OPT_DENSE_V2
    pk, owner = _target_pokemon(state, opt, t, me, player)
    if pk is not None:
        hp = pk.get("hp")
        mx = pk.get("maxHp") or 1
        energies = pk.get("energies") or []
        if hp is not None:
            dense[v + 0] = 1.0                          # a target was resolved
            dense[v + 1] = hp / 300.0
            dense[v + 2] = mx / 300.0
            dense[v + 3] = 1.0 - hp / mx                # damage taken, fraction
            # The exact predicate `chip_target` ranks on: both our chip effects
            # deal exactly 30, so "dies to 30" is the whole rule in one bit.
            dense[v + 4] = 1.0 if hp <= CHIP_DAMAGE else 0.0
            dense[v + 5] = _cdb().prize_value(pk["id"]) / 3.0
            dense[v + 6] = min(len(energies), 6) / 6.0
            # Own-type energy count -- `energy_spread`'s entire signal. A bare
            # Munkidori and a loaded one differ HERE and nowhere else in v2.
            own = _cdb().card(pk["id"]).get("energyType")
            dense[v + 7] = min(sum(1 for e in energies
                                   if e == own or e >= 10), 4) / 4.0
            dense[v + 8] = 1.0 if owner == me else 0.0
            # What our Active could actually do to this target right now. 0.0 on
            # a damage-prevention wall, which is the `wall_defer` condition read
            # off the board instead of hardcoded by card id.
            try:
                mypl, oppl = state["players"][me], state["players"][1 - me]
                act = mypl["active"][0] if mypl.get("active") else None
                dmg = _best_damage()(act, mypl, oppl, pk) if act else 0.0
            except (KeyError, IndexError, TypeError):
                dmg = 0.0
            dense[v + 9] = min(dmg, 400.0) / 400.0
            dense[v + 10] = 1.0 if dmg >= hp else 0.0   # we can KO it now
    # Index disambiguation, and the single most load-bearing scalar here: WITHOUT
    # it two options naming two different bench slots are identical vectors.
    slot_ix = (opt.get("inPlayIndex") if t in (8, 9) else opt.get("index")) or 0
    dense[v + 11] = (min(slot_ix, 5) + 1) / 6.0

    # --- v6: card attributes (indices 37..45) -----------------------------
    # Everything above this line is v3 and must not move.
    w = OPT_DENSE_V3
    if card_id:
        ct = _cdb().card(card_id).get("cardType")
        if ct is not None and 0 <= ct < N_CARD_TYPES:
            dense[w + ct] = 1.0
    if pk is not None:
        tc = _cdb().card(pk["id"])
        dense[w + N_CARD_TYPES + 0] = 1.0 if tc.get("skills") else 0.0
        # Same predicate as features.attr_feats' weakHit, but bound to THIS
        # option rather than to a slot -- the B1 lesson is that the binding is
        # what the net cannot re-derive for itself.
        weak = tc.get("weakness") or 0
        try:
            mypl = state["players"][me]
            act = mypl["active"][0] if mypl.get("active") else None
            atk_t = _cdb().card(act["id"]).get("energyType") if act else None
        except (KeyError, IndexError, TypeError):
            atk_t = None
        dense[w + N_CARD_TYPES + 1] = 1.0 if (weak and atk_t == weak) else 0.0

    if attack_id >= N_ATTACK_IDS:
        attack_id = 0
    return dense, card_id, attack_id, target_id
