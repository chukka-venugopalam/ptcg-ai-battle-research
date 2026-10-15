"""Aim chip damage at the Pokemon it can actually finish.

Measured defect (`scripts/opportunity_audit.py`, 80 games): when the clone
chooses which of the opponent's Pokemon to point an effect at, it picks the
lowest-HP candidate 25.7% of the time for Adrena-Brain's counter move and 42.1%
for Shadow Bullet's bench snipe. With 2-4 candidates on board that is chance.

The cause is in the features, not the weights -- but ⚠ **the original statement of
it here was WRONG, corrected 2026-07-30** (`report/EVIDENCE.md` §8f). It read "no
HP and no damage", which is false: `features.py` has always given the net per-slot
HP, damage fraction, attached energy and prize value for all 12 slots.

**The actual defect is narrower.** The v2 per-option vector encoded position only
as *area* flags (active / bench / hand) and **never encoded `opt["index"]`**. So
two options naming two different benched Pokemon were identical vectors apart from
the card-id embedding -- and two options naming **two copies of the same card were
bitwise identical inputs with different right answers.** The net could see the
board; it could not see which option pointed where. That is why these rules win:
they restore a missing *binding*, not missing arithmetic.

`optfeat` v3 gives the net that binding directly (target HP, dies-to-30, own-type
energy, our damage into it, and the slot index). Whether it makes these rules
redundant is ROADMAP B1, measured head-to-head in the arena -- not by val
accuracy, which has failed to predict strength five times (rule 3).

Both effects deal exactly 30: Shadow Bullet's "30 damage to 1 of your
opponent's Benched Pokemon", and Adrena-Brain's "move up to 3 damage counters".
So the rule is: if anything dies to 30, kill the one worth the most prizes;
otherwise concentrate on the one closest to dying.

Deliberately narrow. It fires only when *every* option in the select resolves
to an opponent Pokemon, which leaves the mixed selects (Adrena-Brain's "from 1
of YOUR Pokemon" source pick) to the net.

`energy_spread` is the same idea one select over: the net cannot see how much
energy an ATTACH target already carries, so it stacks a second {D} on a
Munkidori that already has one instead of arming a bare one.
"""
from __future__ import annotations

from . import cards
from .textdmg import estimate

# SelectContext ints (mirror cg.api without importing it)
DAMAGE_COUNTER = 13
DAMAGE_COUNTER_ANY = 14
DAMAGE = 15
CHIP_CONTEXTS = (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE)

# AreaType ints
_HAND, _ACTIVE, _BENCH = 2, 4, 5

MAIN = 0            # SelectContext.MAIN
SWITCH = 3          # SelectContext.SWITCH -- what Boss's Orders drags with
REMOVE_DAMAGE_COUNTER = 16   # Adrena-Brain's SOURCE pick, all options ours
MAX_MOVE = 3        # Adrena-Brain moves "up to 3 damage counters"
OPT_PLAY = 7        # OptionType.PLAY
OPT_ATTACH = 8      # OptionType.ATTACH
BOSS_ORDERS = 1182
MUNKIDORI = 112
POFFIN = 1086       # Buddy-Buddy Poffin -- E11
DARK_ENERGY = 7     # Basic {D} Energy, card id
DARK_TYPE = 7       # ... and energy type; pk["energies"] holds types

# Shadow Bullet's bench snipe and Adrena-Brain's 3 counters are both 30.
CHIP_DAMAGE = 30

# Pokemon whose ABILITY prevents damage from our {ex} attacks, so our only way
# to remove them is damage counters. Verified in-engine 2026-07-30 over 60 games
# (`scripts/p3_crustle_probe.py`): 209 of 224 attack-damage events onto Crustle
# logged `value: 0` (93.3% prevented), while 1,298 of 1,386 damage-counter events
# landed (93.7%). See `report/EVIDENCE.md` §8d.
#
# ⚠ Hardcoded card ids because the card db exposes no ability text for 345
# (`abilities: None`), so the condition cannot be read off the card, and
# `best_damage` does not model prevention either -- it happily reports 180 for
# Shadow Bullet into a Crustle. V10 hardcodes 344/345 the same way. The general
# version would learn it in-game from the event log (an attack that logged 0
# against this card id), which is the upgrade path if a second wall appears.
WALL_POKEMON = frozenset({345})   # Crustle -- Mysterious Rock Inn

# --- E21: Petrel's fetch (day 30) --------------------------------------------
FETCH = 7            # SelectContext of a "search your deck" resolution
PETREL = 1219        # Team Rocket's Petrel -- "search your deck for a Trainer"
SPIKEMUTH = 1259     # Spikemuth Gym (Stadium, x4)
TOOL_SCRAPPER = 1137  # "discard up to 2 Pokemon Tools" (x1)


def _pokemon_at(state: dict, player: int, area: int, index: int) -> dict | None:
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


def chip_target(obs: dict, wall_defer: bool = False) -> list[int] | None:
    """Ranked option indices for an opponent-targeting select, else None.

    `wall_defer` (`bc:<label>,wall`) is the matchup branch measured on
    2026-07-30. This rule ranks "dies to 30 first, most prizes among those, then
    lowest HP" -- which is right when prizes are the currency, and **wrong
    against a deck whose Active cannot be damaged by our attacks at all**. There,
    damage counters are the only way to remove the blocker, and spending them to
    farm a 1-prize Dwebble loses the game slowly.

    Measured (n=2000 each, fixed `rule:crustle` opponent): `bc` scores 0.559 and
    `bc:x,noChip` scores **0.685** -- this rule is worth **-0.126** in that
    matchup while being worth +0.077 head-to-head in the mirror. The cause is
    measured too: with the rule on, 235 counter-placement events land on Dwebble
    and 1,386 on Crustle; with it off, Dwebble drops to **24** and Crustle rises
    to **1,583** at a higher mean (12.9 -> 15.0). `report/EVIDENCE.md` §8c.

    So when their Active is a wall, hand the select back to the net -- which was
    measured to concentrate counters correctly on its own. Deliberately the
    one-line version of the fix: a bespoke wall-aware ranker is only worth
    building if this fails to recover the -0.126 (HANDOFF §3.3).
    """
    sel = obs.get("select") or {}
    if sel.get("context") not in CHIP_CONTEXTS:
        return None
    options = sel.get("option") or []
    if len(options) < 2:
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None

    if wall_defer:
        try:
            opp_active = state["players"][1 - me]["active"]
            active = opp_active[0] if opp_active else None
        except (KeyError, IndexError, TypeError):
            active = None
        if active is not None and active.get("id") in WALL_POKEMON:
            return None  # counters are our only out here: let the net aim them

    scored: list[tuple[tuple, int]] = []
    for i, opt in enumerate(options):
        player = opt.get("playerIndex")
        if player is None or player == me:
            return None  # mixed or own-side select: leave it to the net
        pk = _pokemon_at(state, player, opt.get("area"), opt.get("index") or 0)
        if pk is None:
            return None  # something we cannot read: leave it to the net
        hp = pk.get("hp")
        if hp is None:
            return None
        kills = hp <= CHIP_DAMAGE
        # kills first, most prizes among those, then closest to dying
        scored.append(((0 if kills else 1,
                        -cards.prize_value(pk["id"]) if kills else 0,
                        hp), i))

    scored.sort()
    return [i for _, i in scored]


def _hand_card_id(state: dict, me: int, index: int) -> int:
    try:
        hand = state["players"][me]["hand"]
        card = hand[index]
        return card["id"] if card else 0
    except (KeyError, IndexError, TypeError):
        return 0


def _dark(pk: dict) -> int:
    return sum(1 for e in (pk.get("energies") or []) if e == DARK_TYPE)


def energy_spread(obs: dict, chosen: list[int]) -> list[int] | None:
    """Redirect a wasted second {D} on a Munkidori to a bare one.

    Verified in-engine over 40 games: Adrena-Brain is once **per Pokemon**
    (we activated it twice in a turn 35 times, and a slot that had used it was
    never re-offered), and its "has any {D} Energy attached" is a **threshold,
    not a cost** -- the energy is never consumed (n=138, unchanged every time).
    So two Munkidori holding one {D} each move 6 damage counters a turn; one
    Munkidori holding two moves 3, and the second {D} does nothing else either:
    Munkidori's only attack is Mind Bend, cost {P}{C}, and this deck runs zero
    Psychic energy. Nor is Munkidori a *Marnie's* Pokemon, so Grimmsnarl ex's
    Punk Up cannot attach to it -- the 1-per-turn hand attach is the only
    source, which is what makes spending it on a no-op expensive.

    Measured on the shipped clone (`opportunity_audit.py`, 150 games): when a
    select offered both a bare and an already-loaded Munkidori and the clone
    attached to one of them, it picked the loaded one **143 times to 94** --
    worse than a coin flip, because `optfeat` gives it no attached-energy count.

    Narrow on purpose: this only reorders *which Munkidori* gets the energy the
    net already decided to attach. It never creates or suppresses an attach.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    if not chosen:
        return None
    pick = chosen[0]
    if not 0 <= pick < len(options):
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None

    bare: list[int] = []
    loaded: list[int] = []
    for i, opt in enumerate(options):
        if opt.get("type") != OPT_ATTACH:
            continue
        if (opt.get("area") or _HAND) != _HAND:
            continue
        if _hand_card_id(state, me, opt.get("index") or 0) != DARK_ENERGY:
            continue
        pk = _pokemon_at(state, me, opt.get("inPlayArea"),
                         opt.get("inPlayIndex") or 0)
        if pk is None or pk.get("id") != MUNKIDORI:
            continue
        (bare if _dark(pk) == 0 else loaded).append(i)

    if pick not in loaded or not bare:
        return None
    return [bare[0]] + [i for i in chosen[1:] if i != bare[0]]


def counter_source(obs: dict, chosen: list[int], rank) -> list[int] | None:
    """Take Adrena-Brain's counters off a Pokemon that HAS three of them.

    Adrena-Brain moves "up to 3 damage counters" from one of our Pokemon to one
    of theirs, and the source is its own select (REMOVE_DAMAGE_COUNTER, all
    options ours). How many it then moves is capped by what the source actually
    carries -- the follow-up REMOVE_DAMAGE_COUNTER_COUNT select offers "1,2,3"
    off a source with 3+ counters but only "1,2" off a source with 2. The clone
    already takes the maximum on that second select **100% of the time**
    (n=481, 120 games), so all the loss is here, one select earlier.

    Measured on the shipped clone (120 games, 291 source selects with >= 2
    options): in **59 of them (20.3%)** it picked a source that moves fewer
    counters than an available alternative -- 10 or 20 damage where 30 was on
    the table. With the rule that goes to 0, and activations that move the full
    3 counters rise from 67.1% to 76.5%.

    arena: `bc:s,src` vs `bc` = **0.534 [0.513, 0.556], n=2000**, mirror.

    This is the `energy_spread` shape, not the `boss_converts` shape: the
    heavily damaged source is better in BOTH directions at once -- it transfers
    more damage AND it heals the Pokemon that actually needed healing -- so
    there is no trade being made and no judgment to override. `optfeat` simply
    has no HP or damage per option, so the net cannot see which is which.

    Deliberately minimal, exactly like `energy_spread`: it never changes
    whether counters move, only which of our Pokemon they come off, and among
    the sources that can pay the full 3 it keeps the net's own preference.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != REMOVE_DAMAGE_COUNTER:
        return None
    options = sel.get("option") or []
    if len(options) < 2 or not chosen:
        return None
    pick = chosen[0]
    if not 0 <= pick < len(options):
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None

    movable: list[int] = []
    for opt in options:
        if opt.get("playerIndex") != me:
            return None  # not the own-side source pick: leave it to the net
        pk = _pokemon_at(state, me, opt.get("area"), opt.get("index") or 0)
        if pk is None:
            return None
        hp, mx = pk.get("hp"), pk.get("maxHp")
        if hp is None or mx is None:
            return None
        movable.append(min(max(0, (mx - hp) // 10), MAX_MOVE))

    best = max(movable)
    if movable[pick] >= best:
        return None  # the net already picked a source that pays in full
    top = {i for i, m in enumerate(movable) if m == best}
    for i in rank():
        if i in top:
            return [i] + [j for j in chosen[1:] if j != i]
    return None


# --- Boss's Orders: drag something we can actually kill -------------------

def best_damage(active: dict, mypl: dict, oppl: dict, target: dict) -> float:
    """Best damage `active` can pay for right now, applied to `target`.

    `estimate` is text-pattern-based and approximate in general, but every
    attack this deck can pay for is flat damage (Shadow Bullet 180, Corkscrew
    Punch 60, Frost Smash 60), so here it is exact. Where it is not, it
    under-reads, which only makes the callers below more conservative."""
    if not active:
        return 0.0
    atk_type = cards.card(active["id"]).get("energyType")
    weak = cards.card(target["id"]).get("weakness")
    mult = 2.0 if (weak is not None and weak == atk_type) else 1.0
    best = 0.0
    for aid in cards.card(active["id"]).get("attacks") or []:
        a = cards.attacks().get(aid)
        if a and cards.energy_satisfied(a["energies"], active["energies"]):
            best = max(best, estimate(aid, active, mypl, oppl) * mult)
    return best


def drag_target(obs: dict, prefer_high_hp: bool = False) -> list[int] | None:
    """Ranked option indices for Boss's Orders' drag, else None.

    Boss's Orders resolves through a **SWITCH** select (not TO_ACTIVE, which is
    our own post-KO promotion) whose options are the opponent's benched
    Pokemon. Same blind spot as `chip_target`: no HP in the features, so the
    net cannot tell which of them dies to the attack we are about to make.

    Measured on the shipped clone (300 games): given a KO-able bench target it
    took the best available KO 85 times out of 99 -- 12 drags of a Pokemon that
    survives, 2 that took fewer prizes than were on offer.

    Rank: dies to our attack first, most prizes among those, then closest to
    dying. Same guard as `chip_target` -- every option must be an opponent's
    benched Pokemon, so our own retreats and switches are left to the net.

    `prefer_high_hp` (`bc:drag,dragHi`) flips the tiebreak **inside the KO-able
    group only**: they all die this turn, so "closest to dying" buys nothing
    there, and the user's argument is that the big one is a developing threat
    while a small basic is cheap for them to replace. The non-KO-able fallback
    keeps ascending HP -- there "closest to dying" is the whole point, and
    flipping it would be a different intervention."""
    sel = obs.get("select") or {}
    if sel.get("context") != SWITCH:
        return None
    options = sel.get("option") or []
    if len(options) < 2:
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None

    scored: list[tuple[tuple, int]] = []
    for i, opt in enumerate(options):
        if opt.get("playerIndex") in (None, me) or opt.get("area") != _BENCH:
            return None
        pk = _pokemon_at(state, 1 - me, _BENCH, opt.get("index") or 0)
        if pk is None or pk.get("hp") is None:
            return None
        kills = best_damage(active, mypl, oppl, pk) >= pk["hp"]
        hp = -pk["hp"] if (kills and prefer_high_hp) else pk["hp"]
        scored.append(((0 if kills else 1,
                        -cards.prize_value(pk["id"]) if kills else 0,
                        hp), i))

    scored.sort()
    return [i for _, i in scored]


def full_rank(net, obs: dict) -> list[int]:
    """The net's complete ranking, not the top-k that `choose` returns.

    A veto needs the runner-up, and every MAIN select measured here has
    maxCount == 1. Plain sort rather than argsort so this module stays
    numpy-free."""
    scores = net.scores(obs)
    return sorted(range(len(scores)), key=lambda i: -float(scores[i]))


def boss_veto(obs: dict, chosen: list[int], rank) -> list[int] | None:
    """Suppress Boss's Orders when their bench holds nothing we can KO.

    **This is the third Boss's Orders intervention, and the only untested one.**
    P4a measured *forcing* the play when it converts (0.493) and *aiming* the
    drag (0.489), both null. Neither touched the case here: the play happening
    at all when it buys nothing. Measured with `scripts/p5_audit.py --matches
    200`: **35 of 108 Boss's Orders plays (32.4%) had no KO-able target on the
    opponent's bench at all.** Those hand the opponent a free promotion --- the
    user watched one drag get evolved into their main attacker the next turn.

    Shape: this deletes an option rather than picking a side in a trade, but the
    option is only *conditionally* dominated (a drag can still strand their
    attacker in the Active, which `best_damage` cannot see), so it sits between
    the P4b class and the P4a class. Per rule 10 it lives or dies by its A/B.

    `rank` is a zero-argument callable returning the net's FULL ranking. It is
    needed because MAIN selects have maxCount == 1 --- `choose` hands back one
    index, so vetoing it leaves nothing to play instead. Called only when the
    veto actually fires, which is why it is lazy.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    if not chosen:
        return None
    pick = chosen[0]
    if not 0 <= pick < len(options):
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    opt = options[pick]
    if opt.get("type") != OPT_PLAY:
        return None
    if _hand_card_id(state, me, opt.get("index") or 0) != BOSS_ORDERS:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None
    if not active:
        return None
    for pk in oppl.get("bench") or []:
        if pk and pk.get("hp") is not None \
                and best_damage(active, mypl, oppl, pk) >= pk["hp"]:
            return None  # the drag buys a prize: let it through

    # Nothing on their bench dies. Fall through to the net's next choice,
    # skipping every other Boss's Orders copy in hand for the same reason.
    vetoed = {i for i, o in enumerate(options)
              if o.get("type") == OPT_PLAY
              and _hand_card_id(state, me, o.get("index") or 0) == BOSS_ORDERS}
    rest = [i for i in rank() if i not in vetoed]
    return rest[:len(chosen)] or None


def boss_converts(obs: dict) -> list[int] | None:
    """Play Boss's Orders when the drag turns a nothing turn into a prize.

    The frequency question was already closed -- we play Boss's Orders on 32.4%
    of legal turns against the demonstrators' 31.4%. The open one was *when*.
    Over 300 games there were 157 turns where our attack would not KO the
    opponent's Active but would KO something on their bench, and the clone
    played Boss's Orders on only 58 of them (36.9%; it plays it on 25.7% of all
    other legal turns, so it does discriminate -- just barely).

    Fires only on that exact shape: we can attack, the Active survives, a
    benched Pokemon does not. It costs the turn's Supporter, which is why it
    stays pinned to the case where the payoff is a guaranteed prize."""
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    boss = [i for i, o in enumerate(options)
            if o.get("type") == OPT_PLAY
            and _hand_card_id(state, me, o.get("index") or 0) == BOSS_ORDERS]
    if not boss:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None
    opp_active = oppl["active"][0] if oppl.get("active") else None
    if not active or not opp_active:
        return None
    if best_damage(active, mypl, oppl, opp_active) >= opp_active["hp"]:
        return None  # the Active already dies; the drag would trade down
    for pk in oppl.get("bench") or []:
        if pk and best_damage(active, mypl, oppl, pk) >= pk["hp"]:
            return [boss[0]]
    return None


def _prize_if_ko(attacker, mypl, oppl, target) -> int:
    """Prizes we take by attacking `target`, or 0 if it survives."""
    if target is None or target.get("hp") is None:
        return 0
    if best_damage(attacker, mypl, oppl, target) < target["hp"]:
        return 0
    return cards.prize_value(target["id"])


def _snipe_prizes(oppl, exclude_index=None) -> int:
    """Best prize Shadow Bullet's 30 bench snipe takes, 0 if none dies to it."""
    best = 0
    for i, pk in enumerate(oppl.get("bench") or []):
        if pk is None or pk.get("hp") is None or i == exclude_index:
            continue
        if pk["hp"] <= CHIP_DAMAGE:
            best = max(best, cards.prize_value(pk["id"]))
    return best


def boss_prize_veto(obs: dict, chosen: list[int], rank) -> list[int] | None:
    """Don't play Boss's Orders when ATTACKING NOW takes strictly more prizes.

    **The fifth Boss's Orders intervention, and `EVIDENCE` §6 said not to write
    it. §6 is wrong, and here is the distinction it missed.** The four nulls all
    answered *which* Pokemon to drag (`drag_target`, `prefer_high_hp`) or
    *whether the drag itself converts* (`boss_converts`, `boss_veto`). **Not one
    of them compares the drag against the attack we already have.**

    The defect this targets was measured on 54 REAL ladder games of the shipped
    v3 agent (`scripts/p8_optv3_replays.py`): of 31 drags where attacking was a
    genuine alternative, **9 (29%) were misplays, and 5 of those threw away a
    DOUBLE KO** -- Shadow Bullet is 180 to the Active *plus 30 to a bench*, so a
    <=30 HP bench sitter means attacking takes two prizes. In every one of those
    five we dragged the very Pokemon we could have sniped for free, converting a
    2-prize turn into a 1-prize turn:

        eg 89011961 t11: could KO Crustle hp=80 (1p) + snipe Dwebble hp=10;
                         dragged Dwebble
        eg 89021174 t9:  could KO Alakazam hp=80 (1p) + snipe Abra hp=20;
                         dragged Abra

    **Why this is the DOMINATED column (rule 11's 3-for-3 side) and the other
    four were not:** both branches are pure arithmetic -- prize values and
    damage-vs-HP, no judgment about tempo or what they might evolve into. We are
    not choosing between two goods; we are deleting a strictly worse option.

    ⚠ The comparison is honest on both sides: a drag can double-KO too (drag a
    KO-able target, snipe a different <=30 HP bench sitter), so `drag_best`
    excludes the dragged Pokemon from its own snipe. The veto fires only on
    **strictly** greater, so ties go to the net.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN or not chosen:
        return None
    options = sel.get("option") or []
    pick = chosen[0]
    if not 0 <= pick < len(options):
        return None
    opt = options[pick]
    if opt.get("type") != OPT_PLAY:
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    if _hand_card_id(state, me, opt.get("index") or 0) != BOSS_ORDERS:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None
    opp_active = oppl["active"][0] if oppl.get("active") else None
    if not active or not opp_active:
        return None

    # What attacking RIGHT NOW is worth: the Active if it dies, plus the snipe.
    attack_now = _prize_if_ko(active, mypl, oppl, opp_active) + _snipe_prizes(oppl)

    # What the best drag is worth: that target if it dies, plus a snipe onto a
    # DIFFERENT bench sitter (the dragged one is no longer benched).
    drag_best = 0
    for i, pk in enumerate(oppl.get("bench") or []):
        if pk is None:
            continue
        got = _prize_if_ko(active, mypl, oppl, pk)
        if got:
            drag_best = max(drag_best, got + _snipe_prizes(oppl, exclude_index=i))

    if attack_now <= drag_best:
        return None  # dragging is at least as good -- leave it to the net

    vetoed = {i for i, o in enumerate(options)
              if o.get("type") == OPT_PLAY
              and _hand_card_id(state, me, o.get("index") or 0) == BOSS_ORDERS}
    rest = [i for i in rank() if i not in vetoed]
    return rest[:len(chosen)] or None


def poffin_force(obs: dict, chosen: list[int]) -> list[int] | None:
    """Play Buddy-Buddy Poffin when the bench has room — E11.

    **The first candidate this project found where WE are the worse player at
    something ordering-free.** `p70_perturn_sweep.py` ranks every option class
    by its per-TURN gap (rule 21) instead of its per-decision gap, and this was
    invisible to the per-decision ranking because the clone is never
    *confidently wrong* here — it simply never gets round to it. Share of
    available turns in which the card is actually played, conditioned on our own
    board occupancy, mirror only (`EVIDENCE` §8bl, `docs/experiments/E11-poffin.md`):

        board  4:  1150+ pilots 70.2%,  our clone 29.4%
        board  5:  1150+ pilots 46.9%,  our clone  7.2%

    Worth **0.80 plays/game**, over the 0.5 sizing gate, and the confound is
    checked: both sides decline at the same mean board size (4.46 vs 4.45), so
    it is the behaviour that differs, not the mix of situations.

    Shape: benching a 70 HP basic is a **tradeoff** (development against giving
    the mirror's own Shadow Bullet snipe another target), so rule 11 would
    normally forbid building it. The governing precedent is `boss_veto`'s —
    rule 10, "it lives or dies by its A/B" — and the A/B is byte-identical-net
    with the rule toggled, so the ±13 Elo seed nuisance cancels exactly.

    ⚠ **Deliberately conservative at board 5.** The experts are themselves a
    coin flip there (46.9%), so forcing that bucket would overshoot the
    behaviour being copied. Fires only with **>= 2 free slots**. Widening it is
    a separate experiment, not a knob to turn after reading the result.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    if not chosen or not 0 <= chosen[0] < len(options):
        return None
    state = obs.get("current") or {}
    try:
        me = state["yourIndex"]
        mypl = state["players"][me]
    except (KeyError, IndexError, TypeError):
        return None

    # Already playing it: nothing to force.
    if _hand_card_id(state, me, options[chosen[0]].get("index") or 0) == POFFIN:
        return None

    # Board occupancy: the Active plus every filled bench slot, out of 6.
    bench = mypl.get("bench") or []
    filled = (1 if (mypl.get("active") and mypl["active"][0]) else 0)
    filled += sum(1 for pk in bench if pk)
    if filled > 4:            # fewer than 2 free slots -- see the docstring
        return None

    for i, o in enumerate(options):
        if (o.get("type") == OPT_PLAY
                and _hand_card_id(state, me, o.get("index") or 0) == POFFIN):
            return [i]
    return None


# --- E21: the Petrel fetch ---------------------------------------------------

def petrel_fetch(obs: dict, stadium: bool = False,
                 scrapper: bool = False) -> list[int] | None:
    """Inject board facts into Petrel's fetch — the one select with NO board.

    **Why this select and not another.** §8br's structural addendum: a fetch
    option's feature vector contains *nothing* about the board. The whole v3
    target block (`hp`, damage taken, dies-to-30, energy count, best_damage) is
    identically zero, because it resolves a Pokemon at (player, area, index) and
    the deck is not an in-play area. **One fetch option differs from another
    only by its card embedding and card type.** Everything situational has to
    arrive through `srepr`, which is concatenated identically to every option
    and can discriminate only through the head MLP's interaction term.

    ⚠ **These rules are NOT derived from what the experts fetch**, and that is
    deliberate: §8u measured that agreement with the FIELD predicts strength
    while agreement with the EXPERT anti-predicts it, and E11 copied a sized,
    ordering-free expert gap (0.80 plays/game) and measured **0.487**. Both
    conditions below come from card text plus the board:

    * `stadium` — Spikemuth Gym is the only Stadium in the 60 and our entire
      evolution line is Marnie's, so it is a repeatable engine tutor for us and
      much weaker for most opponents. Fetch it when **no Stadium is in play, or
      the one in play is theirs** — i.e. when playing it both starts our engine
      and removes theirs. Sized at **0.461 firings/game** over our 76 ladder
      games, the largest rate anything in this seam has had.
    * `scrapper` — Tool Scrapper does *nothing* unless a Tool is attached. So
      fetch it only when a Tool is on THEIR board. Sized at 0.171/game.
      ⚠ Under the 0.5 gate; included because an n=2,000 mirror A/B costs
      ~6 min here, so the gate's original cost model (arena time is scarce)
      does not bind. Rule 14 gates what is worth BUILDING, not what is
      affordable to measure.

    ⛔ Gated on `select.effect` being Petrel itself, so Poke Pad's Supporter
    search and Night Stretcher's discard recovery are untouched — they cannot
    reach a Stadium anyway, and a rule that fires on selects it was not sized
    on is measuring something other than what was sized.

    Returns a single-element order (maxCount is 1 here and minCount is 0, so
    declining is legal and one index is a complete answer), or None to leave the
    select entirely to the net.
    """
    sel = obs.get("select") or {}
    if (sel.get("context") or 0) != FETCH:
        return None
    eff = sel.get("effect")
    if not isinstance(eff, dict) or eff.get("id") != PETREL:
        return None
    options = sel.get("option") or []
    if len(options) < 2:
        return None
    state = obs.get("current") or {}
    try:
        me = state["yourIndex"]
        opp = state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None

    want = None
    if scrapper:
        tools = 0
        for where in ("active", "bench"):
            for pk in (opp.get(where) or []):
                if pk:
                    tools += len((pk or {}).get("tools") or [])
        if tools:
            want = TOOL_SCRAPPER
    if want is None and stadium:
        stad = state.get("stadium") or []
        if (not stad) or (stad[0] or {}).get("playerIndex") != me:
            want = SPIKEMUTH
    if want is None:
        return None

    # The card an option names, via the NET'S OWN extractor -- p76's lesson:
    # a PLAY option carries no `area`, and a hand-rolled mapping produced a
    # table that E11's own measurement contradicted outright.
    from .optfeat import option_features
    for i, o in enumerate(options):
        try:
            if int(option_features(obs, o)[1] or 0) == want:
                return [i]
        except Exception:  # noqa: BLE001
            continue
    return None
