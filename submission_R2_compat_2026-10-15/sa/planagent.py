"""A coherent, plan-conditioned policy for Marnie's Grimmsnarl ex / Munkidori.

⚡ **WHY THIS EXISTS, in one paragraph.** Every hand-written thing this project
has shipped is a *patch on the clone*: `chip_target`, `energy_spread`,
`counter_source`, the wall branch, the Petrel rules. Each answers "can I
override the net at THIS select?" and each was measured alone. The discriminator
that came out of it (rule 11: rules that delete a *dominated* option win 3/3,
rules that pick a side in a *tradeoff* lose 0/4) is therefore a statement about
**patches**, not about policies. This module is the object that was never built:
one plan per turn, and every option scored by whether it advances that plan.

**The three measurements that license it:**

* **§8ch (E26)** — a trained policy's deviations cost **f = 0.758** where
  rate/depth/location-matched random ones cost **0.12**. The local optimum
  forbids **jumps, not paths**. A coherent from-scratch policy is a path.
* **§8bj (F1)** — the 1145–1166 agents' mirror edge dissolved into *timing*:
  identical Munkidori usage (6.42/game vs our 6.23), different ordering. A patch
  cannot express "stop searching after turn 9"; a plan can.
* **§2.8** — "the 1145–1166 agents are not at a better point in our landscape,
  they are in a different one", and §0 records that nothing at the top of this
  board is learned. The observed ceiling of *this* method on *this* board is
  ~1166. Ours is ~990.

**The template is `agents/agentkit/rulebased/sources/v10.py`** — a real
competitor's agent that holds LB 950+ with its MCTS provably never executing.
Its architecture is exactly one `AttackPlan` per turn plus an option ladder
conditioned on it (`Switch` scores 6000 *only if* `plan.attacker > 0`; Boss's
Orders 3200 *only if* `plan.target >= 1`). We are not copying its numbers — its
deck is Mega Lucario, which chooses between several attackers and attacks. Ours
chooses between essentially none: **B2 killed the lethal audit because Marnie's
Grimmsnarl ex has ONE payable attack** (316/316 lethals taken, all forced).

⇒ **So our plan object is not an attack plan. It is a KO SCHEDULE.**
`STRATEGY.md` §7c.1 states the deck's actual game plan: *"the opponent's whole
board accumulates damage it cannot heal, until everything on it dies to a number
smaller than a full attack. The deck wins by making the opponent's board
fragile, not by making our attacker bigger."* And §7d measured that **~84% of
mirror prize takes are single-prize on both sides** — the realised game is a
six-single-prize grind, not the canonical 2-2-2 map. That is a *schedule*: which
six prizes, in what order, and which 30s make them cheap. The clone has no
representation of it — it scores 318 selects independently, which is precisely
why F1 found its Munkidori *counts* right and its *timing* wrong.

**What this module owns and what it does not.** It owns MAIN outright plus the
selects where the schedule is the whole question (damage placement, the
Adrena-Brain source, Boss's drag, post-KO promotion, the deck searches). Every
other context falls through to `fallback`, which is the clone by default. That
is a deliberate build order, not the thesis: `plan:pure` runs the same policy
with the net removed entirely, and the gap between the two arms is the honest
measure of how much of the game the plan layer actually owns. **Coverage is
reported per game (`STATS`) because a policy layer whose firing rate is not
printed cannot be distinguished from one that never fired** — rule 9, and the
§8be family of nulls that bought it.

⚠ **Damage numbers are hardcoded because the card DB cannot supply them.**
`cards.card(cid)["abilities"]` is `None` for every card in our 60 (verified
2026-08-13), exactly as `targeting.WALL_POKEMON` records for Crustle. Punk Up,
Adrena-Brain and Freezing Shroud are engine behaviour with no readable text, so
they are constants here. `targeting.py`'s own verifications are reused rather
than re-derived: Adrena-Brain is once **per Pokemon** and its {D} is a
**threshold, not a cost** (n=138, energy never consumed); the counter move is
capped by what the source actually carries.
"""
from __future__ import annotations

import sys
import traceback

from . import cards

# --- SelectContext (mirror cg.api without importing it) ----------------------
MAIN = 0
SETUP_ACTIVE_POKEMON = 1
SETUP_BENCH_POKEMON = 2
SWITCH = 3
TO_ACTIVE = 4
TO_BENCH = 5
TO_HAND = 7
DISCARD = 8
DAMAGE_COUNTER = 13
DAMAGE_COUNTER_ANY = 14
DAMAGE = 15
REMOVE_DAMAGE_COUNTER = 16
ATTACH_FROM = 21
ATTACH_TO = 22
DAMAGE_COUNTER_COUNT = 39
REMOVE_DAMAGE_COUNTER_COUNT = 40
DRAW_COUNT = 38
IS_FIRST = 41

CHIP_CONTEXTS = (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE)
COUNT_CONTEXTS = (DAMAGE_COUNTER_COUNT, REMOVE_DAMAGE_COUNTER_COUNT, DRAW_COUNT)

# --- OptionType --------------------------------------------------------------
O_NUMBER, O_YES, O_NO, O_CARD = 0, 1, 2, 3
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY = 7, 8, 9, 10
O_DISCARD, O_RETREAT, O_ATTACK, O_END = 11, 12, 13, 14

# --- AreaType ----------------------------------------------------------------
A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE, A_STADIUM = 1, 2, 3, 4, 5, 6, 7

# --- our 60 ------------------------------------------------------------------
DARK_ENERGY = 7          # Basic {D} Energy, card id AND energy type
DARK_TYPE = 7
MUNKIDORI = 112
FROSLASS = 104
SNORUNT = 860
IMPIDIMP = 646           # Marnie's Impidimp
MORGREM = 647            # Marnie's Morgrem
GRIMMSNARL = 648         # Marnie's Grimmsnarl ex
POFFIN = 1086            # Buddy-Buddy Poffin
POKE_PAD = 1152
PETREL = 1219            # Team Rocket's Petrel
LILLIE = 1227            # Lillie's Determination
SPIKEMUTH = 1259         # Spikemuth Gym
RARE_CANDY = 1079
NIGHT_STRETCHER = 1097
BOSS_ORDERS = 1182
UNFAIR_STAMP = 1080
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
DAWN = 1231

MARNIE_LINE = frozenset({IMPIDIMP, MORGREM, GRIMMSNARL})
ATTACKER_LINE = frozenset({GRIMMSNARL, MORGREM})

# Damage constants. Shadow Bullet is 180 to the Active PLUS 30 to a benched
# Pokemon; Adrena-Brain moves up to 3 counters = 30. Both verified in-engine and
# already carried by `targeting.CHIP_DAMAGE` / `optfeat.CHIP_DAMAGE`.
SHADOW_BULLET = 937
SHADOW_BULLET_DMG = 180
SNIPE = 30
CHIP = 30
MAX_MOVE = 3             # "up to 3 damage counters"

# Crustle's Mysterious Rock Inn prevents our {ex} attack damage entirely, so
# counters are the only removal. Same frozenset as `targeting.WALL_POKEMON`,
# duplicated rather than imported so this module owns its own facts.
WALL_POKEMON = frozenset({345})

# Lillie's Determination shuffle-draws 6 (8 while we still hold 6 prizes), and
# decking out is a real loss condition in a grind this long.
LOW_DECK = 8

# --- coverage instrumentation ------------------------------------------------
# 🔴 A policy layer that silently never fires produces a null that means
# nothing (rule 9; the §8be family). `owned` / `deferred` is the treatment size
# and it is printed once per game, never per decision.
STATS = {
    "calls": 0,
    "owned": 0,          # selects the plan layer answered
    "deferred": 0,       # selects handed to `fallback`
    "fallbacks": 0,      # the catch-all fired -- ANY non-zero value is a bug
    "plans": 0,          # turn plans built
    "lethal_plans": 0,   # plans that saw a winning prize count this turn
    "gusts": 0,          # Boss's Orders played because the schedule wanted it
    "first_error": None,
    # --- what the policy ACTUALLY did at MAIN (day 32 diagnostic) -----------
    # 🔴 A score alone cannot separate "this architecture is wrong" from "this
    # implementation never evolved". These count the top-ranked MAIN pick by
    # band, so a structural failure is visible without reading a replay.
    "main": 0,
    "did_attack": 0,
    "did_end": 0,
    "did_evolve_grimm": 0,
    "did_evolve_other": 0,
    "did_play": 0,
    "did_attach": 0,
    "did_ability": 0,
    "did_retreat": 0,
    "attack_avail": 0,   # MAIN selects where an ATTACK option was on offer
    "attack_declined": 0,  # ...and we ranked something else first
    "ready_no_attack": 0,  # ...while the plan said the attacker was armed
}

_BAND = {O_ATTACK: "did_attack", O_END: "did_end", O_PLAY: "did_play",
         O_ATTACH: "did_attach", O_ABILITY: "did_ability",
         O_RETREAT: "did_retreat"}


def health_line() -> str:
    s = STATS
    status = "OK" if s["fallbacks"] == 0 else "DEGRADED"
    owned = s["owned"] + s["deferred"]
    share = (s["owned"] / owned) if owned else 0.0
    return (f"[plan] {status} calls={s['calls']} owned={s['owned']} "
            f"deferred={s['deferred']} share={share:.3f} plans={s['plans']} "
            f"lethal={s['lethal_plans']} gusts={s['gusts']} "
            f"fallbacks={s['fallbacks']}\n"
            f"  [plan/main] main={s['main']} attack={s['did_attack']} "
            f"end={s['did_end']} evolveGrimm={s['did_evolve_grimm']} "
            f"evolveOther={s['did_evolve_other']} play={s['did_play']} "
            f"attach={s['did_attach']} ability={s['did_ability']} "
            f"retreat={s['did_retreat']}\n"
            f"  [plan/atk] avail={s['attack_avail']} "
            f"declined={s['attack_declined']} readyNoAtk={s['ready_no_attack']}")


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = None if k == "first_error" else 0


# --- board helpers -----------------------------------------------------------

def _board(pl: dict) -> list[dict | None]:
    """Slot 0 is the Active, slots 1.. are the bench -- v10's convention."""
    active = pl.get("active") or []
    return [active[0] if active else None] + list(pl.get("bench") or [])


def _slot_of(area: int | None, index: int | None) -> int:
    return (index or 0) + (1 if area == A_BENCH else 0)


def _dmg_on(pk: dict) -> int:
    """Damage already sitting on a Pokemon, in points."""
    hp, mx = pk.get("hp"), pk.get("maxHp")
    if hp is None or mx is None:
        return 0
    return max(0, mx - hp)


def _dark(pk: dict) -> int:
    return sum(1 for e in (pk.get("energies") or []) if e == DARK_TYPE)


def _name(cid: int) -> str:
    return cards.card(cid).get("name") or ""


def _is_marnie(cid: int) -> bool:
    return "Marnie" in _name(cid)


def _hand_ids(pl: dict) -> list[int]:
    return [c["id"] for c in (pl.get("hand") or []) if c]


def _card_at(state: dict, opt: dict, me: int) -> dict | None:
    """Resolve an option's referenced card, for the areas we care about."""
    area, index = opt.get("area"), opt.get("index") or 0
    player = opt.get("playerIndex")
    player = me if player is None else player
    try:
        pl = state["players"][player]
        if area == A_HAND:
            return (pl.get("hand") or [])[index]
        if area == A_DISCARD:
            return (pl.get("discard") or [])[index]
        if area == A_ACTIVE:
            act = pl.get("active") or []
            return act[0] if act else None
        if area == A_BENCH:
            return (pl.get("bench") or [])[index]
        if area == A_DECK:
            return ((state.get("select") or {}).get("deck") or [])[index]
        if area == A_STADIUM:
            st = state.get("stadium") or []
            return st[index] if st else None
    except (KeyError, IndexError, TypeError):
        return None
    return None


# --- the plan ----------------------------------------------------------------

class TurnPlan:
    """One schedule per turn. Every option score below reads this and nothing
    else, which is the entire difference between this policy and a rule bag."""

    __slots__ = ("attack_dmg", "attacker_slot", "target_slot", "snipe_slot",
                 "chip_slot", "need_energy", "need_evolve", "need_gust",
                 "attack_ready", "lethal", "prizes_now", "opp_prizes",
                 "schedule", "wall", "bench_want")

    def __init__(self) -> None:
        self.attack_dmg = 0
        self.attacker_slot = 0
        self.target_slot = 0        # opponent slot the attack should hit
        self.snipe_slot = -1        # opponent bench slot for Shadow Bullet's 30
        self.chip_slot = -1         # opponent slot for Adrena-Brain's counters
        self.need_energy = False
        self.need_evolve = False
        self.need_gust = False
        self.attack_ready = False
        self.lethal = False
        self.prizes_now = 0
        self.opp_prizes = 6
        self.schedule: list[tuple[int, int, int]] = []   # (slot, prizes, hp)
        self.wall = False
        self.bench_want = 0


def _chip_budget(my_board: list[dict | None]) -> int:
    """Damage we can place this turn WITHOUT attacking.

    Adrena-Brain is once per Munkidori and moves up to 3 counters, capped by
    what the source carries -- so the budget is bounded both by how many armed
    Munkidori we have and by how much damage is sitting on our own board to
    move. `targeting.counter_source` establishes both halves.
    """
    armed = sum(1 for pk in my_board
                if pk and pk.get("id") == MUNKIDORI and _dark(pk) >= 1)
    movable = sum(min(_dmg_on(pk) // 10, MAX_MOVE) for pk in my_board if pk)
    return min(armed * CHIP, movable * 10)


def build_plan(obs: dict) -> TurnPlan:
    """Compute the KO schedule for this turn.

    The schedule is the cheapest route to the prizes we still need, in
    *damage* rather than in attacks -- which is the deck's actual win condition
    (`STRATEGY.md` §7c.1) and the thing a per-select scorer cannot represent.
    """
    plan = TurnPlan()
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return plan
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return plan

    my_board, opp_board = _board(mypl), _board(oppl)
    active = my_board[0]
    plan.opp_prizes = len(oppl.get("prize") or [])
    my_prizes = len(mypl.get("prize") or [])

    opp_active = opp_board[0]
    plan.wall = bool(opp_active and opp_active.get("id") in WALL_POKEMON)

    # --- what our Active can actually do right now ---------------------------
    if active:
        plan.attack_dmg = cards.best_usable_damage(
            active["id"], active.get("energies") or [])
        aid = active.get("id")
        if aid == GRIMMSNARL:
            plan.need_energy = _dark(active) < 2
            plan.attack_ready = _dark(active) >= 2
        elif aid in (MORGREM, IMPIDIMP):
            # The evolution is worth more than the 60. Punk Up fires ON evolve
            # and fetches up to 5 {D} out of the deck, so evolving is also the
            # deck's only energy acceleration -- never trade it for a poke.
            plan.need_evolve = True
            plan.attack_ready = plan.attack_dmg > 0
        else:
            plan.attack_ready = plan.attack_dmg > 0

    # Who SHOULD be in front. Our Active is the attacker in almost every turn
    # of this deck, but a Munkidori stranded in the Active spot cannot attack at
    # all (Mind Bend costs {P}{C} and we run zero Psychic), and an armed
    # Grimmsnarl sitting on the bench behind it is a dominated position -- rule
    # 11's WINNING class, not the tradeoff class. `_score_main` turns this into
    # a retreat; without it `attacker_slot` was never set and the retreat band
    # was dead code.
    if not plan.attack_ready:
        for slot, pk in enumerate(my_board):
            if slot == 0 or not pk:
                continue
            if pk.get("id") == GRIMMSNARL and _dark(pk) >= 2:
                plan.attacker_slot = slot
                break

    # A wall takes zero from our attack, so the attack column of the schedule
    # is worthless there and only counters advance it (§8c: aiming chip at the
    # Dwebble instead of the Crustle cost -0.126 in that matchup).
    eff_attack = 0 if plan.wall else plan.attack_dmg

    # --- the schedule --------------------------------------------------------
    budget = _chip_budget(my_board)
    entries: list[tuple[int, int, int]] = []
    for slot, pk in enumerate(opp_board):
        if not pk:
            continue
        entries.append((slot, cards.prize_value(pk["id"]), pk.get("hp") or 0))
    plan.schedule = sorted(entries, key=lambda e: (e[2], -e[1]))

    # Prizes available this turn, counted honestly: the attack takes at most
    # one target, the snipe and the counters take whatever already dies to 30.
    prizes_now = 0
    if eff_attack:
        for slot, prize, hp in entries:
            if hp <= eff_attack and (slot == 0 or _holds(mypl, BOSS_ORDERS)):
                prizes_now = max(prizes_now, prize)

    # --- pick the attack target ---------------------------------------------
    # Their Active is the default. A bench Pokemon becomes the target only when
    # we hold Boss's Orders AND dragging it converts a prize the Active does
    # not -- `targeting.drag_target`'s finding, one level up: the drag is worth
    # spending only when it changes the prize count.
    plan.target_slot = 0
    if eff_attack and opp_active:
        kills_active = (opp_active.get("hp") or 0) <= eff_attack
        best_bench = None
        if _holds(mypl, BOSS_ORDERS):
            for slot, prize, hp in entries:
                if slot == 0 or hp > eff_attack:
                    continue
                gain = prize - (cards.prize_value(opp_active["id"])
                                if kills_active else 0)
                if gain > 0 and (best_bench is None or prize > best_bench[1]):
                    best_bench = (slot, prize)
        # Nothing dies to the attack at all: drag the cheapest prize we can
        # actually finish, rather than hitting a wall or a fresh 320.
        if best_bench is None and not kills_active and _holds(mypl, BOSS_ORDERS):
            for slot, prize, hp in entries:
                if slot != 0 and hp <= eff_attack:
                    best_bench = (slot, prize)
                    break
        if best_bench is not None:
            plan.target_slot = best_bench[0]
            plan.need_gust = True
            prizes_now = max(prizes_now, best_bench[1])

    # --- where the 30s go ----------------------------------------------------
    # Rank: something that DIES to this 30 (most prizes among those), else the
    # Pokemon the schedule wants softened -- the one closest to dying that the
    # attack is not already going to kill. That second clause is the coherence:
    # a greedy ranker spends the snipe on whatever is lowest, a scheduled one
    # spends it where it converts a future KO.
    plan.snipe_slot = _pick_chip(entries, plan, eff_attack, bench_only=True)
    plan.chip_slot = _pick_chip(entries, plan, eff_attack, bench_only=False)

    if budget:
        for slot, prize, hp in entries:
            if hp <= budget and slot != plan.target_slot:
                prizes_now += prize
                break

    plan.prizes_now = prizes_now
    plan.lethal = prizes_now >= plan.opp_prizes and plan.opp_prizes > 0

    # How much bench we still want. Six is the cap; the deck wants Munkidori
    # online early (two armed Munkidori = a 60-point swing per turn) and the
    # Grimmsnarl line developing behind the Active.
    filled = sum(1 for pk in my_board[1:] if pk)
    plan.bench_want = max(0, 5 - filled)

    STATS["plans"] += 1
    if plan.lethal:
        STATS["lethal_plans"] += 1
    if my_prizes and plan.need_gust:
        STATS["gusts"] += 1
    return plan


def _holds(pl: dict, cid: int) -> bool:
    return any(c and c.get("id") == cid for c in (pl.get("hand") or []))


def _pick_chip(entries, plan: TurnPlan, eff_attack: int,
               bench_only: bool) -> int:
    """Which opponent slot a 30 should land on, given the schedule."""
    best, best_key = -1, None
    for slot, prize, hp in entries:
        if bench_only and slot == 0:
            continue
        dies = hp <= CHIP
        # A slot the attack is already killing does not need our 30.
        redundant = (slot == plan.target_slot and hp <= eff_attack and not dies)
        if redundant:
            continue
        # kills first, most prizes among those, then closest to dying
        key = (0 if dies else 1, -prize if dies else 0, hp)
        if best_key is None or key < best_key:
            best, best_key = slot, key
    return best


# --- the policy --------------------------------------------------------------

class PlanAgent:
    """Kaggle-contract callable. `fallback(obs) -> list[int]` answers anything
    the plan layer declines; `pure=True` removes it entirely."""

    def __init__(self, decklist: list[int], fallback=None, pure: bool = False):
        self.decklist = list(decklist)
        self.fallback = None if pure else fallback
        self.pure = pure
        self._turn = -1
        self._plan = TurnPlan()

    # -- entry ---------------------------------------------------------------
    def __call__(self, obs: dict) -> list[int]:
        try:
            if obs.get("select") is None:
                STATS["calls"] += 1
                return list(self.decklist)
            STATS["calls"] += 1
            sel = obs["select"]
            options = sel.get("option") or []
            if not options:
                return []
            state = obs.get("current") or {}
            turn = state.get("turn")
            if turn != self._turn:
                self._turn = turn
                self._plan = build_plan(obs)
            elif sel.get("context") == MAIN:
                # Rebuild at every MAIN: the board changed since the last one
                # (that is what a MAIN loop IS), and a stale plan is exactly the
                # incoherence this module exists to remove.
                self._plan = build_plan(obs)

            ranked = self._decide(obs, sel, options)
            if ranked is None:
                STATS["deferred"] += 1
                if self.fallback is not None:
                    return self.fallback(obs)
                ranked = self._pure_default(obs, sel, options)
            else:
                STATS["owned"] += 1
            return _trim(ranked, sel, len(options))
        except Exception:
            STATS["fallbacks"] += 1
            if STATS["first_error"] is None:
                STATS["first_error"] = traceback.format_exc()
            traceback.print_exc(file=sys.stderr)
            try:
                sel = obs.get("select") or {}
                n = len(sel.get("option") or [])
                return list(range(min(max(1, sel.get("minCount", 0) or 0), n)))
            except Exception:
                return []

    # -- routing -------------------------------------------------------------
    def _decide(self, obs, sel, options) -> list[int] | None:
        ctx = sel.get("context")
        state = obs.get("current") or {}
        me = state.get("yourIndex")
        if me is None:
            return None
        plan = self._plan

        if ctx == MAIN:
            scores = [self._score_main(obs, state, me, plan, o) for o in options]
            ranked = _rank(scores)
            self._tally(state, me, options[ranked[0]])
            if any(o.get("type") == O_ATTACK for o in options):
                STATS["attack_avail"] += 1
                if options[ranked[0]].get("type") != O_ATTACK:
                    STATS["attack_declined"] += 1
                    if plan.attack_ready:
                        STATS["ready_no_attack"] += 1
            return ranked
        if ctx == IS_FIRST:
            # Going first means no attack on turn 1 but a full setup turn and
            # the first Punk Up. This deck wants the development.
            return _rank([1.0 if o.get("type") == O_YES else 0.0
                          for o in options])
        if ctx in COUNT_CONTEXTS:
            # Always move/draw the maximum. The clone already does this 100% of
            # the time (n=481) -- kept explicit so `pure` behaves identically.
            return _rank([float(o.get("number") or 0) for o in options])
        if ctx in CHIP_CONTEXTS:
            return self._aim_chip(state, me, plan, options)
        if ctx == REMOVE_DAMAGE_COUNTER:
            return self._chip_source(state, me, options)
        if ctx == SWITCH:
            return self._drag(state, me, plan, options)
        if ctx == TO_ACTIVE:
            return self._promote(state, me, plan, options)
        if ctx in (SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON):
            return self._setup(state, me, options, active=(ctx == SETUP_ACTIVE_POKEMON))
        if ctx in (TO_HAND, TO_BENCH):
            return self._fetch(obs, state, me, plan, options)
        if ctx in (ATTACH_FROM, ATTACH_TO):
            return self._attach_target(state, me, plan, options)
        return None

    def _tally(self, state, me, opt: dict) -> None:
        """Record the band of the option we actually played at MAIN."""
        STATS["main"] += 1
        t = opt.get("type")
        if t == O_EVOLVE:
            card = _card_at(state, opt, me)
            key = ("did_evolve_grimm"
                   if card and card.get("id") == GRIMMSNARL
                   else "did_evolve_other")
            STATS[key] += 1
            return
        key = _BAND.get(t)
        if key:
            STATS[key] += 1

    # -- MAIN ----------------------------------------------------------------
    def _score_main(self, obs, state, me, plan: TurnPlan, opt: dict) -> float:
        """The ladder. Every band is conditioned on the plan; the bands
        themselves encode "do all the free things before the turn-ending one",
        which is the ordering the clone has no way to represent."""
        t = opt.get("type")
        try:
            mypl = state["players"][me]
        except (KeyError, IndexError, TypeError):
            return 0.0

        if t == O_END:
            # 🔴 END SCORES ZERO, NOT MINUS-A-LOT, and the difference is a real
            # bug caught before the first comparison cell ran. Every option this
            # ladder DECLINES returns -1.0 ("do not do this"), so an END band
            # below -1 makes the agent prefer a declined Boss's Orders to
            # finishing its turn. `v10.py` gets this right by scoring END at 0
            # and declines at -1; the ordering is the contract, not the numbers.
            return 0.0
        if t == O_ABILITY:
            return self._score_ability(state, me, plan, opt)
        if t == O_ATTACK:
            # Attacking ends the turn, so it must lose to every setup action
            # that is still available. It still beats END by a mile.
            if not plan.attack_ready and plan.attack_dmg <= 0:
                return -50.0        # below END: ending beats a 0-damage poke
            return 1000.0 + (500.0 if plan.lethal else 0.0)
        if t == O_PLAY:
            return self._score_play(state, me, mypl, plan, opt)
        if t == O_EVOLVE:
            return self._score_evolve(state, me, plan, opt)
        if t == O_ATTACH:
            return self._score_attach(state, me, plan, opt)
        if t == O_RETREAT:
            # We retreat to put the planned attacker in front, never for value.
            return 6000.0 if plan.attacker_slot > 0 else -100.0
        return 0.0

    def _score_ability(self, state, me, plan: TurnPlan, opt: dict) -> float:
        """Adrena-Brain is free damage AND free healing on the same activation
        (§8c/§8ah), so it is the highest band. Its {D} is a threshold, not a
        cost -- nothing is spent, so there is never a reason to decline it."""
        card = _card_at(state, opt, me)
        cid = card.get("id") if card else None
        if cid == MUNKIDORI:
            return 30000.0
        return 25000.0

    def _score_play(self, state, me, mypl, plan: TurnPlan, opt: dict) -> float:
        card = _card_at(state, opt, me)
        if not card:
            return 0.0
        cid = card.get("id")
        data = cards.card(cid)
        board = _board(mypl)
        counts: dict[int, int] = {}
        for pk in board:
            if pk:
                counts[pk["id"]] = counts.get(pk["id"], 0) + 1

        if data.get("cardType") == 0:      # a Pokemon: bench development
            if plan.bench_want <= 0:
                return -1.0
            if cid == MUNKIDORI:
                # Two armed Munkidori = 60 points of swing per turn. This is
                # the deck's engine, not a filler basic.
                return 20000.0 if counts.get(MUNKIDORI, 0) < 2 else 15000.0
            if cid == IMPIDIMP:
                have = counts.get(IMPIDIMP, 0) + counts.get(MORGREM, 0) \
                    + counts.get(GRIMMSNARL, 0)
                return 19000.0 if have < 2 else 12000.0
            if cid == SNORUNT:
                return 14000.0 if counts.get(SNORUNT, 0) < 1 else 8000.0
            return 13000.0

        if cid == RARE_CANDY:
            # Skips Morgrem so Punk Up lands a turn early. Only worth it with
            # the Grimmsnarl actually in hand.
            return 18000.0 if _holds(mypl, GRIMMSNARL) else -1.0
        if cid == POFFIN:
            return 17000.0 if plan.bench_want > 0 else -1.0
        if cid == BOSS_ORDERS:
            # ⚠ The one place a supporter outranks the draw: it converts a
            # prize the attack cannot otherwise reach. Otherwise it is dead --
            # we already play it on 38% of legal turns and more copies measured
            # null at 0.490, so the fix is timing, not count.
            return 16000.0 if plan.need_gust else -1.0
        if cid == NIGHT_STRETCHER:
            gone = any(c and c.get("id") in ATTACKER_LINE
                       for c in (mypl.get("discard") or []))
            return 11000.0 if gone else -1.0
        if cid == POKE_PAD:
            return 10500.0
        if cid == POKEGEAR:
            # §8ag: 0.27 real choices/game, and we take a free Supporter 39/39.
            # Kept cheap rather than clever.
            return 10400.0
        if cid == TOOL_SCRAPPER:
            opp_tools = any((pk.get("tools") or [])
                            for pk in _board(state["players"][1 - me]) if pk)
            return 9000.0 if opp_tools else -1.0
        if cid == UNFAIR_STAMP:
            # Playable only after we lose a Pokemon; it is the comeback card,
            # so spend it when we are behind on prizes, not on curve.
            mine = len((state["players"][me].get("prize") or []))
            return 9500.0 if mine > plan.opp_prizes else 2000.0
        if cid == SPIKEMUTH:
            # Our whole evolution line is Marnie's, so the tutor is repeatable
            # for us and much weaker for most opponents. ⚡ F1 measured the
            # 1150s STOPPING this search around turn 9.7 while we never stop --
            # the one ordering-free difference it found. Once the line is
            # assembled the search is a no-op that costs us the stadium slot.
            need_line = plan.need_evolve or not _holds(mypl, GRIMMSNARL)
            return 8500.0 if need_line else 500.0
        if data.get("cardType") == 3:      # a Supporter
            deck_c = mypl.get("deckCount")
            deck_c = 60 if deck_c is None else deck_c
            if deck_c <= LOW_DECK:
                return -1.0               # decking out is a real loss here
            if cid == LILLIE:
                return 12000.0
            if cid == PETREL:
                return 11500.0
            if cid == DAWN:
                return 11000.0
            return 10000.0
        return 7000.0

    def _score_evolve(self, state, me, plan: TurnPlan, opt: dict) -> float:
        """Evolving into Grimmsnarl ex is the single most valuable action in
        the deck: it is a 320 HP body AND the energy engine (Punk Up fetches up
        to 5 {D} on the evolve). Nothing outranks it except a free ability."""
        target = _pokemon_at_slot(state, me, opt)
        card = _card_at(state, opt, me)
        into = card.get("id") if card else None
        if into == GRIMMSNARL:
            return 22000.0
        if into == MORGREM:
            return 16500.0
        if into == FROSLASS:
            # Freezing Shroud is unblockable, symmetric, and asymmetric in
            # practice -- Munkidori repairs ours, nothing repairs theirs.
            return 15500.0
        if target is None:
            return 9000.0
        return 9000.0 + len(target.get("energies") or [])

    def _score_attach(self, state, me, plan: TurnPlan, opt: dict) -> float:
        """One hand attach per turn, and it is the scarcest resource in the
        deck -- Punk Up cannot feed Munkidori (not a Marnie's Pokemon), so the
        manual attach is Munkidori's ONLY source. `energy_spread` measured the
        clone stacking a wasted second {D} 143 times to 94 for exactly this."""
        pk = _pokemon_at_slot(state, me, opt)
        if pk is None:
            return 0.0
        cid = pk.get("id")
        slot = _slot_of(opt.get("inPlayArea"), opt.get("inPlayIndex"))
        score = 8000.0
        if cid == MUNKIDORI:
            # A bare Munkidori is +30 damage a turn; a second {D} on a loaded
            # one is worth literally nothing (Mind Bend costs {P}{C} and we run
            # zero Psychic).
            return 8900.0 if _dark(pk) == 0 else 100.0
        if cid == GRIMMSNARL:
            need = max(0, 2 - _dark(pk))
            if need:
                score += 800.0 if slot == 0 else 400.0
            else:
                score -= 200.0
            return score
        if cid in (IMPIDIMP, MORGREM):
            # Energy on the pre-evolution carries up the line, so this is not
            # wasted -- but it loses to arming a Munkidori.
            return score + (200.0 if slot > 0 else 100.0)
        return score - 100.0

    # -- non-MAIN handlers ---------------------------------------------------
    def _aim_chip(self, state, me, plan: TurnPlan, options) -> list[int] | None:
        """Place a 30 where the SCHEDULE wants it.

        This is `chip_target` re-derived from the plan instead of greedily:
        same "dies to 30 first, most prizes among those" head, but the tail
        prefers the slot the plan has already nominated rather than whatever is
        lowest on the board. The wall guard is kept -- against a Crustle the
        counters are our only removal and farming the Dwebble loses slowly
        (§8c, -0.126).
        """
        scored: list[float] = []
        for opt in options:
            player = opt.get("playerIndex")
            if player is None or player == me:
                return None          # mixed or own-side: not ours to answer
            pk = _pokemon_at_opt(state, 1 - me, opt)
            if pk is None or pk.get("hp") is None:
                return None
            slot = _slot_of(opt.get("area"), opt.get("index"))
            hp = pk["hp"]
            dies = hp <= CHIP
            val = 0.0
            if dies:
                val += 10000.0 + 1000.0 * cards.prize_value(pk["id"])
            if plan.wall and pk.get("id") in WALL_POKEMON:
                val += 5000.0        # the only thing we can remove at all
            if slot == plan.chip_slot or slot == plan.snipe_slot:
                val += 400.0
            val += max(0.0, 400.0 - hp)   # closest to dying
            scored.append(val)
        return _rank(scored)

    def _chip_source(self, state, me, options) -> list[int] | None:
        """Take Adrena-Brain's counters off a Pokemon that HAS three, so the
        activation moves the full 30 and heals the body that needed it.
        `counter_source` measured the clone picking a short source in 20.3% of
        291 selects; this is that rule, plan-side."""
        scored: list[float] = []
        for opt in options:
            if opt.get("playerIndex") != me:
                return None
            pk = _pokemon_at_opt(state, me, opt)
            if pk is None:
                return None
            movable = min(max(0, _dmg_on(pk) // 10), MAX_MOVE)
            # tie-break toward the Pokemon we actually want healed
            urgency = 1.0 if pk.get("id") in (GRIMMSNARL, MUNKIDORI) else 0.0
            scored.append(movable * 100.0 + urgency)
        return _rank(scored)

    def _drag(self, state, me, plan: TurnPlan, options) -> list[int] | None:
        """Boss's Orders' drag resolves through SWITCH. Pull the slot the plan
        nominated; fall back to "dies to our attack, most prizes"."""
        eff = 0 if plan.wall else plan.attack_dmg
        scored: list[float] = []
        for opt in options:
            if opt.get("playerIndex") in (None, me):
                return None
            pk = _pokemon_at_opt(state, 1 - me, opt)
            if pk is None or pk.get("hp") is None:
                return None
            slot = _slot_of(opt.get("area"), opt.get("index"))
            hp = pk["hp"]
            val = 0.0
            if slot == plan.target_slot:
                val += 20000.0
            if eff and hp <= eff:
                val += 10000.0 + 1000.0 * cards.prize_value(pk["id"])
            val += max(0.0, 400.0 - hp)
            scored.append(val)
        return _rank(scored)

    def _promote(self, state, me, plan: TurnPlan, options) -> list[int] | None:
        """Post-KO promotion. Put up the body that can attack NEXT turn, which
        for this deck means the most-armed Marnie's Pokemon -- never a bare
        Munkidori, whose only attack we cannot pay for."""
        scored: list[float] = []
        for opt in options:
            if opt.get("playerIndex") not in (None, me):
                return None
            pk = _pokemon_at_opt(state, me, opt)
            if pk is None:
                return None
            cid = pk.get("id")
            val = float(_dark(pk)) * 50.0
            if cid == GRIMMSNARL:
                val += 1000.0 + (500.0 if _dark(pk) >= 2 else 0.0)
            elif cid == MORGREM:
                val += 400.0
            elif cid == IMPIDIMP:
                val += 300.0
            elif cid == MUNKIDORI:
                # Munkidori in front is a 110 HP body that cannot attack and
                # whose ability we still get from the bench. Last resort.
                val -= 200.0
            elif cid == FROSLASS:
                val += 200.0
            val += (pk.get("hp") or 0) / 100.0
            scored.append(val)
        return _rank(scored)

    def _setup(self, state, me, options, active: bool) -> list[int] | None:
        """Opening board. Impidimp in front (it is the line), Munkidori and
        Snorunt on the bench."""
        scored: list[float] = []
        for opt in options:
            card = _card_at(state, opt, me)
            if not card:
                return None
            cid = card.get("id")
            if active:
                val = {IMPIDIMP: 100.0, SNORUNT: 40.0,
                       MUNKIDORI: 30.0}.get(cid, 10.0)
            else:
                val = {MUNKIDORI: 100.0, IMPIDIMP: 90.0,
                       SNORUNT: 50.0}.get(cid, 10.0)
            scored.append(val)
        return _rank(scored)

    def _fetch(self, obs, state, me, plan: TurnPlan, options) -> list[int] | None:
        """Deck searches (Spikemuth, Poffin, Petrel, Night Stretcher).

        ⚡ This is the select E21 called "the ONE select with no board" -- the
        option list is card ids and nothing else, so the net is choosing blind.
        The plan knows what the line is missing, which is the whole point.
        """
        try:
            mypl = state["players"][me]
        except (KeyError, IndexError, TypeError):
            return None
        board = _board(mypl)
        counts: dict[int, int] = {}
        for pk in board:
            if pk:
                counts[pk["id"]] = counts.get(pk["id"], 0) + 1
        hand = _hand_ids(mypl)
        have_grimm = GRIMMSNARL in hand or counts.get(GRIMMSNARL, 0) > 0
        have_candy = RARE_CANDY in hand
        line_on_board = counts.get(IMPIDIMP, 0) + counts.get(MORGREM, 0)

        scored: list[float] = []
        for opt in options:
            card = _card_at(state, opt, me)
            if not card:
                return None
            cid = card.get("id")
            val = 0.0
            if cid == GRIMMSNARL:
                val = 1000.0 if (line_on_board or have_candy) else 600.0
            elif cid == RARE_CANDY:
                val = 900.0 if (have_grimm and line_on_board) else 300.0
            elif cid == IMPIDIMP:
                val = 800.0 if line_on_board == 0 else 250.0
            elif cid == MORGREM:
                val = 500.0 if (line_on_board and not have_candy) else 200.0
            elif cid == MUNKIDORI:
                val = 700.0 if counts.get(MUNKIDORI, 0) < 2 else 200.0
            elif cid == BOSS_ORDERS:
                val = 650.0 if plan.need_gust else 300.0
            elif cid == DARK_ENERGY:
                val = 400.0
            elif cid == SNORUNT:
                val = 350.0 if counts.get(SNORUNT, 0) == 0 else 100.0
            elif cid == POFFIN:
                val = 380.0 if plan.bench_want > 0 else 120.0
            elif cid == LILLIE:
                val = 360.0
            elif cid == PETREL:
                val = 340.0
            else:
                val = 150.0
            # never fetch a duplicate of something already in hand
            val -= 60.0 * hand.count(cid)
            scored.append(val)
        return _rank(scored)

    def _attach_target(self, state, me, plan: TurnPlan, options) -> list[int] | None:
        """Punk Up's placement: up to 5 {D} onto any Marnie's Pokemon. Arm the
        Active to 2 first (that is the attack), then spread onto the bench line
        so the NEXT attacker is already paid for -- the schedule two turns out,
        which is the thing the clone cannot see."""
        scored: list[float] = []
        for opt in options:
            pk = _pokemon_at_opt(state, me, opt)
            if pk is None:
                # ATTACH_FROM often lists energy cards, not Pokemon; defer.
                return None
            cid = pk.get("id")
            slot = _slot_of(opt.get("area"), opt.get("index"))
            d = _dark(pk)
            val = 0.0
            if cid == GRIMMSNARL:
                val = 1000.0 - 100.0 * d + (300.0 if slot == 0 and d < 2 else 0.0)
            elif cid in (IMPIDIMP, MORGREM):
                val = 700.0 - 80.0 * d
            elif cid == MUNKIDORI:
                val = 900.0 if d == 0 else 50.0
            else:
                val = 100.0
            scored.append(val)
        return _rank(scored)

    # -- pure-mode default ---------------------------------------------------
    def _pure_default(self, obs, sel, options) -> list[int]:
        """What `plan:pure` does where the plan layer declines. Deliberately
        the dumbest legal thing (index order) rather than a second hidden
        policy -- so the pure arm measures the PLAN, not a rule bag."""
        return list(range(len(options)))


def _pokemon_at_opt(state: dict, player: int, opt: dict) -> dict | None:
    return _pokemon_slot(state, player, opt.get("area"), opt.get("index") or 0)


def _pokemon_at_slot(state: dict, player: int, opt: dict) -> dict | None:
    return _pokemon_slot(state, player, opt.get("inPlayArea"),
                         opt.get("inPlayIndex") or 0)


def _pokemon_slot(state: dict, player: int, area, index) -> dict | None:
    try:
        pl = state["players"][player]
        if area == A_ACTIVE:
            act = pl.get("active") or []
            return act[0] if act and act[0] else None
        if area == A_BENCH:
            bench = pl.get("bench") or []
            if 0 <= (index or 0) < len(bench):
                return bench[index or 0]
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _rank(scores: list[float]) -> list[int]:
    return [i for i, _ in sorted(enumerate(scores), key=lambda kv: kv[1],
                                 reverse=True)]


def _trim(ranked: list[int], sel: dict, n: int) -> list[int]:
    """Return a legal reply: v10's tail logic, which handles the min/max
    interaction the engine actually enforces."""
    ranked = [i for i in ranked if 0 <= i < n]
    if not ranked:
        return list(range(min(max(1, sel.get("minCount") or 0), n)))
    lo = min(max(1, sel.get("minCount") or 0), n)
    hi = min(sel.get("maxCount") or 1, n)
    return ranked[:max(hi, lo)]
