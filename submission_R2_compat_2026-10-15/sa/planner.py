"""Turn-level determinized search.

For each sampled world (determinization) we search over OUR selects: try each
root candidate, then continue with a greedy 1-step-lookahead playout until the
opponent's next main select (or terminal / depth cap), and score the leaf with
the heuristic eval. Root values are averaged across worlds.

Opponent micro-decisions occurring inside our turn (e.g. their forced discards)
are played greedily *against* us (they minimize our eval).
"""
from __future__ import annotations

import itertools
import math
import random
import time

from . import fastsearch as fs
from . import policynet as pnet
from . import valuenet as vnet
from .evalfn import evaluate
from .worlds import determinize

MAIN = 0  # SelectType.MAIN

ROOT_CAP = 24        # max candidate combos at the root select
ROOT_POLICY_KEEP = 10  # root candidates kept when the policy net prunes
PLAYOUT_CAP = 8      # max candidates per playout decision
MAX_PLAYOUT_STEPS = 160
ROLLOUT_STEPS = 900  # step cap when rolling out all the way to terminal
TANH_SCALE = 6.0     # prize-units -> [-1,1], so truncations mix with results
PRIOR_BONUS = 0.15   # rollout margin needed to overrule the clone (see Planner)
HALVE_AFTER_WORLDS = 4  # rollout mode: don't prune on 1-2 noisy 0/1 samples
MAX_WORLDS = int(__import__("os").environ.get("SA_MAX_WORLDS", "12"))

DEBUG = bool(int(__import__("os").environ.get("SA_DEBUG", "0")))
NO_VNET = bool(int(__import__("os").environ.get("SA_NO_VNET", "0")))
NO_PNET = bool(int(__import__("os").environ.get("SA_NO_PNET", "0")))

STATS = {"decides": 0, "worlds": 0, "begin_fail": 0, "step_fail": 0,
         "playout_steps": 0, "budget_s": 0.0, "spent_s": 0.0,
         # rollout mode: how often we actually reached a terminal state. A
         # truncated rollout falls back to the heuristic leaf, i.e. it measures
         # the thing rollouts are supposed to replace -- watch this number.
         "rollouts": 0, "roll_terminal": 0, "roll_trunc": 0, "roll_steps": 0,
         # how often the search overrules the clone. If this is high, the
         # rollout estimator is not resolving anything -- it is just noise
         # beating the prior (see PRIOR_BONUS).
         "anchored": 0, "deviated": 0}


def candidate_combos(sel: dict, cap: int, rng: random.Random | None = None) \
        -> list[list[int]]:
    """Enumerate (or sample) legal selection index lists."""
    n = len(sel["option"])
    mn = sel["minCount"]
    mx = min(sel["maxCount"], n)
    if mx <= 0:
        return [[]]
    combos: list[list[int]] = []
    if mn == 0:
        combos.append([])
    per_size = max(2, cap // max(1, mx - max(mn, 1) + 1))
    for k in range(max(mn, 1), mx + 1):
        total = math.comb(n, k)
        if total <= per_size:
            combos.extend(list(c) for c in itertools.combinations(range(n), k))
        else:
            seen = set()
            first = tuple(range(k))
            last = tuple(range(n - k, n))
            for c in (first, last):
                if c not in seen:
                    seen.add(c)
                    combos.append(list(c))
            r = rng or random
            while len(seen) < per_size:
                c = tuple(sorted(r.sample(range(n), k)))
                if c not in seen:
                    seen.add(c)
                    combos.append(list(c))
    return combos[:cap] if len(combos) > cap else combos


class Planner:
    def __init__(self, decklist: list[int], no_pnet: bool | None = None,
                 no_vnet: bool | None = None, max_worlds: int | None = None,
                 rollout: bool = False, prior_bonus: float | None = None):
        self.decklist = list(decklist)
        self.rng = random.Random(0xC0FFEE)
        # per-instance knobs; default to the process-wide env flags so two
        # configs can be A/B'd head-to-head inside one arena process
        self.no_pnet = NO_PNET if no_pnet is None else no_pnet
        self.no_vnet = NO_VNET if no_vnet is None else no_vnet
        # the world cap binds on every decision well before the time budget
        # does, so this -- not SA_SPEND_MULT -- is the real compute knob
        self.max_worlds = MAX_WORLDS if max_worlds is None else max_worlds
        # rollout: play the policy to game end and score the real result,
        # instead of stopping after 1.5 turns and trusting the leaf evaluator
        self.rollout = rollout
        # A rollout mean over ~12 worlds of a 0/1 outcome has SE ~= 0.14, so
        # raw Monte-Carlo will happily overrule the clone on pure noise -- and
        # the clone is our strongest component. Require the search to beat the
        # policy's own pick by this margin (in win-probability units) before
        # deviating from it. 0 = trust the rollouts blindly.
        self.prior_bonus = (PRIOR_BONUS if prior_bonus is None
                            else prior_bonus)

    # ---- greedy playout ------------------------------------------------------

    def _playout(self, sid: int, obs: dict, me: int, deadline: float) -> float:
        """Greedy continuation: rest of our turn, the opponent's whole turn
        (played greedily against us), stopping at our next main select.

        In rollout mode there is no horizon: both sides keep playing to a
        terminal state and the leaf is the real result."""
        steps = 0
        opp_turn_seen = False
        step_cap = ROLLOUT_STEPS if self.rollout else MAX_PLAYOUT_STEPS
        while steps < step_cap:
            state = obs.get("current")
            sel = obs.get("select")
            if state is None or sel is None or state["result"] != -1:
                break
            acting = state["yourIndex"]
            if sel["type"] == MAIN and not self.rollout:
                if acting != me:
                    opp_turn_seen = True
                elif opp_turn_seen:
                    break  # back to us: 1.5-turn horizon reached
            if time.perf_counter() > deadline:
                break
            cands = candidate_combos(sel, PLAYOUT_CAP, self.rng)
            if len(cands) == 1:
                try:
                    sid, obs = fs.step(sid, cands[0])
                except fs.SearchError:
                    break
                steps += 1
                continue
            net = None if self.no_pnet else pnet.get()
            if net is not None:
                # clone policy plays both sides: one step, no child evals
                try:
                    sid, obs = fs.step(sid, net.choose(obs))
                    steps += 1
                    continue
                except (fs.SearchError, Exception):
                    pass  # fall through to greedy
            best_v = None
            best = None
            for c in cands:
                try:
                    sid2, obs2 = fs.step(sid, c)
                except fs.SearchError:
                    continue
                steps += 1
                v = evaluate(obs2["current"], me)
                better = (best_v is None
                          or (v > best_v if acting == me else v < best_v))
                if better:
                    best_v = v
                    best = (sid2, obs2)
                if time.perf_counter() > deadline:
                    break
            if best is None:
                break
            sid, obs = best
        if self.rollout:
            STATS["roll_steps"] += steps
            return self._rollout_value(obs, me)
        return self._leaf_value(obs, me)

    def _rollout_value(self, obs: dict, me: int) -> float:
        """Real game result in [-1, 1]. A rollout that ran out of budget or
        steps falls back to a squashed heuristic on the SAME scale, so
        terminated and truncated rollouts can be averaged together."""
        STATS["rollouts"] += 1
        cur = obs.get("current")
        if cur is None:
            STATS["roll_trunc"] += 1
            return 0.0
        res = cur["result"]
        if res != -1:
            STATS["roll_terminal"] += 1
            return 0.0 if res == 2 else (1.0 if res == me else -1.0)
        STATS["roll_trunc"] += 1
        return math.tanh(evaluate(cur, me) / TANH_SCALE)

    def _leaf_value(self, obs: dict, me: int) -> float:
        cur = obs.get("current")
        if cur is None:
            return 0.0
        if cur["result"] != -1:
            return evaluate(cur, me)  # terminal: +/-WIN or 0
        net = None if self.no_vnet else vnet.get()
        if net is not None:
            # scaled so terminals still dominate; comparable across leaves
            return 40.0 * (net.win_prob(cur, me) - 0.5)
        return evaluate(cur, me)

    # ---- root decision -------------------------------------------------------

    def decide(self, obs: dict, known_opp_hand: list[int], budget_s: float,
               combos: list[list[int]]) -> list[int]:
        me = obs["current"]["yourIndex"]
        sbi = obs["search_begin_input"]
        deck_visible = obs["select"].get("deck") is not None

        # policy prior: keep only the top root candidates the clone would
        # consider, plus its outright choice
        net = None if self.no_pnet else pnet.get()
        policy_pick = None   # the clone's own choice, our anchor in rollout mode
        if net is not None:
            try:
                # choose() ranks by score, candidate_combos emits sorted lists,
                # so compare as sets -- otherwise [3,1] != [1,3] and we append
                # a duplicate of a candidate we are already searching.
                policy_pick = sorted(net.choose(obs))
                if len(combos) > ROOT_POLICY_KEEP:
                    sc = net.scores(obs)
                    ranked = sorted(
                        combos,
                        key=lambda c: -(sum(sc[i] for i in c) / max(len(c), 1)
                                        + 0.01 * len(c)))
                    combos = ranked[:ROOT_POLICY_KEEP]
                if policy_pick not in combos:
                    combos.append(policy_pick)
            except Exception:
                policy_pick = None

        t0 = time.perf_counter()
        deadline = t0 + budget_s
        totals = [0.0] * len(combos)
        counts = [0] * len(combos)
        alive = list(range(len(combos)))  # root candidates still in play

        worlds = 0
        while worlds < self.max_worlds and len(alive) >= 1:
            now = time.perf_counter()
            if worlds > 0 and (now > deadline or len(alive) == 1):
                break
            # keep at least one full world even if the budget is tiny
            world = determinize(obs, self.decklist, known_opp_hand, self.rng)
            try:
                root_sid, _root_obs = fs.begin(
                    sbi,
                    [] if deck_visible else world.my_deck,
                    world.my_prize, world.opp_deck, world.opp_prize,
                    world.opp_hand, world.opp_active)
            except (fs.SearchError, ValueError):
                STATS["begin_fail"] += 1
                break  # determinization rejected; bail to what we have
            for i in alive:
                try:
                    sid, o = fs.step(root_sid, combos[i])
                except fs.SearchError:
                    totals[i] += -1e6
                    counts[i] += 1
                    continue
                v = self._playout(sid, o, me, deadline)
                totals[i] += v
                counts[i] += 1
                if time.perf_counter() > deadline and worlds > 0:
                    break
            fs.end()
            worlds += 1
            # successive halving: drop clearly-worse candidates so later
            # worlds concentrate on the contenders.
            # In rollout mode each world contributes a 0/1 outcome, so a mean
            # over 1-2 worlds is nearly pure noise and halving there discards
            # good candidates at random. Collect a few samples first.
            if self.rollout and worlds < HALVE_AFTER_WORLDS:
                continue
            if len(alive) > 3:
                scored = sorted(
                    (totals[i] / max(counts[i], 1), i) for i in alive)
                keep = max(3, len(alive) // 2) if worlds >= 2 else \
                    max(4, int(len(alive) * 0.7))
                alive = sorted(i for _, i in scored[-keep:])

        # In rollout mode, hand the clone's own pick a head start: only a
        # margin the noise is unlikely to manufacture should overrule it.
        bonus_i = -1
        if self.rollout and self.prior_bonus and policy_pick is not None:
            try:
                bonus_i = combos.index(policy_pick)
            except ValueError:
                bonus_i = -1

        best_i = alive[0] if alive else 0
        best_v = None
        for i in (alive or range(len(combos))):
            if counts[i] == 0:
                continue
            v = totals[i] / counts[i]
            if i == bonus_i:
                v += self.prior_bonus
            if best_v is None or v > best_v:
                best_v = v
                best_i = i
        # the anchor must stay reachable: if halving dropped it, keep it only
        # when nothing that survived actually beat it
        if bonus_i >= 0 and counts[bonus_i] and bonus_i not in (alive or []):
            v = totals[bonus_i] / counts[bonus_i] + self.prior_bonus
            if best_v is None or v > best_v:
                best_i = bonus_i

        if bonus_i >= 0:
            STATS["anchored"] += 1
            if best_i != bonus_i:
                STATS["deviated"] += 1

        STATS["decides"] += 1
        STATS["worlds"] += worlds
        STATS["budget_s"] += budget_s
        STATS["spent_s"] += time.perf_counter() - t0
        if DEBUG:
            import sys
            st = obs["current"]
            print(f"[sa] t{st['turn']} sel{obs['select']['type']}"
                  f"/{len(combos)}c w{worlds} "
                  f"best={best_v if best_v is not None else float('nan'):.2f}"
                  f" pick={combos[best_i]} bud={budget_s:.2f}s"
                  f" spent={time.perf_counter() - t0:.2f}s",
                  file=sys.stderr, flush=True)
        return combos[best_i]
