"""B4: choose a MAIN action by looking at where the whole TURN ends up.

**What this is, and what it deliberately is not.** The clone scores each option
in isolation. This scores an option by simulating the **rest of our own turn**
after it and evaluating the end-of-turn board with `evalfn`. That is not the
game-tree search that died in `EVIDENCE` §2: there are **no rollouts to a
terminal state and no simulation of the opponent's reply**, so the 0/1 rollout
variance (SE ~ 0.14) that made that search select noise does not arise here.

**Why we believe the eval can rank these** (`EVIDENCE` §8m, and it is the only
reason this file exists): over 93 real turns, the best-scoring candidate chosen
on one half of the determinizations also won on the *other, independent* half
**62.0% of the time against 6.2% chance**. The eval is ranking transferable
merit, not determinization luck.

**Design decisions, each with its reason:**

* **Re-plan at every MAIN select; use only the first action.** Caching a whole
  turn plan is cheaper, but the plan is computed under an *invented* deck order
  and desyncs the moment a real draw differs. Re-planning is robust and, at
  ~25 ms per select, affordable.
* **Average each candidate over M determinizations.** Hidden information must be
  invented before we can simulate; averaging is what turns §8m's M=1 signal
  (barely above noise) into its M=8 signal (5.7x the standard error).
* **Only fire where sequencing can matter**: MAIN selects with >= 2 options.
  62% of our turns have >= 2 such selects (`EVIDENCE` §8l); the rest are forced
  and greedy is optimal there by definition.
* **Hard time budget, checked every candidate.** Kaggle counts the 600 s pool
  per GAME and exhausting it is a loss (HANDOFF §7). The default is deliberately
  a small fraction of what §8l showed is affordable.
* **Never raise.** Any failure returns None and the caller falls back to the
  clone, which is the shipped agent.
"""
from __future__ import annotations

import random
import time

from . import policynet as pnet
from .evalfn import evaluate

MAIN = 0
MAX_STEPS = 24          # a turn is ~6 real selects; a generous cap
REPLY_STEPS = 24        # same cap for the opponent's reply turn
TOPK = 3                # sample among the net's top-3 at each select
# Terminal value used when the game ends inside the simulation. `evalfn` scores
# a board, not a result, so a line that loses on the spot would otherwise be
# scored on the corpse. Large enough to dominate any board term.
WIN_VALUE = 1e4


class Sequencer:
    """Turn-level lookahead. `plan()` returns a pick, or None to fall back.

    ⚠ `reply=True` is the fix for the DESIGN flaw diagnosed in `EVIDENCE` §8n,
    and it is the whole reason this class is still alive. The prototype scored
    **0.075 [0.026, 0.199] n=40** -- a rout, not a marginal loss -- and two
    candidate *bugs* were eliminated without moving it. The remaining
    explanation is that the objective is wrong: maximising the value of the
    board **at the end of OUR turn** structurally cannot see the opponent's
    reply, so a line that leaves a 2-prize attacker exposed scores well right up
    to the moment it is knocked out. With `reply=True` the simulation continues
    through the opponent's turn and evaluates when control returns to us.
    """

    def __init__(self, decklist: list[int], k: int = 8, dets: int = 4,
                 budget_s: float = 0.35, pool_floor_s: float = 120.0,
                 seed: int = 17, reply: bool = False, net=None):
        self.decklist = list(decklist)
        self.k = k
        self.dets = dets
        self.budget_s = budget_s
        # Stop planning entirely if the remaining pool gets this low: a
        # timeout is a LOSS, and the clone plays fine without us.
        self.pool_floor_s = pool_floor_s
        self.reply = reply
        # Explicit arena checkpoints are per-agent. Keep that exact object for
        # simulated continuations instead of silently using the bundled net.
        self.net = net
        self.rng = random.Random(seed)
        self.stats = {"planned": 0, "fellback": 0, "overruled": 0,
                      "sim_s": 0.0, "aborted_budget": 0}

    # ------------------------------------------------------------------ utils
    def _rollout(self, sid, obs, me, net, rng, forced=None):
        """Play our turn to its end. -> (eval, actions, ok).

        ⚠ `ok` is False unless the turn ACTUALLY ENDED. That is not a detail:
        a candidate stopped mid-turn by the step cap still holds its cards and
        has committed nothing, and `evalfn` scores that higher than the same
        line after it attacks. Comparing a mid-turn state against an
        end-of-turn state makes "stall" the winning move -- which is exactly
        what the first version of this file did (it scored 0.083 and played
        ~50% more turns per game). **All candidates must be evaluated at the
        same point in the turn or the comparison is meaningless.**
        """
        from . import fastsearch as fs
        actions: list[list[int]] = []
        cur_sid, cur = sid, obs
        ended = False
        for depth in range(MAX_STEPS):
            st = cur.get("current") or {}
            sel = cur.get("select") or {}
            opts = sel.get("option") or []
            if not st:
                break
            if st.get("result", -1) != -1:
                ended = True               # game over: a legitimate endpoint
                break
            if st.get("yourIndex") != me or not opts:
                ended = True               # turn passed to the opponent
                break
            pick = None
            if forced is not None and depth < len(forced):
                cand = [i for i in forced[depth] if 0 <= i < len(opts)]
                if len(cand) == len(forced[depth]) and cand:
                    pick = cand
            if pick is None:
                if len(opts) == 1:
                    pick = [0]
                else:
                    try:
                        sc = net.scores(cur) if net is not None else None
                    except Exception:  # noqa: BLE001
                        sc = None
                    if sc is None:
                        pick = [rng.randrange(len(opts))]
                    else:
                        order = sorted(range(len(opts)),
                                       key=lambda i: -float(sc[i]))[:TOPK]
                        pick = [rng.choice(order)]
            try:
                cur_sid, cur = fs.step(cur_sid, pick)
            except Exception:  # noqa: BLE001
                return None, actions, False
            actions.append(list(pick))
        st = cur.get("current") or {}
        if not st or not ended:
            # hit the step cap mid-turn -- NOT comparable, discard it
            return None, actions, False
        if self.reply:
            cur_sid, cur, ok = self._reply(cur_sid, cur, me, net, rng)
            if not ok:
                # The reply did not complete. Same rule as a mid-turn cap: a
                # candidate evaluated BEFORE the reply is not comparable to one
                # evaluated after it, and mixing the two is precisely the bug
                # that made "stall" win the first prototype (§8n).
                return None, actions, False
            st = cur.get("current") or {}
            if not st:
                return None, actions, False
        res = st.get("result", -1)
        if res != -1:
            # Terminal inside the sim: score the RESULT, not the wreckage.
            return (WIN_VALUE if res == me else -WIN_VALUE), actions, True
        return float(evaluate(st, me)), actions, True

    def _reply(self, sid, obs, me, net, rng):
        """Play the OPPONENT's turn out. -> (sid, obs, ok).

        The opponent is piloted by the same clone, which is the best model of
        them we have (they are 46.9%% mirror and the field's modal policy is
        what the net fits best -- `EVIDENCE` §8r). Their hidden cards come from
        the same determinization as ours, so this adds no information we did
        not already invent.
        """
        from . import fastsearch as fs
        cur_sid, cur = sid, obs
        for _ in range(REPLY_STEPS):
            st = cur.get("current") or {}
            sel = cur.get("select") or {}
            opts = sel.get("option") or []
            if not st:
                return cur_sid, cur, False
            if st.get("result", -1) != -1:
                return cur_sid, cur, True      # game ended in their turn
            if st.get("yourIndex") == me:
                return cur_sid, cur, True      # control is back with us
            if not opts:
                return cur_sid, cur, False
            if len(opts) == 1:
                pick = [0]
            else:
                try:
                    sc = net.scores(cur) if net is not None else None
                except Exception:  # noqa: BLE001
                    sc = None
                if sc is None:
                    pick = [rng.randrange(len(opts))]
                else:
                    order = sorted(range(len(opts)),
                                   key=lambda i: -float(sc[i]))[:TOPK]
                    pick = [rng.choice(order)]
            try:
                cur_sid, cur = fs.step(cur_sid, pick)
            except Exception:  # noqa: BLE001
                return cur_sid, cur, False
        return cur_sid, cur, False             # cap hit: not comparable

    # ------------------------------------------------------------------- plan
    def plan(self, obs: dict, clone_pick: list[int]) -> list[int] | None:
        """-> the first action of the best turn continuation, or None."""
        from . import fastsearch as fs
        from .worlds import determinize

        sel = obs.get("select") or {}
        st = obs.get("current") or {}
        opts = sel.get("option") or []
        if (sel.get("context") != MAIN or len(opts) < 2
                or "search_begin_input" not in obs
                or st.get("result", -1) != -1):
            return None
        # A timeout is a loss; below the floor we simply stop planning.
        pool = st.get("pool")
        if isinstance(pool, (int, float)) and pool < self.pool_floor_s:
            return None

        t0 = time.perf_counter()
        deadline = t0 + self.budget_s
        me = st["yourIndex"]
        net = self.net or pnet.get()
        sbi = obs["search_begin_input"]
        deck_visible = sel.get("deck") is not None

        def begin_root(seed):
            w = determinize(obs, self.decklist, [], random.Random(seed))
            return fs.begin(sbi, [] if deck_visible else w.my_deck,
                            w.my_prize, w.opp_deck, w.opp_prize,
                            w.opp_hand, w.opp_active)

        def release_root(sid):
            try:
                fs.release(sid)
            except Exception:  # noqa: BLE001
                pass

        try:
            # 1. propose K candidate continuations under one determinization.
            # Release immediately: confirm (M=32) used to keep every world live
            # until the end of plan(), which OOM'd 8 GB laptops.
            base = self.rng.randrange(1 << 30)
            sid0, o0 = begin_root(base)
            acts: list[list[list[int]]] = []
            try:
                for j in range(self.k):
                    if time.perf_counter() > deadline:
                        self.stats["aborted_budget"] += 1
                        break
                    _v, a, ok = self._rollout(sid0, o0, me, net,
                                              random.Random(base + 101 + j))
                    if ok and a:
                        acts.append(a)
            finally:
                release_root(sid0)
            # distinct first actions only -- two candidates that open the same
            # way cannot inform THIS select, and they cost the same to score
            seen, uniq = set(), []
            for a in acts:
                key = tuple(a[0])
                if key not in seen:
                    seen.add(key)
                    uniq.append(a)
            if len(uniq) < 2:
                return None

            # 2. score each candidate, averaged over M determinizations.
            # One live world at a time; planner.py likewise frees per world.
            totals = [0.0] * len(uniq)
            counts = [0] * len(uniq)
            for m in range(self.dets):
                if time.perf_counter() > deadline:
                    self.stats["aborted_budget"] += 1
                    break
                try:
                    sidm, om = begin_root(base + 5000 + m)
                except Exception:  # noqa: BLE001
                    continue
                try:
                    for i, a in enumerate(uniq):
                        if time.perf_counter() > deadline:
                            break
                        v, _a, ok = self._rollout(
                            sidm, om, me, net, random.Random(base + 7),
                            forced=a)
                        if ok and v is not None:
                            totals[i] += v
                            counts[i] += 1
                finally:
                    release_root(sidm)
                    # Free stepped child nodes from this determinization's
                    # rollouts. Safe here: only the sequencer uses search.
                    try:
                        fs.end()
                    except Exception:  # noqa: BLE001
                        pass
            scored = [(totals[i] / counts[i], i)
                      for i in range(len(uniq)) if counts[i] >= 2]
            if len(scored) < 2:
                return None
            scored.sort(reverse=True)
            best = uniq[scored[0][1]][0]
            self.stats["planned"] += 1
            if list(best) != list(clone_pick):
                self.stats["overruled"] += 1
            return list(best)
        except Exception:  # noqa: BLE001
            self.stats["fellback"] += 1
            return None
        finally:
            self.stats["sim_s"] += time.perf_counter() - t0
            try:
                fs.end()
            except Exception:  # noqa: BLE001
                pass
