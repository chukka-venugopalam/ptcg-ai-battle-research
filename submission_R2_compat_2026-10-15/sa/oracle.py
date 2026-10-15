"""The clock: a two-stage rollout oracle over the net's OWN top-k options.

Licensed by **E17** (`docs/experiments/E17-self-oracle-value.md`, `EVIDENCE`
§8by), which measured what this is worth before it was built:

    realized gain, control-corrected, over the net's own top-3   +0.0139/decision
    the same with this module's two-stage gate                   +0.0120/decision
    ...for 17% of the compute                                    70 s of the 600 s

**What it does.** At a live MAIN decision it forks the real position, plays each
of the net's top-k options out to a terminal state with the clone piloting both
seats, and plays whichever actually wins more. No corpus, no mode, no
conformity — the one instrument this project owns that is not a *conformity*
metric (§8r) or a weak evaluator (`evalfn`).

**Why it is gated rather than always-on, and this is the whole design.** E17
measured that **57% of our decisions carry no value at all** (win probability
> 0.85 ⇒ +0.0015/decision) and that the value is concentrated where we are
**LOSING** (< 0.15 ⇒ +0.0743). So:

  stage 0  a FREE trigger — option count <= max_opts (+0.0373 on 36% of
           decisions, E17's Q6; independent of the turn>=11 trigger, 19
           positions overlap against 23 expected by chance)
  stage 1  a CHEAP probe — `probe` rollouts of the agent's own pick. If the
           position is already won, decline and spend nothing more. ⚠ The probe
           is noisy by construction (SE ~0.16 at 10 rollouts); E17 simulated
           the gate *including* that noise and it still keeps 89% of the value.
  stage 2  the contest — `r_sel` rollouts of each of the top-k arms on shared
           worlds, then argmax.

⚠ **Known and stated, not discovered later:**
- The value is win probability under **clone-vs-clone continuation** (§8bw). In
  a mirror A/B the opponent *is* the clone, so the rollout model is exactly
  right there and **flatters this agent**; against the real ladder it is a model
  mismatch.
- Determinization samples the opponent's *whole deck* from a library by best
  overlap, and inside each sampled world the rollout plays as if the hidden
  cards were known. Averaging outcomes does not restore the information set.
- Per-decision win-probability gains **do not add** across a game. E17 cannot
  produce a win rate; only the A/B can.
- `tau` (a minimum margin before overruling) is **post-hoc** — six values swept,
  none pre-registered. Default 0.0 = the pre-registered behaviour.

🔴 **Frees with `fs.end()` per rollout.** The first version released only the
root search id and LEAKED ~1 MB per rollout — 1.68 GB in 8 minutes, against
0.12 GB for E17's collection doing the same work through `p80.rollout`. See
`_rollout` for the invariant that makes `end()` safe inside a live game.
"""
from __future__ import annotations

import random
import time
from collections import Counter

import numpy as np

from . import fastsearch as fs
from . import policynet as pnet
from .worlds import determinize

MAIN = 0
ROLLOUT_CAP = 1500
RESERVE_S = 45.0        # never plan to dip into the last chunk (timemgr's line)

STATS: Counter = Counter()
ERR: dict = {}


def health_line() -> str:
    """⚠ The skip reasons are kept SEPARATE on purpose. A single `skip_budget`
    counter conflated "the clock is gone" with "too few rollouts finished", and
    the two demand opposite fixes -- the first is a real budget problem, the
    second was a guard misfiring on a small `r_sel`.
    """
    s = STATS
    hist = " ".join(f"{k[5:]}:{v}" for k, v in sorted(s.items())
                    if k.startswith("nopt_"))
    return (f"[oracle] fired={s['fired']} probed={s['probed']} "
            f"skip_won={s['skip_won']} skip_trigger={s['skip_trigger']} "
            f"skip_noclock={s['skip_noclock']} skip_thin={s['skip_thin']} "
            f"skip_shape={s['skip_shape']} skip_capped={s['skip_capped']} "
            f"overruled={s['overruled']} "
            f"aborted={s['aborted']} errors={s['errors']} "
            f"rollouts={s['rollouts']} secs={s['secs_x1000'] / 1000.0:.1f}"
            + (f" first_error={ERR['err']!r}" if 'err' in ERR else "")
            + _change_line()
            + (f" | nopt {hist}" if hist else ""))


def _change_line() -> str:
    """What the extra time CHANGED, not merely that it was spent.

    Without this a null reads the same whether search agreed with the net at
    every decision (nothing to change) or overruled it constantly and gained
    nothing (the interesting failure). Reports the overrule rate, which rank
    search promoted, and how far down the NET's own ranking the promoted
    option sat.
    """
    s = STATS
    o = s["overruled"]
    if not o:
        return ""
    parts = [f" | CHANGED {o}/{s['fired']} fired ({o / max(s['fired'], 1):.0%})"]
    ranks = [f"r{k}:{s[f'promoted_rank{k}']}" for k in (2, 3, 4)
             if s.get(f"promoted_rank{k}")]
    if ranks:
        parts.append("promoted " + ",".join(ranks))
    parts.append(f"net-margin mean {s['margin_x1000'] / 1000.0 / o:+.2f}")
    buckets = [f"{b}:{s[f'netmargin_{b}']}" for b in
               ("lt0p5", "lt1p5", "lt3", "ge3") if s.get(f"netmargin_{b}")]
    if buckets:
        parts.append("(" + " ".join(buckets) + ")")
    parts.append(f"rollout-gap mean {s['gap_x1000'] / 1000.0 / o:+.3f}")
    return "  ".join(parts)


def reset_stats() -> None:
    STATS.clear()
    ERR.clear()


class RolloutOracle:
    def __init__(self, decklist: list[int], net=None, arms: int = 3,
                 probe: int = 10, r_sel: int = 20, wp_skip: float = 0.85,
                 max_opts: int = 5, min_opts: int = 3, tau: float = 0.0,
                 decision_cap_s: float = 12.0, reserve_s: float = RESERVE_S,
                 min_turn: int = 2, seed: int = 0, cap: int = 0):
        # E19a: at most `cap` overrules per GAME (0 = unlimited). This exists to
        # test the one-step-deviation hypothesis: every value this component
        # acts on is Q^π(s,a), the worth of deviating ONCE with the clone
        # continuing, and E18 deviated 3.32 times a game. At cap=1 the
        # assumption is exactly satisfied, so the measured +0.035 should arrive
        # intact -- or not, which would mean the rollout value never transferred.
        self.cap = int(cap)
        self._used = 0
        self.decklist = list(decklist)
        self.net = net
        self.arms = int(arms)
        self.probe = int(probe)
        self.r_sel = int(r_sel)
        self.wp_skip = float(wp_skip)
        self.max_opts = int(max_opts)
        self.min_opts = int(min_opts)
        self.tau = float(tau)
        self.decision_cap_s = float(decision_cap_s)
        self.reserve_s = float(reserve_s)
        self.min_turn = int(min_turn)
        self.rng = random.Random(seed)

    def new_game(self) -> None:
        """Reset per-game state. 🔴 Called from `PolicyAgent.__call__` on the
        deck-registration select, which is the only reliable game boundary the
        agent sees: `arena.py` constructs the agent ONCE and plays every match
        through it, so a cap that never reset would fire once per SHARD -- a
        null by construction, and one that would have looked like a result.
        """
        self._used = 0

    # -- the fork, one rollout to terminal ------------------------------
    def _rollout(self, obs: dict, world, first: list[int], me: int, net,
                 deadline: float):
        """-> win probability in {0, 0.5, 1} from `me`'s view, or None.

        🔴 **Frees with `fs.end()`, and the first version's `fs.release(root)`
        was a LEAK.** Measured: an arena process running this at
        `release`-per-rollout reached **1.68 GB in 8 minutes** (~1 MB per
        rollout, unbounded), while E17's collection processes — identical work
        through `p80.rollout`, which calls `fs.end()` — sat flat at **0.12 GB**.
        Six shards of the leaking version do not fit in 8 GB.

        `release` frees one search id; the rollout creates a fresh id at every
        one of up to `ROLLOUT_CAP` steps, so releasing the root reclaims one
        node out of ~100. `end()` frees the whole arena, which the C library
        then reuses for the next search — that is its documented behaviour and
        it costs nothing measurable (p80 ran at the same 76–100 ms/rollout).

        ⚠ **Invariant this relies on:** no other `fs` search may be live across
        an oracle rollout. True here — the harness drives the real game through
        `ptcg.env`, not `fs`, and `sequencer.plan()` has already returned and
        released its own roots before the oracle is consulted.
        """
        sel = obs["select"]
        root = None
        try:
            root, o = fs.begin(
                obs["search_begin_input"],
                [] if sel.get("deck") is not None else world.my_deck,
                world.my_prize, world.opp_deck, world.opp_prize,
                world.opp_hand, world.opp_active)
            sid, o = fs.step(root, first)
            steps = 1
            while steps < ROLLOUT_CAP:
                cur, s2 = o.get("current"), o.get("select")
                if cur is None or s2 is None:
                    return None
                if cur["result"] != -1:
                    r = cur["result"]
                    return 0.5 if r == 2 else (1.0 if r == me else 0.0)
                if time.perf_counter() > deadline:
                    STATS["aborted"] += 1
                    return None
                sid, o = fs.step(sid, net.choose(o))
                steps += 1
            return None
        except Exception as e:
            STATS["errors"] += 1
            # A swallowed exception is a silent null: the component reports
            # "no gain" when it never ran. Record the first one so the health
            # line says WHAT failed, the way bcagent's `first_error` does.
            if "err" not in ERR:
                ERR["err"] = f"{type(e).__name__}: {e}"[:200]
            return None
        finally:
            STATS["rollouts"] += 1
            if root is not None:
                try:
                    fs.end()          # the whole arena, not one node -- see above
                except Exception:
                    pass

    def _arms_for(self, net, obs: dict, picked: list[int]) -> list[int] | None:
        """Arm 0 is the agent's OWN pick, not merely the net's argmax.

        With rules off these coincide (verified 106/106: `argmax(scores) ==
        net.choose`), but if a rule fired, the thing to improve on is what the
        agent would actually have played.
        """
        try:
            sc = np.asarray(net.scores(obs), dtype=float)
        except Exception:
            return None
        order = [int(i) for i in np.argsort(-sc)]
        a0 = int(picked[0])
        rest = [i for i in order if i != a0]
        return [a0] + rest[:max(0, self.arms - 1)]

    # -- the decision ---------------------------------------------------
    def choose(self, obs: dict, picked: list[int]) -> list[int] | None:
        """-> a replacement single-index pick, or None to keep `picked`.

        Returns None on ANY doubt. This runs in a live game where a raised
        exception is a lost match, and `timeout = loss` is a failure mode this
        project has never been near and must not meet now.
        """
        t0 = time.perf_counter()
        try:
            sel = obs.get("select") or {}
            cur = obs.get("current") or {}
            if sel.get("context") != MAIN or not obs.get("search_begin_input"):
                STATS["skip_shape"] += 1
                return None
            if not (sel.get("minCount", 1) <= 1 <= sel.get("maxCount", 1)):
                STATS["skip_shape"] += 1
                return None
            if cur.get("result", -1) != -1 or cur.get("turn", 0) < self.min_turn:
                STATS["skip_shape"] += 1
                return None
            if self.cap and self._used >= self.cap:
                STATS["skip_capped"] += 1     # spend nothing once capped out
                return None
            n = len(sel.get("option") or [])
            STATS[f"nopt_{min(n, 12):02d}"] += 1   # live option-count histogram
            if not (self.min_opts <= n <= self.max_opts):
                STATS["skip_trigger"] += 1        # stage 0, and it is FREE
                return None
            if not picked or len(picked) != 1:
                STATS["skip_shape"] += 1
                return None

            # budget: refuse rather than risk the clock. `remainingOverageTime`
            # is the pool the engine actually enforces.
            rem = float(obs.get("remainingOverageTime", 600.0))
            budget = min(self.decision_cap_s, rem - self.reserve_s)
            if budget <= 1.0:
                STATS["skip_noclock"] += 1
                return None
            deadline = t0 + budget

            net = self.net or pnet.get()
            if net is None:
                STATS["skip_shape"] += 1
                return None
            me = cur["yourIndex"]
            arms = self._arms_for(net, obs, picked)
            if arms is None or len(arms) < 2:
                STATS["skip_shape"] += 1
                return None

            base = self.rng.randrange(1 << 30)

            # ---- stage 1: the cheap probe -----------------------------
            # E17: 57% of decisions sit above 0.85 win probability and are
            # worth +0.0015. Buying that information for `probe` rollouts and
            # declining is the single largest compute saving in the design.
            if self.probe > 0:
                vals = []
                for k in range(self.probe):
                    if time.perf_counter() > deadline:
                        break
                    w = determinize(obs, self.decklist, [],
                                    random.Random(base + k))
                    v = self._rollout(obs, w, [arms[0]], me, net, deadline)
                    if v is not None:
                        vals.append(v)
                STATS["probed"] += 1
                if len(vals) < max(3, self.probe // 3):
                    STATS["skip_thin"] += 1
                    return None
                if (sum(vals) / len(vals)) > self.wp_skip:
                    STATS["skip_won"] += 1
                    return None

            # ---- stage 2: the contest, paired on a shared world -------
            # ⚠ NOT common random numbers — the engine draws its own shuffles
            # and coins beyond the determinized world (§8bw C2). A shared world
            # is the only pairing there is, and it is worth rho~0.53.
            tot = [0.0] * len(arms)
            cnt = [0] * len(arms)
            for k in range(self.r_sel):
                if time.perf_counter() > deadline:
                    break
                seed = base + 100_003 + k
                for j, a in enumerate(arms):
                    w = determinize(obs, self.decklist, [],
                                    random.Random(seed))
                    v = self._rollout(obs, w, [a], me, net, deadline)
                    if v is not None:
                        tot[j] += v
                        cnt[j] += 1
            if min(cnt) < min(3, self.r_sel):
                STATS["skip_thin"] += 1
                return None

            means = [tot[j] / cnt[j] for j in range(len(arms))]
            gaps = [m - means[0] for m in means]
            gaps[0] = 0.0
            best = int(max(range(len(arms)), key=lambda j: gaps[j]))
            STATS["fired"] += 1
            if best == 0 or gaps[best] <= self.tau:
                return None                       # not earned: keep the pick
            STATS["overruled"] += 1
            self._used += 1
            # ── what did the extra time actually CHANGE? ──────────────────
            # A null is uninterpretable without this: it separates "search
            # agreed with the net, so of course nothing moved" from "search
            # overruled the net constantly and still gained nothing".
            STATS[f"promoted_rank{best + 1}"] += 1
            try:
                sc = np.asarray(net.scores(obs), dtype=float)
                # how far down the NET's own ranking was the option we took?
                margin = float(sc[arms[0]] - sc[arms[best]])
                STATS["margin_x1000"] += int(margin * 1000)
                # bucket it: a switch the net nearly agreed with is a very
                # different event from one it scored far worse.
                b = ("lt0p5" if margin < 0.5 else "lt1p5" if margin < 1.5
                     else "lt3" if margin < 3.0 else "ge3")
                STATS[f"netmargin_{b}"] += 1
            except Exception:
                pass
            STATS["gap_x1000"] += int(gaps[best] * 1000)
            return [arms[best]]
        except Exception as e:
            STATS["errors"] += 1
            if "err" not in ERR:
                ERR["err"] = f"choose {type(e).__name__}: {e}"[:200]
            return None
        finally:
            STATS["secs_x1000"] += int((time.perf_counter() - t0) * 1000)
