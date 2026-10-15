"""E20 — ONE-PLY lookahead scored by a learned value function.

**This is `oracle.py` with the evaluator swapped, and that swap is the whole
hypothesis.** The clock forked a position and rolled ~100 engine steps to a
terminal 0/1; E19 (§8ca) measured that estimate as biased and bought nothing.
Here the fork advances **exactly one step** and the resulting state is scored by
`valuenet`, trained on 20,000 self-play outcomes (`scripts/train_value.py`).

Why each prior death does not transfer:

  * §2's search died of variance — a terminal rollout returns 0/1, so a mean
    over 12 determinizations carries SE ~ 0.14 and the max over ~9 arms sits
    0.21-0.28 above truth by chance. **V returns a calibrated scalar. That
    variance term does not exist here.**
  * B4/E5 died with `evalfn`, a handcrafted scorer. **V is learned from
    realized outcomes.**
  * The clock died (most likely) of strategy fusion compounding over ~100
    simulated steps inside a determinized world. **One step cannot compound.**

⚠ Determinization is still present — `fs.begin` needs a full world — so fusion
is *reduced*, not eliminated. Averaging over `worlds` sampled worlds is what
marginalizes it, and W is frozen at 4 by the pre-registration.

🔴 **THE SEAT TRAP, and it is the one bug here that would look like a result.**
After `fs.step` the successor's `current["yourIndex"]` is frequently the
OPPONENT, because control passes. V must be read from **our** seat, captured
before the step. Reading `yourIndex` instead yields a V that is systematically
inverted exactly when control changes hands — an agent that plays to lose,
confidently, with every health counter green.
"""
from __future__ import annotations

import random
import time
from collections import Counter

import numpy as np

from . import fastsearch as fs
from . import policynet as pnet
from . import valuenet as vnet
from .worlds import determinize

MAIN = 0
RESERVE_S = 45.0

STATS: Counter = Counter()
ERR: dict = {}


def health_line() -> str:
    s = STATS
    ev = s["evals"]
    out = (f"[vlook] fired={s['fired']} overruled={s['overruled']} "
           f"evals={ev} skip_shape={s['skip_shape']} "
           f"skip_trigger={s['skip_trigger']} skip_noclock={s['skip_noclock']} "
           f"skip_thin={s['skip_thin']} skip_novnet={s['skip_novnet']} "
           f"errors={s['errors']} secs={s['secs_x1000'] / 1000.0:.1f}")
    if ev:
        out += f" ms/eval={s['secs_x1000'] / ev:.1f}"
    if "err" in ERR:
        out += f" first_error={ERR['err']!r}"
    if s["sd_n"]:
        out += f" ens_sd={s['sd_x1000'] / 1000.0 / s['sd_n']:.3f}"
    for e in (0.01, 0.02, 0.04, 0.08, 0.16, 0.32):
        if s[f"gapge_{e}"]:
            out += f" ge{e}={s[f'gapge_{e}']}"
    if s["fired"]:
        # E21's decisive diagnostic: E20 kept the clone's pick 23% of the time
        # here and 6.1% over p87's wider decision set, both under a ~20% chance
        # rate. If pessimism identified the mechanism this rises.
        out += (f" agree={1 - s['overruled'] / s['fired']:.1%}")
    o = s["overruled"]
    if o:
        # Same reasoning as oracle._change_line: a null is uninterpretable
        # unless we know whether V agreed with the net (nothing to change) or
        # overruled it constantly and still gained nothing.
        out += (f" | CHANGED {o}/{s['fired']} ({o / max(s['fired'], 1):.0%})"
                f" net-margin mean {s['margin_x1000'] / 1000.0 / o:+.2f}")
        buckets = [f"{b}:{s[f'netmargin_{b}']}" for b in
                   ("lt0p5", "lt1p5", "lt3", "ge3") if s.get(f"netmargin_{b}")]
        if buckets:
            out += " (" + " ".join(buckets) + ")"
        out += f" v-gap mean {s['gap_x1000'] / 1000.0 / o:+.3f}"
    return out


def reset_stats() -> None:
    STATS.clear()
    ERR.clear()


class ValueLookahead:
    """One ply into the real engine, scored by V. Returns None on any doubt."""

    def __init__(self, decklist: list[int], net=None, worlds: int = 4,
                 max_opts: int = 12, min_opts: int = 2, tau: float = 0.0,
                 decision_cap_s: float = 5.0, reserve_s: float = RESERVE_S,
                 min_turn: int = 2, seed: int = 0, vnet_path: str | None = None,
                 lcb: float = 0.0, arms: int = 0, rand_p: float = 0.0):
        self.decklist = list(decklist)
        self.net = net
        # E21: `vnet=a.npz+b.npz+...` is an ENSEMBLE. Its members must all load
        # or the pessimism term is computed over a different set than the
        # identity records -- so a missing member is fatal, never silent.
        self.Vs = []
        if vnet_path:
            for p in vnet_path.split("+"):
                if not p:
                    continue
                m = vnet.load(p)
                if m is None:
                    raise ValueError(f"value net failed to load: {p}")
                self.Vs.append(m)
        self.V = self.Vs[0] if self.Vs else None
        # K in `mean - K*sd`. Frozen at 1.0 by E21; 0.0 reproduces E20 exactly.
        self.lcb = float(lcb)
        # 0 = every option (E20). >0 = the net's top-`arms`, which is a
        # COVERAGE constraint: successors of options the clone ranks last are
        # precisely the ones V never saw.
        self.arms = int(arms)
        # E22 AUDIT ARM. `rand_p` replaces V's argmax with a coin flip over the
        # SAME covered arms, at a MATCHED deviation rate, and is the control
        # that decides what E22's 0.1580 means. E20 -> E22 moved the win rate
        # 0.0065 -> 0.1580 while the average override's net-margin fell
        # +6.02 -> +2.33: the deviations got CLOSER TO THE CLONE at the same
        # time as they got better-selected, and the arm cannot separate those.
        # Rate-matching is the whole point -- an unmatched random arm deviates
        # on 2/3 of firings against E22's 0.555 and would be measuring the
        # deviation rate, which is the confound it exists to remove.
        self.rand_p = float(rand_p)
        if self.rand_p > 0.0 and self.arms <= 0:
            # The control is defined only relative to a covered arm set -- with
            # `arms=0` it would deviate uniformly over EVERY option, which is a
            # different treatment than the one E22 ran.
            raise ValueError("vrnd requires varm>0: the control is a coin flip "
                             "over the SAME covered arms, not over all options")
        self.worlds = int(worlds)
        self.max_opts = int(max_opts)
        self.min_opts = int(min_opts)
        self.tau = float(tau)
        self.decision_cap_s = float(decision_cap_s)
        self.reserve_s = float(reserve_s)
        self.min_turn = int(min_turn)
        self.rng = random.Random(seed)

    def new_game(self) -> None:
        return None

    # -- the fork, ONE step, then V -------------------------------------
    def _evaluate(self, obs: dict, world, first: list[int], me: int, Vs):
        """-> (mean, sd) of the ensemble at the successor, from `me`'s seat.

        The fork is paid ONCE and every member scores the same successor, so a
        5-net ensemble costs 5 cheap forward passes, not 5 engine steps.

        Frees with `fs.end()`, never `fs.release()`. §N.6's defect 2: `release`
        reclaims ONE search id while a fork creates one per step, which leaked
        1.68 GB in 8 minutes and probably caused an intermittent 6.9% error
        rate through allocation failure.
        """
        root = None
        try:
            sel = obs["select"]
            root, o = fs.begin(
                obs["search_begin_input"],
                [] if sel.get("deck") is not None else world.my_deck,
                world.my_prize, world.opp_deck, world.opp_prize,
                world.opp_hand, world.opp_active)
            _sid, o = fs.step(root, first)
            cur = o.get("current")
            if cur is None:
                return None
            r = cur.get("result", -1)
            if r != -1:
                # One ply ended the game: ground truth, and it carries ZERO
                # epistemic uncertainty, so pessimism must not penalise it.
                return (0.5 if r == 2 else (1.0 if r == me else 0.0)), 0.0
            # 🔴 `me`, never cur["yourIndex"] -- p86 verified indexing is
            # absolute, so our pre-step seat stays valid after the step.
            vs = [m.win_prob(cur, me) for m in Vs]
            mu = sum(vs) / len(vs)
            sd = (sum((x - mu) ** 2 for x in vs) / len(vs)) ** 0.5 if len(vs) > 1 else 0.0
            return mu, sd
        except Exception as e:
            STATS["errors"] += 1
            if "err" not in ERR:
                ERR["err"] = f"{type(e).__name__}: {e}"[:200]
            return None
        finally:
            STATS["evals"] += 1
            if root is not None:
                try:
                    fs.end()
                except Exception:
                    pass

    def choose(self, obs: dict, picked: list[int]) -> list[int] | None:
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
            if not picked or len(picked) != 1:
                STATS["skip_shape"] += 1
                return None
            n = len(sel.get("option") or [])
            STATS[f"nopt_{min(n, 12):02d}"] += 1
            # ⚠ NOT a value trigger. E20 pre-registers "no trigger"; this is a
            # shape bound only -- fewer than 2 options is not a decision, and
            # the upper bound caps worst-case cost, not expected value.
            if not (self.min_opts <= n <= self.max_opts):
                STATS["skip_trigger"] += 1
                return None

            Vs = self.Vs or ([vnet.get()] if vnet.get() else [])
            if not Vs:
                STATS["skip_novnet"] += 1       # a missing npz must not read as a null
                return None
            net = self.net or pnet.get()
            if net is None:
                STATS["skip_shape"] += 1
                return None

            rem = float(obs.get("remainingOverageTime", 600.0))
            budget = min(self.decision_cap_s, rem - self.reserve_s)
            if budget <= 0.2:
                STATS["skip_noclock"] += 1
                return None
            deadline = t0 + budget

            me = cur["yourIndex"]               # OUR seat, captured before any step
            a0 = int(picked[0])
            # E21 COVERAGE: the net's top-`arms` options, with the agent's own
            # pick as arm 0. E20 scored every option and its argmax agreed with
            # the clone 6.1% against a 20.3% chance rate -- successors of
            # options the clone ranks last are exactly the ones V never saw.
            if self.arms > 0:
                try:
                    sc0 = np.asarray(net.scores(obs), dtype=float)
                    order = [int(i) for i in np.argsort(-sc0)]
                    arms = [a0] + [i for i in order if i != a0][:self.arms - 1]
                except Exception:
                    STATS["skip_shape"] += 1
                    return None
            else:
                arms = list(range(n))
            if self.rand_p > 0.0:
                # No fork and no V: the arm set above is already the whole
                # treatment being controlled for. Fires on the identical
                # decision set (every gate above is pre-fork), so `fired` is
                # comparable to E22's counter by construction.
                STATS["fired"] += 1
                others = [j for j in arms if j != a0]
                if not others or self.rng.random() >= self.rand_p:
                    return None
                best = int(self.rng.choice(others))
                STATS["overruled"] += 1
                try:
                    margin = float(sc0[a0] - sc0[best])
                    STATS["margin_x1000"] += int(margin * 1000)
                    b = ("lt0p5" if margin < 0.5 else "lt1p5" if margin < 1.5
                         else "lt3" if margin < 3.0 else "ge3")
                    STATS[f"netmargin_{b}"] += 1
                except Exception:
                    pass
                return [best]
            base = self.rng.randrange(1 << 30)

            tot = {j: 0.0 for j in arms}
            sdt = {j: 0.0 for j in arms}
            cnt = {j: 0 for j in arms}
            for k in range(self.worlds):
                if time.perf_counter() > deadline:
                    break
                # A SHARED world per replicate — the only pairing available
                # (§8bw C2: the engine draws its own shuffles, so common random
                # numbers do not exist here). Worth rho ~ 0.53.
                seed = base + 100_003 + k
                for j in arms:
                    w = determinize(obs, self.decklist, [], random.Random(seed))
                    r = self._evaluate(obs, w, [j], me, Vs)
                    if r is not None:
                        tot[j] += r[0]
                        sdt[j] += r[1]
                        cnt[j] += 1
            if min(cnt.values()) < 1:
                STATS["skip_thin"] += 1
                return None

            # E21 PESSIMISM: mean - K*sd. The disagreement between
            # independently-initialised members is largest exactly where the
            # successor is off-distribution, which is where E20 failed.
            means = {j: tot[j] / cnt[j] - self.lcb * (sdt[j] / cnt[j])
                     for j in arms}
            STATS["sd_x1000"] += int(1000 * sum(sdt[j] / cnt[j]
                                                for j in arms) / len(arms))
            STATS["sd_n"] += 1
            best = int(max(arms, key=lambda j: means[j]))
            gap = means[best] - means[a0]
            STATS["fired"] += 1
            if best == a0 or gap <= self.tau:
                return None
            STATS["overruled"] += 1
            try:
                sc = np.asarray(net.scores(obs), dtype=float)
                margin = float(sc[a0] - sc[best])
                STATS["margin_x1000"] += int(margin * 1000)
                b = ("lt0p5" if margin < 0.5 else "lt1p5" if margin < 1.5
                     else "lt3" if margin < 3.0 else "ge3")
                STATS[f"netmargin_{b}"] += 1
            except Exception:
                pass
            STATS["gap_x1000"] += int(gap * 1000)
            # E25 probe: the DISTRIBUTION of V's own confidence in its
            # overrides. `vtau` is set from this to hit a pre-registered
            # deviation RATE, never to maximise a score -- the distinction
            # between a measurement and E17's post-hoc arm selection.
            for edge in (0.01, 0.02, 0.04, 0.08, 0.16, 0.32):
                if gap >= edge:
                    STATS[f"gapge_{edge}"] += 1
            return [best]
        except Exception as e:
            STATS["errors"] += 1
            if "err" not in ERR:
                ERR["err"] = f"choose {type(e).__name__}: {e}"[:200]
            return None
        finally:
            STATS["secs_x1000"] += int((time.perf_counter() - t0) * 1000)
