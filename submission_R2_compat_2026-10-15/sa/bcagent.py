"""Policy-only agent: pure behavioral clone, near-instant decisions."""
from __future__ import annotations

import sys
import traceback

from . import policynet, targeting

# --- health counters (day 15) -------------------------------------------------
# 🔴 WHY THIS EXISTS. `__call__`'s catch-all returns `range(minCount)` -- the
# first N options in INDEX ORDER -- and prints a traceback to stderr. On Kaggle
# that traceback goes nowhere anyone reads, so **a submission can run the
# index-order fallback on EVERY decision and look completely normal from
# outside**: it still returns legal moves, still finishes games, still gets a
# rating. §8g had to detect exactly this indirectly, by arguing from a 40.7%
# index-0 rate over 4,682 selects against the 100% a real fallback would show.
#
# These counters make it a direct read. They are free on the happy path (one
# dict increment per select, against ~1 ms of decision time) and the summary is
# ONE LINE per game, never per-decision spam.
STATS = {
    "calls": 0,          # selects seen
    "fallbacks": 0,      # times the catch-all fired -- ANY non-zero value is a bug
    "net_missing": 0,    # net failed to load: also index-order, also silent
    "deck_returns": 0,   # the pre-battle deck handshake
    "first_error": None,  # the first traceback, verbatim, for the log
    # E3 near-tie probe (day 23). An intervention whose firing rate is not
    # printed cannot be distinguished from one that never fired -- rule 9, and
    # the reason §8am reports "off-argmax selects" next to every score.
    "flip_eligible": 0,
    "flips": 0,
    # E21 (day 30). A rule that silently never fires produces a null that means
    # nothing -- the §8be family, and control 2 of every pre-registration since.
    # `fetch_seen` counts Petrel fetches reached; `fetch_fired` counts those the
    # rule actually redirected.
    #
    # ⚡ E23 (day 30): `fetch_fired` is NOT the treatment size. The rule can fire
    # on a fetch the net would have made anyway, and then the two arms of an A/B
    # are identical at that decision. `fetch_diff` counts the firings where the
    # rule's pick differs from `net.choose` -- the only ones that can move a
    # score. Diagnostic only: it costs one extra forward pass per firing (~0.3
    # per game), and it never changes what is returned.
    "fetch_seen": 0,
    "fetch_fired": 0,
    "fetch_diff": 0,
    # E26 (day 31). Same distinction one level up, for a whole substituted
    # POLICY: `x_fired` is every eligible single-pick decision the wrapper
    # visited, `x_diff` only those where the played option differs from ours.
    # `x_diff / x_fired` IS the deviation rate cell B must be matched to, and
    # it is realised on-policy -- offline sizing has missed in both directions
    # (§8cc 1.6x over, §8ce under).
    "x_fired": 0,
    "x_diff": 0,
    "x_skip": 0,     # multi-select or <2 options: both arms fall through here
    "x_error": 0,    # must be 0; the wrapper is fail-soft, so this is silent harm
}


def _option_sig(obs: dict, option: dict) -> bytes:
    """Bitwise identity of an option under the shipped encoding (§8x)."""
    from .optfeat import option_features
    import numpy as _np

    dense, card_id, attack_id, target_id = option_features(obs, option)
    return (_np.asarray(dense, dtype=_np.float32).tobytes()
            + _np.asarray([card_id, attack_id, target_id],
                          dtype=_np.int32).tobytes())


def health_line() -> str:
    """One-line health summary -- the highest value-per-byte thing to log.

    Print this once per game from the agent wrapper; a submission log built
    from it answers "was the net actually live?" without any per-move output.
    """
    s = STATS
    bad = s["fallbacks"] + s["net_missing"]
    status = "OK" if bad == 0 else "DEGRADED"
    line = (f"[health] {status} calls={s['calls']} fallbacks={s['fallbacks']} "
            f"net_missing={s['net_missing']} deck={s['deck_returns']}")
    if s["fetch_seen"]:
        line += (f" fetch={s['fetch_fired']}/{s['fetch_seen']}"
                 f" ({s['fetch_fired'] / s['fetch_seen']:.1%})"
                 f" diff={s['fetch_diff']}")
    if s["flip_eligible"]:
        line += (f" flips={s['flips']}/{s['flip_eligible']}"
                 f" ({s['flips'] / s['flip_eligible']:.1%})")
    if s["x_fired"]:
        line += (f" x={s['x_diff']}/{s['x_fired']}"
                 f" ({s['x_diff'] / s['x_fired']:.1%})"
                 f" skip={s['x_skip']} xerr={s['x_error']}")
    if s["first_error"]:
        line += f" first_error={s['first_error'][:200]!r}"
    # The oracle spends real wall-clock time, so "did it fire, how often, and
    # what did it cost" has to be readable without per-move logging -- E15's
    # null was only interpretable because `sym8` could be shown firing on 8.36%
    # of selects, and a component that silently never fires reads as a null.
    try:
        from .oracle import STATS as OSTATS, health_line as ohealth

        if OSTATS:
            line += " | " + ohealth()
        from .vlook import STATS as VSTATS, health_line as vhealth

        if VSTATS:
            line += " | " + vhealth()
        from .xpolicy import LIVE as XLIVE, health_line as xhealth

        if XLIVE:
            line += " | " + xhealth()
    except Exception:
        pass
    return line


def reset_stats() -> None:
    STATS.update(calls=0, fallbacks=0, net_missing=0, deck_returns=0,
                 first_error=None, flip_eligible=0, flips=0,
                 fetch_seen=0, fetch_fired=0, fetch_diff=0,
                 x_fired=0, x_diff=0, x_skip=0, x_error=0)


class PolicyAgent:
    def __init__(self, decklist: list[int], net_path: str | None = None,
                 chip_targeting: bool = True, energy_spread: bool = True,
                 drag_target: bool = False, boss_converts: bool = False,
                 drag_high_hp: bool = False, boss_veto: bool = False,
                 counter_source: bool = True, chip_wall_defer: bool = True,
                 boss_prize_veto: bool = False,
                 sequencer: bool = False, seq_k: int = 8, seq_dets: int = 4,
                 seq_budget: float = 0.35, seq_reply: bool = False,
                 flip_margin: float | None = None,
                 poffin_force: bool = False, sym_k: int = 0,
                 fetch_stadium: bool = False, fetch_scrapper: bool = False,
                 oracle: bool = False, orc_probe: int = 10,
                 orc_sel: int = 20, orc_arms: int = 3,
                 orc_wp: float = 0.85, orc_maxopt: int = 5,
                 orc_tau: float = 0.0, orc_cap: float = 12.0,
                 orc_maxdev: int = 0,
                 vlook: bool = False, vlk_worlds: int = 4,
                 vlk_maxopt: int = 12, vlk_cap: float = 5.0,
                 vlk_path: str | None = None,
                 vlk_lcb: float = 0.0, vlk_arms: int = 0,
                 vlk_rand: float = 0.0, vlk_tau: float = 0.0,
                 xnet_path: str | None = None, x_rand: float = 0.0,
                 x_rank: str = "", x_rankfile: str = "", x_dump: str = "",
                 x_accept: float = 1.0,
                 handoff_threshold: float | None = None):
        self.decklist = list(decklist)
        # R2 (day 27): average the decision over K bench-slot relabellings, a
        # nuisance variable the net demonstrably reads -- 16.9% of decisions
        # flip under one (EVIDENCE 8bt, sa/symavg.py). 0 = off; 1 = the no-op
        # control (identity relabelling only, still pays the extra plumbing).
        self.sym_k = int(sym_k or 0)
        self._sym_rng = None
        # E3's teacher-free gate (day 23). Take the OTHER side of a near-tie:
        # when the logit gap between the lowest-scored SELECTED option and the
        # highest-scored UNSELECTED one is below this, swap them. This is not a
        # candidate rule -- it is a probe that measures whether the band E3
        # wants a human to relabel is indifferent, using no teacher at all.
        # None = off, and `bc` with no flag is byte-identical in behaviour to
        # what it was before this existed.
        self.flip_margin = flip_margin
        # The FIFTH Boss's Orders rule: suppress the play when attacking
        # right now takes strictly more prizes than any drag can. The
        # other four picked a side in a trade and all measured null; this
        # one deletes a dominated option (EVIDENCE 6 vs 8g). Opt-in until
        # it clears the five anchors.
        self.boss_prize_veto = boss_prize_veto
        # B4: turn-level lookahead (sequencer.py). OFF by default and opt-in
        # via `bc:<label>,seq` until it clears an arena A/B -- it is an
        # experiment, not a shipped component (EVIDENCE 8m).
        # Load an explicit net before constructing the sequencer so simulated
        # continuations use the same policy as the owning agent. Falling back
        # to policynet.get() here would silently use the bundled checkpoint.
        # `net=a.npz+b.npz` loads an ENSEMBLE: the members vote per option
        # (softmax each, then average). One path behaves exactly as before.
        # `net=dev.npz|mid.npz` loads a PHASE HANDOFF: readiness_score picks
        # mid once the board looks past setup (see sa/readiness.py). `+` and
        # `|` are mutually exclusive specs.
        if net_path and "+" in net_path and "|" in net_path:
            raise ValueError(
                f"net {net_path!r}: use either '+' (ensemble) or '|' "
                f"(dev→mid handoff), not both")
        if net_path and "|" in net_path:
            parts = [p for p in net_path.split("|") if p]
            if len(parts) != 2:
                raise ValueError(
                    f"net {net_path!r}: phase handoff needs exactly "
                    f"early|mid (got {len(parts)} paths)")
            from .readiness import (DEFAULT_THRESHOLD, load_phase_handoff)
            thr = (DEFAULT_THRESHOLD if handoff_threshold is None
                   else float(handoff_threshold))
            self.net = load_phase_handoff(parts[0], parts[1], threshold=thr)
        elif net_path and "+" in net_path:
            parts = [p for p in net_path.split("+") if p]
            self.net = policynet.load_ensemble(parts)
        else:
            self.net = policynet.load(net_path) if net_path else None
        # 🔴 AN EXPLICIT `net=` THAT DOES NOT LOAD MUST NEVER BECOME THE
        # SINGLETON. `policynet.load` returns None on any guard failure or
        # exception rather than raising, and `__call__` below falls back to
        # `policynet.get()` -- the tracked `sa/policy_net.npz`, which is the old
        # width-496 `policy_lw2`. So before day 22, `bc:v7,net=<a net that
        # fails the dim or vocab guard>` played lw2, archived under the name of
        # the net it was ASKED for, and printed a perfectly ordinary score. A
        # whole A/B could run that way and read as a result. Demonstrated with a
        # v7 net whose vocab map was one entry short -- exactly the "rebuild the
        # corpus and a net's map is stale" hazard §8aw names -- which loaded as
        # None, was accepted by `arena.build_agent` (it checks only that the
        # path EXISTS), and would have played 496-wide lw2 against a 708-wide
        # control. Fail loudly instead: this is the fifth "plausible number, not
        # a crash" in this repo (rule 18).
        #
        # ⚠ Only the `net=` path is strict. The submission never passes
        # `net_path` -- `build_submission.py` ships the candidate AS
        # `sa/policy_net.npz` and verifies it with `policynet.load` at build
        # time -- so the shipped agent keeps its fail-soft behaviour, where
        # degrading and logging beats forfeiting a live episode.
        #
        # ⚠ MERGE NOTE (day 22): this guard sits immediately after the load and
        # BEFORE the sequencer is built, because the beyond-BC branch moved the
        # load earlier so `Sequencer` shares the agent's net. Failing here means
        # we never construct a Sequencer around a silently-null net, which is
        # strictly better than where the guard originally landed on `main`.
        if net_path and self.net is None:
            raise ValueError(
                f"net {net_path!r} exists but FAILED policynet.load's guard "
                f"(feature dims, n_pool/n_attr, or the v7 vocab row count). "
                f"Refusing to fall back to the tracked sa/policy_net.npz, "
                f"which is a different net and would have scored silently.")
        self.seq = None
        if sequencer:
            from .sequencer import Sequencer
            self.seq = Sequencer(decklist, k=seq_k, dets=seq_dets,
                                 budget_s=seq_budget, reply=seq_reply,
                                 net=self.net)
        # An explicit net lets two candidate policies play each other inside
        # ONE arena process. Comparing them via a third opponent instead needs
        # ~2x the games for the same resolution, and the module-level
        # policynet.get() singleton cannot hold two nets at once.
        # The net cannot see option HP at all (see targeting.py), so it aims
        # chip damage at chance. Per-instance so the two sides of an A/B can
        # differ inside one process.
        self.chip_targeting = chip_targeting
        # Same blindness on the other side of the board: no attached-energy
        # count per option, so it stacks a dead second {D} on one Munkidori.
        self.energy_spread = energy_spread
        # Boss's Orders: which benched Pokemon to drag, and when the drag is
        # worth the Supporter. Both need damage-vs-HP arithmetic, and both
        # default OFF: together they measured 0.452 [0.435, 0.470] over 3000
        # mirror games. Whatever the per-rule isolation says, `_A(_deck)` in
        # the submission's main.py takes these defaults -- so a rule turns on
        # here only once it has cleared 0.5 on its own.
        self.drag_target = drag_target
        self.boss_converts = boss_converts
        # `drag_high_hp` only reorders the KO-able group inside drag_target, so
        # it does nothing unless drag_target is on too.
        self.drag_high_hp = drag_high_hp
        # The third Boss's Orders intervention (P5b): suppress the play when
        # their bench holds nothing we can KO -- 32.4% of our plays. Off until
        # its own A/B clears 0.5, same discipline as the two above.
        self.boss_veto = boss_veto
        # Adrena-Brain's source pick: same HP blindness, and the source caps
        # how many counters the ability can move at all. ON by default -- it
        # cleared 0.5 alone (0.534 [0.513, 0.556] n=2000 mirror) and an
        # independent opponent agreed (0.626 [0.604, 0.647] vs rule:v10,noS
        # against 0.593 for a bare bc). `bc:<label>,noSrc` turns it off.
        self.counter_source = counter_source
        # E11: play Buddy-Buddy Poffin when the bench has >=2 free slots. The
        # 1150+ pilots play it in 70.2% of available turns at board size 4 and
        # our clone in 29.4% -- 0.80 plays/game, ordering-free (rule 21). OFF by
        # default until its own A/B clears the bar, same discipline as every
        # rule above it.
        self.poffin_force = poffin_force
        # E21: inject board facts into Petrel's fetch -- the ONE select whose
        # option vector carries no board at all (§8br). Both OFF by default and
        # opt-in via `bc:<label>,fstad` / `,fscrap` until an A/B clears the bar,
        # same discipline as every rule above.
        self.fetch_stadium = fetch_stadium
        self.fetch_scrapper = fetch_scrapper
        # The matchup branch (2026-07-30): `chip_target` is worth +0.077 in the
        # mirror and **-0.126 against `rule:crustle`**, because "kill what dies
        # to 30" farms Dwebbles while the undamageable wall survives. This defers
        # the select to the net whenever their Active is a wall.
        #
        # ON by default -- it cleared its bar on the anchor that motivated it
        # (0.663 [0.642, 0.684] vs 0.559 [0.537, 0.581] for unconditional
        # chip_target, n=2000 each vs `rule:crustle`) and it cannot fire in the
        # matchups where chip_target pays, so the mirror is untouched by
        # construction -- confirmed at 0.521 [0.490, 0.552] n=1000, containing
        # 0.5. `bc:<label>,noWall` turns it off. See report/EVIDENCE.md §8c.
        self.chip_wall_defer = chip_wall_defer
        # E17 / ROADMAP §2.7 — the clock. A two-stage rollout oracle over the
        # net's OWN top-k options, gated to the decisions E17 measured value at
        # (option count <= orc_maxopt, and not already won). OFF by default and
        # opt-in via `bc:<label>,orc` until it clears an arena A/B: it is an
        # experiment, not a shipped component, and it is the only component
        # here that can spend real wall-clock time. See sa/oracle.py.
        self.orc = None
        if oracle:
            from .oracle import RolloutOracle

            self.orc = RolloutOracle(
                decklist, net=self.net, arms=orc_arms, probe=orc_probe,
                r_sel=orc_sel, wp_skip=orc_wp, max_opts=orc_maxopt,
                tau=orc_tau, decision_cap_s=orc_cap,
                cap=orc_maxdev)

        # E20 — one-ply lookahead scored by a LEARNED value function, the
        # evaluator every dead search here lacked (§2 rollout variance, B4's
        # handcrafted evalfn, the clock's fused rollout). Opt-in via
        # `bc:<label>,vlp` and OFF by default: it is an experiment until it
        # clears the A/B pre-registered in docs/experiments/E20.
        self.vlk = None
        if vlook:
            from .vlook import ValueLookahead

            self.vlk = ValueLookahead(
                decklist, net=self.net, worlds=vlk_worlds,
                max_opts=vlk_maxopt, decision_cap_s=vlk_cap,
                vnet_path=vlk_path, lcb=vlk_lcb, arms=vlk_arms,
                rand_p=vlk_rand, tau=vlk_tau)

        # E26 — substitute a whole different POLICY's pick (`xnet=`), or a
        # rate- and rank-matched RANDOM one (`xrnd`), so that coherence is the
        # only difference between the two arms. Both arms construct this object
        # and pay the same second forward pass. OFF by default.
        self.xsub = None
        if xnet_path or x_rand or x_dump:
            from .xpolicy import Substitute, parse_rank_hist

            xnet = None
            if xnet_path:
                xnet = policynet.load(xnet_path)
                if xnet is None:
                    # Same guard, same reason as `net=`: a substitute that fails
                    # to load would silently leave the treatment arm playing the
                    # BASE policy, and the cell would read as a clean null.
                    raise ValueError(
                        f"xnet {xnet_path!r} exists but FAILED policynet.load's "
                        f"guard. Refusing to run an E26 treatment arm that is "
                        f"bitwise the control.")
            by_n, cal_ranks = None, None
            if x_rankfile:
                import json as _json
                from pathlib import Path as _P
                raw = _json.loads(_P(x_rankfile).read_text(encoding="utf-8"))
                by_n = {int(k): list(v) for k, v in raw["by_n"].items()}
                cal_ranks = list(raw.get("ranks") or [])
                if not by_n:
                    raise ValueError(
                        f"xrankfile {x_rankfile!r} carries an EMPTY histogram; "
                        f"the control would fall back to uniform ranks and "
                        f"would not be depth-matched.")
            self.xsub = Substitute(xnet=xnet, rand_rate=x_rand,
                                   rank_hist=parse_rank_hist(x_rank),
                                   rank_by_n=by_n, dump_path=x_dump,
                                   cal_ranks=cal_ranks, accept=x_accept)

    def _flip_near_tie(self, net, obs: dict, picked: list[int]) -> list[int]:
        """Swap the boundary pair when their logit gap is under `flip_margin`.

        The definition is `p43_dagger_queue.make_candidate`'s, deliberately and
        exactly: same boundary pair (lowest selected vs highest unselected),
        same exclusion of bitwise-equivalent options -- two copies of one card
        in one role are a free tie by construction (§8x) and flipping them
        would dilute the treatment with a known no-op. If the two scripts ever
        disagree, the sizing no longer describes the intervention.

        ⚠ Costs a second forward pass (~1.2 ms). `bc` has no time-budgeted
        component and uses 1.12 s of a 1,800 s pool, so this is not a compute
        confound the way E5's planner was -- but it IS an asymmetry between the
        arms, and it is stated rather than assumed away.
        """
        options = (obs.get("select") or {}).get("option") or []
        chosen = set(picked)
        unchosen = set(range(len(options))) - chosen
        if not chosen or not unchosen:
            return picked
        STATS["flip_eligible"] += 1
        scores = net.scores(obs)
        low = min(chosen, key=lambda i: float(scores[i]))
        high = max(unchosen, key=lambda i: float(scores[i]))
        if float(scores[low]) - float(scores[high]) >= self.flip_margin:
            return picked
        if _option_sig(obs, options[low]) == _option_sig(obs, options[high]):
            return picked
        STATS["flips"] += 1
        return [high if i == low else i for i in picked]

    def _sym_choose(self, net, obs: dict) -> list[int]:
        """`net.choose` with the bench relabelling averaged out (R2).

        Falls back to the plain path on any failure -- this is an experiment
        wrapped around the shipped agent, and it must never be the reason a
        live episode forfeits.
        """
        from . import symavg
        import random as _random
        if self._sym_rng is None:
            self._sym_rng = _random.Random(17)
        try:
            sc = symavg.sym_scores(net, obs, self.sym_k, self._sym_rng)
            if sc is None:
                return net.choose(obs)
            # `pick` owns the count rule; srepr is only read by the `learned`
            # count head, which the shipped COUNT_MODE ("table") does not use.
            return net.pick(obs, sc, None)
        except Exception:  # noqa: BLE001
            return net.choose(obs)

    def __call__(self, obs: dict) -> list[int]:
        STATS["calls"] += 1
        try:
            if obs.get("select") is None:
                STATS["deck_returns"] += 1
                # The deck registration is the only reliable game boundary an
                # agent sees -- arena builds the agent ONCE and plays every
                # match through it, so per-game oracle state must reset here.
                if self.orc is not None:
                    self.orc.new_game()
                if self.vlk is not None:
                    self.vlk.new_game()
                return list(self.decklist)
            sel = obs["select"]
            n = len(sel.get("option") or [])
            mn = sel.get("minCount", 0)
            mx = sel.get("maxCount", 0)
            if n == 0 or mx == 0:
                return []
            if mn == mx == n:
                return list(range(n))
            want = max(min(mn, mx, n), 1)
            if self.chip_targeting:
                order = targeting.chip_target(obs, self.chip_wall_defer)
                if order is not None:
                    return order[:want]
            if self.drag_target:
                order = targeting.drag_target(obs, self.drag_high_hp)
                if order is not None:
                    return order[:want]
            if self.boss_converts:
                order = targeting.boss_converts(obs)
                if order is not None:
                    return order[:want]
            if self.fetch_stadium or self.fetch_scrapper:
                # Counted before the rule runs so `seen` is the denominator
                # even when the condition does not hold.
                _eff = sel.get("effect")
                if (sel.get("context") == targeting.FETCH
                        and isinstance(_eff, dict)
                        and _eff.get("id") == targeting.PETREL):
                    STATS["fetch_seen"] += 1
                order = targeting.petrel_fetch(
                    obs, self.fetch_stadium, self.fetch_scrapper)
                if order is not None:
                    STATS["fetch_fired"] += 1
                    # Diagnostic only (E23): a firing that agrees with the net
                    # is a no-op for the A/B, so `fired` overstates the
                    # treatment. Never allowed to change the return value, and
                    # never allowed to raise -- §8bz's lesson is that a counter
                    # is what makes a null readable, not a reason to forfeit.
                    try:
                        _n = self.net or policynet.get()
                        if _n is not None and _n.choose(obs)[:want] != order[:want]:
                            STATS["fetch_diff"] += 1
                    except Exception:  # noqa: BLE001
                        pass
                    return order[:want]
            net = self.net or policynet.get()
            if net is None:
                # index order, silently, forever -- the failure §8g had to
                # infer. Counted so it can be read instead.
                STATS["net_missing"] += 1
                return list(range(mn))
            if self.sym_k:
                picked = self._sym_choose(net, obs)
            else:
                picked = net.choose(obs)
            if self.xsub is not None:
                picked = self.xsub.apply(net, obs, picked, want, STATS)
            if self.flip_margin is not None:
                picked = self._flip_near_tie(net, obs, picked)
            if self.boss_veto:
                # lazy: the full ranking costs a second forward pass, and the
                # veto fires only when the net's top pick is Boss's Orders
                fixed = targeting.boss_veto(
                    obs, list(picked), lambda: targeting.full_rank(net, obs))
                if fixed is not None:
                    return fixed
            if self.boss_prize_veto:
                fixed = targeting.boss_prize_veto(
                    obs, list(picked), lambda: targeting.full_rank(net, obs))
                if fixed is not None:
                    return fixed
            if self.poffin_force:
                forced = targeting.poffin_force(obs, list(picked))
                if forced is not None:
                    return forced[:want]
            if self.counter_source:
                fixed = targeting.counter_source(
                    obs, list(picked), lambda: targeting.full_rank(net, obs))
                if fixed is not None:
                    return fixed
            if self.energy_spread:
                fixed = targeting.energy_spread(obs, list(picked))
                if fixed is not None:
                    picked = fixed
            # B4 last: it overrules the clone AND the rules, because it is the
            # only component that scores a whole turn rather than one option.
            # Returns None (fall through) whenever it cannot plan safely.
            if self.seq is not None:
                planned = self.seq.plan(obs, list(picked))
                if planned is not None:
                    return planned
            # The oracle goes LAST, for the same reason the sequencer does: it
            # is the only component that scores an option by simulated OUTCOME
            # rather than by resemblance to the corpus. It returns None -- keep
            # `picked` -- on any doubt, any budget pressure and any exception.
            if self.orc is not None:
                better = self.orc.choose(obs, list(picked))
                if better is not None:
                    return better
            # E20 goes after the oracle for the same reason: it scores an
            # option by SIMULATED OUTCOME rather than resemblance to the
            # corpus, and returns None on any doubt. The two are never both on.
            if self.vlk is not None:
                better = self.vlk.choose(obs, list(picked))
                if better is not None:
                    return better
            return picked
        except Exception:
            STATS["fallbacks"] += 1
            if STATS["first_error"] is None:
                STATS["first_error"] = traceback.format_exc()
            traceback.print_exc(file=sys.stderr)
            try:
                return list(range((obs.get("select") or {}).get("minCount", 0)))
            except Exception:
                return []
