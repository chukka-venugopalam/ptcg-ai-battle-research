"""E26: substitute a DIFFERENT POLICY's pick, or a rate- and rank-matched
random one, so that coherence is the only thing that differs between the arms.

**The question** (docs/experiments/E26-coherence-at-matched-rate.md, §N.3).
E25 closed "override the clone with a better *evaluator*": the evaluator's
confidence and its distance from the clone are the same variable, so the ranking
signal is the damage signal (§8cg). A different *policy* is not an evaluator --
it is on-distribution for itself and its deviations are chosen in sequence by a
net trained to make them. Whether that matters is H1 vs H2, and it has been the
project's named open question since day 28.

**The design.** Both arms run this same wrapper and pay the same second forward
pass -- the `sym1` / `flip0` discipline, which is what made §8bu's null readable:

* `xnet=<path>`  -- TREATMENT: play that net's pick.
* `xrnd<p>`      -- CONTROL: with probability `p`, play a random option drawn at
                    a rank sampled from `xrank`, else play our own pick.

⚠ **Rank matching is the part E25 wished it had.** §8cf recorded that ~20% of
V's apparent edge might have been net-margin rather than selection, because the
control deviated DEEPER than the treatment. Here the control's rank histogram is
the treatment's *measured* one, so depth is matched by construction and the
residual is direction alone.

⚠ **Only single-pick decisions are eligible.** "Rank" is meaningless for a
multi-select, and a treatment that replaces a whole pick list while the control
replaces one element would not be matched. Both arms fall through identically
on `want > 1`, and the counters say how often that happened.

⚠ `x_fired` is the denominator; `x_diff` is the treatment. A firing whose pick
the base net would have made anyway is NOT a treatment -- E24's lesson (§8ce),
where `fired` overstated the treated set by ~7%.
"""
from __future__ import annotations

import random

import numpy as np


def parse_rank_hist(spec: str) -> list[float] | None:
    """`"1:0.62;2:0.21;3:0.10"` -> a normalised weight per rank (index 0 unused).

    Ranks are 1-based because rank 0 is our own pick, i.e. no deviation at all.

    ⚠ **The separator is `;`, not `,`, and that is not cosmetic:** an agent spec
    is itself comma-delimited (`arena.build_agent` splits on `,`), so a
    comma-separated histogram is parsed as a run of unknown flags. It failed
    loudly the first time it was run, which is what that guard exists for --
    `,` is still accepted here so a histogram pasted from `hist_spec()` into a
    script rather than a spec string does not silently mean something else.
    """
    if not spec:
        return None
    w: dict[int, float] = {}
    for part in spec.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        r = int(k)
        if r < 1:
            raise ValueError(f"rank {r} < 1: rank 0 is our own pick, not a deviation")
        w[r] = w.get(r, 0.0) + float(v)
    if not w:
        return None
    hi = max(w)
    out = [0.0] * (hi + 1)
    for r, v in w.items():
        out[r] = v
    tot = sum(out)
    if tot <= 0:
        raise ValueError(f"rank histogram sums to {tot}")
    return [v / tot for v in out]


# Every live Substitute, so `health_line` can print the rank histogram the way
# oracle/vlook print theirs. The arena builds both seats in ONE process, so this
# is a list rather than a singleton -- a singleton would silently report only
# whichever agent was constructed last.
LIVE: list["Substitute"] = []


def health_line() -> str:
    return " | ".join(s.line() for s in LIVE) if LIVE else ""


class Substitute:
    """Holds whichever arm's machinery is in use, plus the shared counters."""

    def __init__(self, xnet=None, rand_rate: float = 0.0,
                 rank_hist: list[float] | None = None, seed: int = 26,
                 rank_by_n: dict[int, list[float]] | None = None,
                 dump_path: str | None = None,
                 cal_ranks: list[float] | None = None,
                 accept: float = 1.0):
        LIVE.append(self)
        self.xnet = xnet
        self.rand_rate = float(rand_rate or 0.0)
        self.rank_hist = rank_hist
        # 🔴 THE MARGINAL HISTOGRAM IS NOT A MATCHED CONTROL, measured rather
        # than assumed: sampling ranks from the pooled histogram and clipping
        # them to the option count made the control's mean deviation rank 1.59
        # against the treatment's 1.92, with 17% of draws clipped. Clipping can
        # only make the control SHALLOWER, i.e. cheaper, i.e. it inflates the
        # coherence effect this experiment is trying to measure -- a bias in
        # the direction of the hypothesis, which is the kind this project has
        # been burned by (§8cf's own recorded fault, one level up).
        #
        # The fix is to condition on what makes clipping necessary: a decision
        # with 3 options cannot host a rank-5 deviation for EITHER arm, and the
        # treatment's histogram already reflects that mix. Matching per option
        # count removes the clipping entirely instead of counting it.
        self.rank_by_n = rank_by_n or {}
        self.cal_ranks = cal_ranks
        # 🔴 AND THE RATE IS NOT FLAT EITHER -- measured in the same calibration
        # pass, and it is the LARGER of the two mismatches. The expert deviates
        # on 9.4% of 2-option decisions and ~50% of 10-option ones. A control
        # deviating at the pooled 28.7% everywhere would therefore deviate in
        # the WRONG PLACES: too often at 2-option decisions (which §8bd measured
        # as nearly free) and too rarely at the wide ones. With `rank_by_n`
        # loaded, `rand_rate` stops being a probability and becomes a SCALE on
        # the treatment's own per-option-count rate, so 1.0 means "match".
        self.scaled = bool(self.rank_by_n)
        # E27 cell 0: P(accept a deviation the substitute proposes). 1.0 is E26.
        self.accept = float(accept)
        self._rng = random.Random(seed)
        # rank of the played option under OUR net's ordering; index 0 = our own
        # pick, so `ranks[0]` counts the firings where nothing changed.
        self.ranks = [0] * 24
        # rank counts split by how many options the decision offered -- the
        # object cell B is actually matched to.
        self.by_n: dict[int, list[int]] = {}
        self.rank_over = 0        # deviation deeper than the histogram is long
        self.clipped = 0          # sampled rank >= option count, clipped down
        self.dump_path = dump_path
        self._dump_due = 0

    # -- reporting -------------------------------------------------------
    def hist_spec(self) -> str:
        """The `xrank=` string cell B should be run with (deviations only)."""
        tot = sum(self.ranks[1:]) + self.rank_over
        if tot <= 0:
            return ""
        parts = [f"{r}:{self.ranks[r] / tot:.4f}"
                 for r in range(1, len(self.ranks)) if self.ranks[r]]
        return ";".join(parts)

    def line(self) -> str:
        tot = sum(self.ranks)
        dev = tot - self.ranks[0]
        mean_rank = (sum(r * c for r, c in enumerate(self.ranks)) / dev
                     if dev else 0.0)
        return (f"x_ranks={self.ranks[:8]} over={self.rank_over} "
                f"clipped={self.clipped} mean_dev_rank={mean_rank:.2f} "
                f"xrank={self.hist_spec()}")

    def dump(self) -> dict:
        """The matched-control calibration: rank counts per option count."""
        return {"by_n": {str(k): v for k, v in sorted(self.by_n.items())},
                "ranks": self.ranks, "over": self.rank_over}

    def _maybe_dump(self) -> None:
        if not self.dump_path:
            return
        self._dump_due += 1
        if self._dump_due % 1000:
            return
        # Written periodically rather than at exit: the arena has no per-agent
        # finish hook, and a calibration file that only appears on a clean
        # shutdown is one interrupted run away from not existing.
        try:
            import json
            from pathlib import Path as _P
            p = _P(self.dump_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.dump()), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # -- the two arms ----------------------------------------------------
    def apply(self, base_net, obs: dict, picked: list[int],
              want: int, stats: dict) -> list[int]:
        """Return the pick this arm plays. Never raises: on any failure the
        base pick stands, because an experiment must not forfeit a live
        episode (the fail-soft rule the shipped agent is built on)."""
        options = (obs.get("select") or {}).get("option") or []
        if want != 1 or len(options) < 2 or len(picked) != 1:
            stats["x_skip"] += 1
            return picked
        stats["x_fired"] += 1
        try:
            sc = np.asarray(base_net.scores(obs), dtype=np.float64)
            order = list(np.argsort(-sc))
            rank_of = {int(o): r for r, o in enumerate(order)}

            if self.xnet is not None:
                xp = self.xnet.choose(obs)
                if len(xp) != 1 or not 0 <= int(xp[0]) < len(options):
                    return picked
                new = int(xp[0])
                # E27 cell 0: accept the substitute's DEVIATIONS only a
                # fraction of the time, which walks the same policy's cost
                # curve down in rate without changing WHICH decisions or HOW
                # DEEP -- the two axes E26 had to fix. Agreements are
                # unaffected because accepting them changes nothing.
                if (self.accept < 1.0 and new != int(picked[0])
                        and self._rng.random() >= self.accept):
                    self._record(0, len(order))
                    self.ranks[0] += 1
                    return picked
            elif self.rand_rate > 0.0:
                if self._rng.random() >= self._dev_p(len(order)):
                    self._record(0, len(order))
                    self.ranks[0] += 1
                    return picked
                new = int(order[self._sample_rank(len(order))])
            else:
                return picked

            r = rank_of.get(new, 0)
            self._record(r, len(order))
            if r < len(self.ranks):
                self.ranks[r] += 1
            else:
                self.rank_over += 1
            if new != int(picked[0]):
                stats["x_diff"] += 1
            return [new]
        except Exception:  # noqa: BLE001
            stats["x_error"] += 1
            return picked

    def _dev_p(self, n_opts: int) -> float:
        """P(deviate) at a decision offering `n_opts` options.

        Without a calibration this is the flat `rand_rate` -- which is what E25
        did and what this class now refuses to call matched. With one, it is
        the treatment's OWN rate at this option count, scaled by `rand_rate`.
        """
        if not self.scaled:
            return self.rand_rate
        h = self.rank_by_n.get(int(n_opts)) or self.cal_ranks
        if not h:
            return self.rand_rate
        tot = sum(h)
        if tot <= 0:
            return self.rand_rate
        return self.rand_rate * (1.0 - h[0] / tot)

    def _record(self, rank: int, n_opts: int) -> None:
        row = self.by_n.setdefault(int(n_opts), [0] * 24)
        row[rank if rank < 24 else 23] += 1
        self._maybe_dump()

    def _sample_rank(self, n_opts: int) -> int:
        """A deviation rank >= 1, drawn from the treatment's histogram FOR THIS
        OPTION COUNT.

        Conditioning on `n_opts` is what makes the control matched: a 3-option
        decision cannot host a rank-5 deviation for either arm, so the
        treatment's own conditional distribution is the only sampler that
        cannot be systematically shallower. Clipping remains as a fallback
        (an option count the treatment never met) and is still COUNTED, because
        clipping can only bias the control cheap.
        """
        h = self.rank_by_n.get(int(n_opts)) or self.rank_hist
        if not h or sum(h[1:]) <= 0:
            return self._rng.randrange(1, n_opts)
        w = [0.0] + list(h[1:])
        r = self._rng.choices(range(len(w)), weights=w, k=1)[0]
        r = max(1, r)
        if r >= n_opts:
            self.clipped += 1
            r = n_opts - 1
        return r
