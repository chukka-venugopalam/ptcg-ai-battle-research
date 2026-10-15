"""R2: marginalise out the BENCH SLOT NUMBER, a nuisance variable the net reads.

**The measurement this exists for** (`scripts/p78_symmetry_probe.py`, EVIDENCE
§8bt). A bench slot number carries no game meaning -- moving a Pokemon from slot
1 to slot 3 changes nothing -- but it changes `opt["index"]`, which §8f encoded
deliberately (+115 Elo) because it was the only thing separating two options
naming two copies of one card. Measured on the SHIPPED net: **7.7% of
relabellings change the chosen option, and 16.9% of decisions are unstable under
at least one of six** (MAIN 24.8%).

So the net's decision is partly a function of an arbitrary labelling. This
averages that labelling out: score K relabellings of the same position and rank
by the mean. At K=1 the single relabelling is the IDENTITY, so `sym1` is
bitwise today's agent while still paying the copy -- the no-op control, exactly
like `flip0`.

⚠ **What this is NOT.** §8bd measured the near-tie band and found it
INDIFFERENT (flipping the k-th choice for the (k+1)-th reads 0.494 [0.467,
0.520]), and §8bt found the unstable decisions sit in exactly that band (median
margin 0.310 vs 1.298 overall). **The prior is against this paying.** It is a
variance reduction over a nuisance variable, which is a different operation from
deliberately taking a worse-ranked option -- but that difference is a hypothesis
until an arena A/B says otherwise.

⚠ **Probabilities, not raw logits** -- the same reasoning `Ensemble` records.
Each relabelling gets one vote; averaging raw logits would silently weight the
relabellings the net happens to be most confident about.
"""
from __future__ import annotations

import random

import numpy as np

_BENCH = 5


def _relabel(obs: dict, seat: int, perm: list[int]) -> dict | None:
    """`obs` with `seat`'s bench relabelled by `perm` (perm[new] = old).

    Copies only along the mutated path -- an observation carries hand, discard
    and log arrays, and a deepcopy per relabelling would cost more than the
    forward pass it feeds.

    ⚠ Every field that indexes the bench must move together. There are TWO:
    `opt["index"]` for options that name a bench slot, and `opt["inPlayIndex"]`
    for ATTACH/EVOLVE (types 8, 9), which `optfeat._target_pokemon` and
    `optfeat`'s `slot_ix` both read. Missing the second does not relabel the
    position, it corrupts the option -- that bug inflated §8bt's first reading
    and was caught by an option-identity control, not by inspection.
    """
    st = obs.get("current") or {}
    try:
        pl = st["players"][seat]
        bench = pl["bench"]
    except (KeyError, IndexError, TypeError):
        return None
    n = len(bench)
    if n < 2 or len(perm) != n:
        return None

    new_pl = dict(pl)
    new_pl["bench"] = [bench[perm[i]] for i in range(n)]
    new_players = list(st["players"])
    new_players[seat] = new_pl
    new_st = dict(st)
    new_st["players"] = new_players

    inv = [0] * n
    for new, old in enumerate(perm):
        inv[old] = new

    me = st.get("yourIndex")
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    new_opts = []
    for o in opts:
        pi = o.get("playerIndex")
        pi = me if pi is None else pi
        changed = None
        if o.get("area") == _BENCH and pi == seat:
            k = o.get("index") or 0
            if not 0 <= k < n:
                return None
            changed = dict(o)
            changed["index"] = inv[k]
        if o.get("inPlayArea") == _BENCH and seat == me:
            k = o.get("inPlayIndex") or 0
            if not 0 <= k < n:
                return None
            changed = dict(changed or o)
            changed["inPlayIndex"] = inv[k]
        new_opts.append(changed if changed is not None else o)

    new_sel = dict(sel)
    new_sel["option"] = new_opts
    new_obs = dict(obs)
    new_obs["current"] = new_st
    new_obs["select"] = new_sel
    return new_obs


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max()
    e = np.exp(np.clip(z, -60.0, 0.0))
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def sym_scores(net, obs: dict, k: int,
               rng: random.Random | None = None) -> np.ndarray | None:
    """Mean option probability over `k` bench relabellings, or None to fall back.

    The FIRST relabelling is always the identity, so `k=1` reproduces the
    unmodified net exactly and every larger k contains it.
    """
    if k <= 0:
        return None
    st = obs.get("current") or {}
    seat = st.get("yourIndex")
    if seat not in (0, 1):
        return None
    try:
        n = len(st["players"][seat]["bench"])
    except (KeyError, IndexError, TypeError):
        return None

    base = net.scores(obs)
    if base is None:
        return None
    acc = _softmax(base)
    if k == 1 or n < 2:
        return acc

    rng = rng or random.Random(17)
    used = 1
    for _ in range(k - 1):
        perm = list(range(n))
        rng.shuffle(perm)
        o2 = _relabel(obs, seat, perm)
        if o2 is None:
            continue
        try:
            s2 = net.scores(o2)
        except Exception:  # noqa: BLE001
            continue
        if s2 is None or len(s2) != len(base):
            continue
        acc = acc + _softmax(s2)
        used += 1
    return acc / used
