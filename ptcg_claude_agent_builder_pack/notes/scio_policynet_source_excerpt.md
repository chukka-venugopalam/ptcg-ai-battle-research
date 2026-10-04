```


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/agents/sa/policynet.py
====================================================================================================
"""Numpy inference for the cloned policy (see scripts/train_policy.py)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .features import attr_feats, extra_feats, featurize
from .optfeat import option_features, pool_scalars, pool_width
from .routing import ROUTE_GENERAL, route_from_obs

# SA_PNET_PATH lets an arena run score a candidate net without overwriting the
# shipped one. Kaggle sets no env vars, so there it is always the bundled npz.
_PATH = Path(os.environ.get("SA_PNET_PATH")
             or Path(__file__).resolve().parent / "policy_net.npz")
# how many options to take on a variable-count select; see Net.choose
COUNT_MODE = os.environ.get("SA_COUNT_MODE", "table")
_BAGS = ("my_hand", "my_discard", "opp_discard")
SEL_DENSE = 14

_net = None
_tried = False


def _sel_features(sel: dict) -> np.ndarray:
    v = np.zeros(SEL_DENSE, dtype=np.float32)
    t = sel.get("type") or 0
    if t < 11:
        v[t] = 1.0
    v[11] = sel.get("minCount", 0) / 5.0
    v[12] = sel.get("maxCount", 0) / 5.0
    v[13] = (sel.get("context") or 0) / 50.0
    return v


class Net:
    def __init__(self, z):
        self.slot_emb = z["slot_emb"]
        self.bag_emb = z["bag_emb"]
        self.card_emb = z["card_emb"]
        self.atk_emb = z["atk_emb"]
        # Layers are stored generically (`sfc{i}_w` / `head{i}_w`) so the net
        # can be made deeper without touching this file. Nets exported before
        # that change used fixed ws/w1/w2 names -- still loadable.
        if "n_sfc" in z:
            self.state_layers = [(z[f"sfc{i}_w"], z[f"sfc{i}_b"])
                                 for i in range(int(z["n_sfc"][0]))]
            self.head_layers = [(z[f"head{i}_w"], z[f"head{i}_b"])
                                for i in range(int(z["n_head"][0]))]
        else:
            self.state_layers = [(z["ws"], z["bs"])]
            self.head_layers = [(z["w1"], z["b1"]), (z["w2"], z["b2"])]
        self.count_frac = z["count_frac"] if "count_frac" in z else None
        # E1 heads are optional and append-only. Legacy checkpoints keep the
        # exact policy path; multitask checkpoints expose these predictions for
        # diagnostics, learned count selection, and later planning.
        self.outcome_head = ((z["outcome_w"], z["outcome_b"])
                             if "outcome_w" in z else None)
        self.count_head = ((z["count_w"], z["count_b"])
                           if "count_w" in z else None)
        # The v5 pooled option-set block, 0 for every net before day 13. Recorded
        # rather than derived: the v4 and v5 state widths are both legal, so
        # `state_in` alone cannot tell them apart.
        self.n_pool = int(z["n_pool"][0]) if "n_pool" in z else 0
        # Which members of the v4 block this net was shown (features.X_GROUPS).
        # Absent = all of them, which is every net before day 13.
        self.x_mask = z["x_mask"] if "x_mask" in z else None
        # E2 residual adapters. Absent keys keep the exact legacy policy path.
        self.adapters: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.adapter_route_ids: dict[str, int] = {}
        if "adapter_names" in z:
            names = [str(x) for x in z["adapter_names"].tolist()]
            route_ids = (z["adapter_route_ids"].tolist()
                         if "adapter_route_ids" in z else [])
            for i, name in enumerate(names):
                n_layers = int(z[f"adapter_{name}_n"][0])
                layers = [(z[f"adapter_{name}{j}_w"],
                           z[f"adapter_{name}{j}_b"])
                          for j in range(n_layers)]
                self.adapters[name] = layers
                if i < len(route_ids):
                    self.adapter_route_ids[name] = int(route_ids[i])
                else:
                    from .routing import NAME_TO_ROUTE
                    self.adapter_route_ids[name] = NAME_TO_ROUTE[name]
        # --- and the main-side optional blocks (v6 attr, v7 vocab) ---
        # The v6 card-attribute block, 0 for every net before day 20. Recorded
        # for the same reason as n_pool: with three optional blocks, `state_in`
        # no longer identifies the layout on its own.
        self.n_attr = int(z["n_attr"][0]) if "n_attr" in z else 0
        # Which members of the v6 block this net was shown (features.A_GROUPS).
        self.a_mask = z["a_mask"] if "a_mask" in z else None
        # The v7 vocabulary remap, absent on every net before day 21. Each table
        # was collapsed to the rows the corpus trained: row 0 = PAD, row 1 = UNK,
        # rows 2.. = `vocab_<table>` in order. Without it a card the corpus never
        # contained reads an untrained N(0,1) row whose norm is indistinguishable
        # from a trained one's, so the net cannot tell "unknown" from "known".
        self.lut = None
        if "vocab_slot_emb" in z:
            from .features import N_CARD_IDS
            from .optfeat import N_ATTACK_IDS
            self.lut = {}
            for t in ("slot_emb", "bag_emb", "card_emb", "atk_emb"):
                ids = z[f"vocab_{t}"].astype(np.int64)
                size = N_ATTACK_IDS if t == "atk_emb" else N_CARD_IDS
                lut = np.full(max(size, int(ids[-1]) + 1 if ids.size else size),
                              1, dtype=np.int64)      # 1 = UNK
                lut[0] = 0                            # 0 = PAD
                lut[ids] = np.arange(2, 2 + ids.size, dtype=np.int64)
                self.lut[t] = lut

    def _m(self, table: str, ids):
        """Raw card/attack id -> this net's row. Identity on a pre-v7 net."""
        if self.lut is None:
            return ids
        lut = self.lut[table]
        a = np.asarray(ids)
        # Out-of-range cannot happen -- features.py clamps to 0 -- but an id past
        # the table falls to UNK rather than raising in the middle of a match.
        return np.where(a < len(lut), lut[np.clip(a, 0, len(lut) - 1)], 1)

    @property
    def state_in(self) -> int:
        return self.state_layers[0][0].shape[1]

    @property
    def state_out(self) -> int:
        return self.state_layers[-1][0].shape[0]

    @property
    def head_in(self) -> int:
        return self.head_layers[0][0].shape[1]

    @property
    def opt_in(self) -> int:
        """How many per-option dense features THIS net was trained on.

        Derived rather than read from `optfeat.OPT_DENSE`, because a v2 net
        (25) and a v3 net (37) have to be able to run in the same process for a
        head-to-head A/B across the feature change (HANDOFF rule 4). The v3 block
        is appended, so slicing to this width gives a v2 net byte-identical input
        to what it was trained on."""
        return (self.head_in - self.state_out
                - 2 * self.card_emb.shape[1] - self.atk_emb.shape[1])

    def _forward(self, obs: dict) -> tuple[np.ndarray, np.ndarray | None]:
        """Return option logits and the shared state representation."""
        state = obs["current"]
        sel = obs["select"]
        me = state["yourIndex"]
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return np.zeros(0, dtype=np.float32), None
        # The per-option encoding is built BEFORE the state, because the v5 pool
        # is a summary of it. Nets without the pool ignore it and slice it off,
        # so this costs them nothing but the loop order.
        emb = self.card_emb.shape[1]
        ow = self.opt_in
        oenc = np.empty((n, ow + 3 * emb), dtype=np.float32)
        for i, o in enumerate(opts):
            od, cid, aid, tid = option_features(obs, o)
            # Slice to the width this net was trained at -- the v3 target block
            # is appended, so a v2 net simply does not see it.
            oenc[i, :ow] = od[:ow]
            oenc[i, ow:ow + emb] = self.card_emb[self._m("card_emb", cid)]
            oenc[i, ow + emb:ow + 2 * emb] = self.atk_emb[
                self._m("atk_emb", aid)]
            oenc[i, ow + 2 * emb:] = self.card_emb[self._m("card_emb", tid)]

        dense, bags = featurize(state, me)
        parts = [dense,
                 self.slot_emb[self._m("slot_emb", bags["slots"])].reshape(-1)]
        for name in _BAGS:
            b = bags[name]
            parts.append(self.bag_emb[self._m("bag_emb", b)].mean(axis=0)
                         if len(b)
                         else np.zeros(self.bag_emb.shape[1],
                                       dtype=np.float32))
        parts.append(_sel_features(sel))
        # The v4 block goes LAST, so slicing to this net's own `state_in` feeds
        # a v3 net byte-identical input (features.py, "APPENDED, NEVER
        # INSERTED"). Same trick as `opt_in` one level up.
        xd, xids = extra_feats(state, sel, me)
        if self.x_mask is not None:     # a drop-one ablation arm (day 13)
            from .features import N_EXTRA
            xd = xd * self.x_mask[:N_EXTRA]
            xids = np.where(self.x_mask[N_EXTRA:] > 0, xids, 0)
        parts.append(xd)
        parts.append(self.slot_emb[self._m("slot_emb", xids)].reshape(-1))
        # ...and the v5 pool goes after v4, same rule (optfeat.pool_width).
        if self.n_pool:
            parts += [oenc.mean(axis=0), oenc.max(axis=0), pool_scalars(n)]
        # ...and the v6 attribute block goes after v5, same rule again. Computed
        # only when the net was trained with it -- attr_feats walks 12 slots and
        # a v5 net would pay for a vector it then slices off.
        if self.n_attr:
            a = attr_feats(state, me)
            parts.append(a * self.a_mask if self.a_mask is not None else a)
        x = np.concatenate(parts)
        srepr = x[:self.state_in]
        for w, b in self.state_layers:      # every state layer is relu'd
            srepr = np.maximum(w @ srepr + b, 0.0)

        sw = len(srepr)
        feats = np.empty((n, self.head_in), dtype=np.float32)
        feats[:, :sw] = srepr
        feats[:, sw:] = oenc
        h = feats
        for j, (w, b) in enumerate(self.head_layers):
            h = h @ w.T + b
            if j < len(self.head_layers) - 1:   # last layer is the raw logit
                h = np.maximum(h, 0.0)
        logits = h.reshape(-1)
        if self.adapters:
            route = route_from_obs(obs)
            if route != ROUTE_GENERAL:
                for name, route_id in self.adapter_route_ids.items():
                    if route_id != route:
                        continue
                    residual = feats
                    layers = self.adapters[name]
                    for j, (w, b) in enumerate(layers):
                        residual = residual @ w.T + b
                        if j < len(layers) - 1:
                            residual = np.maximum(residual, 0.0)
                    logits = logits + residual.reshape(-1)
                    break
        return logits, srepr

    def scores(self, obs: dict) -> np.ndarray:
        """Logit per option of obs['select']."""
        return self._forward(obs)[0]

    @staticmethod
    def _head_value(head, srepr: np.ndarray) -> float | None:
        if head is None:
            return None
        w, b = head
        return float((w @ srepr + b).reshape(-1)[0])

    def win_prob(self, obs: dict) -> float | None:
        """Auxiliary E1 outcome estimate, or None for legacy checkpoints."""
        _, srepr = self._forward(obs)
        if srepr is None:
            return None
        logit = self._head_value(self.outcome_head, srepr)
        if logit is None:
            return None
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def choose(self, obs: dict) -> list[int]:
        """Rank options by logit; how MANY to take is the harder half.

        `table` (default): a data-derived per-(selectType, context) mean count
        fraction -- one number for the whole bucket, so it is wrong on every
        select whose true count is bimodal.
        `expect`: sum the per-option sigmoids, i.e. the model's own expected
        number of chosen options. Only meaningful for a net trained with a
        pointwise (BCE) term -- a pure listwise net's logits are not calibrated
        probabilities, only a valid ranking.
        """
        sc, srepr = self._forward(obs)
        return self.pick(obs, sc, srepr)

    def pick(self, obs: dict, sc: np.ndarray,
             srepr: np.ndarray | None) -> list[int]:
        """Rank by `sc` and decide HOW MANY to take.

        Split out of `choose` so an ensemble can supply its own combined
        scores without re-implementing the count rule -- rule 18: do not
        re-derive a statistic the tool already computes. `choose` is exactly
        `pick` applied to this net's own forward pass.
        """
        sel = obs["select"]
        mn = sel.get("minCount", 0)
        mx = sel.get("maxCount", 0)
        order = list(np.argsort(-sc))
        k = mx
        if mx > mn:
            if COUNT_MODE == "expect":
                probs = 1.0 / (1.0 + np.exp(-np.clip(sc, -30.0, 30.0)))
                k = int(round(float(probs.sum())))
            elif COUNT_MODE == "learned" and self.count_head is not None:
                logit = self._head_value(self.count_head, srepr)
                frac = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                k = mn + int(round(float(frac) * (mx - mn)))
            else:
                frac = 1.0
                if self.count_frac is not None:
                    t = min(sel.get("type") or 0, 10)
                    ctx = min(sel.get("context") or 0, 63)
                    frac = float(self.count_frac[t, ctx])
                k = mn + int(round(frac * (mx - mn)))
            k = max(mn, min(k, mx))
        return [int(i) for i in order[:k]]


class Ensemble:
    """Several independently-trained nets voting on one decision.

    🔴 **Why this is not the closed capacity axis.** §8w made ONE net bigger
    (2.6x and 8.2x the parameters) and bought two decisions out of 12,939 and
    then lost 43 -- the features, not the parameter count, were binding. An
    ensemble does something else: it averages functions that were fitted
    *independently*, which cancels the part of each net's error that is
    idiosyncratic to its own initialisation rather than shared.

    ⚡ **And this project has already measured that the idiosyncratic part is
    large.** §5.6/E8 found two same-recipe nets differing only in `--seed`
    swinging **0.073** against each other in a direct mirror head-to-head,
    against ±0.036 of sampling noise -- they disagree far more than the games
    alone explain, i.e. they make DIFFERENT mistakes. That measurement was
    filed as a warning about our instrument; it is also the precondition for
    averaging to pay.

    ⚠ **Probabilities, not raw logits.** A listwise loss fixes the ranking, not
    the scale: two nets can be equally good and differ by a constant factor in
    logit magnitude, and a raw-logit mean would then be a weighted vote with
    weights nobody chose. Softmax each net over the option set first, then
    average, so every member gets exactly one vote. `--raw` overrides.

    ⚠ **The count comes from the FIRST member**, via its own `pick`. Ensembling
    the count fraction as well would confound "which options" with "how many",
    and the count rule is a per-(type, context) table, not a scored quantity.
    """

    def __init__(self, nets: list["Net"], raw: bool = False):
        if not nets:
            raise ValueError("Ensemble needs at least one net")
        self.nets = nets
        self.raw = raw
        # exposed so callers that introspect a Net (x_mask checks, vocab
        # guards, the flip probe) see the primary member's shape
        self.primary = nets[0]

    def __len__(self) -> int:
        return len(self.nets)

    # Passthroughs so anything that introspects a net (the build smoke, the
    # dim guard's callers) sees the shape it is actually being fed. Every
    # member is verified same-architecture by `load` before it gets here.
    @property
    def opt_in(self) -> int:
        return self.primary.opt_in

    @property
    def state_in(self) -> int:
        return self.primary.state_in

    def scores(self, obs: dict) -> np.ndarray:
        acc = None
        for net in self.nets:
            s = np.asarray(net.scores(obs), dtype=np.float64)
            if not self.raw:
                z = s - float(s.max())
                e = np.exp(z)
                s = e / e.sum()
            acc = s if acc is None else acc + s
        return acc / float(len(self.nets))

    def choose(self, obs: dict) -> list[int]:
        sel = obs.get("select") or {}
        n = len(sel.get("option") or [])
        if n == 0:
            return []
        # srepr is only consulted by the `learned` count mode, which the
        # shipped nets do not use; the primary's own forward supplies it when
        # it is needed rather than being faked.
        srepr = None
        if COUNT_MODE == "learned" and self.primary.count_head is not None:
            srepr = self.primary._forward(obs)[1]
        return self.primary.pick(obs, self.scores(obs), srepr)


def load_ensemble(paths: list[str], raw: bool = False) -> Ensemble | None:
    """Load several nets for voting. Returns None if ANY member fails.

    Strict on purpose: a silently-dropped member is a different agent playing
    under the ensemble's name, which is day 22's defect 2 with extra steps.
    """
    nets = []
    for p in paths:
        net = load(p)
        if net is None:
            return None
        nets.append(net)
    return Ensemble(nets, raw=raw)


def load(path) -> Net | None:
    """Load a specific npz, returning None unless it matches the CURRENT
    feature dims. The guard is what stops a stale net from being used
    silently after a feature change -- never remove it."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        net = Net(np.load(path))
        from .features import DENSE_DIM, N_ATTR, N_EXTRA, N_XSLOT
        from .optfeat import KNOWN_OPT_DENSE
        emb = net.slot_emb.shape[1]
        base = (DENSE_DIM + 12 * emb + 3 * net.bag_emb.shape[1] + SEL_DENSE)
        # Four legitimate state widths now, exactly as with the option block:
        # the v3 layout, + the appended v4 block, + the appended v5 pool, + the
        # appended v6 attribute block. `n_pool` and `n_attr` say which one a net
        # is, and they must AGREE with the width -- a net claiming a block it was
        # not trained with would silently read hundreds of columns of garbage.
        v4 = base + N_EXTRA + N_XSLOT * emb
        v5 = v4 + pool_width(net.opt_in, emb)
        want = v5 if net.n_pool else v4
        if net.n_attr:
            want += N_ATTR
        # A v7 net's tables are sized BY the map that travels with them. If the
        # two disagree, every lookup is off by however far they drifted and the
        # agent plays a scrambled net at full confidence -- the exact failure
        # E6 measured at -0.251. Check them against each other, not against a
        # constant: the row count IS 2 + len(vocab), PAD and UNK.
        if net.lut is not None:
            for t, w in (("slot_emb", net.slot_emb), ("bag_emb", net.bag_emb),
                         ("card_emb", net.card_emb), ("atk_emb", net.atk_emb)):
                if w.shape[0] != int((net.lut[t] > 1).sum()) + 2:
                    return None
        if (net.state_in in (base, want) and net.opt_in in KNOWN_OPT_DENSE
                and net.n_pool in (0, pool_width(net.opt_in, emb))
                and net.n_attr in (0, N_ATTR)
                and (net.a_mask is None or net.a_mask.shape == (N_ATTR,))):
            return net
    except Exception:
        pass
    return None


def get() -> Net | None:
    """The process-wide singleton, loaded from `SA_PNET_PATH` or the bundle.

    🔴 **Announces WHICH net it just loaded, once, on stderr.** E33 rolled out
    with this singleton (`policy_net.npz`, the v2 clone) while its seats played
    `out/policy_v5_s2.npz`, published a calibration verdict, and had to
    withdraw it; `p82` had already warned in writing that scoring one net's
    options against another net's games *"returns a plausible number, not an
    error"*. Which net a probe actually used was recoverable only by reading
    the source and knowing the environment, so nothing in a log could ever
    contradict a wrong assumption. Now every run says so itself.

    ⚠ **stderr, not stdout, and that is load-bearing**: `kaggle/score.py` and
    the p5x drivers parse stdout for the arena's score line and drop the rest,
    so a notice printed there is a notice nobody reads -- the same reasoning
    `arena.build_agent` gives for its deck-mismatch warning.
    """
    global _net, _tried
    if not _tried:
        _tried = True
        _net = load(_PATH)
        try:
            tag = "MISSING"
            if _net is not None:
                import hashlib
                tag = "#" + hashlib.md5(
                    Path(_PATH).read_bytes()).hexdigest()[:8]
            # ⚠ Build this OUTSIDE the f-string. Kaggle's episode runner is
            # Python 3.11 (`kaggle_environments` under python3.11/dist-packages)
            # even though Kaggle *notebooks* are 3.12, and a replacement field
            # that spans lines is PEP 701 -- i.e. 3.12+ ONLY. As a multi-line
            # f-string this raised `SyntaxError: unterminated string literal` at
            # IMPORT, so `main.py`'s `from sa.bcagent import PolicyAgent` never
            # completed, both seats died in 0.04s and submission 55489084 came
            # back "Validation Episode failed." A logging nicety took the whole
            # agent down, on the one path no local smoke could see.
            note = ("" if os.environ.get("SA_PNET_PATH")
                    else "  (repo default -- set SA_PNET_PATH to pin a "
                         "different one)")
            print(f"[policynet] singleton = {_PATH} {tag}{note}",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
    return _net


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/agents/sa/valuenet.py
====================================================================================================
"""Numpy-only inference for the trained value net (see scripts/train_value.py).

If agents/sa/value_net.npz is absent, `get()` returns None and callers fall
back to the handcrafted eval.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .features import featurize

# `SA_VNET_PATH` mirrors `policynet.SA_PNET_PATH`. It exists so an A/B can pin
# WHICH value net played (rule 20: the identity a result is filed under must
# contain everything that can change the result) instead of silently using
# whatever `value_net.npz` happens to be on disk -- the exact defect rule 19
# describes, and `_net_fp` in arena.py is what records the bytes.
_PATH = Path(os.environ.get("SA_VNET_PATH")
             or Path(__file__).resolve().parent / "value_net.npz")
_BAGS = ("my_hand", "my_discard", "opp_discard")

_net = None
_tried = False


def load(path: str | Path) -> "Net | None":
    """Load a specific value net, bypassing the module-level singleton.

    Two `ValueLookahead` instances in one process (a head-to-head A/B, rule 4)
    must be able to hold different value nets, which a singleton cannot do.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        net = Net(np.load(p))
    except Exception:
        return None
    from .features import DENSE_DIM
    expect = (DENSE_DIM + 12 * net.slot_emb.shape[1]
              + 3 * net.bag_emb.shape[1])
    return net if net.w1.shape[1] == expect else None


class Net:
    def __init__(self, z):
        self.slot_emb = z["slot_emb"]
        self.bag_emb = z["bag_emb"]
        self.w1 = z["w1"]
        self.b1 = z["b1"]
        self.w2 = z["w2"]
        self.b2 = z["b2"]
        self.w3 = z["w3"]
        self.b3 = z["b3"]

    def win_prob(self, state: dict, me: int) -> float:
        dense, bags = featurize(state, me)
        return self.forward(dense, bags)

    def forward(self, dense, bags) -> float:
        """The scoring path, split out so it can be tested against the trainer
        on raw corpus rows (`p88_value_equivalence.py`). `win_prob` is then
        only `featurize` + this, and the equivalence test exercises the code
        that actually plays rather than a reimplementation of it."""
        parts = [dense, self.slot_emb[bags["slots"]].reshape(-1)]
        for name in _BAGS:
            b = bags[name]
            # 🔴 EMPTY BAG -> row 0, NOT zeros. `train_value.py` pads an empty
            # bag with row 0 (EmbeddingBag mode="mean" returns NaN on a truly
            # empty bag), so the weights were fitted against `bag_emb[0]`.
            # Substituting zeros here computes a DIFFERENT FUNCTION from the
            # one that was trained: p88 measured max |diff| 0.126 on the 7.0%
            # of rows with an empty bag, against a within-position sibling
            # range of 0.186 -- i.e. comparable to the whole signal an argmax
            # depends on, and structured, because hands empty exactly when we
            # have played them out. E20 spent 2,000 games before this was
            # checked. Rule 18: compute it a second way and reconcile FIRST.
            parts.append(self.bag_emb[b].mean(axis=0) if len(b)
                         else self.bag_emb[0])
        x = np.concatenate(parts)
        h = np.maximum(self.w1 @ x + self.b1, 0.0)
        h = np.maximum(self.w2 @ h + self.b2, 0.0)
        logit = float(self.w3 @ h + self.b3)
        return 1.0 / (1.0 + np.exp(-logit))


def get() -> Net | None:
    global _net, _tried
    if not _tried:
        _tried = True
        if _PATH.exists():
            try:
                net = Net(np.load(_PATH))
                from .features import DENSE_DIM
                expect = (DENSE_DIM + 12 * net.slot_emb.shape[1]
                          + 3 * net.bag_emb.shape[1])
                if net.w1.shape[1] == expect:
                    _net = net
            except Exception:
                _net = None
    return _net


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/agents/sa/bcagent.py
====================================================================================================
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


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/train_policy.py
====================================================================================================
"""Train the policy net (behavioral cloning of top players' selects).

    python scripts/train_policy.py --ds artifacts/pds --out agents/sa/policy_net.npz

The state is encoded once per row, then scored against each option.

    state:  dense + slot_emb(12x16) + 3 bag means(16) + seld(14)
            -> MLP(--state-h) -> state_repr
    option: opt_dense + card_emb(16) + atk_emb(16) + tgt_emb(16)
    score:  MLP([state_repr, option], --head-h) -> 1

`--loss listwise` optimizes softmax cross-entropy over each select's option
set, which is what the agent actually does at inference (rank the options and
take the top k). `--loss bce` is the original pointwise objective; it treats
every option independently and does not model "which of these is best".

Layer sizes are exported generically (`sfc{i}_w` / `head{i}_w` + counts), so
sa/policynet.py mirrors any depth without a code change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import (A_GROUPS, DENSE_DIM, N_ATTR,  # noqa: E402
                         N_CARD_IDS, N_EXTRA, N_XSLOT, X_GROUPS)
from sa.optfeat import (OPT_DENSE, OPT_DENSE_V2, N_ATTACK_IDS,  # noqa: E402
                        pool_width)
from sa.routing import (NAME_TO_ROUTE, ROUTE_NAMES,  # noqa: E402
                        routes_from_corpus)

EMB = 16
SEL_DENSE = 14
BAGS = ("my_hand", "my_discard", "opp_discard")
# The four embedding tables and the id space each is indexed by. Used only by
# --vocab; see build_remap.
EMB_TABLES = ("slot_emb", "bag_emb", "card_emb", "atk_emb")
PAD_IX, UNK_IX = 0, 1
# The band the LB's top ~40 teams sit in; `val_top1@1120+` says how well the
# net fits STRONG demonstrators as opposed to the mixture (ROADMAP B7).
VAL_HI_RATING = 1120.0


def build_remap(vocab_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-table id -> row map with a PAD row and a shared UNK row.

    The shipped tables are allocated over the RAW id space (1300 card ids, 1600
    attack ids) but the corpus only ever touches 104/134/135/57 of those rows.
    The other ~90% are exported at their random init -- harmless while they are
    never read, and NOT harmless at inference, where an out-of-vocabulary
    opponent card lands on a random unit-normal vector whose norm (3.91-3.95) is
    indistinguishable from a trained row's (3.97-4.07). The net cannot tell "a
    card I have never seen" from "a card I know", so it reads confident garbage.

    Remapping collapses each table to exactly the rows that got a gradient, and
    routes everything else to ONE row that is trained, by construction, to mean
    "unknown card". Row 0 is PAD (empty slot / no stadium / no effect); with
    `padding_idx=0` it is pinned at zero and takes no gradient, which is where
    the v5 net was heading on its own -- it drove |slot_emb[0]| to 2.337 against
    a 3.958 table mean, the 11th smallest of 1,300 rows, on 25.5% of lookups.

    ⚠ Per-table, not shared: a card seen in hand but never on an opponent's
    board is trained in `bag_emb` and untrained in `slot_emb`. One shared vocab
    would re-introduce the exact defect this removes, just for fewer rows.
    """
    from sa.features import N_CARD_IDS
    from sa.optfeat import N_ATTACK_IDS
    tabs = json.loads(vocab_path.read_text(encoding="utf-8"))["tables"]
    sizes = {"atk_emb": N_ATTACK_IDS}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in EMB_TABLES:
        if t not in tabs:
            raise SystemExit(f"{vocab_path} has no census for {t}")
        ids = np.array(sorted(int(k) for k in tabs[t] if int(k) != 0),
                       dtype=np.int64)
        size = sizes.get(t, N_CARD_IDS)
        if ids.size and int(ids[-1]) >= size:
            raise SystemExit(f"{t}: census id {int(ids[-1])} >= {size}; the "
                             "vocab was built against a different id space")
        lut = np.full(size, UNK_IX, dtype=np.int64)
        lut[PAD_IX] = PAD_IX
        lut[ids] = np.arange(2, 2 + ids.size, dtype=np.int64)
        out[t] = (ids, lut)
    return out


def apply_remap(data: "Data", remap: dict[str, tuple[np.ndarray, np.ndarray]]
                ) -> None:
    """Rewrite every id column in place. Ids at or past a table's raw size
    cannot appear -- features.py already clamps them to 0 -- but clip anyway so
    a corpus built by an older builder fails to UNK rather than IndexError."""
    def m(t: str, a: np.ndarray) -> np.ndarray:
        lut = remap[t][1]
        return lut[np.clip(a, 0, len(lut) - 1)]
    data.slots = m("slot_emb", data.slots)
    data.xslots = m("slot_emb", data.xslots)
    for nm in BAGS:
        data.bag_flat[nm] = m("bag_emb", data.bag_flat[nm])
    data.opt_card = m("card_emb", data.opt_card)
    data.opt_tgt = m("card_emb", data.opt_tgt)
    data.opt_atk = m("atk_emb", data.opt_atk)
    for t in EMB_TABLES:
        ids = remap[t][0]
        print(f"  {t:9s} {len(remap[t][1]):5d} raw ids -> {ids.size + 2:4d} "
              f"rows (PAD + UNK + {ids.size} seen)")


def load_init(model: PolicyNet, path: Path) -> None:
    """Warm-start from an exported .npz (the fine-tuning arm of B7). Refuses on
    any shape mismatch rather than partially loading -- a silently half-loaded
    net trains fine and measures like a fresh one."""
    z = np.load(path)
    with torch.no_grad():
        for name, emb in (("slot_emb", model.slot_emb), ("bag_emb",
                          model.bag_emb), ("card_emb", model.card_emb),
                         ("atk_emb", model.atk_emb)):
            w = z[name]
            if w.shape != tuple(emb.weight.shape):
                raise SystemExit(f"--init {path.name}: {name} is {w.shape}, "
                                 f"model wants {tuple(emb.weight.shape)}")
            emb.weight.copy_(torch.from_numpy(w))
        for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
            lins = [m for m in seq if isinstance(m, nn.Linear)]
            n = int(z[f"n_{prefix}"][0])
            if n != len(lins):
                raise SystemExit(f"--init {path.name}: {n} {prefix} layers, "
                                 f"model has {len(lins)}")
            for i, lin in enumerate(lins):
                w, b = z[f"{prefix}{i}_w"], z[f"{prefix}{i}_b"]
                if w.shape != tuple(lin.weight.shape):
                    raise SystemExit(
                        f"--init {path.name}: {prefix}{i}_w is {w.shape}, "
                        f"model wants {tuple(lin.weight.shape)}")
                lin.weight.copy_(torch.from_numpy(w))
                lin.bias.copy_(torch.from_numpy(b))
        # E1 auxiliary heads are append-only. A plain v5 checkpoint has no
        # auxiliary tensors, so warm-starting it deliberately leaves these
        # heads at their seeded initialization while loading the policy
        # byte-for-byte. A later multitask checkpoint restores them as well.
        for prefix, head in (("outcome", model.outcome_head),
                             ("count", model.count_head)):
            if head is None or f"{prefix}_w" not in z:
                continue
            w, b = z[f"{prefix}_w"], z[f"{prefix}_b"]
            if w.shape != tuple(head.weight.shape):
                raise SystemExit(f"--init {path.name}: {prefix}_w is {w.shape}, "
                                 f"model wants {tuple(head.weight.shape)}")
            head.weight.copy_(torch.from_numpy(w))
            head.bias.copy_(torch.from_numpy(b))
        # E2 adapters are append-only. A plain v5 checkpoint has none, so
        # warm-starting leaves the zero-initialized residuals in place.
        if model.adapters is not None and "adapter_names" in z:
            names = [str(x) for x in z["adapter_names"].tolist()]
            for name in names:
                if name not in model.adapters:
                    raise SystemExit(
                        f"--init {path.name}: unknown adapter {name!r}")
                seq = model.adapters[name]
                lins = [m for m in seq if isinstance(m, nn.Linear)]
                n = int(z[f"adapter_{name}_n"][0])
                if n != len(lins):
                    raise SystemExit(
                        f"--init {path.name}: adapter {name} has {n} layers, "
                        f"model has {len(lins)}")
                for i, lin in enumerate(lins):
                    w, b = (z[f"adapter_{name}{i}_w"],
                            z[f"adapter_{name}{i}_b"])
                    if w.shape != tuple(lin.weight.shape):
                        raise SystemExit(
                            f"--init {path.name}: adapter_{name}{i}_w is "
                            f"{w.shape}, model wants "
                            f"{tuple(lin.weight.shape)}")
                    lin.weight.copy_(torch.from_numpy(w))
                    lin.bias.copy_(torch.from_numpy(b))
    print(f"warm-started from {path}")


def _mlp(sizes: list[int], dropout: float, out_dim: int | None) -> nn.Sequential:
    """ReLU MLP over `sizes` hidden widths; `out_dim` appends a linear head."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
    if out_dim is not None:
        layers.append(nn.Linear(sizes[-1], out_dim))
    return nn.Sequential(*layers)


def _make_adapter(in_dim: int, hidden: int) -> nn.Sequential:
    """Residual logit MLP; final layer is zero-initialized for v5 equivalence."""
    seq = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    nn.init.zeros_(seq[-1].weight)
    nn.init.zeros_(seq[-1].bias)
    return seq


class PolicyNet(nn.Module):
    def __init__(self, state_h: tuple[int, ...] = (256,),
                 head_h: tuple[int, ...] = (128,), dropout: float = 0.1,
                 opt_cols: int = OPT_DENSE, extra: bool = True,
                 pool: bool = False, outcome: bool = False,
                 count: bool = False, adapter_names: list[str] | None = None,
                 adapter_h: int = 64, adapters_off: bool = False,
                 attr: bool = False,
                 rows: dict[str, int] | None = None, pad: bool = False):
        super().__init__()
        self.opt_cols = opt_cols
        self.extra = extra
        self.pool = pool
        self.adapters_off = adapters_off
        self.adapter_h = adapter_h
        self.attr = attr
        # `rows` shrinks each table to its own vocabulary (--vocab); absent, the
        # tables span the raw id space exactly as v3-v6 did. Only the ROW count
        # changes -- EMB is untouched -- so every downstream width, and hence
        # every exported layer shape, is identical to the control's.
        r = rows or {}
        pi = PAD_IX if pad else None
        self.slot_emb = nn.Embedding(r.get("slot_emb", N_CARD_IDS), EMB,
                                     padding_idx=pi)
        self.bag_emb = nn.EmbeddingBag(r.get("bag_emb", N_CARD_IDS), EMB,
                                       mode="mean", include_last_offset=True,
                                       padding_idx=pi)
        self.card_emb = nn.Embedding(r.get("card_emb", N_CARD_IDS), EMB,
                                     padding_idx=pi)
        self.atk_emb = nn.Embedding(r.get("atk_emb", N_ATTACK_IDS), EMB,
                                    padding_idx=pi)
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        if extra:                       # the v4 block, appended (features.py)
            in_state += N_EXTRA + N_XSLOT * EMB
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)
        # Constructed AFTER every policy parameter. Resetting the seed therefore
        # gives a control and an auxiliary treatment identical policy weights;
        # only the treatment consumes additional RNG after that point.
        self.outcome_head = (nn.Linear(state_h[-1], 1) if outcome else None)
        self.count_head = (nn.Linear(state_h[-1], 1) if count else None)
        # E2 adapters are also append-only and zero-initialized, so an untrained
        # treatment matches the frozen base logits exactly.
        self.adapter_names = list(adapter_names or [])
        self.adapter_route_ids: dict[str, int] = {}
        if self.adapter_names:
            unknown = [n for n in self.adapter_names if n not in NAME_TO_ROUTE
                       or NAME_TO_ROUTE[n] == 0]
            if unknown:
                raise SystemExit(
                    f"adapters must be non-general route names; got {unknown}")
            self.adapter_route_ids = {n: NAME_TO_ROUTE[n]
                                      for n in self.adapter_names}
            self.adapters = nn.ModuleDict({
                n: _make_adapter(in_head, adapter_h)
                for n in self.adapter_names
            })
        else:
            self.adapters = None

    def forward(self, dense, slots, bag_flat, bag_off, seld,
                opt_dense, opt_card, opt_atk, opt_tgt, opt_row,
                xdense=None, xslots=None, attrs=None, routes=None,
                return_state: bool = False):
        # The per-option encoding is built FIRST, because the v5 pool feeds it
        # into the state. It is the same tensor the head consumes below, so the
        # pool costs one reduction and no extra embedding lookups.
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,
        # so `--opt-cols 25` trains the exact v2-feature control on the identical
        # rows -- same games, same selects, same labels, only the features differ.
        # That is a cleaner control than comparing against the shipped net, which
        # also differs in corpus (2,810 games vs whatever is on disk now).
        oenc = torch.cat([opt_dense[:, :self.opt_cols],
                          self.card_emb(opt_card),
                          self.atk_emb(opt_atk),
                          self.card_emb(opt_tgt)], dim=1)     # (O, D)
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        parts.append(seld)
        # v4 goes LAST so that `--no-extra` reproduces the v3 state vector
        # byte-for-byte on the identical rows -- the same control discipline as
        # `--opt-cols 25` for the option block.
        if self.extra:
            parts.append(xdense)
            parts.append(self.slot_emb(xslots).flatten(1))
        # ...and v5 goes after v4, so `pool=False` reproduces the v4 state
        # vector byte-for-byte. Same discipline, third generation.
        if self.pool:
            parts.append(self._pool(oenc, opt_row, dense.shape[0]))
        # ...and v6 goes after v5, so `attr=False` reproduces the v5 state
        # vector byte-for-byte. Same discipline, fourth generation.
        if self.attr:
            parts.append(attrs)
        srepr = self.state_fc(torch.cat(parts, dim=1))       # (B, H)
        per_opt = torch.cat([srepr[opt_row], oenc], dim=1)   # (O, ...)
        logits = self.head(per_opt).squeeze(1)               # (O,)
        if (self.adapters is not None and not self.adapters_off
                and routes is not None):
            residual = torch.zeros_like(logits)
            row_route = routes[opt_row]
            for name, route_id in self.adapter_route_ids.items():
                mask = row_route == route_id
                if mask.any():
                    residual[mask] = self.adapters[name](
                        per_opt[mask]).squeeze(1)
            logits = logits + residual
        return (logits, srepr) if return_state else logits

    def _pool(self, oenc: torch.Tensor, opt_row: torch.Tensor,
              n_rows: int) -> torch.Tensor:
        """Segment mean/max of the option encodings + two count scalars.

        A permutation-invariant summary of the option SET, which is the one
        thing an independently-scored option can never carry. Empty selects
        (none exist in the corpus, but the arena can produce one) pool to zero
        rather than to -inf."""
        d = oenc.shape[1]
        idx = opt_row.unsqueeze(1).expand(-1, d)
        cnt = torch.zeros(n_rows, device=oenc.device).index_add_(
            0, opt_row, torch.ones_like(opt_row, dtype=oenc.dtype))
        mean = torch.zeros(n_rows, d, device=oenc.device).index_add_(
            0, idx[:, 0], oenc) / cnt.clamp_min(1.0).unsqueeze(1)
        mx = torch.full((n_rows, d), -1e30, device=oenc.device).scatter_reduce(
            0, idx, oenc, reduce="amax", include_self=True)
        nz = (cnt > 0).unsqueeze(1)
        mx = torch.where(nz, mx, torch.zeros_like(mx))
        scal = torch.stack([cnt.clamp_max(40.0) / 40.0,
                            torch.log1p(cnt) / float(np.log(41.0))], dim=1)
        return torch.cat([mean, mx, scal], dim=1)


def parse_episode_span(spec: str) -> tuple[float, float]:
    """Parse `--episode-span START:END` into inclusive-exclusive fractions."""
    if ":" not in spec:
        raise SystemExit("--episode-span needs START:END (e.g. 0:0.5)")
    a, b = (s.strip() for s in spec.split(":", 1))
    try:
        start, end = float(a), float(b)
    except ValueError as exc:
        raise SystemExit(f"--episode-span {spec!r}: {exc}") from exc
    if not (0.0 <= start < end <= 1.0):
        raise SystemExit("--episode-span requires 0 <= START < END <= 1")
    return start, end


def episode_span_mask(gid: np.ndarray, start: float, end: float) -> np.ndarray:
    """True for rows in [floor(n*start), floor(n*end)) within each gid.

    Row order is the order they appear in `gid` (shard-concat chronological
    order from build_policy_dataset). Odd-length games put the middle row in
    the second half when start=0.5 (floor splits)."""
    keep = np.zeros(len(gid), dtype=bool)
    # First pass: counts per gid in appearance order, without sorting the
    # whole array (gids are not contiguous across day dirs).
    order: dict[int, list[int]] = {}
    for i, g in enumerate(gid.tolist()):
        order.setdefault(g, []).append(i)
    for idxs in order.values():
        n = len(idxs)
        lo = int(n * start)
        hi = int(n * end)
        for j in idxs[lo:hi]:
            keep[j] = True
    return keep


def listwise_loss(out: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor, n_rows: int,
                  w: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax cross-entropy within each select's option set, averaged over
    the chosen options of that select. This is the objective that matches
    inference: the agent ranks the options and takes the top k.

    `w` is an optional per-ROW weight (ROADMAP B7): the loss becomes a weighted
    mean, so a strong demonstrator's selects pull the mode further than a weak
    one's. Weights are normalised to mean 1 by the caller, which keeps the
    effective step size comparable to the unweighted control."""
    # log-softmax per row, computed with a segmented max for stability
    big = torch.full((n_rows,), -1e30, device=out.device)
    mx = big.scatter_reduce(0, opt_row, out, reduce="amax", include_self=True)
    ex = torch.exp(out - mx[opt_row])
    denom = torch.zeros(n_rows, device=out.device).index_add_(0, opt_row, ex)
    logp = out - mx[opt_row] - torch.log(denom + 1e-12)[opt_row]
    picked = torch.zeros(n_rows, device=out.device).index_add_(
        0, opt_row, logp * chosen)
    cnt = torch.zeros(n_rows, device=out.device).index_add_(0, opt_row, chosen)
    valid = cnt > 0
    per_row = -(picked[valid] / cnt[valid])
    if w is None:
        return per_row.mean()
    wv = w[valid]
    return (per_row * wv).sum() / wv.sum().clamp_min(1e-8)


def count_targets(seld: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target fraction and validity mask for variable-count selects.

    This is the row-level equivalent of `count_fraction_table`: the old table
    averages these targets within `(selectType, context)` buckets, while E1
    asks the shared state representation to predict each row separately.
    """
    n_rows = seld.shape[0]
    picked = torch.zeros(n_rows, dtype=chosen.dtype,
                         device=chosen.device).index_add_(0, opt_row, chosen)
    mn = seld[:, 11] * 5.0
    mx = seld[:, 12] * 5.0
    valid = mx > mn + 1e-6
    target = (picked - mn) / (mx - mn).clamp_min(1e-6)
    return target.clamp(0.0, 1.0), valid


class Data:
    def __init__(self, paths: list[Path], want_attr: bool = True):
        # ⚠ `want_attr=False` skips the v6 attribute block entirely instead of
        # materialising it. It is 276 float32 per row -- 27.6% of this object --
        # and `Model.forward` only reads it when the net was built with
        # `--attr`, so under the v5 recipe every one of those bytes is loaded,
        # copied per batch and moved to the device to be discarded. On the
        # 40.1M-row corpus that is 44.3 GB resident and ~89 GB of load peak for
        # nothing. Verified no-op: train loss identical to 4 dp with and
        # without. Default True so every existing caller is unchanged.
        sd, slots, seld, gid, won, rating = [], [], [], [], [], []
        xd, xs, at = [], [], []
        # B8. Present only in shards written by p26_selfplay_gen.py; a BC
        # corpus gets NaN, which is what `--advantage` refuses to weight.
        margin: list = []
        adv: list = []
        self.rows_per_path: list[int] = []
        od, oc, oa, ot, om = [], [], [], [], []
        self.opt_rows: list[tuple[int, int]] = []  # (start,end) per row
        # ⚠ Bags are kept FLAT (one array + one offset array per bag), exactly
        # as the shards store them. The previous version materialised one small
        # numpy array per row per bag -- 249k rows x 3 bags = ~750k objects --
        # and that allocation, not the model, is what OOM'd this 7.3 GB machine
        # on any net above ~1.5M params. Same semantics, ~1 GB less resident.
        bag_flats: dict[str, list] = {n: [] for n in BAGS}
        bag_lens: dict[str, list] = {n: [] for n in BAGS}
        base = 0
        for p in paths:
            z = np.load(p)
            n = len(z["gid"])
            sd.append(z["dense"])
            slots.append(z["slots"])
            seld.append(z["seld"])
            # Corpora built before day 12 have no v4 block; zeros keep them
            # loadable, and `--extra` on such a corpus is refused in main().
            xd.append(z["xdense"] if "xdense" in z
                      else np.zeros((n, N_EXTRA), dtype=np.float32))
            xs.append(z["xslots"] if "xslots" in z
                      else np.zeros((n, N_XSLOT), dtype=np.int32))
            # Same contract for the v6 block: corpora built before day 20 get
            # zeros so they stay loadable, and `--attr` on such a corpus is
            # refused in main() rather than silently training on nothing.
            at.append(z["attr"] if (want_attr and "attr" in z)
                      else np.zeros((n, N_ATTR if want_attr else 0),
                                    dtype=np.float32))
            gid.append(z["gid"])
            won.append(z["won"])
            self.rows_per_path.append(n)
            margin.append(z["margin"] if "margin" in z
                          else np.full(n, np.nan, dtype=np.float32))
            # E27: the per-decision TD residual written by p92_td_advantage.py.
            # NaN for every corpus built before it, which --advantage-col
            # refuses to weight rather than treating as zero.
            adv.append(z["adv"] if "adv" in z
                       else np.full(n, np.nan, dtype=np.float32))
            # Corpora built before `--ratings` have no per-row demonstrator.
            rating.append(z["rating"] if "rating" in z
                          else np.full(n, np.nan, dtype=np.float32))
            # Pre-v6 corpora store OPT_DENSE_V3 (37) cols; current builds store
            # OPT_DENSE (46). Pad the short layout with zeros so mixed --ds
            # unions concatenate; --opt-cols then slices the prefix it needs.
            od_arr = z["opt_dense"]
            w = int(od_arr.shape[1])
            if w < OPT_DENSE:
                od_arr = np.pad(od_arr, ((0, 0), (0, OPT_DENSE - w)))
            elif w > OPT_DENSE:
                od_arr = od_arr[:, :OPT_DENSE]
            od.append(od_arr)
            oc.append(z["opt_card"])
            oa.append(z["opt_attack"])
            ot.append(z["opt_target"] if "opt_target" in z
                      else np.zeros_like(z["opt_card"]))
            om.append(z["opt_chosen"])
            off = z["opt_off"]
            for i in range(n):
                self.opt_rows.append((base + off[i], base + off[i + 1]))
            base += off[-1]
            for nm in BAGS:
                flat = z[f"bag_{nm}_flat"]
                boff = z[f"bag_{nm}_off"]
                bag_flats[nm].append(flat.astype(np.int64, copy=False))
                bag_lens[nm].append(np.diff(boff).astype(np.int64))
        self.dense = np.concatenate(sd)
        self.slots = np.concatenate(slots).astype(np.int64)
        self.seld = np.concatenate(seld)
        self.xdense = np.concatenate(xd)
        self.xslots = np.concatenate(xs).astype(np.int64)
        self.attr = np.concatenate(at)
        self.has_extra = all("xdense" in np.load(p) for p in paths)
        self.has_attr = all("attr" in np.load(p) for p in paths)
        self.gid = np.concatenate(gid)
        self.won = np.concatenate(won)
        self.rating = np.concatenate(rating)
        self.margin = np.concatenate(margin)
        self.adv = np.concatenate(adv)
        self.w = np.ones(len(self.gid), dtype=np.float32)
        self.opt_dense = np.concatenate(od)
        self.opt_card = np.concatenate(oc).astype(np.int64)
        self.opt_atk = np.concatenate(oa).astype(np.int64)
        self.opt_tgt = np.concatenate(ot).astype(np.int64)
        self.opt_chosen = np.concatenate(om)
        # Flats concatenate in row order, so a cumsum over the per-row lengths
        # gives GLOBAL offsets across shards.
        self.bag_flat: dict[str, np.ndarray] = {}
        self.bag_off: dict[str, np.ndarray] = {}
        for nm in BAGS:
            lens = np.concatenate(bag_lens[nm])
            off = np.zeros(len(lens) + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            self.bag_off[nm] = off
            self.bag_flat[nm] = (np.concatenate(bag_flats[nm]) if off[-1]
                                 else np.zeros(0, dtype=np.int64))
        self.n = len(self.gid)
        # E2 route labels from observable opponent slots + discard only.
        self.routes = routes_from_corpus(
            self.slots, self.bag_flat["opp_discard"],
            self.bag_off["opp_discard"])

    def batches(self, idx: np.ndarray, bs: int,
                rng: np.random.Generator | None):
        order = rng.permutation(idx) if rng is not None else idx
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            bag_flat, bag_off = {}, {}
            for nm in BAGS:
                go = self.bag_off[nm]
                lens = go[sel + 1] - go[sel]
                off = np.zeros(len(sel) + 1, dtype=np.int64)
                np.cumsum(lens, out=off[1:])
                if off[-1]:
                    idx = np.concatenate([np.arange(go[k], go[k + 1])
                                          for k in sel if go[k + 1] > go[k]])
                    gathered = self.bag_flat[nm][idx]
                else:
                    gathered = np.zeros(0, dtype=np.int64)
                bag_flat[nm] = torch.from_numpy(gathered)
                bag_off[nm] = torch.from_numpy(off)
            spans = [self.opt_rows[k] for k in sel]
            opt_idx = np.concatenate([np.arange(a, b) for a, b in spans])
            opt_row = np.concatenate(
                [np.full(b - a, j) for j, (a, b) in enumerate(spans)])
            yield (torch.from_numpy(self.dense[sel]),
                   torch.from_numpy(self.slots[sel]),
                   bag_flat, bag_off,
                   torch.from_numpy(self.seld[sel]),
                   torch.from_numpy(self.opt_dense[opt_idx]),
                   torch.from_numpy(self.opt_card[opt_idx]),
                   torch.from_numpy(self.opt_atk[opt_idx]),
                   torch.from_numpy(self.opt_tgt[opt_idx]),
                   torch.from_numpy(opt_row),
                   torch.from_numpy(self.opt_chosen[opt_idx]),
                   spans,
                   torch.from_numpy(self.w[sel]),
                   sel,
                   torch.from_numpy(self.xdense[sel]),
                   torch.from_numpy(self.xslots[sel]),
                   torch.from_numpy(self.attr[sel]),
                   torch.from_numpy(self.routes[sel]))


def td_advantage_weights(data: "Data", is_rl: np.ndarray, beta: float,
                         anchor_w: float) -> np.ndarray:
    """E27: AWR over the PER-DECISION TD residual, not the game result.

    `w = exp(beta * A / sd(A))` on RL rows. **The normalisation by sd(A) is the
    part that has to be argued, and it is frozen in the pre-registration rather
    than tuned**: a TD residual has sd ~0.07 while B8's `won - baseline` has
    sd ~0.5, so passing the same beta to both would ma

... TRUNCATED ...

====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/train_value.py
====================================================================================================
#!/usr/bin/env python
"""Train V(s) -> P(win) on SELF-PLAY OUTCOMES. E20, pre-registered.

⚡ **`agents/sa/valuenet.py` has cited this script since day 1 and it has never
existed.** The inference side is already written, already dimension-guarded, and
already takes `me` explicitly -- so this trainer's only job is to produce an npz
in exactly that layout: `slot_emb`, `bag_emb`, `w1/b1`, `w2/b2`, `w3/b3`, with
torch's (out, in) weight orientation, because `valuenet.Net` computes `w @ x`.

**The input is the pure STATE**: `features.featurize` -> dense(242) + slot ids +
three card bags. Deliberately NOT `seld`/`xdense`, which are select-conditional
-- at play time V scores a SUCCESSOR observation returned by `fs.step`, so its
input must be a function of state alone or training and inference diverge and
produce a plausible number (rule 18).

**The label is `won`, from the acting seat's point of view** (p26 writes it that
way; draws are 0.5). This is the one column the BC corpus has always carried and
that nothing has ever trained on.

⚠ §8az's warning is why early stopping is not optional: E1's outcome head on the
HUMAN corpus overfit after its first epoch. Split is by `gid` -- a row-wise
split leaks a game across both sides (rule 17's shape).

    python -X utf8 scripts/train_value.py --data artifacts/rl_v5_t05 \\
        --out out/value_v1.npz --epochs 30
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / _sub))

# `sa.features` -> `sa.cards` -> `cg.sim`, so the SDK has to be on the path
# before the first feature import. Same bootstrap as `train_policy.py`.
from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import DENSE_DIM, N_CARD_IDS  # noqa: E402

EMB = 16
BAGS = ("my_hand", "my_discard", "opp_discard")


class ValueNet(nn.Module):
    """Mirrors `valuenet.Net` exactly. Any change here is a change there."""

    def __init__(self, h1: int = 256, h2: int = 128, dropout: float = 0.1):
        super().__init__()
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)
        in_dim = DENSE_DIM + 12 * EMB + len(BAGS) * EMB
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, dense, slots, bag_flat, bag_off):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        x = torch.cat(parts, dim=1)
        h = self.drop(torch.relu(self.fc1(x)))
        h = self.drop(torch.relu(self.fc2(h)))
        return self.fc3(h).squeeze(1)


class Data:
    def __init__(self, paths: list[Path]):
        d, s, w, g = [], [], [], []
        bags: dict[str, list] = {b: [] for b in BAGS}
        offs: dict[str, list] = {b: [] for b in BAGS}
        for p in paths:
            z = np.load(p)
            d.append(z["dense"])
            s.append(z["slots"])
            w.append(z["won"])
            g.append(z["gid"])
            for b in BAGS:
                bags[b].append(z[f"bag_{b}_flat"])
                offs[b].append(z[f"bag_{b}_off"])
        self.dense = np.concatenate(d)
        self.slots = np.concatenate(s)
        self.won = np.concatenate(w).astype(np.float32)
        self.gid = np.concatenate(g)
        # Offsets are per-shard and must be rebased before concatenation, or
        # every shard after the first indexes into the wrong card bag -- a
        # silent corruption that trains fine and evaluates to noise.
        self.bag_flat, self.bag_off = {}, {}
        for b in BAGS:
            flats, off_out, base, row = [], [], 0, 0
            for fl, of in zip(bags[b], offs[b]):
                flats.append(fl)
                off_out.append(of[:-1] + base if row else of[:-1])
                base += len(fl)
                row += 1
            self.bag_flat[b] = np.concatenate(flats).astype(np.int64)
            self.bag_off[b] = np.concatenate(off_out + [np.array([base])]
                                             ).astype(np.int64)
        n = len(self.won)
        for b in BAGS:
            assert len(self.bag_off[b]) == n + 1, (b, len(self.bag_off[b]), n)

    def __len__(self) -> int:
        return len(self.won)

    def batch(self, idx: np.ndarray, dev) -> tuple:
        dense = torch.from_numpy(self.dense[idx]).to(dev)
        slots = torch.from_numpy(self.slots[idx].astype(np.int64)).to(dev)
        bf, bo = {}, {}
        for b in BAGS:
            off, flat = self.bag_off[b], self.bag_flat[b]
            segs = [flat[off[i]:off[i + 1]] for i in idx]
            lens = np.array([len(s) for s in segs])
            # EmbeddingBag with mode="mean" divides by zero on an empty bag and
            # returns NaN; pad an empty bag with row 0, matching valuenet.Net's
            # explicit zeros() branch closely enough for a mean over one row.
            segs = [s if len(s) else np.zeros(1, dtype=np.int64) for s in segs]
            lens = np.maximum(lens, 1)
            bf[b] = torch.from_numpy(np.concatenate(segs).astype(np.int64)).to(dev)
            bo[b] = torch.from_numpy(
                np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)).to(dev)
        y = torch.from_numpy(self.won[idx]).to(dev)
        return dense, slots, bf, bo, y


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC over decided rows only (draws carry no order)."""
    m = y != 0.5
    y, p = y[m], p[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0,
                    help="init + data order. Ensemble members vary THIS.")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="game split. Ensemble members must SHARE this, or "
                         "each member holds out different games and the "
                         "ensemble's spread confounds init variance with "
                         "data variance. Defaults to --seed.")
    args = ap.parse_args()
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    paths = sorted(Path(p) for d in args.data
                   for p in glob.glob(f"{d}/**/shard_*.npz", recursive=True))
    if not paths:
        sys.exit(f"no shards under {args.data}")
    print(f"loading {len(paths)} shards ...", flush=True)
    data = Data(paths)

    # 🔴 SPLIT BY GAME. Rows from one game are near-duplicates of each other and
    # share a label exactly; a row-wise split puts both sides of the same game
    # in train and val and reports a val number that means nothing.
    gids = np.unique(data.gid)
    np.random.default_rng(split_seed).shuffle(gids)
    rng = np.random.default_rng(args.seed)      # batch order only
    n_val = max(1, int(len(gids) * args.val_frac))
    val_g = set(gids[:n_val].tolist())
    is_val = np.fromiter((g in val_g for g in data.gid), dtype=bool,
                         count=len(data))
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"rows {len(data):,}  games {len(gids):,}  "
          f"train {len(tr_idx):,} / val {len(va_idx):,}  "
          f"base rate {data.won.mean():.4f}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = ValueNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            dense, slots, bf, bo, y = data.batch(idx, dev)
            opt.zero_grad()
            loss = lossf(model(dense, slots, bf, bo), y)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(va_idx), args.bs):
                idx = va_idx[i:i + args.bs]
                dense, slots, bf, bo, y = data.batch(idx, dev)
                ps.append(torch.sigmoid(model(dense, slots, bf, bo)).cpu().numpy())
                ys.append(y.cpu().numpy())
        p, y = np.concatenate(ps), np.concatenate(ys)
        vl = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
        print(f"ep {ep:2d}  train {tot/len(perm):.4f}  val {vl:.4f}  "
              f"AUC {auc(y, p):.4f}  {time.time()-t0:.0f}s", flush=True)

        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best val {best:.4f})", flush=True)
                break

    # Export rule pinned in advance (E20 / rule 18's corollary): BEST val
    # logloss on the gid-disjoint split, patience 3. Recorded, not chosen after.
    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(va_idx), args.bs):
            idx = va_idx[i:i + args.bs]
            dense, slots, bf, bo, y = data.batch(idx, dev)
            ps.append(torch.sigmoid(model(dense, slots, bf, bo)).cpu().numpy())
            ys.append(y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)

    # ⚠ THE ORIENTATION CONTROL. A sign-flipped V is the single failure that
    # would produce a confident, plausible, exactly-wrong agent: it would play
    # to LOSE and the arena would report ~0.0 rather than an error.
    mw, ml = float(p[y == 1].mean()), float(p[y == 0].mean())
    print(f"\nORIENTATION  mean V | won = {mw:.4f}   mean V | lost = {ml:.4f}   "
          f"AUC {auc(y, p):.4f}")
    if not mw > ml:
        sys.exit("ORIENTATION FAILED: V does not score won states above lost")

    sd = model.state_dict()
    np.savez(
        args.out,
        slot_emb=sd["slot_emb.weight"].numpy(),
        bag_emb=sd["bag_emb.weight"].numpy(),
        w1=sd["fc1.weight"].numpy(), b1=sd["fc1.bias"].numpy(),
        w2=sd["fc2.weight"].numpy(), b2=sd["fc2.bias"].numpy(),
        w3=sd["fc3.weight"].numpy()[0], b3=sd["fc3.bias"].numpy()[0],
    )
    print(f"wrote {args.out}  (val logloss {best:.4f}, AUC {auc(y, p):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/build_policy_dataset.py
====================================================================================================
"""Build policy-cloning shards from replay JSONs.

    python scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26

One row per select with >=2 options: state features + per-option features +
multi-hot chosen mask (from the replay's actual action).

⚠ By default this clones BOTH seats of every game -- every archetype and every
skill level in the dump. `--player NAME` keeps only the seats belonging to the
named team(s), which is how an EXPERT corpus is built from a third-party dump:

    python scripts/build_policy_dataset.py --out artifacts/pds_expert \\
        --player "Raja Biswas" --player "Sixth Sense" \\
        replays/sixth_sense_31-07-2026

`--ratings` tags every row with the LB score of the demonstrator who made that
choice, so a corpus can be reweighted or sliced by demonstrator strength (B7):

    python scripts/build_policy_dataset.py --out artifacts/pds_v3r \\
        --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import attr_feats, extra_feats, featurize  # noqa: E402
from sa.optfeat import option_features, OPT_DENSE  # noqa: E402

SHARD_ROWS = 60_000
SEL_DENSE = 14
NO_RATING = np.float32("nan")


def load_ratings(path: Path) -> tuple[dict[str, float], dict[int, float]]:
    """Kaggle LB export -> (by team name, by teamId). Accepts the .zip that
    `competition_leaderboard_download` writes or the .csv inside it."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if not names:
                raise SystemExit(f"{path}: no .csv inside")
            text = z.read(names[0]).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    by_name: dict[str, float] = {}
    by_id: dict[int, float] = {}
    by_user: dict[str, float] = {}
    n_teams = 0
    for row in csv.DictReader(text.splitlines()):
        try:
            score = float(row["Score"])
        except (TypeError, ValueError, KeyError):
            continue
        n_teams += 1
        by_name[row["TeamName"]] = score
        try:
            by_id[int(row["TeamId"])] = score
        except (TypeError, ValueError, KeyError):
            pass
        # A replay's TeamNames can hold a MEMBER's username rather than the
        # team's display name (teams merge and rename; `zoroark190` is how the
        # LB's #1 `James Cox & Henry Chao` appears in 07-26 replays). Exact
        # member matches are safe; ambiguous ones are dropped, not guessed.
        for user in (row.get("TeamMemberUserNames") or "").split(","):
            user = user.strip()
            if user:
                by_user[user] = score if user not in by_user else float("nan")
    # Display names always win; a username that collides with some other team's
    # display name, or with a second team, is dropped rather than resolved.
    n_user = 0
    for user, score in by_user.items():
        if user not in by_name and score == score:
            by_name[user] = score
            n_user += 1
    print(f"ratings: {n_teams} teams (+{n_user} member usernames) "
          f"from {path.name}")
    return by_name, by_id


def name_id(name: str) -> int:
    """Stable per-team id for dumps with no `episodes_meta.json` sidecar (the
    daily replay dirs). Deterministic across runs and processes, unlike
    `hash()`, so a corpus rebuilt tomorrow keeps the same ids."""
    import hashlib
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return -int.from_bytes(h[:6], "big")   # negative = derived, not Kaggle's


def load_episode_meta(d: Path) -> dict[int, dict[int, tuple[int, int]]]:
    """`episodes_meta.json` -> {episode_id: {seat: (submissionId, teamId)}}.
    Only targeted per-team dumps carry it; day dumps return {}."""
    p = d / "episodes_meta.json"
    if not p.exists():
        return {}
    out: dict[int, dict[int, tuple[int, int]]] = {}
    for ep in json.loads(p.read_text(encoding="utf-8")):
        seats = {}
        for a in ep.get("agents") or []:
            seats[int(a.get("index") or 0)] = (int(a.get("submissionId") or -1),
                                               int(a.get("teamId") or -1))
        out[int(ep["id"])] = seats
    return out


def sel_features(sel: dict) -> np.ndarray:
    v = np.zeros(SEL_DENSE, dtype=np.float32)
    t = sel.get("type") or 0
    if t < 11:
        v[t] = 1.0
    v[11] = sel.get("minCount", 0) / 5.0
    v[12] = sel.get("maxCount", 0) / 5.0
    v[13] = (sel.get("context") or 0) / 50.0
    return v


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.reset()

    def reset(self):
        self.sd, self.slots, self.seld, self.gid = [], [], [], []
        self.xd, self.xslots, self.attr = [], [], []
        self.bags = {"my_hand": [], "my_discard": [], "opp_discard": []}
        self.od, self.ocard, self.oatk, self.otgt, self.chosen = [], [], [], [], []
        self.off = [0]
        self.won = []
        self.rating, self.opp_rating, self.team_id, self.sub_id = [], [], [], []

    def add(self, dense, bags, seld, opts, chosen_mask, gid, won,
            rating=NO_RATING, opp_rating=NO_RATING, team_id=-1, sub_id=-1,
            extra=None, attr=None):
        self.sd.append(dense)
        # The v4 state block (features.extra_feats). Written unconditionally --
        # a trainer that does not want it simply does not read these arrays,
        # which is what makes the v3 control run on the IDENTICAL rows.
        xd, xids = extra
        self.xd.append(xd)
        self.xslots.append(xids)
        # The v6 card-attribute block (features.attr_feats), same contract:
        # always written, so a v5 control trains on the IDENTICAL rows.
        self.attr.append(attr)
        self.slots.append(bags["slots"])
        for k in self.bags:
            self.bags[k].append(bags[k])
        self.seld.append(seld)
        od, oc, oa, ot = opts
        self.od.append(od)
        self.ocard.append(oc)
        self.oatk.append(oa)
        self.otgt.append(ot)
        self.chosen.append(chosen_mask)
        self.off.append(self.off[-1] + len(oc))
        self.gid.append(gid)
        self.won.append(won)
        self.rating.append(rating)
        self.opp_rating.append(opp_rating)
        self.team_id.append(team_id)
        self.sub_id.append(sub_id)
        if len(self.sd) >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.sd:
            return
        arrs = {
            "dense": np.stack(self.sd),
            "slots": np.stack(self.slots),
            "seld": np.stack(self.seld),
            "xdense": np.stack(self.xd),
            "xslots": np.stack(self.xslots),
            "attr": np.stack(self.attr),
            "gid": np.asarray(self.gid, dtype=np.int64),
            "won": np.asarray(self.won, dtype=np.float32),
            # B7: who made this choice, and how good are they? NaN = the team
            # was not on the LB snapshot (renamed, or withdrawn).
            "rating": np.asarray(self.rating, dtype=np.float32),
            "opp_rating": np.asarray(self.opp_rating, dtype=np.float32),
            "team_id": np.asarray(self.team_id, dtype=np.int64),
            "sub_id": np.asarray(self.sub_id, dtype=np.int64),
            "opt_dense": np.concatenate(self.od),
            "opt_card": np.concatenate(self.ocard),
            "opt_attack": np.concatenate(self.oatk),
            "opt_target": np.concatenate(self.otgt),
            "opt_chosen": np.concatenate(self.chosen),
            "opt_off": np.asarray(self.off, dtype=np.int64),
        }
        for k, lists in self.bags.items():
            off = np.zeros(len(lists) + 1, dtype=np.int64)
            for i, a in enumerate(lists):
                off[i + 1] = off[i] + len(a)
            arrs[f"bag_{k}_flat"] = (np.concatenate(lists) if off[-1]
                                     else np.zeros(0, dtype=np.int32))
            arrs[f"bag_{k}_off"] = off
        path = self.out_dir / f"shard_{self.idx:03d}.npz"
        np.savez_compressed(path, **arrs)
        print(f"  wrote {path.name}: {len(self.sd)} rows")
        self.idx += 1
        self.reset()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--player", action="append", default=[],
                    help="keep only seats owned by this team name (repeatable). "
                         "Default: clone both seats of every game.")
    ap.add_argument("--players-file",
                    help="UTF-8 file of team names, one per line, added to "
                         "--player (for name lists the shell cannot quote)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop these team names even if --player/--players-file "
                         "names them. ⚠ Always exclude OURSELVES from a control "
                         "population: our own agent's selects are what the net "
                         "was fitted to, so leaving `Scio` in inflates "
                         "agreement toward 100%% for those rows.")
    ap.add_argument("--ratings",
                    help="Kaggle leaderboard export (.zip or .csv). Tags every "
                         "row with the demonstrator's LB score (B7). Without "
                         "it every row's rating is NaN.")
    ap.add_argument("--aliases", default="replays/team_aliases.tsv",
                    help="TSV of `replay name<TAB>LB team name` for teams that "
                         "renamed or merged and cannot be matched exactly. "
                         "Missing file = no aliases.")
    args = ap.parse_args()

    keep = set(args.player)
    if args.players_file:
        keep |= {ln.strip() for ln
                 in Path(args.players_file).read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")}
    # ⚠ An empty filter means "clone both seats of every game" -- silently the
    # OPPOSITE of what was asked. An empty --players-file used to build a
    # whole-dump corpus under an expert corpus's name.
    if (args.player or args.players_file) and not keep:
        raise SystemExit("--player/--players-file given but resolved to zero "
                         "names; refusing to build an unfiltered corpus")
    drop = set(args.exclude)
    if drop:
        hit = keep & drop
        keep -= drop
        print(f"--exclude: dropped {sorted(hit)} from the demonstrator set")
        if (args.player or args.players_file) and not keep:
            raise SystemExit("--exclude removed every demonstrator")
    rate_name, rate_id = ({}, {})
    alias: dict[str, str] = {}
    team_name: dict[int, str] = {}
    ap_path = ROOT / args.aliases
    if ap_path.exists():
        for ln in ap_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or ln.startswith("#") or "\t" not in ln:
                continue
            old, new = (s.strip() for s in ln.split("\t", 1))
            alias[old] = new
    if args.ratings:
        rate_name, rate_id = load_ratings(Path(args.ratings))
        n_alias = 0
        for old, new in alias.items():
            if new in rate_name and old not in rate_name:
                rate_name[old] = rate_name[new]
                n_alias += 1
            elif new not in rate_name:
                print(f"  alias target not on the LB, ignored: {new!r}",
                      file=sys.stderr)
        print(f"  aliases: {n_alias} applied from {ap_path.name}")
    writer = Writer(ROOT / args.out)
    n_games = n_rows = n_err = n_skip_game = n_skip_seat = 0
    n_rated = n_unrated = 0
    unrated_names: dict[str, int] = {}
    for d in args.dirs:
        ep_meta = load_episode_meta(Path(d))
        for path in sorted(Path(d).glob("*.json")):
            if path.name == "manifest.json" or not path.stem.isdigit():
                continue   # sidecars: manifest.json, episodes_meta.json, ...
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
                rewards = rep["rewards"]
                if rewards[0] is None or rewards[1] is None:
                    continue
                names = (rep.get("info") or {}).get("TeamNames") or []
                seats = ({i for i, n in enumerate(names) if n in keep}
                         if keep else
                         ({i for i, n in enumerate(names) if n not in drop}
                          if drop else None))
                if seats is not None and not seats:
                    n_skip_game += 1
                    continue
                vis = rep["steps"][0][0].get("visualize") or []
                try:
                    gid = int(path.stem)
                except ValueError:
                    gid = hash(path.stem) & 0x7FFFFFFF
                n_games += 1
                # Per-seat identity. The sidecar's teamId is authoritative when
                # present -- `info.TeamNames` is a DISPLAY name and teams rename
                # mid-window (§8q: one demonstrator appeared as two).
                meta = ep_meta.get(gid, {})
                seat_rating: dict[int, float] = {}
                seat_team: dict[int, int] = {}
                seat_sub: dict[int, int] = {}
                for i in range(max(len(names), len(meta))):
                    sub, tid = meta.get(i, (-1, -1))
                    r = rate_id.get(tid) if tid >= 0 else None
                    nm = names[i] if i < len(names) else ""
                    if r is None and nm:
                        r = rate_name.get(nm)
                    seat_rating[i] = float(r) if r is not None else float("nan")
                    if tid < 0 and nm:
                        # No sidecar: identify the demonstrator by name. The
                        # alias file has already merged renames, so this does
                        # not split one team in two (§8q).
                        tid = name_id(alias.get(nm, nm))
                        team_name[tid] = alias.get(nm, nm)
                    elif tid >= 0 and nm:
                        team_name[tid] = nm
                    seat_team[i] = tid
                    seat_sub[i] = sub
                    if args.ratings:
                        if r is None:
                            n_unrated += 1
                            nm = names[i] if i < len(names) else f"seat{i}"
                            unrated_names[nm] = unrated_names.get(nm, 0) + 1
                        else:
                            n_rated += 1
                for v in vis:
                    obs = v.get("obs")
                    if not obs or not obs.get("current") or not obs.get("select"):
                        continue
                    state = obs["current"]
                    if state["result"] != -1:
                        continue
                    sel = obs["select"]
                    opts = sel.get("option") or []
                    if len(opts) < 2:
                        continue
                    action = v.get("selected")
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]
                    if len(picked) != len(action):
                        continue
                    me = state["yourIndex"]
                    if seats is not None and me not in seats:
                        n_skip_seat += 1
                        continue
                    won = 1.0 if rewards[me] > rewards[1 - me] else 0.0
                    dense, bags = featurize(state, me)
                    od = np.zeros((len(opts), OPT_DENSE), dtype=np.float32)
                    oc = np.zeros(len(opts), dtype=np.int32)
                    oa = np.zeros(len(opts), dtype=np.int32)
                    ot = np.zeros(len(opts), dtype=np.int32)
                    for i, o in enumerate(opts):
                        od[i], oc[i], oa[i], ot[i] = option_features(obs, o)
                    mask = np.zeros(len(opts), dtype=np.float32)
                    mask[picked] = 1.0
                    writer.add(dense, bags, sel_features(sel),
                               (od, oc, oa, ot), mask, gid, won,
                               extra=extra_feats(state, sel, me),
                               attr=attr_feats(state, me),
                               rating=seat_rating.get(me, float("nan")),
                               opp_rating=seat_rating.get(1 - me,
                                                          float("nan")),
                               team_id=seat_team.get(me, -1),
                               sub_id=seat_sub.get(me, -1))
                    n_rows += 1
            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"  {path.name}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
    writer.flush()
    if team_name:
        # `team_id` in the shards is an int; this is how a report table gets a
        # name next to it. Merged with any existing map so a corpus built one
        # day-dir at a time accumulates rather than overwrites.
        tp = ROOT / args.out / "teams.json"
        old = (json.loads(tp.read_text(encoding="utf-8"))
               if tp.exists() else {})
        old.update({str(k): v for k, v in team_name.items()})
        tp.write_text(json.dumps(old, ensure_ascii=False, indent=1),
                      encoding="utf-8")
        print(f"  wrote {tp.name}: {len(old)} demonstrators")
    # ⚠ rule 9: a filter that matches nothing writes an empty corpus and exits 0.
    # A mistyped team name (CJK homoglyph, a rename) looks exactly like this.
    if keep and not n_games:
        raise SystemExit(f"player filter {sorted(keep)} matched ZERO games in "
                         f"{args.dirs}; check the exact team name")
    print(f"games={n_games} rows={n_rows} errors={n_err}")
    if args.ratings:
        tot = n_rated + n_unrated
        print(f"ratings: {n_rated}/{tot} seats matched the LB snapshot "
              f"({n_rated / max(tot, 1):.1%})")
        # ⚠ rule 9: an unmatched name is a SILENT NaN. Name the biggest misses
        # so a rename or an encoding mismatch cannot hide as "sparse data".
        for nm, c in sorted(unrated_names.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  unrated: {nm!r} x{c}")
    if keep:
        print(f"player filter {sorted(keep)}: skipped {n_skip_game} games with "
              f"no matching seat, {n_skip_seat} opponent-seat rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/build_submission.py
====================================================================================================
"""Assemble the sa search-agent submission and tar.gz it for Kaggle.

    python scripts/build_submission.py [--deck grimmsnarl] [--agent search|bc]

Bundle layout (Kaggle: .tar.gz, main.py at TOP level):
    main.py            entrypoint defining agent(obs) -> list[int]
    deck.csv           60 card ids, one per line
    cg/                engine (from the local SDK)
    sa/                agent package (+ value_net.npz / policy_net.npz /
                       deck_library.json if present)

Then smoke-runs the *extracted* bundle from a temp dir: full self-play game,
crash = build failure. Prints latency + time-pool stats.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg import config  # noqa: E402

SIZE_CAP_MIB = 197.7

# 🔴 Kaggle's EPISODE RUNNER is Python 3.11 -- `kaggle_environments` lives under
# `/usr/local/lib/python3.11/dist-packages/`. Kaggle *notebooks* are 3.12, and
# so is this dev box, so 3.12-only syntax parses everywhere we normally look and
# then raises SyntaxError at IMPORT on the grader. Submission 55489084 died that
# way: one PEP 701 multi-line f-string in `sa/policynet.py` (a logging nicety)
# stopped `from sa.bcagent import PolicyAgent`, both seats crashed in 0.04s, and
# the LB said only "Validation Episode failed." The smoke test CANNOT catch this
# -- it runs on the local interpreter, where the file is valid.
KAGGLE_PY = (3, 11)


def _find_kaggle_python() -> str | None:
    """An interpreter matching Kaggle's episode runner, or None.

    `SA_KAGGLE_PYTHON` wins; otherwise ask uv for a managed 3.11. Returning
    None is not fatal -- the caller falls back to a narrower static check.
    """
    pinned = os.environ.get("SA_KAGGLE_PYTHON")
    if pinned and Path(pinned).exists():
        return pinned
    ver = "%d.%d" % KAGGLE_PY
    try:
        proc = subprocess.run(["uv", "python", "find", ver],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    cand = proc.stdout.strip()
    return cand if proc.returncode == 0 and cand and Path(cand).exists() else None


def _scan_pep701(root: Path) -> list[str]:
    """Static fallback: f-strings that only parse on 3.12+ (PEP 701).

    Catches the multi-line replacement field that actually shipped. It is a
    subset of what a real 3.11 parse catches, so it is the fallback, never the
    primary check.
    """
    import io
    import tokenize
    fs_start = getattr(tokenize, "FSTRING_START", None)
    fs_end = getattr(tokenize, "FSTRING_END", None)
    if fs_start is None:  # pre-3.12 host: it would have failed outright
        return []
    hits = []
    for f in sorted(root.rglob("*.py")):
        try:
            toks = list(tokenize.generate_tokens(
                io.StringIO(f.read_text(encoding="utf-8")).readline))
        except Exception:  # noqa: BLE001  -- a parse failure is caught elsewhere
            continue
        depth, start = 0, None
        for t in toks:
            if t.type == fs_start:
                if depth == 0:
                    start = t
                depth += 1
            elif t.type == fs_end:
                depth -= 1
                if depth == 0 and start is not None:
                    quoted = start.string.lstrip("fFrRbB")
                    if t.end[0] != start.start[0] and quoted not in ('"""', "'''"):
                        hits.append(f"{f.relative_to(root)}:{start.start[0]}"
                                    " multi-line f-string (PEP 701, 3.12+)")
                    start = None
    return hits


def _check_kaggle_syntax(build: Path) -> None:
    """Compile every bundled .py the way the grader's interpreter will."""
    py = _find_kaggle_python()
    if py:
        proc = subprocess.run([py, "-m", "compileall", "-q", str(build)],
                              capture_output=True, text=True, timeout=600)
        for cache in build.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:] or proc.stderr[-3000:])
            raise SystemExit(
                "syntax check FAILED under Python %d.%d -- this bundle would "
                "crash at import on Kaggle, exactly like 55489084" % KAGGLE_PY)
        print("  kaggle syntax: OK under %d.%d (%s)" % (*KAGGLE_PY, py))
        return
    hits = _scan_pep701(build)
    if hits:
        for h in hits:
            print(f"    {h}")
        raise SystemExit(
            "bundle uses 3.12-only f-string syntax and Kaggle runs %d.%d"
            % KAGGLE_PY)
    print("  kaggle syntax: no %d.%d interpreter found (`uv python install "
          "%d.%d`); ran the PEP-701 scan only -- WEAKER" % (*KAGGLE_PY,
                                                            *KAGGLE_PY))

MAIN_PY = '''\
import os
import sys

# Kaggle loads this file with exec(code_object, env) -- `__file__` is NOT
# defined there (kaggle_environments/agent.py:get_last_callable). Never rely
# on it: fall back to the documented agent dir, then cwd.
_CANDS = []
try:
    _CANDS.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
_CANDS.append("/kaggle_simulations/agent")
_CANDS.append(os.getcwd())
_HERE = next((p for p in _CANDS
              if p and os.path.exists(os.path.join(p, "deck.csv"))), _CANDS[-1])
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

AGENT_KIND = {agent_kind!r}
# Extra ensemble members, relative to the bundle root. Empty = single net, i.e.
# exactly the behaviour every submission before this one had. When non-empty the
# agent votes across sa/policy_net.npz + these (EVIDENCE: E9).
ENSEMBLE_EXTRA = {ensemble_extra!r}
# Explicit rule flags, pinned at BUILD time. The defaults in sa/bcagent.py are
# tuned for the lw2 net; an optfeat-v3 net wants them OFF (the three rules
# measure 0.427 against it -- report/EVIDENCE.md 8f). Pinning the (net, flags)
# PAIR here keeps both configurations shippable without a global default flip.
AGENT_KWARGS = {agent_kwargs!r}


def _read_deck():
    path = os.path.join(_HERE, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as fh:
        return [int(line) for line in fh.read().split()[:60]]


_deck = _read_deck()

if AGENT_KIND == "bc":
    from sa.bcagent import PolicyAgent as _A
    _agent = None
    if ENSEMBLE_EXTRA:
        # 🔴 FAIL-SOFT IS NON-NEGOTIABLE ON THE SHIPPED PATH. An explicit
        # `net=` is STRICT by design (day 22: a net that failed its guard used
        # to silently play a different one), and strict means it RAISES. In the
        # arena that is correct -- a bad cell must not print a score. Here it
        # would forfeit a live episode. So: try the vote, and on ANY failure
        # fall back to the single bundled net, which is member 0 -- i.e. the
        # agent that shipped before this change. Degrade, log, keep playing.
        try:
            _paths = [os.path.join(_HERE, "sa", "policy_net.npz")]
            _paths += [os.path.join(_HERE, p) for p in ENSEMBLE_EXTRA]
            _agent = _A(_deck, "+".join(_paths), **AGENT_KWARGS)
            print("[health] ENSEMBLE OK members=%d" % len(_paths), flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("[health] ENSEMBLE FAILED TO LOAD -- falling back to the "
                  "single bundled net", flush=True)
            _agent = None
    if _agent is None:
        _agent = _A(_deck, **AGENT_KWARGS)
else:
    from sa.agent import SearchAgent as _A
    _agent = _A(_deck)


# --- health logging (day 15) -------------------------------------------------
# Kaggle keeps our own submission's stdout, and until now we printed NOTHING on
# the happy path, so the logs were empty. The thing worth the bytes is whether
# the NET IS ACTUALLY LIVE: `bcagent.__call__`'s catch-all returns
# `range(minCount)` -- index order -- and a submission running that on every
# decision still returns legal moves, still finishes games and still gets a
# rating, so it looks normal from outside. EVIDENCE 8g had to infer the net was
# live from a 40.7% index-0 rate; this makes it a direct read.
#
# Discipline: ONE line per game (the deck handshake fires once per game), plus a
# heartbeat every 1000 selects so a single long episode still reports, plus a
# one-shot line the first time the fallback ever fires. Never per-decision.
#
# EVERYTHING here is wrapped: a logging bug must never be able to break the
# agent, which would trade a real rating for a diagnostic.
#
# ⚠ MAIN_PY is a str.format() template -- every literal brace below is doubled.
_LOG = {{"next": 0, "failed": False}}


def _log_health(force=False):
    try:
        from sa import bcagent as _bc
        s = _bc.STATS
        if s["fallbacks"] and not _LOG["failed"]:
            _LOG["failed"] = True
            print("[health] FIRST FALLBACK -- net path raised; now playing "
                  "index order", flush=True)
            print(str(s["first_error"])[:1500], flush=True)
            return
        if force or s["calls"] >= _LOG["next"]:
            _LOG["next"] = s["calls"] + 1000
            print(_bc.health_line(), flush=True)
    except Exception:
        pass


def agent(obs):
    out = _agent(obs)
    _log_health(force=obs.get("select") is None)
    return out
'''

SMOKE = r'''
import os, sys, time
sys.path.insert(0, ".")

# Load main.py THE WAY KAGGLE DOES: exec the source with no __file__ in globals
# (kaggle_environments/agent.py does exec(code_object, env)). `import main`
# would define __file__ and hide a whole class of crash -- it did once.
with open("main.py", "rb") as _fh:
    _src = _fh.read()
_env = {}
exec(compile(_src, "main.py", "exec"), _env)
assert "__file__" not in _env, "smoke must not leak __file__ into agent globals"


class _M:
    pass


main = _M()
main._deck = _env["_deck"]
main.agent = _env["agent"]

deck = list(main._deck)
assert len(deck) == 60, len(deck)

# The agent "runs" perfectly well with a rejected net -- it just plays
# list(range(minCount)), i.e. random-legal, and scores ~600. Kaggle sets no env
# vars, so the bundled npz is the only thing that can save it. Assert the net is
# actually live, and print the pinned rule flags so the build log records the
# exact configuration that was shipped.
_ag = _env.get("_agent")
if _ag is not None and type(_ag).__name__ == "PolicyAgent":
    from sa import policynet as _pn
    _live = _ag.net or _pn.get()
    assert _live is not None, "POLICY NET NOT LOADED -- agent would play random-legal"
    print(f"NET_OK opt_in={_live.opt_in} state_in={_live.state_in}")
    # An ensemble that quietly loaded ONE member is a different agent shipping
    # under the measured one's name, and it would score like the single net
    # while the build log said nothing. Assert the member count explicitly.
    _want = int(os.environ.get("SMOKE_MEMBERS", "1"))
    _got = len(getattr(_live, "nets", []) or [1])
    print(f"MEMBERS={_got} want={_want}")
    assert _got == _want, f"ensemble has {_got} members, expected {_want}"
    print("FLAGS chip=%s spread=%s src=%s wall=%s"
          % (_ag.chip_targeting, _ag.energy_spread, _ag.counter_source,
             _ag.chip_wall_defer))

import cg.game as game

class B:  # opposing agent: trivial legal
    def __call__(self, obs):
        if obs.get("select") is None:
            return deck
        return list(range(obs["select"]["minCount"]))

opp = B()
obs, _ = game.battle_start(deck, deck)
overage = [600.0, 600.0]
selects = 0
lat_max = 0.0
try:
    while True:
        st = obs.get("current")
        if st is not None and st["result"] != -1:
            print(f"RESULT={st['result']} turns={st['turn']} selects={selects} "
                  f"agent_pool_left={overage[0]:.1f}s lat_max={lat_max:.2f}s")
            break
        who = st["yourIndex"]
        obs["remainingOverageTime"] = overage[who]
        t0 = time.perf_counter()
        choice = main.agent(obs) if who == 0 else opp(obs)
        dt = time.perf_counter() - t0
        overage[who] -= dt
        if who == 0:
            lat_max = max(lat_max, dt)
        obs = game.battle_select([int(c) for c in choice])
        selects += 1
        assert overage[0] > 0, "agent exhausted its time pool"
        if selects > 6000:
            raise SystemExit("game did not terminate")
finally:
    game.battle_finish()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--agent", default="search", choices=["search", "bc"])
    ap.add_argument("--no-smoke", action="store_true")
    # Which nets to ship. The SA_NO_* kill-switches are env vars and Kaggle
    # sets none, so anything bundled is LIVE there. Omitting an npz makes the
    # matching net.get() return None and the agent fall back to the
    # handcrafted eval -- the only way to pin the config for the grader.
    ap.add_argument("--nets", default="both",
                    choices=["both", "policy", "value", "none"],
                    help="nets to include (default both)")
    # Ship a candidate net instead of agents/sa/policy_net.npz. The bundle is
    # what the grader runs, so the net and the rule flags must be pinned
    # TOGETHER -- a v3 net with the lw2 defaults is the 0.427 configuration.
    ap.add_argument("--policy-net", default=None,
                    help="path to an npz to ship as sa/policy_net.npz")
    # Extra ensemble members (EVIDENCE E9). Each is copied into the bundle as
    # sa/policy_net_<i>.npz and voted with sa/policy_net.npz. Member ORDER is
    # part of the agent's identity -- member 0 supplies the count rule.
    ap.add_argument("--ensemble-net", action="append", default=[],
                    help="extra npz to vote with the shipped policy net; "
                         "repeatable")
    ap.add_argument("--no-rules", action="store_true",
                    help="disable chip_targeting/energy_spread/counter_source. "
                         "Required with an optfeat-v3 net (EVIDENCE 8f).")
    args = ap.parse_args()
    if args.agent == "bc" and args.nets in ("value", "none"):
        raise SystemExit("--agent bc requires the policy net (--nets both|policy)")

    agent_kwargs: dict[str, bool] = {}
    if args.no_rules:
        agent_kwargs = {"chip_targeting": False, "energy_spread": False,
                        "counter_source": False}

    sdk_dir = config.find_sdk_dir()
    if sdk_dir is None:
        raise SystemExit("cg engine not found under data/")

    build = config.DIST_DIR / f"_build_{args.agent}-{args.deck}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    ensemble_extra = [f"sa/policy_net_{i + 1}.npz"
                      for i in range(len(args.ensemble_net))]
    (build / "main.py").write_text(
        MAIN_PY.format(agent_kind=args.agent, agent_kwargs=agent_kwargs,
                       ensemble_extra=ensemble_extra),
        encoding="utf-8")
    print(f"agent kwargs: {agent_kwargs or '(bcagent.py defaults)'}")

    deck = importlib.import_module(f"decks.{args.deck}").DECK
    (build / "deck.csv").write_text(deck.to_csv(), encoding="utf-8")
    print(f"deck.csv: decks/{args.deck}.py ({deck.size} cards)")

    shutil.copytree(sdk_dir / "cg", build / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "agents" / "sa", build / "sa",
                    ignore=shutil.ignore_patterns("__pycache__"))
    keep_policy = args.nets in ("both", "policy")
    keep_value = args.nets in ("both", "value")
    for npz, keep in (("policy_net.npz", keep_policy),
                      ("value_net.npz", keep_value)):
        if not keep:
            (build / "sa" / npz).unlink(missing_ok=True)
    if args.policy_net:
        src = Path(args.policy_net)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit(f"--policy-net not found: {src}")
        if not keep_policy:
            raise SystemExit("--policy-net needs --nets both|policy")
        shutil.copy2(src, build / "sa" / "policy_net.npz")
        print(f"  sa/policy_net.npz <- {src}")
        # The dim guard silently returns None for a net whose feature layout
        # this code cannot feed, and the agent then falls back to
        # list(range(minCount)) -- i.e. a broken agent that still "runs".
        # Verifying HERE turns that into a build failure. (See policynet.load.)
        # sa.cards imports cg.sim, so the SDK has to be on the path first. Done
        # here rather than at module import to keep the builder light.
        from ptcg.env import sdk as _sdk
        _sdk.load()
        from sa import policynet as _pn
        if _pn.load(build / "sa" / "policy_net.npz") is None:
            raise SystemExit(f"{src} FAILS the dim guard -- it would silently "
                             "fall back to random-legal on Kaggle")
        print("  dim guard: net loads OK")
    if args.ensemble_net:
        if not keep_policy:
            raise SystemExit("--ensemble-net needs --nets both|policy")
        from ptcg.env import sdk as _sdk2
        _sdk2.load()
        from sa import policynet as _pn2
        for i, extra_net in enumerate(args.ensemble_net):
            src = Path(extra_net)
            if not src.is_absolute():
                src = ROOT / src
            if not src.exists():
                raise SystemExit(f"--ensemble-net not found: {src}")
            dst = build / "sa" / f"policy_net_{i + 1}.npz"
            shutil.copy2(src, dst)
            if _pn2.load(dst) is None:
                raise SystemExit(f"{src} FAILS the dim guard")
            print(f"  sa/{dst.name} <- {src} (dim guard OK)")
        # The members must be DIFFERENT policies. Two files holding one policy
        # would give it two votes -- silently weighting a vote whose whole
        # point is one member one vote (E9: policy_v5c_s1 is 100% identical to
        # policy_v5_s1 despite a different md5).
        import hashlib
        seen: dict[str, str] = {}
        for f in ["policy_net.npz"] + [f"policy_net_{i + 1}.npz"
                                       for i in range(len(args.ensemble_net))]:
            digest = hashlib.md5((build / "sa" / f).read_bytes()).hexdigest()
            if digest in seen:
                raise SystemExit(
                    f"ensemble members {seen[digest]} and {f} are the SAME "
                    "bytes -- that is a weighted vote, not an ensemble")
            seen[digest] = f
        print(f"  ensemble: {len(seen)} distinct members")
    for extra in ("value_net.npz", "policy_net.npz", "deck_library.json"):
        present = (build / "sa" / extra).exists()
        note = "present" if present else "excluded -> handcrafted fallback"
        print(f"  sa/{extra}: {note}")

    # Before tarring: would the GRADER's interpreter even parse this? The smoke
    # below runs on the local one and cannot answer that (see KAGGLE_PY).
    _check_kaggle_syntax(build)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (config.DIST_DIR /
           f"submission_{args.agent}-{args.deck}-nets{args.nets}_{stamp}.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for item in sorted(build.iterdir()):
            tar.add(item, arcname=item.name)
    shutil.rmtree(build)

    size_mib = out.stat().st_size / (1024 * 1024)
    print(f"built {out} ({size_mib:.1f} MiB)")
    if size_mib > SIZE_CAP_MIB:
        raise SystemExit(f"exceeds cap {SIZE_CAP_MIB} MiB")

    if not args.no_smoke:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(out) as tar:
                tar.extractall(tmp)
            env = dict(os.environ,
                       SMOKE_MEMBERS=str(1 + len(args.ensemble_net)))
            proc = subprocess.run([sys.executable, "-X", "utf8", "-c", SMOKE],
                                  cwd=tmp, capture_output=True, text=True,
                                  timeout=1800, env=env)
            ok = proc.returncode == 0 and "RESULT=" in proc.stdout
            if args.agent == "bc" and "NET_OK" not in proc.stdout:
                ok = False
                print("  smoke: net was NOT live in the extracted bundle")
            print(f"  smoke: {'OK' if ok else 'FAILED'}")
            for line in proc.stdout.strip().splitlines()[-4:]:
                print(f"    {line}")
            if not ok:
                print(proc.stderr[-3000:])
                raise SystemExit("smoke test failed")

    latest = config.DIST_DIR / "submission.tar.gz"
    shutil.copy2(out, latest)
    print(f"latest -> {latest}\nupload: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/pckhoa/model.py
====================================================================================================
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PokemonFeatureExtractor(BaseFeaturesExtractor):
    """Custom PyTorch Feature Extractor for encoding Pokemon TCG state space."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Simple multi-layer perceptron to encode the 1D state representation vector
        self.encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.encoder(observations)


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/pckhoa/main.py
====================================================================================================
import os
import random

from cg.api import (
    Observation,
    to_observation_class,
    OptionType,
    SelectType,
    SelectContext,
    AreaType,
    CardType,
    all_card_data,
    all_attack,
    search_begin,
    search_step,
    search_end,
    search_release
)

# Global card and attack database maps for Heuristic logic
try:
    CARD_DATA_MAP = {c.cardId: c for c in all_card_data()}
except Exception:
    CARD_DATA_MAP = {}

try:
    ATTACK_DMG_MAP = {a.attackId: a.damage for a in all_attack()}
except Exception:
    ATTACK_DMG_MAP = {}

def _read_deck_csv_top() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r") as file:
            csv = file.read().split("\n")
        return [int(line) for line in csv if line.strip()]
    except Exception:
        return []

try:
    _CURRENT_DECK = _read_deck_csv_top()
except Exception:
    _CURRENT_DECK = []

def is_playing_abomasnow() -> bool:
    return 723 in _CURRENT_DECK

# Global variables for ONNX inference session
_ONNX_SESSION = None
_ONNX_LOADED = False

def init_onnx_model():
    """Attempt to initialize the ONNX model for deep reinforcement learning inference."""
    global _ONNX_SESSION, _ONNX_LOADED
    if _ONNX_LOADED:
        return
        
    onnx_path = "model.onnx"
    if not os.path.exists(onnx_path):
        # Check Kaggle submission directory
        onnx_path = "/kaggle_simulations/agent/" + onnx_path
        
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            # Use CPU execution provider for CPU efficiency and safety on Kaggle
            _ONNX_SESSION = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            _ONNX_LOADED = True
            print("ONNX model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load ONNX model ({e}). Falling back to Heuristic Agent.")
            _ONNX_LOADED = False
    else:
        _ONNX_LOADED = False

def extract_state_for_onnx(obs: Observation) -> list:
    """Extract flat state vector of size 164 matching the DRL Env extraction."""
    state = [0.0] * 164
    if not obs or not obs.current:
        return state
        
    your_idx = obs.current.yourIndex
    player = obs.current.players[your_idx]
    opponent = obs.current.players[1 - your_idx]
    
    # Player 0 (Agent) features
    idx = 0
    # Active Pokemon
    active = player.active[0] if player.active else None
    state[idx] = active.hp / active.maxHp if active else 0.0; idx += 1
    state[idx] = len(active.energies) / 5.0 if active else 0.0; idx += 1
    state[idx] = active.id / 1500.0 if active else 0.0; idx += 1
    
    # Bench Pokemon (max 5)
    for i in range(5):
        pkmn = player.bench[i] if i < len(player.bench) else None
        state[idx] = pkmn.hp / pkmn.maxHp if pkmn else 0.0; idx += 1
        state[idx] = len(pkmn.energies) / 5.0 if pkmn else 0.0; idx += 1
        state[idx] = pkmn.id / 1500.0 if pkmn else 0.0; idx += 1
        
    # Hand & Deck counts
    state[idx] = len(player.hand or []) / 10.0; idx += 1
    state[idx] = player.deckCount / 60.0; idx += 1
    state[idx] = len(player.prize) / 6.0; idx += 1
    
    # Hand Cards (max 10)
    hand_cards = player.hand or []
    for i in range(10):
        card = hand_cards[i] if i < len(hand_cards) else None
        card_data = CARD_DATA_MAP.get(card.id) if card else None
        state[idx] = card.id / 1500.0 if card else 0.0; idx += 1
        state[idx] = int(card_data.cardType) / 7.0 if card_data else 0.0; idx += 1
    
    # Player 1 (Opponent) features
    opp_active = opponent.active[0] if opponent.active else None
    state[idx] = opp_active.hp / opp_active.maxHp if opp_active else 0.0; idx += 1
    state[idx] = len(opp_active.energies) / 5.0 if opp_active else 0.0; idx += 1
    state[idx] = opp_active.id / 1500.0 if opp_active else 0.0; idx += 1
    
    for i in range(5):
        pkmn = opponent.bench[i] if i < len(opponent.bench) else None
        state[idx] = pkmn.hp / pkmn.maxHp if pkmn else 0.0; idx += 1
        state[idx] = len(pkmn.energies) / 5.0 if pkmn else 0.0; idx += 1
        state[idx] = pkmn.id / 1500.0 if pkmn else 0.0; idx += 1
        
    state[idx] = opponent.handCount / 10.0; idx += 1
    state[idx] = opponent.deckCount / 60.0; idx += 1
    state[idx] = len(opponent.prize) / 6.0; idx += 1
    
    # Game Phase / Turn features
    state[idx] = obs.current.turn / 50.0; idx += 1
    state[idx] = obs.select.context / 50.0 if obs.select else 0.0; idx += 1
    
    # Options features
    options = obs.select.option if (obs.select and obs.select.option) else []
    for i in range(50):
        if i < len(options):
            opt = options[i]
            state[idx] = opt.type / 16.0; idx += 1
            state[idx] = (getattr(opt, "attackId", 0) or 0) / 1200.0; idx += 1
        else:
            idx += 2 # pad with zeros
            
    return state

def run_onnx_inference(obs: Observation) -> list[int]:
    """Execute ONNX model inference and return best action indices."""
    global _ONNX_SESSION
    import numpy as np
    
    state_vec = extract_state_for_onnx(obs)
    
    # Expand dims to batch size 1 (1, 144)
    input_data = np.expand_dims(np.array(state_vec, dtype=np.float32), axis=0)
    
    # Feed to ONNX session
    input_name = _ONNX_SESSION.get_inputs()[0].name
    output_name = _ONNX_SESSION.get_outputs()[0].name
    logits = _ONNX_SESSION.run([output_name], {input_name: input_data})[0][0] # shape (50,)
    
    options = obs.select.option
    max_count = obs.select.maxCount
    your_idx = obs.current.yourIndex
    
    # 1. Primary choice is chosen purely by the ONNX RL Model
    best_logit = -999999.0
    primary_choice = 0
    for i in range(len(options)):
        val = float(logits[i]) if i < len(logits) else -999999.0
        if val > best_logit:
            best_logit = val
            primary_choice = i
            
    choices = [primary_choice]
    
    # 2. Secondary choices are chosen by Heuristic (matching training env logic)
    if max_count > 1 and len(options) > 1:
        scored_options = []
        for i in range(len(options)):
            if i == primary_choice:
                continue
            try:
                score = score_option(obs, options[i], obs.select.context, your_idx)
            except Exception:
                score = 100.0
            scored_options.append((score, i))
            
        scored_options.sort(key=lambda x: x[0], reverse=True)
        for _, idx in scored_options[:max_count - 1]:
            choices.append(idx)
            
    return choices

def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def get_pokemon_from_option(obs: Observation, opt, your_idx: int):
    player_idx = opt.playerIndex if getattr(opt, "playerIndex", None) is not None else your_idx
    player = obs.current.players[player_idx]
    
    # 1. Check inPlayArea and inPlayIndex
    in_play_area = getattr(opt, "inPlayArea", None)
    in_play_index = getattr(opt, "inPlayIndex", None)
    if in_play_area is not None:
        if in_play_area == AreaType.ACTIVE:
            return player.active[0] if player.active else None
        elif in_play_area == AreaType.BENCH:
            if in_play_index is not None and 0 <= in_play_index < len(player.bench):
                return player.bench[in_play_index]
                
    # 2. Check area and index (when the option itself represents a card in play)
    area = getattr(opt, "area", None)
    index = getattr(opt, "index", None)
    if area is not None:
        if area == AreaType.ACTIVE:
            return player.active[0] if player.active else None
        elif area == AreaType.BENCH:
            if index is not None and 0 <= index < len(player.bench):
                return player.bench[index]
                
    return None

def get_card_id(obs: Observation, opt, your_idx: int) -> int:
    # Check cardId directly
    card_id = getattr(opt, "cardId", None)
    if card_id is not None and card_id != 0:
        return card_id
        
    # Check card in area
    area = getattr(opt, "area", None)
    index = getattr(opt, "index", None)
    player_idx = opt.playerIndex if getattr(opt, "playerIndex", None) is not None else your_idx
    
    if area is not None and index is not None:
        player = obs.current.players[player_idx]
        if area == AreaType.HAND:
            if player.hand and 0 <= index < len(player.hand):
                return player.hand[index].id
        elif area == AreaType.LOOKING:
            if obs.current.looking and 0 <= index < len(obs.current.looking):
                card = obs.current.looking[index]
                return card.id if card else 0
        elif area == AreaType.DISCARD:
            if player.discard and 0 <= index < len(player.discard):
                return player.discard[index].id
        elif area == AreaType.DECK:
            if obs.select and obs.select.deck and 0 <= index < len(obs.select.deck):
                return obs.select.deck[index].id
        elif area == AreaType.ACTIVE:
            if player.active:
                return player.active[0].id
        elif area == AreaType.BENCH:
            if 0 <= index < len(player.bench):
                return player.bench[index].id
                
    return 0

def get_max_attack_damage(obs, your_idx: int) -> int:
    """Calculate maximum damage our active Pokemon can deal."""
    player = obs.current.players[your_idx]
    active_pkmn = player.active[0] if player.active else None
    if not active_pkmn:
        return 0
    if active_pkmn.id == 46: # Gouging Fire ex
        return 260 # Blaze Blitz
    elif active_pkmn.id == 31: # Chi-Yu
        return 60 # Ground Melter
    elif active_pkmn.id == 77: # Litten
        return 10
    elif active_pkmn.id == 97: # Litwick
        return 20
    elif active_pkmn.id == 76: # Slugma
        return 10
    elif active_pkmn.id == 723: # Mega Abomasnow ex
        return 200 # Hammer-lanche / Frost Barrier
    elif active_pkmn.id == 722: # Snover
        return 30 # Icy Snow
    return 10

def calculate_enemy_max_damage_next_turn(obs: Observation, opp_idx: int) -> int:
    """Safely estimate the maximum attack damage of the opponent active Pokemon next turn."""
    opponent = obs.current.players[opp_idx]
    opp_active = opponent.active[0] if opponent.active else None
    if not opp_active:
        return 0
    card_data = CARD_DATA_MAP.get(opp_active.id)
    if not card_data or not card_data.attacks:
        return 30 # default baseline damage
        
    max_dmg = 0
    for attack_id in card_data.attacks:
        dmg = ATTACK_DMG_MAP.get(attack_id, 0)
        if dmg > max_dmg:
            max_dmg = dmg
    return max_dmg if max_dmg > 0 else 30

def score_option(obs, opt, context, your_idx: int) -> float:
    opt_type = opt.type
    player = obs.current.players[your_idx]
    opponent = obs.current.players[1 - your_idx]
    
    score = 100.0
    card_id = get_card_id(obs, opt, your_idx)
    
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        if card_id in (46, 723): score = 1000.0
        elif card_id in (31, 722): score = 500.0
        else: score = 100.0
            
    elif context == SelectContext.SETUP_BENCH_POKEMON:
        if card_id in (46, 723): score = 1000.0
        elif card_id in (31, 77, 97, 76, 722): score = 800.0
        else: score = 100.0
            
    elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        pkmn = get_pokemon_from_option(obs, opt, your_idx)
        if pkmn:
            energy_count = len(pkmn.energies)
            if pkmn.id in (46, 723):
                score = 10000.0 + energy_count * 1000.0 + pkmn.hp
            elif pkmn.id == 31:
                score = 8000.0 + energy_count * 1000.0 + pkmn.hp
            else:
                score = 500.0 + pkmn.hp
        else:
            score = 100.0
            
    elif context == SelectContext.ATTACH_FROM:
        pkmn = get_pokemon_from_option(obs, opt, your_idx)
        if pkmn:
            energy_count = len(pkmn.energies)
            if pkmn.id in (46, 723) and energy_count < 3:
                score = 3200.0 if opt.area == AreaType.ACTIVE else 3000.0
            elif pkmn.id == 31 and energy_count < 2:
                score = 2900.0 if opt.area == AreaType.ACTIVE else 2400.0
            else:
                score = 1000.0
        else:
            score = 100.0
            
    elif context == SelectContext.ATTACH_TO:
        if card_id in (2, 3): score = 1000.0
        else: score = 100.0
            
    elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        if card_id in (46, 723): score = 3000.0
        elif card_id == 31: score = 2500.0
        elif card_id in (1235, 1205, 1227): score = 1800.0
        elif card_id in (2, 3): score = 1000.0
        else: score = 500.0
            
    elif context in (SelectContext.ACTIVATE, SelectContext.MULLIGAN, SelectContext.COIN_HEAD, SelectContext.IS_FIRST) or opt_type in (OptionType.YES, OptionType.NO):
        if opt_type == OptionType.YES: score = 1000.0
        elif opt_type == OptionType.NO: score = 100.0
            
    elif context == SelectContext.MAIN:
        max_dmg = get_max_attack_damage(obs, your_idx)
        opp_active = opponent.active[0] if opponent.active else None
        opp_hp = opp_active.hp if opp_active else 999
        can_ko_active = (max_dmg >= opp_hp)
        
        active_pkmn = player.active[0] if player.active else None
        enemy_max_dmg = calculate_enemy_max_damage_next_turn(obs, 1 - your_idx)
        is_in_lethal_range = active_pkmn and active_pkmn.hp <= enemy_max_dmg
        
        if opt_type == OptionType.ATTACK:
            if can_ko_active: score = 15000.0 
            else: score = 7000.0 
                
        elif opt_type == OptionType.EVOLVE:
            score = 9500.0
            
        elif opt_type == OptionType.ATTACH:
            pkmn = get_pokemon_from_option(obs, opt, your_idx)
            if is_in_lethal_range and opt.inPlayArea == AreaType.ACTIVE:
                score = 3000.0
            else:
                card = None
                if opt.area == AreaType.HAND and player.hand and 0 <= opt.index < len(player.hand):
                    card = player.hand[opt.index]
                if card and pkmn:
                    if card.id in (2, 3): 
                        energy_count = len(pkmn.energies)
                        if opt.inPlayArea == AreaType.ACTIVE:
                            if (pkmn.id == 46 or pkmn.id == 723) and energy_count < 3: score = 8800.0
                            elif pkmn.id == 31 and energy_count < 2: score = 8750.0
                            else: score = 1000.0
                        elif opt.inPlayArea == AreaType.BENCH:
                            if (pkmn.id == 46 or pkmn.id == 723) and energy_count < 3: score = 8600.0
                            elif pkmn.id == 31 and energy_count < 2: score = 8500.0
                            else: score = 900.0
                
        elif opt_type == OptionType.PLAY:
            card = get_card_id(obs, opt, your_idx)
            if card == 1205: # Cyrano (Search ex)
                has_ex = False
                active_pkmn = player.active[0] if player.active else None
                if active_pkmn and active_pkmn.id in (46, 723):
                    has_ex = True
                for pkmn in player.bench:
                    if pkmn.id in (46, 723):
                        has_ex = True
                for c in (player.hand or []):
                    if c.id in (46, 723):
                        has_ex = True
                
                # If we don't have any ex, Cyrano is top priority. Otherwise, save supporter turn for drawing.
                score = 9400.0 if not has_ex else 100.0
            elif card == 1235: # Waitress (Draw)
                score = 9400.0
            elif card == 1227: # Lillie (Draw)
                score = 9300.0
            elif card == 1145: # Mega Signal
                score = 9500.0 if is_playing_abomasnow() else 100.0
            else: score = 8000.0
                
        elif opt_type == OptionType.RETREAT:
            score = 100.0
                
    return score

def predict_card_lists(obs: Observation):
    your_idx = obs.current.yourIndex
    player = obs.current.players[your_idx]
    opponent = obs.current.players[1 - your_idx]
    
    # 1. Calculate remaining cards in our deck & prize
    start_deck = list(_CURRENT_DECK)
    if not start_deck:
        start_deck = [3] * 60
        
    known_cards = []
    if player.active and player.active[0] is not None:
        known_cards.append(player.active[0].id)
        known_cards.extend([c.id for c in player.active[0].energyCards])
    for pk in player.bench:
        if pk is not None:
            known_cards.append(pk.id)
            known_cards.extend([c.id for c in pk.energyCards])
    if player.hand:
        known_cards.extend([c.id for c in player.hand])
    if player.discard:
        known_cards.extend([c.id for c in player.discard])
        
    remaining_pool = list(start_deck)
    for cid in known_cards:
        if cid in remaining_pool:
            remaining_pool.remove(cid)
            
    prize_count = len(player.prize)
    your_prize = remaining_pool[:prize_count]
    your_deck = remaining_pool[prize_count:]
    
    while len(your_prize) < prize_count:
        your_prize.append(3)
    while len(your_deck) < player.deckCount:
        your_deck.append(3)
    your_deck = your_deck[:player.deckCount]
    
    # 2. Opponent predictions
    opp_start_deck = list(_CURRENT_DECK)
    if not opp_start_deck:
        opp_start_deck = [3] * 60
        
    opp_known = []
    if opponent.active and opponent.active[0] is not None:
        opp_known.append(opponent.active[0].id)
        opp_known.extend([c.id for c in opponent.active[0].energyCards])
    for pk in opponent.bench:
        if pk is not None:
            opp_known.append(pk.id)
            opp_known.extend([c.id for c in pk.energyCards])
    if opponent.discard:
        opp_known.extend([c.id for c in opponent.discard])
        
    opp_remaining = list(opp_start_deck)
    for cid in opp_known:
        if cid in opp_remaining:
            opp_remaining.remove(cid)
            
    opp_prize_count = len(opponent.prize)
    opp_hand_count = opponent.handCount
    
    opponent_prize = opp_remaining[:opp_prize_count]
    opponent_hand = opp_remaining[opp_prize_count : opp_prize_count + opp_hand_count]
    opponent_deck = opp_remaining[opp_prize_count + opp_hand_count :]
    
    while len(opponent_prize) < opp_prize_count:
        opponent_prize.append(3)
    while len(opponent_hand) < opp_hand_count:
        opponent_hand.append(3)
    while len(opponent_deck) < opponent.deckCount:
        opponent_deck.append(3)
    opponent_deck = opponent_deck[:opponent.deckCount]
    
    opponent_active = []
    active = opponent.active
    if len(active) > 0 and active[0] is None:
        basic_id = 722 if is_playing_abomasnow() else 77
        for cid in opp_remaining:
            card_data = CARD_DATA_MAP.get(cid)
            if card_data and card_data.basic:
                basic_id = cid
                break
        opponent_active = [basic_id]
        
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.
    
    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()
        
    # --- PHASE 0: LOOK-AHEAD SEARCH VERIFICATION ---
    options = obs.select.option
    max_count = obs.select.maxCount
    your_idx = obs.current.yourIndex
    
    if max_count == 1 and options and len(options) > 1:
        best_search_score = -999999.0
        best_search_idx = -1
        
        try:
            your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active = predict_card_lists(obs)
            
            for i in range(len(options)):
                try:
                    root = search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)
                    child = search_step(root.searchId, [i])
                    
                    child_obs = child.observation
                    child_current = child_obs.current
                    child_result = child_current.result if child_current else -1
                    
                    base_score = score_option(obs, options[i], obs.select.context, your_idx)
                    
                    if child_result == your_idx:
                        search_score = 999999.0
                    elif child_result != -1:
                        search_score = -999999.0
                    else:
                        search_score = base_score
                        player_before = obs.current.players[your_idx]
                        player_after = child_current.players[your_idx] if child_current else None
                        
                        if player_after:
                            prizes_before = sum(1 for p in player_before.prize if p is not None)
                            prizes_after = sum(1 for p in player_after.prize if p is not None)
                            if prizes_after < prizes_before:
                                search_score += 5000.0 * (prizes_before - prizes_after)
                                
                            active_after = player_after.active if player_after else []
                            if not active_after or active_after[0] is None:
                                search_score -= 3000.0
                                
                    search_release(child.searchId)
                    
                    if search_score > best_search_score:
                        best_search_score = search_score
                        best_search_idx = i
                except Exception:
                    pass
            
            search_end()
            if best_search_idx != -1 and best_search_score > -500000.0:
                return [best_search_idx]
        except Exception:
            try:
                search_end()
            except Exception:
                pass
                
    # Attempt DRL inference if ONNX is available and loaded
    try:
        init_onnx_model()
        if _ONNX_LOADED:
            # --- PHASE 1: HYBRID HEURISTIC OVERRIDE ---
            options = obs.select.option
            max_count = obs.select.maxCount
            your_idx = obs.current.yourIndex
            
            best_score = -999999.0
            best_idx = 0
            
            if max_count == 1 and len(options) > 1:
                for i, opt in enumerate(options):
                    try:
                        score = score_option(obs, opt, obs.select.context, your_idx)
                    except Exception:
                        score = 100.0
                    if score > best_score:
                        best_score = score
                        best_idx = i
                if best_score >= 12000.0:
                    # GOD MOVE DETECTED! OVERRIDE RL!
                    return [best_idx]
            
            # --- PHASE 2: ONNX INFERENCE ---
            return run_onnx_inference(obs)
    except Exception as e:
        # Fallback to heuristics silently
        pass
    
    # --- HEURISTIC FALLBACK AGENT ---
    context = obs.select.context
    options = obs.select.option
    your_idx = obs.current.yourIndex
    
    # Score all options
    scored_options = []
    for i, opt in enumerate(options):
        try:
            score = score_option(obs, opt, context, your_idx)
        except Exception:
            score = 100.0
        scored_options.append((score, i))
        
    # Sort options by score descending
    scored_options.sort(key=lambda x: x[0], reverse=True)
    
    # Select maxCount elements
    k = obs.select.maxCount
    selected_indices = [idx for score, idx in scored_options[:k]]
    
    return selected_indices


====================================================================================================
FILE: /kaggle/working/ptcg_research_index/cloned_repos/pckhoa/train.py
====================================================================================================
import os
import sys
import argparse

# Check requirements
try:
    import gymnasium as gym
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
except ImportError as e:
    print(f"Error: Required library is missing ({e}).")
    print("Please install requirements using the following command:")
    print("  pip install gymnasium stable-baselines3 sb3-contrib onnx onnxruntime torch")
    sys.exit(1)

import numpy as np
from env import PokemonTCGEnv
from model import PokemonFeatureExtractor
from main import agent as heuristic_agent_main, score_option
from cg.api import to_observation_class

class OpponentPolicy:
    """Wrapper for selecting opponent moves during self-play and evaluation."""
    def __init__(self, mode="heuristic", model=None):
        self.mode = mode
        self.model = model

    def __call__(self, obs_dict):
        select_data = obs_dict.get("select")
        if not select_data:
            return []
        
        # Mode 1: Heuristic opponent
        if self.mode == "heuristic":
            try:
                # Wrap the existing main.py agent function
                return heuristic_agent_main(obs_dict)
            except Exception:
                # Fallback to random
                return self._random_policy(select_data)
                
        # Mode 2: Self-play neural network opponent
        elif self.mode == "rl" and self.model is not None:
            try:
                # Extract state representation
                env_temp = PokemonTCGEnv()
                obs_obj = to_observation_class(obs_dict)
                state_vec = env_temp._extract_state(obs_obj)
                
                # Get action mask
                options = select_data.get("option", [])
                mask = np.zeros(50, dtype=bool)
                for i in range(min(len(options), 50)):
                    mask[i] = True
                    
                # Predict action
                action, _ = self.model.predict(state_vec, action_masks=mask, deterministic=True)
                
                # Fill remaining actions using heuristics if maxCount > 1
                max_count = select_data.get("maxCount", 1)
                primary_choice = min(int(action), len(options) - 1)
                choices = [primary_choice]
                
                if max_count > 1 and len(options) > 1:
                    your_idx = obs_obj.current.yourIndex
                    scored_options = []
                    for i in range(len(options)):
                        if i == primary_choice:
                            continue
                        try:
                            score = score_option(obs_obj, options[i], obs_obj.select.context, your_idx)
                        except Exception:
                            score = 100.0
                        scored_options.append((score, i))
                    scored_options.sort(key=lambda x: x[0], reverse=True)
                    for _, idx in scored_options[:max_count - 1]:
                        choices.append(idx)
                return choices[:max_count]
            except Exception:
                return self._random_policy(select_data)
                
        return self._random_policy(select_data)

    def _random_policy(self, select_data):
        max_count = select_data.get("maxCount", 1)
        options_len = len(select_data.get("option", []))
        import random
        return random.sample(list(range(options_len)), max_count)

def evaluate_agent(eval_env, model, eval_games=20) -> float:
    """Evaluate current RL model against Heuristic baseline."""
    wins = 0
    for _ in range(eval_games):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action_masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
        current = eval_env.unwrapped.obs_dict.get("current") if eval_env.unwrapped.obs_dict is not None else None
        if current and current.get("result") == 0:
            wins += 1
            
    win_rate = wins / eval_games
    return win_rate

def export_to_onnx(model, onnx_path="model.onnx"):
    """Export the trained SB3 PPO model to ONNX for lightweight inference on Kaggle."""
    import torch as th
    
    # Define a helper wrapper class to output policy action probabilities
    class ONNXWrapper(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.extractor = policy.features_extractor
            self.mlp = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, x):
            features = self.extractor(x)
            latent_pi, _ = self.mlp(features)
            action_logits = self.action_net(latent_pi)
            return action_logits

    # Wrap the policy
    onnx_wrapper = ONNXWrapper(model.policy)
    onnx_wrapper.eval()
    
    # Create dummy input tensor on the same device as the model
    device = next(model.policy.parameters()).device
    dummy_input = th.randn(1, 164, device=device)
    
    # Export to ONNX file
    th.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Successfully exported model to ONNX: {onnx_path} (size ~ {os.path.getsize(onnx_path) / 1024:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description="Self-play DRL training pipeline for Pokemon TCG.")
    parser.add_argument("--steps", type=int, default=50000, help="Total number of steps to train.")
    args = parser.parse_args()

    # 1. Setup Environment
    # Default opponent is the Heuristic baseline agent
    opp_policy = OpponentPolicy(mode="heuristic")
    env = PokemonTCGEnv(opponent_policy=opp_policy)

    # 2. Define Custom Neural Network policy model structure
    policy_kwargs = dict(
        features_extractor_class=PokemonFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # 3. Initialize Masked PPO Agent
    print("Initializing MaskablePPO Agent...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99
    )
    
    # Attach model back to env for evaluation wrapper
    env.model = model

    # 4. Training loop with self-play evaluation
    total_steps = args.steps
    steps_per_epoch = 10000
    current_step = 0
    opponent_mode = "heuristic"

    # Create a completely separate environment for evaluation to avoid state pollution
    eval_env = PokemonTCGEnv(opponent_policy=OpponentPolicy(mode="heuristic"))

    print(f"Starting Training: Total steps = {total_steps}, Epoch steps = {steps_per_epoch}")
    
    while current_step < total_steps:
        # Train for 1 epoch
        model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)
        current_step += steps_per_epoch
        print(f"Completed step {current_step}/{total_steps}")
        
        # Evaluate performance against heuristic baseline using the dedicated eval_env
        win_rate = evaluate_agent(eval_env, model, eval_games=20)
        print(f"Evaluation against Heuristic Baseline: Win Rate = {win_rate * 100:.1f}%")
        
        # Check if we should update opponent for Self-play
        if win_rate > 0.60:
            print("RL Agent achieved > 60% win rate! Updating opponent to current RL checkpoint (Self-play)...")
            opp_policy.mode = "rl"
            opp_policy.model = model
            opponent_mode = "self-play (rl)"
        else:
            print("Win rate under 60%. Continuing training against current opponent...")
            
        # Re-set training opponent to active self-play or baseline pool
        env.opponent_policy = opp_policy
        
        # Force reset the training environment and update model's last observation buffer
        # to clear any C++ global state pollution caused by the evaluation environment games.
        model._last_obs = model.env.reset()
        
        # Save checkpoints
        model.save("model_checkpoint.zip")
        
    print("Training Complete. Saving final model and exporting to ONNX...")
    model.save("model_final.zip")
    export_to_onnx(model, "model.onnx")

if __name__ == "__main__":
    main()

```