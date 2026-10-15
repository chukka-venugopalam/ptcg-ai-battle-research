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
        # Vocab checkpoints map unseen IDs to UNK row 1.
        return np.where(a < len(lut), lut[np.clip(a, 0, len(lut) - 1)], 1)

    def _emb_row(self, table: str, ids):
        """Map IDs to safe rows even for pre-v7 identity-mapped checkpoints.

        The current R2 card pool contains IDs beyond the 1300-row embedding
        tables used by this checkpoint. State/bag feature builders already map
        those IDs to row 0; option card and target embeddings must do the same
        rather than raising IndexError and forcing the agent's broad fallback.
        Vocab-remapped checkpoints preserve their normal UNK mapping.
        """
        row = np.asarray(self._m(table, ids))
        size = getattr(self, table).shape[0]
        if self.lut is not None:
            # _m maps OOV IDs to row 1, but keep the guard for malformed maps.
            return np.where((row >= 0) & (row < size), row, 1)
        return np.where((row >= 0) & (row < size), row, 0)

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
            oenc[i, ow:ow + emb] = self.card_emb[self._emb_row("card_emb", cid)]
            oenc[i, ow + emb:ow + 2 * emb] = self.atk_emb[
                self._emb_row("atk_emb", aid)]
            oenc[i, ow + 2 * emb:] = self.card_emb[self._emb_row("card_emb", tid)]

        dense, bags = featurize(state, me)
        parts = [dense,
                 self.slot_emb[self._emb_row("slot_emb", bags["slots"])].reshape(-1)]
        for name in _BAGS:
            b = bags[name]
            parts.append(self.bag_emb[self._emb_row("bag_emb", b)].mean(axis=0)
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
        parts.append(self.slot_emb[self._emb_row("slot_emb", xids)].reshape(-1))
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
