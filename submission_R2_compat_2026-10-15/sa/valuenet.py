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
