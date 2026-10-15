"""Raw-dict wrapper over the engine's search API.

Bypasses cg.api's recursive dataclass conversion (the hot path) and works on
plain dicts straight from json.loads.
"""
from __future__ import annotations

import ctypes
import json

from cg.sim import lib

_agent_ptr = None


def _ptr():
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = lib.AgentStart()
    return _agent_ptr


def _arr(xs):
    return (ctypes.c_int * len(xs))(*[int(x) for x in xs])


class SearchError(RuntimeError):
    def __init__(self, where: str, code: int):
        super().__init__(f"{where} error {code}")
        self.code = code


def begin(sbi: str, my_deck, my_prize, opp_deck, opp_prize, opp_hand,
          opp_active, manual_coin: bool = False):
    """-> (search_id, observation_dict)."""
    bs = lib.SearchBegin(_ptr(), sbi.encode("ascii"), len(sbi),
                         _arr(my_deck), _arr(my_prize), _arr(opp_deck),
                         _arr(opp_prize), _arr(opp_hand), _arr(opp_active),
                         int(manual_coin))
    res = json.loads(bs.decode())
    if res.get("error"):
        raise SearchError("SearchBegin", res["error"])
    st = res["state"]
    return st["searchId"], st["observation"]


def step(search_id: int, select):
    """-> (search_id, observation_dict)."""
    bs = lib.SearchStep(_ptr(), ctypes.c_int64(search_id), _arr(select),
                        len(select))
    res = json.loads(bs.decode())
    if res.get("error"):
        raise SearchError("SearchStep", res["error"])
    st = res["state"]
    return st["searchId"], st["observation"]


def release(search_id: int) -> None:
    lib.SearchRelease(_ptr(), ctypes.c_int64(search_id))


def end() -> None:
    """Free all search memory (reused next search)."""
    lib.SearchEnd(_ptr())
