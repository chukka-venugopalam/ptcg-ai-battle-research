"""Mirror-check tests on REAL engine observations (captured from fixture agents; engine required, never skipped)."""
import json, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import mirror_check as M


@pytest.fixture(scope="module")
def obs():
    o = M.capture(3, None, seed=11)
    assert len(o) > 100
    return o


def test_mirror_is_involution(obs):
    for o in obs[:300]:
        assert M.mirror_obs(M.mirror_obs(o)) == json.loads(json.dumps(o))


def test_mirror_swaps_perspective_but_keeps_own_state(obs):
    o = next(x for x in obs if x["current"]["yourIndex"] == 1 and x["current"]["players"][1]["hand"])
    m = M.mirror_obs(o)
    assert m["current"]["yourIndex"] == 0 and m["current"]["players"][0]["deckCount"] == o["current"]["players"][1]["deckCount"]
    assert len(m["current"]["players"][0]["hand"]) == len(o["current"]["players"][1]["hand"])
    assert all(c["playerIndex"] == 0 for c in m["current"]["players"][0]["hand"])      # ownership indices flipped with the player


def sym_agent():                                                         # uses players[yourIndex]; seat-symmetric by construction
    def ag(o):
        s = o["select"]
        if s is None: return [0] * 60
        me = o["current"]["players"][o["current"]["yourIndex"]]
        n = len(s["option"]); k = max(s["minCount"], min(s["maxCount"], n))
        return list(range(n - k, n)) if me["handCount"] % 2 else list(range(k))
    return ag


def biased_agent():                                                      # BUG PATTERN: treats players[0] as "me"
    def ag(o):
        s = o["select"]
        if s is None: return [0] * 60
        me = o["current"]["players"][0]
        n = len(s["option"]); k = max(s["minCount"], min(s["maxCount"], n))
        return list(range(n - k, n)) if me["handCount"] % 2 else list(range(k))
    return ag


def test_symmetric_agent_passes(obs):
    r = M.check_agent(sym_agent, obs)
    assert r["n_decisions"] > 100 and r["n_mismatch"] == 0


def test_seat_biased_agent_is_detected(obs):
    r = M.check_agent(biased_agent, obs)
    assert r["n_mismatch"] > 0 and all(seat in (0, 1) for seat, _ in r["by_seat_context"])
