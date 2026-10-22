#!/usr/bin/env python3
"""mirror_check.py -- seat-symmetry (mirror) invariance test for a cabt agent (Mission 6, Phase C hypothesis 1).
Schema facts used (official docs matsuoinstitute.github.io/cabt/api.html, accessed 2026-10-17, + local engine):
  State.yourIndex/firstPlayer/result and every Card/Option/Log `playerIndex` are ABSOLUTE player indices; State.players is
  indexed by absolute player; `select.option` is relative to the selecting player and is NOT reordered by a mirror.
Mirror(obs): swap players[0]/[1], flip yourIndex/firstPlayer/result and every playerIndex.  A seat-symmetric agent must choose
the SAME option indices on obs and mirror(obs) (same RNG seed, stateless or fed the same sequence).  LIMITATIONS: the opaque
`search_begin_input` cannot be mirrored (agents that call search_begin are not testable this way); serial numbers are kept;
stochastic agents need a fixed seed; an agent that really should treat seats differently (e.g. IS_FIRST policy by seat
on purpose) will show mismatches that must be read, not auto-failed.
    mirror_check.py capture --out obs.json --games N [--main DIR/main.py]     (needs the real engine)
    mirror_check.py check --main DIR/main.py --obs obs.json"""
import argparse, collections, copy, json, os, random, sys


def _flip(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "playerIndex" and v in (0, 1): o[k] = 1 - v
            else: _flip(v)
        if o.get("type") == 23 and o.get("result") in (0, 1): o["result"] = 1 - o["result"]   # LogType.RESULT
    elif isinstance(o, list):
        for x in o: _flip(x)


def mirror_obs(obs: dict) -> dict:
    m = json.loads(json.dumps(obs)); cur = m.get("current")
    if cur:
        cur["yourIndex"] = 1 - cur["yourIndex"]
        if cur.get("firstPlayer", -1) in (0, 1): cur["firstPlayer"] = 1 - cur["firstPlayer"]
        if cur.get("result", -1) in (0, 1): cur["result"] = 1 - cur["result"]
        cur["players"] = cur["players"][::-1]
    _flip(cur); _flip(m.get("select")); _flip(m.get("logs"))
    return m


def check_agent(make_agent, observations, seed=0):
    """make_agent() -> fresh callable.  Feeds the recorded sequence to one instance and the mirrored sequence to another."""
    a, b = make_agent(), make_agent(); mism = collections.Counter(); n = 0; first = []
    for o in observations:
        if o.get("select") is None: continue
        random.seed(seed + n); x = list(a(copy.deepcopy(o)))
        random.seed(seed + n); y = list(b(mirror_obs(o)))
        n += 1
        if x != y:
            mism[(o["current"]["yourIndex"], o["select"]["context"])] += 1
            if len(first) < 3: first.append(dict(step=n, yourIndex=o["current"]["yourIndex"], ctx=o["select"]["context"], orig=x, mirrored=y))
    return dict(n_decisions=n, n_mismatch=sum(mism.values()), by_seat_context=dict(mism), examples=first)


def load_main(path):
    from kaggle_environments.agent import get_last_callable
    os.chdir(os.path.dirname(os.path.abspath(path)))                    # agents commonly read deck.csv relative to cwd
    return get_last_callable(open(path).read(), path=path)             # SDK rule: no __file__, LAST callable wins


def capture(games, main=None, seed=0, deck=None):
    from kaggle_environments import make
    if deck is not None: DECK = [int(x) for x in open(deck).read().replace(',', ' ').split()]
    else: from kaggle_environments.envs.cabt.cabt import deck as DECK
    rnd = random.Random(seed); obs_log = []

    def rec(inner):
        def ag(obs):
            if obs["select"] is not None: obs_log.append(json.loads(json.dumps(obs)))
            return inner(obs)
        return ag
    def rand_fixture(obs):
        s = obs["select"]
        if s is None: return list(DECK)
        n = len(s["option"]); return rnd.sample(range(n), max(s["minCount"], min(s["maxCount"], n)))
    for g in range(games):
        env = make("cabt", configuration={"bo": 3}); a = rec(load_main(main) if main else rand_fixture)
        env.run([a, rand_fixture] if g % 2 == 0 else [rand_fixture, a])
    return obs_log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture"); c.add_argument("--out", required=True); c.add_argument("--games", type=int, default=4); c.add_argument("--main"); c.add_argument("--deck")
    k = sp.add_parser("check"); k.add_argument("--main", required=True); k.add_argument("--obs", required=True)
    a = ap.parse_args()
    if a.cmd == "capture":
        o = capture(a.games, os.path.abspath(a.main) if a.main else None, deck=a.deck); json.dump(o, open(a.out, "w")); print(f"captured {len(o)} observations -> {a.out}")
    else:
        obs = json.load(open(a.obs)); m = os.path.abspath(a.main)
        r = check_agent(lambda: load_main(m), obs); print(json.dumps(r, indent=1)); sys.exit(0 if r["n_mismatch"] == 0 else 2)
