#!/usr/bin/env python3
"""Real-engine local probe (in-process, fixture agents only).  Purpose: (1) verify seat/first-player/BO3 semantics of the
INSTALLED engine, (2) cross-check canonical match_result_from_games against engine rewards, (3) save 5 local replays for
schema inspection.  NOT a strength test and NOT the live ladder.  Needs kaggle-environments (fails loudly otherwise)."""
import argparse, collections, gzip, hashlib, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
try:
    from kaggle_environments import make
    from kaggle_environments.envs.cabt.cabt import deck as DECK
except Exception as e:  # hard failure, never a silent fallback
    sys.exit(f"ENGINE UNAVAILABLE ({type(e).__name__}: {e}) -- refusing to run")
import ptcg_eval_canonical as C

IS_FIRST = 41
def pick(kind, mode, sel, rnd):
    n = len(sel["option"]); k = max(sel["minCount"], min(sel["maxCount"], n))
    if mode and sel["context"] == IS_FIRST:
        want = 1 if mode == "YES" else 2
        for i, o in enumerate(sel["option"]):
            if o["type"] == want: return [i]
    return rnd.sample(range(n), k) if kind == "random" else list(range(k))

def make_agent(kind, seat, ev, rnd, mode=None):
    def agent(obs):
        sel = obs["select"]
        if sel is None: return list(DECK)
        cur = obs["current"]; act = pick(kind, mode, sel, rnd)
        ev.append(dict(seat=seat, your=cur["yourIndex"], round=cur.get("round"), turn=cur["turn"], first=cur["firstPlayer"],
                       tally=[cur["players"][0]["win"], cur["players"][1]["win"], cur.get("draw", 0)], ctx=sel["context"],
                       stype=sel["type"], nopt=len(sel["option"]),
                       otypes=[o["type"] for o in sel["option"]] if sel["context"] == IS_FIRST else None, act=act))
        return act
    return agent

def games_from(ev, final):
    first_tally = {}
    for e in ev: first_tally.setdefault(e["round"], e["tally"])
    rs = sorted(first_tally); out = []
    for i, r in enumerate(rs):
        nxt = first_tally[rs[i + 1]] if i + 1 < len(rs) else list(final)
        d = [nxt[j] - first_tally[r][j] for j in range(3)]
        out.append({0: "W", 1: "L", 2: "D"}[next(j for j in range(3) if d[j] > 0)])   # seat-0 view
    return out

def run_match(k0, k1, mode, seed):
    rnd = random.Random(seed); ev = []
    env = make("cabt", configuration={"bo": 3})
    t0 = time.time()
    env.run([make_agent(k0, 0, ev, rnd, mode), make_agent(k1, 1, ev, rnd, mode)])
    fin = env.steps[-1]; res = list(env.result)
    rw = [s.reward for s in fin]; st = [s.status for s in fin]
    games = games_from(ev, res)
    exp = {1: "W", -1: "L", 0: "D"}.get(rw[0]) if rw[0] is not None else None
    try: rule = C.match_result_from_games(games)
    except ValueError as e: rule = f"ERR {e}"
    return env, dict(seed=seed, k0=k0, k1=k1, mode=mode, rewards=rw, statuses=st, result=res, games_seat0=games,
                     reward_result=exp, rule_result=rule, rule_ok=(exp == rule), n_steps=len(env.steps),
                     n_calls=len(ev), your_ne_seat=sum(e["your"] != e["seat"] for e in ev), wall_s=round(time.time() - t0, 2)), ev

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--n-probe", type=int, default=8); ap.add_argument("--n-smoke", type=int, default=30); a = ap.parse_args()
    os.makedirs(a.out + "/replays", exist_ok=True); rows = []; isf = []; saved = []; T = time.time()
    plan = [("random", "random", "YES" if i % 2 == 0 else "NO") for i in range(a.n_probe)]
    plan += [("random", "first", None) if i % 2 == 0 else ("first", "random", None) for i in range(a.n_smoke)]
    for i, (k0, k1, mode) in enumerate(plan):
        if time.time() - T > a.budget: print(f"BUDGET reached after {i} matches", file=sys.stderr); break
        env, row, ev = run_match(k0, k1, mode, 1000 + i); rows.append(row)
        for e in ev:
            if e["ctx"] == IS_FIRST: isf.append(dict(match=i, mode=mode, round=e["round"], chooser=e["your"], first=None, otypes=e["otypes"], act=e["act"]))
        fp = {}
        for e in ev:
            if e["first"] >= 0: fp.setdefault(e["round"], e["first"])
        for x in isf:
            if x["match"] == i: x["first"] = fp.get(x["round"])
        if len(saved) < 5 and mode is None:
            p = f"{a.out}/replays/local_replay_{len(saved) + 1}.json.gz"
            with gzip.open(p, "wt") as f: json.dump(env.toJSON(), f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest(); saved.append((p, h, os.path.getsize(p), row["wall_s"]))
        with open(a.out + "/local_matches.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
    json.dump(isf, open(a.out + "/is_first_events.json", "w"))
    json.dump(saved, open(a.out + "/replay_hashes.json", "w"))
    n = len(rows); print(f"matches run: {n}  wall {time.time() - T:.0f}s")
    print("statuses:", collections.Counter(tuple(r["statuses"]) for r in rows))
    print("BO3-rule cross-check (canonical vs engine reward): ", sum(r["rule_ok"] for r in rows), "/", n, "agree")
    print("yourIndex != agent seat on any call:", sum(r["your_ne_seat"] for r in rows))
    print("result tallies [w0,w1,draw] distribution:", collections.Counter(tuple(r["result"]) for r in rows))
    print("games per match:", collections.Counter(len(r["games_seat0"]) for r in rows), " game-level draws:", sum(g == "D" for r in rows for g in r["games_seat0"]))
    print("IS_FIRST chooser by round:", collections.Counter((x["round"], x["chooser"]) for x in isf))
    for mode in ("YES", "NO"):
        xs = [x for x in isf if x["mode"] == mode and x["first"] is not None]
        print(f"mode {mode}: chooser went FIRST in", sum(x["first"] == x["chooser"] for x in xs), "of", len(xs), "games; option types seen:", collections.Counter(tuple(x["otypes"]) for x in xs))
    print("saved replays:", saved)
