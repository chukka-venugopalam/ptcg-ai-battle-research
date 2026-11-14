import sys, os, json, time, subprocess, random
# usage: proc_arena.py CGDIR BUNDLE_A BUNDLE_B GAMES [DECK_A] [DECK_B]   (Bo3 with reverse start in game 2, seats swap per match)
cgdir, A, B, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
deckA = sys.argv[5] if len(sys.argv) > 5 else None
deckB = sys.argv[6] if len(sys.argv) > 6 else None
sys.path.insert(0, cgdir)
import cg.game as G
class Proc:
    def __init__(self, bundle, tag):
        self.p = subprocess.Popen([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"), bundle],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(f"logs/proc_{tag}_{os.getpid()}.err", "w"), text=True, bufsize=1)
    def __call__(self, obs):
        self.p.stdin.write(json.dumps(obs) + "\n"); self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "err" in r: raise RuntimeError(r["err"])
        return r["act"]
agA, agB = Proc(A, "A"), Proc(B, "B")
RESET = {"select": None, "remainingOverageTime": 600, "logs": [], "current": None}
def deck_of(ag, path):
    if path: return [int(x) for x in open(path).read().split()]
    return list(ag(RESET))
decks = [deck_of(agA, deckA), deck_of(agB, deckB)]
res = {"A": 0, "B": 0, "D": 0, "err": 0}; t0 = time.time(); times = [0.0, 0.0]
for g in range(N):
    swap = (g // 3) % 2 == 1
    ags = [(agB, decks[1]), (agA, decks[0])] if swap else [(agA, decks[0]), (agB, decks[1])]
    who = ["B", "A"] if swap else ["A", "B"]
    if g % 3 == 1: obs, sd = G.battle_start(ags[0][1], ags[1][1], True)
    else: obs, sd = G.battle_start(ags[0][1], ags[1][1])
    if obs is None: print("deck error", sd.errorPlayer, sd.errorType); res["err"] += 1; continue
    for a in ags:
        try: a[0](RESET)
        except Exception: pass
    rem = [600.0, 600.0]; steps = 0; bad = None
    while obs["current"]["result"] < 0 and steps < 5000:
        p = obs["current"]["yourIndex"]; o = dict(obs); o["remainingOverageTime"] = rem[p]
        t = time.time()
        try: act = ags[p][0](o)
        except Exception as e: bad = (p, repr(e)); break
        dt = time.time() - t; rem[p] -= dt; times[p] += dt
        try: obs = G.battle_select([int(x) for x in act])
        except Exception as e: bad = (p, "illegal " + repr(e) + str(act)[:60]); break
        steps += 1
    if bad: res["err"] += 1; res[who[1 - bad[0]]] += 1; print("game", g, "agent", who[bad[0]], "failed:", bad[1], flush=True)
    else:
        r = obs["current"]["result"]
        if r == 2: res["D"] += 1
        else: res[who[r]] += 1
    G.battle_finish(); print("PROG", g, json.dumps(res), "%.0fs" % (time.time() - t0), flush=True)
print("FINAL", json.dumps(res), "secs A %.1f B %.1f" % tuple(times))
