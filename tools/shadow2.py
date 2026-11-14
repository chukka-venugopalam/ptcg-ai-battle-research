import sys, os, json, time, subprocess, collections
# usage: shadow.py CGDIR DRIVER_BUNDLE SHADOW_BUNDLE GAMES DECK_P DECK_Q
cgdir, P, Q, N, dP, dQ = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
SEAT_ONLY = int(os.environ.get('SEAT_ONLY', '-1'))
sys.path.insert(0, cgdir)
import cg.game as G
here = os.path.dirname(os.path.abspath(__file__))
class Proc:
    def __init__(self, bundle, tag):
        self.p = subprocess.Popen([sys.executable, "-I", os.path.join(here, "worker.py"), bundle], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(f"logs/shadow_{tag}_{os.getpid()}.err", "w"), text=True, bufsize=1)
    def __call__(self, obs):
        self.p.stdin.write(json.dumps(obs) + "\n"); self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "err" in r: raise RuntimeError(r["err"])
        return r["act"]
PS = [Proc(P, "P0"), Proc(P, "P1")]; QS = [Proc(Q, "Q0"), Proc(Q, "Q1")]
RESET = {"select": None, "remainingOverageTime": 600, "logs": [], "current": None}
deckP = [int(x) for x in open(dP).read().split()]; deckQ = [int(x) for x in open(dQ).read().split()]
tot = 0; dis = 0; bytype = collections.Counter(); bytype_dis = collections.Counter(); t0 = time.time()
for g in range(N):
    for a in PS + QS:
        a(RESET)
    swap = (g // 3) % 2 == 1
    d0, d1 = (deckQ, deckP) if swap else (deckP, deckQ)
    fseat = 1 if swap else 0
    if g % 3 == 1: obs, sd = G.battle_start(d0, d1, True)
    else: obs, sd = G.battle_start(d0, d1)
    steps = 0
    while obs["current"]["result"] < 0 and steps < 4000:
        p = obs["current"]["yourIndex"]; o = dict(obs); o["remainingOverageTime"] = 500.0
        ap = PS[p](o); aq = QS[p](o)
        if SEAT_ONLY < 0 or p == fseat:
            tot += 1; st = obs["select"]["type"]; bytype[st] += 1
            if ap != aq: dis += 1; bytype_dis[st] += 1
        obs = G.battle_select([int(x) for x in ap]); steps += 1
    G.battle_finish()
    print("game", g, "decisions", tot, "disagree", dis, "%.0fs" % (time.time() - t0), flush=True)
print("FINAL decisions", tot, "disagree", dis, "rate %.4f" % (dis / max(1, tot)), "by type", {k: (bytype_dis[k], bytype[k]) for k in bytype})
