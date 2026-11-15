import sys, os, json, time, subprocess
# Ladder-faithful best-of-3 arena (mirrors kaggle_environments cabt interpreter):
#   * each agent is asked for its deck ONCE per match (select=None); games 2/3 start with NO reset call
#   * current.round / players[].win / draw are filled in like the interpreter does
#   * reverse start flag = (games played so far) is odd; the 600 s pool is reset per game (ladder behaviour)
# usage: bo3_arena.py CGPARENT BUNDLE_A BUNDLE_B MATCHES [DECK_A] [DECK_B]    (absolute paths)
# env: RESET_EACH_GAME=1 reproduces the OLD local behaviour (reset before every game) for comparison.
cgdir, A, B, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
deckA = sys.argv[5] if len(sys.argv) > 5 else None
deckB = sys.argv[6] if len(sys.argv) > 6 else None
OFFSET = int(os.environ.get("MATCH_OFFSET", "0"))
RESET_EACH = os.environ.get("RESET_EACH_GAME", "0") == "1"
sys.path.insert(0, cgdir)
import cg.game as G
LOGDIR = os.environ.get("ARENA_LOGDIR", "logs"); os.makedirs(LOGDIR, exist_ok=True)

class Proc:
    def __init__(self, bundle, tag):
        self.p = subprocess.Popen([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"), bundle],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(f"{LOGDIR}/bo3_{tag}_{os.getpid()}.err", "w"), text=True, bufsize=1)
    def __call__(self, obs):
        self.p.stdin.write(json.dumps(obs) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line: raise RuntimeError("agent process died")
        r = json.loads(line)
        if "err" in r: raise RuntimeError(r["err"])
        return r["act"]

agA, agB = Proc(A, "A"), Proc(B, "B")
RESET = {"select": None, "remainingOverageTime": 600, "logs": [], "current": None}
def deck_of(ag, path):
    d = list(ag(RESET))
    if path: d = [int(x) for x in open(path).read().split()]
    return d
decks = [deck_of(agA, deckA), deck_of(agB, deckB)]

def annotate(obs, result, count):
    cur = obs["current"]
    cur["players"][0]["win"] = result[0]; cur["players"][1]["win"] = result[1]
    cur["draw"] = result[2]; cur["round"] = count + 1

res = {"A": 0, "B": 0, "D": 0, "err": 0}
gstat = {}   # game index -> [A game wins, B game wins, draws]
t0 = time.time()
for m in range(OFFSET, OFFSET + N):
    swap = m % 2 == 1
    seat_ag = [agB, agA] if swap else [agA, agB]
    seat_deck = [decks[1], decks[0]] if swap else [decks[0], decks[1]]
    names = ["B", "A"] if swap else ["A", "B"]
    for ag in seat_ag:
        ag(RESET)                                   # the single deck request of the match
    result = [0, 0, 0]; count = 0; bad = None
    while True:
        obs, sd = G.battle_start(seat_deck[0], seat_deck[1], count % 2 != 0)
        if obs is None:
            bad = (-1, "deck error %s" % sd.errorPlayer); break
        if RESET_EACH and count > 0:
            for ag in seat_ag: ag(RESET)
        annotate(obs, result, count)
        rem = [600.0, 600.0]; steps = 0
        while obs["current"]["result"] < 0 and steps < 5000:
            p = obs["current"]["yourIndex"]; o = dict(obs); o["remainingOverageTime"] = rem[p]
            t = time.time()
            try: act = seat_ag[p](o)
            except Exception as e: bad = (p, repr(e)); break
            rem[p] -= time.time() - t
            try: obs = G.battle_select([int(x) for x in act])
            except Exception as e: bad = (p, "illegal " + repr(e) + str(act)[:60]); break
            if obs["current"]["result"] < 0: annotate(obs, result, count)
            steps += 1
        if bad: break
        r = obs["current"]["result"]; G.battle_finish()
        result[r if r < 2 else 2] += 1
        gs = gstat.setdefault(count, [0, 0, 0])
        if r < 2: gs[0 if names[r] == "A" else 1] += 1
        else: gs[2] += 1
        count += 1
        remain = 3 - count
        if result[0] > result[1] + remain or result[1] > result[0] + remain or remain <= 0: break
    if bad:
        res["err"] += 1; print("match", m, "failed:", bad, flush=True)
        try: G.battle_finish()
        except Exception: pass
        if bad[0] >= 0: res[names[1 - bad[0]]] += 1
    else:
        if result[0] > result[1]: res[names[0]] += 1
        elif result[1] > result[0]: res[names[1]] += 1
        else: res["D"] += 1
    print("PROG", m, json.dumps(res), "games", json.dumps(gstat), "%.0fs" % (time.time() - t0), flush=True)
print("FINAL", json.dumps(res), "games", json.dumps(gstat))
