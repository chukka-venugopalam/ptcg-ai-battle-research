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

AGENT_KIND = 'bc'
# Extra ensemble members, relative to the bundle root. Empty = single net, i.e.
# exactly the behaviour every submission before this one had. When non-empty the
# agent votes across sa/policy_net.npz + these (EVIDENCE: E9).
ENSEMBLE_EXTRA = []
# Explicit rule flags, pinned at BUILD time. The defaults in sa/bcagent.py are
# tuned for the lw2 net; an optfeat-v3 net wants them OFF (the three rules
# measure 0.427 against it -- report/EVIDENCE.md 8f). Pinning the (net, flags)
# PAIR here keeps both configurations shippable without a global default flip.
AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}


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
_LOG = {"next": 0, "failed": False}


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
