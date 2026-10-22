"""Test-agent fixtures for the harness tests.  Generated at run time (no engine data is stored in the repository: the
60-card deck is read from the installed SDK).  These are NOT strength baselines -- they only exercise plumbing."""
import os

RAND_BODY = '''
import random
{deck_loader}
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    n = len(sel["option"])
    k = max(sel["minCount"], min(sel["maxCount"], n))
    return random.sample(range(n), k)
'''

FIRST_BODY = '''
{deck_loader}
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    n = len(sel["option"])
    k = max(sel["minCount"], min(sel["maxCount"], n))
    return list(range(k))
'''

LOADERS = {
    "cwd": 'DECK = [int(x) for x in open("deck.csv").read().split()]',
    "kpath": 'DECK = [int(x) for x in open("/kaggle_simulations/agent/deck.csv").read().split()]',
    "file_unguarded": 'import os\nDECK = [int(x) for x in open(os.path.join(os.path.dirname(__file__), "deck.csv")).read().split()]',
    "file_guarded": ('import os\ntry:\n    _H = os.path.dirname(__file__)\nexcept NameError:\n    _H = os.getcwd()\n'
                     'DECK = [int(x) for x in open(os.path.join(_H, "deck.csv")).read().split()]'),
    "helper": "from helper import DECK",
}

CRASH_BODY = '''
import random
{deck_loader}
_n = 0
def agent(obs):
    global _n
    sel = obs["select"]
    if sel is None:
        return DECK
    _n += 1
    if _n == 5:
        raise RuntimeError("injected crash")
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
'''

ILLEGAL_BODY = '''
{deck_loader}
_n = 0
def agent(obs):
    global _n
    sel = obs["select"]
    if sel is None:
        return DECK
    _n += 1
    if _n == 3:
        return [len(sel["option"]) + 7]
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
'''

HANG_BODY = '''
import time
{deck_loader}
_n = 0
def agent(obs):
    global _n
    sel = obs["select"]
    if sel is None:
        return DECK
    _n += 1
    if _n == 3:
        time.sleep(10**6)
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
'''

COUNTER_BODY = '''
{deck_loader}
FALLBACK_COUNT = 0
def agent(obs):
    global FALLBACK_COUNT
    sel = obs["select"]
    if sel is None:
        return DECK
    FALLBACK_COUNT += 1
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
'''

BADDECK_BODY = '''
{deck_loader}
DECK = DECK[:-1] + [1432]
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
'''

LAST_CALLABLE_TRAP = '''
{deck_loader}
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
def helper_defined_after_agent():   # the SDK picks the LAST callable in globals -> this one, not `agent`
    return 1
'''


def sdk_deck():
    from kaggle_environments.envs.cabt.cabt import deck
    return list(deck)


def deck_variant(deck):
    """Second legal deck: swap two Basic {W} Energy (id 3) for Basic {G} Energy (id 1)."""
    d = list(deck)
    swapped = 0
    for i, c in enumerate(d):
        if c == 3 and swapped < 2:
            d[i] = 1
            swapped += 1
    return d


def _write(root, name, body, loader_key, deck, extra=None):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "main.py"), "w").write(body.format(deck_loader=LOADERS[loader_key]))
    open(os.path.join(d, "deck.csv"), "w").write("\n".join(map(str, deck)) + "\n")
    for fn, txt in (extra or {}).items():
        open(os.path.join(d, fn), "w").write(txt)
    return d


def make_agents(root):
    deck, deck2 = sdk_deck(), None
    deck2 = deck_variant(deck)
    out = {}
    out["rand_cwd"] = _write(root, "rand_cwd", RAND_BODY, "cwd", deck)
    out["first_kpath"] = _write(root, "first_kpath", FIRST_BODY, "kpath", deck)
    out["file_unguarded"] = _write(root, "file_unguarded", FIRST_BODY, "file_unguarded", deck)
    out["file_guarded"] = _write(root, "file_guarded", FIRST_BODY, "file_guarded", deck)
    out["crash5"] = _write(root, "crash5", CRASH_BODY, "cwd", deck)
    out["illegal3"] = _write(root, "illegal3", ILLEGAL_BODY, "cwd", deck)
    out["hang3"] = _write(root, "hang3", HANG_BODY, "cwd", deck)
    out["counter"] = _write(root, "counter", COUNTER_BODY, "cwd", deck)
    out["trap"] = _write(root, "trap", LAST_CALLABLE_TRAP, "cwd", deck)
    out["baddeck"] = _write(root, "baddeck", BADDECK_BODY, "cwd", deck)
    out["helper_x"] = _write(root, "helper_x", FIRST_BODY, "helper", deck, {"helper.py": f"DECK = {deck!r}\n"})
    out["helper_y"] = _write(root, "helper_y", FIRST_BODY, "helper", deck2, {"helper.py": f"DECK = {deck2!r}\n"})
    out["_deck"], out["_deck2"] = deck, deck2
    return out
