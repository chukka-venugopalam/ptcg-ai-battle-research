
import random
import os
try:
    _H = os.path.dirname(__file__)
except NameError:
    _H = os.getcwd()
DECK = [int(x) for x in open(os.path.join(_H, "deck.csv")).read().split()]
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    n = len(sel["option"])
    k = max(sel["minCount"], min(sel["maxCount"], n))
    return random.sample(range(n), k)
