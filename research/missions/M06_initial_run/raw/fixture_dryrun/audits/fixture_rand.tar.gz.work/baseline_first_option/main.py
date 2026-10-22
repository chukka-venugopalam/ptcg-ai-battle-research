DECK = [int(x) for x in open("deck.csv").read().split()]
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))
    return list(range(k))
