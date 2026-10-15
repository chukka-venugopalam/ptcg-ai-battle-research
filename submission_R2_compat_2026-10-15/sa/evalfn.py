"""Heuristic state evaluation, in rough prize-card units, from `me`'s view."""
from __future__ import annotations

from . import cards as cdb

WIN = 1000.0


def _pokemon_value(pk: dict, opp_best_damage: int) -> float:
    """Value of having this Pokemon on my board."""
    cid = pk["id"]
    hp = pk["hp"]
    max_hp = pk["maxHp"] or 1
    energies = pk["energies"]
    c = cdb.card(cid)
    # base material: bigger/evolved pokemon are worth more
    base = 0.10 + 0.0022 * (c.get("hp") or 60)
    if c.get("stage1"):
        base += 0.10
    if c.get("stage2"):
        base += 0.22
    # attached energy is invested tempo
    base += 0.06 * min(len(energies), 5)
    # damage taken devalues it in proportion to what the opponent gains by
    # finishing it off
    pv = cdb.prize_value(cid)
    base -= (1.0 - hp / max_hp) * 0.30 * pv
    # nearly dead to the opponent's current attacker
    if opp_best_damage >= hp:
        base -= 0.22 * pv
    return base


def _side_threat(mypl: dict, oppl: dict) -> float:
    """Best expected damage this side can bring to bear next turn: its active
    at full weight, benched attackers slightly discounted (they must promote)."""
    from .textdmg import best_estimated_damage

    best = 0.0
    act = mypl["active"][0] if mypl["active"] else None
    if act is not None:
        best = best_estimated_damage(act, mypl, oppl)
    for pk in mypl["bench"]:
        if pk is not None:
            best = max(best, 0.8 * best_estimated_damage(pk, mypl, oppl))
    return best


def _effective_damage(dmg: int, attacker_cid: int, defender: dict) -> int:
    if dmg <= 0 or defender is None:
        return 0
    dc = cdb.card(defender["id"])
    a_type = cdb.card(attacker_cid).get("energyType")
    if dc.get("weakness") is not None and dc.get("weakness") == a_type:
        dmg *= 2
    if dc.get("resistance") is not None and dc.get("resistance") == a_type:
        dmg -= 30
    return max(dmg, 0)


def evaluate(state: dict, me: int) -> float:
    result = state["result"]
    if result != -1:
        if result == 2:
            return 0.0
        return WIN if result == me else -WIN

    opp = 1 - me
    mypl = state["players"][me]
    oppl = state["players"][opp]

    score = 0.0

    # prizes taken (I start with 6 in my pile; each KO I score removes one)
    my_taken = 6 - len(mypl["prize"])
    opp_taken = 6 - len(oppl["prize"])
    score += 1.0 * (my_taken - opp_taken)
    # closing bonus: being near 6 matters superlinearly
    score += 0.06 * my_taken * my_taken
    score -= 0.06 * opp_taken * opp_taken

    # attack threat (mutual), including benched attackers and text damage
    my_act = mypl["active"][0] if mypl["active"] else None
    opp_act = oppl["active"][0] if oppl["active"] else None
    my_dmg = _side_threat(mypl, oppl)
    opp_dmg = _side_threat(oppl, mypl)
    my_eff = _effective_damage(my_dmg, my_act["id"], opp_act) \
        if my_act and opp_act else my_dmg
    opp_eff = _effective_damage(opp_dmg, opp_act["id"], my_act) \
        if my_act and opp_act else opp_dmg

    score += 0.0016 * min(my_eff, 400)
    score -= 0.0016 * min(opp_eff, 400)
    if opp_act is not None and my_act is not None:
        if my_eff >= opp_act["hp"]:
            score += 0.45 * cdb.prize_value(opp_act["id"])
        if opp_eff >= my_act["hp"]:
            score -= 0.45 * cdb.prize_value(my_act["id"])

    # board material
    opp_best = opp_eff
    my_best = my_eff
    for pk in list(mypl["active"]) + list(mypl["bench"]):
        if pk is not None:
            score += _pokemon_value(pk, opp_best if pk is my_act else 0)
    for pk in list(oppl["active"]) + list(oppl["bench"]):
        if pk is not None:
            score -= _pokemon_value(pk, my_best if pk is opp_act else 0)

    # damage spread on their side is progress; on mine is exposure
    # (already inside _pokemon_value)

    # hand and deck
    score += 0.035 * min(mypl["handCount"], 9)
    score -= 0.035 * min(oppl["handCount"], 9)
    if mypl["deckCount"] == 0:
        score -= 1.4  # deck-out looms at my next turn start
    if oppl["deckCount"] == 0:
        score += 1.4
    elif oppl["deckCount"] <= 3:
        score += 0.15

    # status on actives
    my_status = (0.30 * mypl["paralyzed"] + 0.25 * mypl["asleep"]
                 + 0.12 * mypl["confused"] + 0.10 * mypl["poisoned"]
                 + 0.08 * mypl["burned"])
    opp_status = (0.30 * oppl["paralyzed"] + 0.25 * oppl["asleep"]
                  + 0.12 * oppl["confused"] + 0.10 * oppl["poisoned"]
                  + 0.08 * oppl["burned"])
    score -= my_status
    score += opp_status

    # a bench exists so a KO doesn't end the game
    my_bench = sum(1 for pk in mypl["bench"] if pk is not None)
    opp_bench = sum(1 for pk in oppl["bench"] if pk is not None)
    if my_bench == 0:
        score -= 0.5
    if opp_bench == 0:
        score += 0.5

    return score
