"""Expected-damage estimation from attack text (pattern-based, approximate).

Used only inside the heuristic eval's threat terms; playouts always see real
simulated damage. Estimates are compiled once per attackId.
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import cards as cdb

_ENERGY_LETTER = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7,
                  "M": 8, "N": 9}


@lru_cache(maxsize=4096)
def _analyze(aid: int) -> tuple:
    """-> tuple of (kind, per, arg) modifiers for this attack's text."""
    a = cdb.attacks().get(aid) or {}
    text = (a.get("text") or "").replace("\n", " ")
    mods: list[tuple] = []

    m = re.search(r"Flip (\d+) coins?\. This attack does (\d+) damage for each"
                  r" heads", text)
    if m:
        mods.append(("flat", 0.5 * int(m.group(1)) * int(m.group(2)), None))
    if re.search(r"Flip a coin until you get tails", text):
        m2 = re.search(r"(\d+) damage for each heads", text)
        if m2:
            mods.append(("flat", float(m2.group(1)), None))
    m = re.search(r"Flip a coin\. If heads, this attack does (\d+) more damage",
                  text)
    if m:
        mods.append(("flat", 0.5 * int(m.group(1)), None))

    for m in re.finditer(
            r"does (\d+)(?: more)? damage for each (?:\{(\w)\} )?Energy "
            r"attached to (this Pok\S+mon|all of your[^,.]*|your opponent"
            r"\S+s Active Pok\S+mon|both Active Pok\S+mon)", text):
        per = int(m.group(1))
        etype = _ENERGY_LETTER.get(m.group(2)) if m.group(2) else None
        scope = m.group(3)
        if scope.startswith("this"):
            mods.append(("energy_self", per, etype))
        elif scope.startswith("all of your"):
            mods.append(("energy_all_mine", per, etype))
        elif scope.startswith("both"):
            mods.append(("energy_both_actives", per, etype))
        else:
            mods.append(("energy_opp_active", per, etype))

    m = re.search(r"does (\d+)(?: more)? damage for each of your Benched",
                  text)
    if m:
        mods.append(("bench_mine", int(m.group(1)), None))
    m = re.search(r"does (\d+)(?: more)? damage for each of your opponent"
                  r"\S+s Benched", text)
    if m:
        mods.append(("bench_opp", int(m.group(1)), None))

    m = re.search(r"does (\d+)(?: more)? damage for each damage counter on "
                  r"(this Pok\S+mon|your opponent\S+s Active)", text)
    if m:
        mods.append(("counters_self" if m.group(2).startswith("this")
                     else "counters_opp", int(m.group(1)), None))

    m = re.search(r"does (\d+) damage for each card in your opponent\S+s hand",
                  text)
    if m:
        mods.append(("opp_hand", int(m.group(1)), None))
    m = re.search(r"does (\d+) damage for each Prize card your opponent has "
                  r"taken", text)
    if m:
        mods.append(("opp_prizes_taken", int(m.group(1)), None))

    m = re.search(r"Discard (?:up to (\d+)|all) (?:\{(\w)\} )?Energy.{0,40}?"
                  r"does (\d+) damage for each", text)
    if m:
        cap = int(m.group(1)) if m.group(1) else 99
        etype = _ENERGY_LETTER.get(m.group(2)) if m.group(2) else None
        mods.append(("discard_energy", int(m.group(3)), (cap, etype)))

    # generic conditional bonus: assume it's live half the time
    if not mods:
        m = re.search(r"If [^.]{3,80}, this attack does (\d+) more damage",
                      text)
        if m:
            mods.append(("flat", 0.5 * int(m.group(1)), None))

    return tuple(mods)


def _count_energy(energies: list[int], etype: int | None) -> int:
    if etype is None:
        return len(energies)
    return sum(1 for e in energies if e == etype or e >= 10)


def estimate(aid: int, attacker: dict, mypl: dict, oppl: dict) -> float:
    """Expected damage of attack `aid` used by `attacker` (a pokemon dict on
    mypl's side) against oppl's active, before weakness/resistance."""
    a = cdb.attacks().get(aid) or {}
    dmg = float(a.get("damage") or 0)
    for kind, per, arg in _analyze(aid):
        if kind == "flat":
            dmg += per
        elif kind == "energy_self":
            dmg += per * _count_energy(attacker["energies"], arg)
        elif kind == "energy_all_mine":
            total = 0
            for pk in list(mypl["active"]) + list(mypl["bench"]):
                if pk is not None:
                    total += _count_energy(pk["energies"], arg)
            dmg += per * total
        elif kind == "energy_opp_active":
            act = oppl["active"][0] if oppl["active"] else None
            if act:
                dmg += per * _count_energy(act["energies"], arg)
        elif kind == "energy_both_actives":
            for pl in (mypl, oppl):
                act = pl["active"][0] if pl["active"] else None
                if act:
                    dmg += per * _count_energy(act["energies"], arg)
        elif kind == "bench_mine":
            dmg += per * sum(1 for pk in mypl["bench"] if pk is not None)
        elif kind == "bench_opp":
            dmg += per * sum(1 for pk in oppl["bench"] if pk is not None)
        elif kind == "counters_self":
            dmg += per * max(0, (attacker["maxHp"] - attacker["hp"]) // 10)
        elif kind == "counters_opp":
            act = oppl["active"][0] if oppl["active"] else None
            if act:
                dmg += per * max(0, (act["maxHp"] - act["hp"]) // 10)
        elif kind == "opp_hand":
            dmg += per * oppl["handCount"]
        elif kind == "opp_prizes_taken":
            dmg += per * (6 - len(oppl["prize"]))
        elif kind == "discard_energy":
            cap, etype = arg
            dmg += per * min(cap, _count_energy(attacker["energies"], etype))
    return dmg


def best_estimated_damage(attacker: dict, mypl: dict, oppl: dict) -> float:
    """Max expected damage over the attacker's payable attacks."""
    best = 0.0
    for aid in cdb.card(attacker["id"]).get("attacks") or []:
        a = cdb.attacks().get(aid)
        if a and cdb.energy_satisfied(a["energies"], attacker["energies"]):
            best = max(best, estimate(aid, attacker, mypl, oppl))
    return best
