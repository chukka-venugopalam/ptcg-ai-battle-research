"""Static card/attack data pulled once from the engine, dict-indexed."""
from __future__ import annotations

import json

from cg.sim import lib

# EnergyType ints (mirror cg.api without importing it)
COLORLESS = 0
RAINBOW = 10
TEAM_ROCKET = 11

_cards: dict[int, dict] | None = None
_attacks: dict[int, dict] | None = None


def cards() -> dict[int, dict]:
    global _cards
    if _cards is None:
        _cards = {c["cardId"]: c
                  for c in json.loads(lib.AllCard().decode())}
    return _cards


def attacks() -> dict[int, dict]:
    global _attacks
    if _attacks is None:
        _attacks = {a["attackId"]: a
                    for a in json.loads(lib.AllAttack().decode())}
    return _attacks


def card(cid: int) -> dict:
    return cards().get(cid) or {}


def is_pokemon(cid: int) -> bool:
    return card(cid).get("cardType") == 0


def is_basic_pokemon(cid: int) -> bool:
    c = card(cid)
    return c.get("cardType") == 0 and bool(c.get("basic"))


def is_basic_energy(cid: int) -> bool:
    return card(cid).get("cardType") == 5


def prize_value(cid: int) -> int:
    """Prizes the opponent takes when this Pokemon is KO'd."""
    c = card(cid)
    if c.get("megaEx"):
        return 3
    if c.get("ex"):
        return 2
    return 1


def energy_satisfied(cost: list[int], have: list[int]) -> bool:
    """Can `have` (attached energy units) pay `cost`? Colorless is wildcard;
    RAINBOW counts as any type; TEAM_ROCKET as psychic(5)/darkness(7)."""
    have = list(have)
    for e in cost:
        if e == COLORLESS:
            continue
        if e in have:
            have.remove(e)
        elif RAINBOW in have:
            have.remove(RAINBOW)
        elif e in (5, 7) and TEAM_ROCKET in have:
            have.remove(TEAM_ROCKET)
        else:
            return False
    n_colorless = sum(1 for e in cost if e == COLORLESS)
    return len(have) >= n_colorless


def best_usable_damage(cid: int, energies: list[int]) -> int:
    """Max printed damage among this Pokemon's attacks payable with
    `energies`. Text effects (bonus damage, coins) are not modeled."""
    best = 0
    atk_db = attacks()
    for aid in card(cid).get("attacks") or []:
        a = atk_db.get(aid)
        if a and energy_satisfied(a["energies"], energies):
            best = max(best, a.get("damage") or 0)
    return best
