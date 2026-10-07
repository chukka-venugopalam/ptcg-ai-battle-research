"""Archaludon ex / Cinderace specialist for the PTCG AI Battle Playground.

Clean-room reimplementation of the publicly documented community strategy:
- Cinderace-first opener, then Duraludon/Relicanth
- protect the empty bench invariant
- build Archaludon through Metal Energy / Alloy
- prefer actual KOs and prize-efficient attacks
- preserve Boss for concrete KO/gust opportunities
- Crustle-specific safeguard against Metal Defender / ex evolution traps
- dead-active tempo correction

No model/dependency other than the supplied competition SDK.
"""
from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (
    AreaType,
    LogType,
    OptionType,
    SelectContext,
    CardType,
    EnergyType,
    all_attack,
    all_card_data,
    to_observation_class,
)

# ---------------------------------------------------------------------------
# Public card/attack tables supplied by the competition SDK
# ---------------------------------------------------------------------------
CARD_DB = {c.cardId: c for c in all_card_data()}
ATTACK_DB = {a.attackId: a for a in all_attack()}

# Archaludon ex / Cinderace public list
DURALUDON = 169
ARCHALUDON_EX = 190
CINDERACE = 666
RELICANTH = 57
METAL_ENERGY = 8

POKE_PAD = 1152
ULTRA_BALL = 1121
POKEGEAR = 1122
NIGHT_STRETCHER = 1097
JUMBO_ICE_CREAM = 1147
HERO_CAPE = 1159
BOSS = 1182
EXPLORER = 1185
LILLIE = 1227
FULL_METAL_LAB = 1244
RAGING_HAMMER = 224
METAL_DEFENDER = 253
MEGA_BRAVE = 983

CRUSTLE_IDS = {344, 345, 532}
STARMIE_IDS = {1030, 1031}
LUCARIO_IDS = {677, 678}
HOP_IDS = {288, 289, 299, 304, 307, 308, 309, 310, 878, 879}
HOP_SNORLAX = 304

ITEM_IDS = {POKE_PAD, ULTRA_BALL, POKEGEAR, NIGHT_STRETCHER, JUMBO_ICE_CREAM, HERO_CAPE}

# Published deck list, embedded so the package is self-contained.
DECK = [
    169,169,169,169,190,190,190,190,666,666,666,666,1244,57,
    1152,1152,1152,1152,1121,1121,1121,1121,1122,1122,1122,1122,
    1097,1097,1097,1147,1147,1147,1147,1159,1182,1182,1182,1182,
    1185,1185,1185,1185,1227,1227,1227,1227,1244,1244,1244,
    8,8,8,8,8,8,8,8,8,8,8
]

# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------
def _player(obs, idx):
    return obs.current.players[idx]


def _me_idx(obs):
    return obs.current.yourIndex


def me(obs):
    return _player(obs, _me_idx(obs))


def opp(obs):
    return _player(obs, 1 - _me_idx(obs))


def active(ps):
    a = getattr(ps, "active", None) or []
    return a[0] if a else None


def bench(ps):
    return [p for p in (getattr(ps, "bench", None) or []) if p]


def hand(ps):
    return [c for c in (getattr(ps, "hand", None) or []) if c]


def discard(ps):
    return [c for c in (getattr(ps, "discard", None) or []) if c]


def hand_ids(obs):
    return [c.id for c in hand(me(obs))]


def discard_ids(obs):
    return [c.id for c in discard(me(obs))]


def count_hand(obs, cid):
    return hand_ids(obs).count(cid)


def count_discard(obs, cid):
    return discard_ids(obs).count(cid)


def energy_count(p):
    if not p:
        return 0
    x = getattr(p, "energyCards", None)
    if x is not None:
        return len(x)
    return len(getattr(p, "energies", None) or [])


def tool_count(p):
    return len(getattr(p, "tools", None) or [])


def damage_taken(p):
    if not p:
        return 0
    mx = getattr(p, "maxHp", None) or getattr(p, "hp", 0)
    return max(0, mx - getattr(p, "hp", 0))


def get_card(obs, area, index, player_idx):
    try:
        ps = _player(obs, player_idx)
        if area == AreaType.DECK:
            d = getattr(obs.select, "deck", None)
            return d[index] if d is not None and 0 <= index < len(d) else None
        if area == AreaType.HAND:
            x = getattr(ps, "hand", None) or []
            return x[index] if 0 <= index < len(x) else None
        if area == AreaType.DISCARD:
            x = getattr(ps, "discard", None) or []
            return x[index] if 0 <= index < len(x) else None
        if area == AreaType.ACTIVE:
            x = getattr(ps, "active", None) or []
            return x[index] if 0 <= index < len(x) else None
        if area == AreaType.BENCH:
            x = getattr(ps, "bench", None) or []
            return x[index] if 0 <= index < len(x) else None
        if area == AreaType.PRIZE:
            x = getattr(ps, "prize", None) or []
            return x[index] if 0 <= index < len(x) else None
        if area == AreaType.LOOKING:
            x = getattr(obs.current, "looking", None) or []
            return x[index] if 0 <= index < len(x) else None
        if area == AreaType.STADIUM:
            x = getattr(obs.current, "stadium", None) or []
            return x[index] if 0 <= index < len(x) else None
    except Exception:
        return None
    return None


def option_card(obs, opt):
    if opt.type == OptionType.PLAY:
        return get_card(obs, AreaType.HAND, getattr(opt, "index", -1), _me_idx(obs))
    pi = getattr(opt, "playerIndex", None)
    if pi is None:
        pi = _me_idx(obs)
    return get_card(obs, getattr(opt, "area", None), getattr(opt, "index", -1), pi)


def option_target(obs, opt):
    area = getattr(opt, "inPlayArea", None)
    idx = getattr(opt, "inPlayIndex", None)
    if area is None or idx is None:
        return None
    return get_card(obs, area, idx, _me_idx(obs))


def all_my_pokemon(obs):
    ps = me(obs)
    return ([p for p in (getattr(ps, "active", None) or []) if p]
            + [p for p in (getattr(ps, "bench", None) or []) if p])


def all_opp_pokemon(obs):
    ps = opp(obs)
    return ([p for p in (getattr(ps, "active", None) or []) if p]
            + [p for p in (getattr(ps, "bench", None) or []) if p])


def count_my(obs, cid):
    return sum(1 for p in all_my_pokemon(obs) if p.id == cid)


def has_my(obs, cid):
    return count_my(obs, cid) > 0


def prize_value(p):
    if not p:
        return 0
    d = CARD_DB.get(p.id)
    if d is None:
        return 1
    if getattr(d, "megaEx", False):
        return 3
    if getattr(d, "ex", False):
        return 2
    return 1


def weakness(target, energy_type):
    d = CARD_DB.get(target.id) if target else None
    if not d:
        return False
    w = getattr(d, "weakness", None)
    return getattr(w, "value", w) == energy_type


def base_attack_damage(attack_id, attacker):
    if attack_id == RAGING_HAMMER:
        return 80 + (damage_taken(attacker) // 10) * 10
    if attack_id == METAL_DEFENDER:
        return 220
    # Read the engine's attack table where possible.
    a = ATTACK_DB.get(attack_id)
    if a is not None:
        for attr in ("damage", "baseDamage"):
            v = getattr(a, attr, None)
            if isinstance(v, (int, float)):
                return int(v)
    return 0


def effective_damage(attack_id, attacker, target):
    dmg = base_attack_damage(attack_id, attacker)
    if weakness(target, EnergyType.METAL):
        dmg *= 2
    return dmg


def is_ko(attack_id, attacker, target):
    return bool(target and effective_damage(attack_id, attacker, target) >= getattr(target, "hp", 10**9))


def detect_matchup(obs):
    ids = {p.id for p in all_opp_pokemon(obs)}
    if ids & CRUSTLE_IDS:
        return "crustle"
    if ids & HOP_IDS:
        return "hop"
    if ids & STARMIE_IDS:
        return "starmie"
    if ids & LUCARIO_IDS:
        return "lucario"
    return "generic"

# ---------------------------------------------------------------------------
# Stateful attack/tempo memory
# ---------------------------------------------------------------------------
_last_opp_attack = None
_seen_logs = 0
_turn_seen = -1


def update_opponent_attack(obs):
    global _last_opp_attack, _seen_logs, _turn_seen
    try:
        logs = list(getattr(obs, "logs", None) or [])
    except Exception:
        return
    if getattr(obs.current, "turn", -1) != _turn_seen:
        _turn_seen = getattr(obs.current, "turn", -1)
        _seen_logs = 0
    for entry in logs[_seen_logs:]:
        typ = getattr(entry, "type", None)
        if typ == LogType.ATTACK:
            pi = getattr(entry, "playerIndex", None)
            if pi is None or pi != _me_idx(obs):
                _last_opp_attack = getattr(entry, "attackId", None)
    _seen_logs = len(logs)

# ---------------------------------------------------------------------------
# Setup / routing heuristics
# ---------------------------------------------------------------------------
def choose_setup(obs, opt):
    ctx = obs.select.context
    c = option_card(obs, opt)
    cid = c.id if c else None
    if ctx == SelectContext.MULLIGAN:
        return 10000 if opt.type == OptionType.NO else 0
    if ctx == SelectContext.IS_FIRST:
        # Published pilot intentionally chooses the second seat.
        return 10000 if opt.type == OptionType.NO else 0
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return {CINDERACE: 100000, DURALUDON: 20000, RELICANTH: 5000}.get(cid, 0)
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return {DURALUDON: 25000, RELICANTH: 22000}.get(cid, -10000)
    return 0


def bench_empty(obs):
    return len(bench(me(obs))) == 0


def main_has_basic_bench_play(obs):
    if obs.select.context != SelectContext.MAIN:
        return False
    for o in obs.select.option:
        if o.type == OptionType.PLAY:
            c = option_card(obs, o)
            if c and c.id in (DURALUDON, RELICANTH):
                return True
    return False


def planned_attackers(obs):
    ps = me(obs)
    out = []
    for p in [active(ps)] + bench(ps):
        if p and p.id in (DURALUDON, ARCHALUDON_EX):
            out.append(p)
    return out


def can_attack_this_turn(obs, p):
    if not p:
        return False, False
    e = energy_count(p)
    has_attached = not bool(getattr(obs.current, "energyAttached", False))
    if p.id == ARCHALUDON_EX:
        if e >= 3:
            return True, False
        if e == 2 and has_attached and METAL_ENERGY in hand_ids(obs):
            return True, True
        return False, False
    if p.id == DURALUDON:
        # direct attack or next evolution into ex using Alloy
        if e >= 3:
            return True, False
        metals = count_discard(obs, METAL_ENERGY)
        if getattr(p, "appearThisTurn", False):
            return False, False
        if ARCHALUDON_EX in hand_ids(obs) and e + min(2, metals) >= 3:
            return True, False
        if e == 2 and has_attached and METAL_ENERGY in hand_ids(obs):
            return True, True
    return False, False


def ready_attack_targets(obs):
    out = []
    for p in planned_attackers(obs):
        ok, attach = can_attack_this_turn(obs, p)
        if ok:
            out.append((p, attach))
    return out


def best_attack_for(obs, attacker):
    choices = []
    target = active(opp(obs))
    if attacker.id == ARCHALUDON_EX:
        choices = [a for a in getattr(obs.select, "option", []) if a.type == OptionType.ATTACK]
    elif attacker.id == DURALUDON:
        choices = [a for a in getattr(obs.select, "option", []) if a.type == OptionType.ATTACK]
    best = None
    for o in choices:
        aid = getattr(o, "attackId", None)
        if aid is None:
            continue
        dmg = effective_damage(aid, attacker, target)
        score = dmg
        if target and dmg >= target.hp:
            score += 50000 + 5000 * prize_value(target)
        if aid == RAGING_HAMMER and detect_matchup(obs) == "crustle":
            score += 3000
        if aid == METAL_DEFENDER and detect_matchup(obs) == "crustle":
            score -= 10000
        if best is None or score > best[0]:
            best = (score, o)
    return best

# ---------------------------------------------------------------------------
# Per-context scores
# ---------------------------------------------------------------------------
def score_main_play(obs, opt):
    c = option_card(obs, opt)
    cid = c.id if c else None
    ps = me(obs)
    if cid is None:
        return 0
    if cid in (DURALUDON, RELICANTH):
        return 50000 if bench_empty(obs) else 18000
    if cid == FULL_METAL_LAB:
        a = active(ps)
        return 20000 if (a and a.id in (DURALUDON, ARCHALUDON_EX)) else -200
    if cid == ULTRA_BALL:
        if bench_empty(obs):
            return -5000
        if count_hand(obs, METAL_ENERGY) and count_discard(obs, METAL_ENERGY) < 2:
            return 20000
        if (not count_discard(obs, DURALUDON) and not has_my(obs, DURALUDON)
                and (count_hand(obs, EXPLORER) + count_hand(obs, LILLIE)) > 0):
            return 16000
        return 3000
    if cid == EXPLORER:
        return -5000 if getattr(ps, "deckCount", 60) <= 10 else 16000
    if cid == LILLIE:
        if getattr(obs.current, "supporterPlayed", False):
            return -1000
        # Keep draw when Boss + attack route is ready.
        if count_hand(obs, BOSS) and ready_attack_targets(obs):
            return 2500
        return 7000
    if cid == BOSS:
        if getattr(obs.current, "supporterPlayed", False):
            return -1000
        # Stronger than generic draw only when it changes the prize race.
        attackers = ready_attack_targets(obs)
        if not attackers:
            return 0
        score = 12000
        opp_b = bench(opp(obs))
        for t in [active(opp(obs))] + opp_b:
            if not t:
                continue
            if any(is_ko(getattr(o, "attackId", 0), p, t)
                   for p, _ in attackers for o in getattr(obs.select, "option", [])
                   if o.type == OptionType.ATTACK):
                score += 7000 + prize_value(t) * 2000
        return score
    if cid in ITEM_IDS:
        if cid == NIGHT_STRETCHER:
            urgent = (DURALUDON in discard_ids(obs) or ARCHALUDON_EX in discard_ids(obs))
            return 16000 if urgent else -500
        if cid == JUMBO_ICE_CREAM:
            a = active(ps)
            if not a or a.id != ARCHALUDON_EX:
                return -500
            match = detect_matchup(obs)
            thresholds = {"crustle":120, "lucario":270, "starmie":210, "hop":220, "generic":230}
            return 18000 if a.hp <= thresholds.get(match, 220) else -500
        if cid == HERO_CAPE:
            return 17000 if any(p.id in (DURALUDON, ARCHALUDON_EX) and tool_count(p) == 0 for p in all_my_pokemon(obs)) else -500
        return 10000
    return 1000


def score_attach(obs, opt):
    c = option_card(obs, opt)
    t = option_target(obs, opt)
    cid = c.id if c else None
    if not t:
        return -500
    if cid == HERO_CAPE:
        if t.id == ARCHALUDON_EX and tool_count(t) == 0:
            return 12000 + max(0, 300 - t.hp)
        if t.id == DURALUDON and tool_count(t) == 0 and energy_count(t) >= 1:
            return 8000
        return -1000
    if cid != METAL_ENERGY:
        return -500
    e = energy_count(t)
    if e >= 3:
        return -4000
    if t.id == ARCHALUDON_EX:
        return 16000 + (9000 if e == 2 else 5000 if e == 1 else 2000)
    if t.id == DURALUDON:
        return 15000 + (8000 if e == 2 else 4500 if e == 1 else 2000)
    if t.id == CINDERACE:
        return 11000 if e == 0 else -2000
    return 2000


def score_evolve(obs, opt):
    c = option_card(obs, opt)
    t = option_target(obs, opt)
    if not c or not t:
        return 0
    if c.id == ARCHALUDON_EX and t.id == DURALUDON:
        metals = count_discard(obs, METAL_ENERGY)
        if detect_matchup(obs) == "crustle":
            return -10000
        if t in ([active(me(obs))] + bench(me(obs))) and metals >= 2:
            return 28000 + 2000 * metals
        if energy_count(t) >= 3:
            return 17000
        if metals == 1:
            return 7000
        return -500
    return 9000


def score_switch(obs, opt):
    c = option_card(obs, opt)
    if not c:
        return 0
    pi = getattr(opt, "playerIndex", _me_idx(obs))
    if pi != _me_idx(obs):
        # Boss target: prefer a concrete KO and/or high-prize target.
        attackers = ready_attack_targets(obs)
        if any(is_ko(getattr(aopt, "attackId", 0), p, c)
               for p, _ in attackers for aopt in getattr(obs.select, "option", [])
               if aopt.type == OptionType.ATTACK):
            return 23000 + prize_value(c) * 3000 + energy_count(c) * 100
        return 5000 + prize_value(c) * 1000 + energy_count(c) * 200
    if c.id == CINDERACE:
        return 17000
    if c.id == ARCHALUDON_EX:
        return 16000
    if c.id == DURALUDON:
        return 13000
    if c.id == RELICANTH:
        return 6000
    return 2000 + energy_count(c) * 200


def score_to_hand(obs, opt):
    c = option_card(obs, opt)
    if not c:
        return 0
    cid = c.id
    ids = hand_ids(obs)
    if cid == ARCHALUDON_EX:
        return 22000 if count_my(obs, ARCHALUDON_EX) < 2 else 6000
    if cid == DURALUDON:
        return 23000 if count_my(obs, DURALUDON) + count_my(obs, ARCHALUDON_EX) < 2 else 8000
    if cid == RELICANTH:
        return 12000 if not has_my(obs, RELICANTH) else 3000
    if cid == METAL_ENERGY:
        return 9000
    if cid == HERO_CAPE:
        return 8000 if any(p.id in (DURALUDON, ARCHALUDON_EX) and tool_count(p) == 0 for p in all_my_pokemon(obs)) else 1000
    if cid == EXPLORER:
        return 8000
    if cid == LILLIE:
        return 7000
    if cid == BOSS:
        return 5000
    if cid == CINDERACE:
        return -2000
    if cid in ids:
        return 500
    return 1500


def score_discard(obs, opt):
    c = option_card(obs, opt)
    if not c:
        return 0
    cid = c.id
    ids = hand_ids(obs)
    md = count_discard(obs, METAL_ENERGY)
    if cid == METAL_ENERGY:
        return 16000 if md < 2 else (11000 if ids.count(METAL_ENERGY) > 1 else -1000)
    if cid == CINDERACE:
        return 12000
    if cid in (BOSS, POKEGEAR, FULL_METAL_LAB):
        return 7000
    if cid in (LILLIE, EXPLORER) and ids.count(cid) > 1:
        return 6500
    if cid in (DURALUDON, ARCHALUDON_EX):
        return -5000
    if cid == HERO_CAPE:
        return -4000
    return 1000


def score_target(obs, opt):
    ctx = obs.select.context
    c = option_card(obs, opt)
    if not c:
        return 0
    cid = c.id
    if ctx == SelectContext.ATTACH_FROM:
        return score_attach(obs, opt)
    if ctx == SelectContext.ATTACH_TO:
        return 5000 if cid == METAL_ENERGY else 1000
    if ctx in (SelectContext.TO_FIELD, SelectContext.TO_BENCH):
        return {ARCHALUDON_EX:18000, DURALUDON:16000, CINDERACE:3000, RELICANTH:5000}.get(cid, 1000)
    if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        return score_switch(obs, opt)
    if ctx == SelectContext.DAMAGE:
        return 12000 - getattr(c, "hp", 999)
    if ctx == SelectContext.HEAL:
        return 20000 + damage_taken(c) if cid == ARCHALUDON_EX else damage_taken(c)
    return 1000

# ---------------------------------------------------------------------------
# Final option ranker
# ---------------------------------------------------------------------------
def score_option(obs, opt):
    ctx = obs.select.context
    t = opt.type

    if ctx in (SelectContext.MULLIGAN, SelectContext.IS_FIRST,
               SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        s = choose_setup(obs, opt)
    elif t in (OptionType.YES, OptionType.NO):
        if ctx == SelectContext.ACTIVATE:
            s = 100000 if t == OptionType.YES else -100000
        else:
            s = 100 if t == OptionType.YES else 0
    elif t == OptionType.NUMBER:
        s = getattr(opt, "number", 0) or 0
    elif ctx == SelectContext.MAIN:
        if t == OptionType.PLAY:
            s = score_main_play(obs, opt)
        elif t == OptionType.ATTACH:
            s = score_attach(obs, opt)
        elif t == OptionType.EVOLVE:
            s = score_evolve(obs, opt)
        elif t == OptionType.RETREAT:
            s = 12000 if ready_attack_targets(obs) and active(me(obs)) not in [p for p,_ in ready_attack_targets(obs)] else -100
        elif t == OptionType.ATTACK:
            a = active(me(obs))
            target = active(opp(obs))
            aid = getattr(opt, "attackId", None)
            s = effective_damage(aid, a, target)
            if target and s >= target.hp:
                s += 50000 + prize_value(target) * 5000
            # Do not throw Metal Defender into Crustle / Safeguard-like targets.
            if detect_matchup(obs) == "crustle" and aid == METAL_DEFENDER:
                s -= 15000
            if detect_matchup(obs) == "crustle" and aid == RAGING_HAMMER:
                s += 5000
            # When behind on prizes, a legal KO gets extra priority.
            if len(getattr(me(obs), "prize", None) or []) > len(getattr(opp(obs), "prize", None) or []):
                if target and s >= target.hp:
                    s += 8000
        elif t == OptionType.ABILITY:
            s = 10000
        elif t == OptionType.END:
            s = -1000 if bench_empty(obs) and main_has_basic_bench_play(obs) else 0
        else:
            s = 500
    elif ctx == SelectContext.TO_HAND:
        s = score_to_hand(obs, opt)
    elif ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
        s = score_discard(obs, opt)
    elif ctx in (SelectContext.ATTACH_TO, SelectContext.TO_FIELD, SelectContext.TO_BENCH,
                 SelectContext.ATTACH_FROM, SelectContext.SWITCH, SelectContext.TO_ACTIVE,
                 SelectContext.HEAL, SelectContext.DAMAGE):
        s = score_target(obs, opt)
    elif ctx == SelectContext.ATTACK:
        a = active(me(obs))
        target = active(opp(obs))
        s = effective_damage(getattr(opt, "attackId", 0), a, target)
    elif t == OptionType.ENERGY:
        s = 1000
    elif t == OptionType.END:
        s = 0
    else:
        s = 100

    # Global safety overlays, deliberately narrow and context-aware.
    if bench_empty(obs):
        c = option_card(obs, opt)
        if c and c.id in (DURALUDON, RELICANTH) and ctx == SelectContext.MAIN and t == OptionType.PLAY:
            s = max(s, 50000)
        if ctx == SelectContext.MAIN and t == OptionType.PLAY and c and c.id == ULTRA_BALL:
            s = min(s, -5000)
        if ctx == SelectContext.MAIN and t == OptionType.END and main_has_basic_bench_play(obs):
            s = min(s, -50000)

    # Dead-active tempo: prefer powering a bench attacker or retreating to it.
    a = active(me(obs))
    dead = bool(a and ((getattr(a, "hp", 0) <= (getattr(a, "maxHp", getattr(a, "hp", 0)) or 1) * 0.25)
                       or (energy_count(a) == 0 and getattr(a, "hp", 0) > 0
                           and getattr(a, "hp", 0) <= (getattr(a, "maxHp", getattr(a, "hp", 0)) or 1) * 0.4)))
    if ctx == SelectContext.MAIN and dead:
        c = option_target(obs, opt)
        if t == OptionType.ATTACH and c in bench(me(obs)) and c and c.id in (DURALUDON, ARCHALUDON_EX):
            s = max(s, 35000)
        elif t == OptionType.RETREAT and any(p.id in (DURALUDON, ARCHALUDON_EX) for p in bench(me(obs))):
            s = max(s, 30000)
        elif t == OptionType.END and not getattr(obs.current, "energyAttached", False):
            if METAL_ENERGY in hand_ids(obs):
                s = min(s, -15000)

    # Concrete Crustle safeguards.
    if detect_matchup(obs) == "crustle":
        c = option_card(obs, opt)
        cid = c.id if c else None
        if t == OptionType.EVOLVE and cid == ARCHALUDON_EX:
            s = min(s, -10000)
        if t == OptionType.ATTACK and getattr(opt, "attackId", None) == METAL_DEFENDER:
            s = min(s, -10000)
        if ctx == SelectContext.TO_HAND and t == OptionType.CARD and cid == ARCHALUDON_EX:
            s = min(s, -3000)
        if t == OptionType.PLAY and cid == RELICANTH:
            s = min(s, -5000)

    # Last-turn Mega Brave can leave the opponent awkward; preserve Boss.
    if t == OptionType.PLAY and option_card(obs, opt) and option_card(obs, opt).id == BOSS and _last_opp_attack == MEGA_BRAVE:
        s = min(s, 2000)

    return float(s)


def legal_rank(obs):
    opts = list(obs.select.option or [])
    scored = []
    for i, o in enumerate(opts):
        try:
            s = score_option(obs, o)
        except Exception:
            s = -1e9
        scored.append((s, -i, i))
    scored.sort(reverse=True)

    chosen = []
    minc = int(getattr(obs.select, "minCount", 0) or 0)
    maxc = int(getattr(obs.select, "maxCount", 0) or 0)
    # Never exceed legal max. Skip negative choices unless minCount requires them.
    for s, _, i in scored:
        if len(chosen) >= maxc:
            break
        if s < 0 and len(chosen) >= minc:
            continue
        chosen.append(i)
    if len(chosen) < minc:
        chosen = [i for _, _, i in scored[:minc]]
    return chosen


def legal_fallback(obs_dict):
    sel = obs_dict.get("select")
    if sel is None:
        return list(DECK)
    n = len(sel.get("option", []) or [])
    mn = int(sel.get("minCount", 0) or 0)
    mx = int(sel.get("maxCount", 0) or 0)
    if n == 0 or mx <= 0:
        return []
    k = min(max(mn, 1), mx, n)
    return list(range(k))


def agent(obs_dict):
    global _last_opp_attack
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return legal_fallback(obs_dict)

    if obs.select is None:
        _last_opp_attack = None
        return list(DECK)

    try:
        update_opponent_attack(obs)
    except Exception:
        pass

    try:
        out = legal_rank(obs)
        n = len(obs.select.option)
        mn = int(getattr(obs.select, "minCount", 0) or 0)
        mx = int(getattr(obs.select, "maxCount", 0) or 0)
        out = [int(i) for i in out if 0 <= int(i) < n]
        if len(out) < mn:
            out = legal_fallback(obs_dict)
        return out[:mx]
    except Exception:
        return legal_fallback(obs_dict)
