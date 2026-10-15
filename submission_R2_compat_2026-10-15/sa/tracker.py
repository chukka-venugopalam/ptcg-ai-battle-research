"""Cross-turn knowledge tracker fed by observation logs.

Tracks which cards we know are in the opponent's hand (cards that moved there
face-up: bounced from discard/field, revealed searches, etc.). Serial numbers
make this exact until a face-down move out of their hand voids it.
"""
from __future__ import annotations

# LogType ints
_MOVE_CARD = 6
_MOVE_CARD_REVERSE = 7
_PLAY = 10
_ATTACH = 11
_EVOLVE = 12

# AreaType ints
_DECK = 1
_HAND = 2


class Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.known: dict[int, int] = {}        # serial -> cardId (opp cards)
        self.opp_hand_serials: set[int] = set()
        self.selects_seen = 0
        self.last_turn = -1

    def maybe_reset(self, obs: dict) -> None:
        cur = obs.get("current")
        if cur is None:
            self.reset()
            return
        turn = cur.get("turn", 0)
        if turn < self.last_turn:
            self.reset()
        self.last_turn = turn

    def update(self, obs: dict) -> None:
        cur = obs.get("current")
        if cur is None:
            return
        me = cur["yourIndex"]
        opp = 1 - me
        self.selects_seen += 1
        for log in obs.get("logs") or []:
            t = log.get("type")
            pi = log.get("playerIndex")
            if pi != opp:
                continue
            serial = log.get("serial")
            if t == _MOVE_CARD:
                cid = log.get("cardId")
                if serial is not None and cid:
                    self.known[serial] = cid
                    if log.get("toArea") == _HAND:
                        self.opp_hand_serials.add(serial)
                    else:
                        self.opp_hand_serials.discard(serial)
            elif t == _MOVE_CARD_REVERSE:
                if log.get("fromArea") == _HAND:
                    # face-down exit from hand (e.g. hand shuffled into deck):
                    # we no longer know what's there
                    self.opp_hand_serials.clear()
            elif t in (_PLAY, _ATTACH, _EVOLVE):
                if serial is not None:
                    self.opp_hand_serials.discard(serial)

        # never claim to know more than their hand size
        hand_count = cur["players"][opp]["handCount"]
        while len(self.opp_hand_serials) > hand_count:
            self.opp_hand_serials.pop()

    def known_opp_hand(self) -> list[int]:
        return [self.known[s] for s in self.opp_hand_serials
                if s in self.known]
