# Between-game reset (found 2026-11-15)

**Finding.** The official `cabt` interpreter (kaggle-environments 1.33.0, `envs/cabt/cabt.py`) asks each
agent for its deck once per match (observation with `select=None`). Games 2 and 3 of the best-of-3 start with
`battle_start(...)` and NO call to the agent. Agents that reset their per-game memory only on `select=None`
(B / B2: recurrent state, prize tracker, KO tracker; A2: InfoTracker) therefore carry game-1 memory into
games 2 and 3. The old local arena reset before every game, which hid this.

**Other facts from the interpreter.** `remainingOverageTime` starts at 600 s; `actTimeout` is 0. The reverse-start
flag for game k is "games played so far is odd". `current.round`, `players[].win` and `draw` are filled in by the
interpreter on every observation. (Kelly Mao notes the ladder resets the 600 s pool per game.)

**Fix.** On every non-deck observation compare `current.round` with the last seen value (and treat a falling
`current.turn` as a new game); on change run the same reset as the deck request.

```python
def _detect_new_game(obs_dict):
    cur = obs_dict.get("current") or {}
    rnd, turn = cur.get("round"), cur.get("turn")
    new = False
    if rnd is not None:
        new = new or (_last_round is not None and rnd != _last_round)
        _last_round = rnd
    if turn is not None:
        new = new or (_last_turn is not None and turn < _last_turn)
        _last_turn = turn
    if new:
        reset_recurrent_state_and_trackers()
```

**Measured (bo3_arena.py, 200 matches, same deck, fixed B2 vs current B2).** Fixed won 108-92 (54%), p about 0.17.
Game-by-game there is no extra edge in games 2/3 beyond game 1 (game 1 itself showed 111-89 for the identical
agent, so the noise is about +-3.5 points). The reset fires exactly once per extra game, no false triggers.
Conclusion: correct and harmless, gain not distinguishable from zero for B. A2's tracker fix is built but untested
(too slow to run here).
