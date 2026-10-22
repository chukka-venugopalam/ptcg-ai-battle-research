# ARCHIVE_DIFFS
Base = Bv1 baseline. Exact bytes extracted from the supplied tarballs; nothing modified.


## candidate vs Bv1
Identical files: 30. Only in Bv1: none. Only in candidate: none. **Changed (7):**
| path | size Bv1→candidate | sha256 Bv1→candidate |
|---|---|---|
| cg/cg.dll | 1525248→1659392 | 9ea2b0a75102→e084185a17fa |
| cg/game.py | 2225→2434 | 3bd3d4f4a369→2558b876af59 |
| cg/libcg-arm64.so | 1300584→1413168 | 030b4728ce9f→95628f0bd20a |
| cg/libcg.dylib | 1256448→1345208 | a9a78cd0e61f→da53cb331277 |
| cg/libcg.so | 1342400→1458504 | ffd89bf92352→c699020e086e |
| cg/sim.py | 2273→2385 | 1555f57f5d22→e83ca0430099 |
| sa/policynet.py | 22264→23036 | 2545ed74c91e→00b8cfc210d2 |

### binary `cg/cg.dll`: sizes 1525248→1659392 (hash only; native/SDK file)

### source diff `cg/game.py` (25 lines)
```diff
--- Bv1/cg/game.py
+++ candidate/cg/game.py
@@ -17,5 +17,5 @@
 
 
-def battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict, StartData]:
+def battle_start(deck0: list[int], deck1: list[int], reverse_player = False) -> tuple[dict, StartData]:
     """Start the battle.
 
@@ -23,4 +23,5 @@
         deck0: List of card IDs included in the first player’s deck.
         deck1: List of card IDs included in the second player’s deck.
+        reverse_player: When True, the second player can choose whether to go first or second.
 
     Returns:
@@ -33,5 +34,8 @@
     cards = deck0 + deck1
     arg = (ctypes.c_int * len(cards))(*cards)
-    start_data = lib.BattleStart(arg)
+    if reverse_player:
+        start_data = lib.BattleStartReverse(arg)
+    else:
+        start_data = lib.BattleStart(arg)
     Battle.battle_ptr = start_data.battlePtr
     if Battle.battle_ptr == None or Battle.battle_ptr == 0:
```

### binary `cg/libcg-arm64.so`: sizes 1300584→1413168 (hash only; native/SDK file)

### binary `cg/libcg.dylib`: sizes 1256448→1345208 (hash only; native/SDK file)

### binary `cg/libcg.so`: sizes 1342400→1458504 (hash only; native/SDK file)

### source diff `cg/sim.py` (10 lines)
```diff
--- Bv1/cg/sim.py
+++ candidate/cg/sim.py
@@ -33,4 +33,7 @@
 lib.BattleStart.restype = StartData
 lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
+
+lib.BattleStartReverse.restype = StartData
+lib.BattleStartReverse.argtypes = [ctypes.POINTER(ctypes.c_int)]
 
 lib.AgentStart.restype = ctypes.c_void_p
```

### source diff `sa/policynet.py` (55 lines)
```diff
--- Bv1/sa/policynet.py
+++ candidate/sa/policynet.py
@@ -118,7 +118,22 @@
         lut = self.lut[table]
         a = np.asarray(ids)
-        # Out-of-range cannot happen -- features.py clamps to 0 -- but an id past
-        # the table falls to UNK rather than raising in the middle of a match.
+        # Vocab checkpoints map unseen IDs to UNK row 1.
         return np.where(a < len(lut), lut[np.clip(a, 0, len(lut) - 1)], 1)
+
+    def _emb_row(self, table: str, ids):
+        """Map IDs to safe rows even for pre-v7 identity-mapped checkpoints.
+
+        The current R2 card pool contains IDs beyond the 1300-row embedding
+        tables used by this checkpoint. State/bag feature builders already map
+        those IDs to row 0; option card and target embeddings must do the same
+        rather than raising IndexError and forcing the agent's broad fallback.
+        Vocab-remapped checkpoints preserve their normal UNK mapping.
+        """
+        row = np.asarray(self._m(table, ids))
+        size = getattr(self, table).shape[0]
+        if self.lut is not None:
+            # _m maps OOV IDs to row 1, but keep the guard for malformed maps.
+            return np.where((row >= 0) & (row < size), row, 1)
+        return np.where((row >= 0) & (row < size), row, 0)
 
     @property
@@ -166,15 +181,15 @@
             # is appended, so a v2 net simply does not see it.
             oenc[i, :ow] = od[:ow]
-            oenc[i, ow:ow + emb] = self.card_emb[self._m("card_emb", cid)]
+            oenc[i, ow:ow + emb] = self.card_emb[self._emb_row("card_emb", cid)]
             oenc[i, ow + emb:ow + 2 * emb] = self.atk_emb[
-                self._m("atk_emb", aid)]
-            oenc[i, ow + 2 * emb:] = self.card_emb[self._m("card_emb", tid)]
+                self._emb_row("atk_emb", aid)]
+            oenc[i, ow + 2 * emb:] = self.card_emb[self._emb_row("card_emb", tid)]
 
         dense, bags = featurize(state, me)
         parts = [dense,
-                 self.slot_emb[self._m("slot_emb", bags["slots"])].reshape(-1)]
+                 self.slot_emb[self._emb_row("slot_emb", bags["slots"])].reshape(-1)]
         for name in _BAGS:
             b = bags[name]
-            parts.append(self.bag_emb[self._m("bag_emb", b)].mean(axis=0)
+            parts.append(self.bag_emb[self._emb_row("bag_emb", b)].mean(axis=0)
                          if len(b)
                          else np.zeros(self.bag_emb.shape[1],
@@ -190,5 +205,5 @@
             xids = np.where(self.x_mask[N_EXTRA:] > 0, xids, 0)
         parts.append(xd)
-        parts.append(self.slot_emb[self._m("slot_emb", xids)].reshape(-1))
+        parts.append(self.slot_emb[self._emb_row("slot_emb", xids)].reshape(-1))
         # ...and the v5 pool goes after v4, same rule (optfeat.pool_width).
         if self.n_pool:
```

### Runtime flags (exact source lines)
AGENT_KWARGS Bv1: `{'main.py:28': {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}}`
AGENT_KWARGS candidate: `{'main.py:28': {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}}`
```diff
```
### Deck multiset: identical

## B2 vs Bv1
Identical files: 36. Only in Bv1: none. Only in B2: none. **Changed (1):**
| path | size Bv1→B2 | sha256 Bv1→B2 |
|---|---|---|
| main.py | 4597→4596 | c7055754b58c→3c5fab333f87 |

### source diff `main.py` (9 lines)
```diff
--- Bv1/main.py
+++ B2/main.py
@@ -26,5 +26,5 @@
 # measure 0.427 against it -- report/EVIDENCE.md 8f). Pinning the (net, flags)
 # PAIR here keeps both configurations shippable without a global default flip.
-AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}
+AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': True}
 
 
```

### Runtime flags (exact source lines)
AGENT_KWARGS Bv1: `{'main.py:28': {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}}`
AGENT_KWARGS B2: `{'main.py:28': {'chip_targeting': False, 'energy_spread': False, 'counter_source': True}}`
```diff
--- Bv1
+++ B2
@@ -1 +1 @@
-main.py:28: AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}
+main.py:28: AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': True}
```
### Deck multiset: identical

## Bv2 vs Bv1
Identical files: 30. Only in Bv1: none. Only in Bv2: none. **Changed (7):**
| path | size Bv1→Bv2 | sha256 Bv1→Bv2 |
|---|---|---|
| cg/cg.dll | 1525248→1659392 | 9ea2b0a75102→e084185a17fa |
| cg/game.py | 2225→2434 | 3bd3d4f4a369→2558b876af59 |
| cg/libcg-arm64.so | 1300584→1413168 | 030b4728ce9f→95628f0bd20a |
| cg/libcg.dylib | 1256448→1345208 | a9a78cd0e61f→da53cb331277 |
| cg/libcg.so | 1342400→1458504 | ffd89bf92352→c699020e086e |
| cg/sim.py | 2273→2385 | 1555f57f5d22→e83ca0430099 |
| main.py | 4597→4609 | c7055754b58c→6ed0be986b3c |

### binary `cg/cg.dll`: sizes 1525248→1659392 (hash only; native/SDK file)

### source diff `cg/game.py` (25 lines)
```diff
--- Bv1/cg/game.py
+++ Bv2/cg/game.py
@@ -17,5 +17,5 @@
 
 
-def battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict, StartData]:
+def battle_start(deck0: list[int], deck1: list[int], reverse_player = False) -> tuple[dict, StartData]:
     """Start the battle.
 
@@ -23,4 +23,5 @@
         deck0: List of card IDs included in the first player’s deck.
         deck1: List of card IDs included in the second player’s deck.
+        reverse_player: When True, the second player can choose whether to go first or second.
 
     Returns:
@@ -33,5 +34,8 @@
     cards = deck0 + deck1
     arg = (ctypes.c_int * len(cards))(*cards)
-    start_data = lib.BattleStart(arg)
+    if reverse_player:
+        start_data = lib.BattleStartReverse(arg)
+    else:
+        start_data = lib.BattleStart(arg)
     Battle.battle_ptr = start_data.battlePtr
     if Battle.battle_ptr == None or Battle.battle_ptr == 0:
```

### binary `cg/libcg-arm64.so`: sizes 1300584→1413168 (hash only; native/SDK file)

### binary `cg/libcg.dylib`: sizes 1256448→1345208 (hash only; native/SDK file)

### binary `cg/libcg.so`: sizes 1342400→1458504 (hash only; native/SDK file)

### source diff `cg/sim.py` (10 lines)
```diff
--- Bv1/cg/sim.py
+++ Bv2/cg/sim.py
@@ -33,4 +33,7 @@
 lib.BattleStart.restype = StartData
 lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
+
+lib.BattleStartReverse.restype = StartData
+lib.BattleStartReverse.argtypes = [ctypes.POINTER(ctypes.c_int)]
 
 lib.AgentStart.restype = ctypes.c_void_p
```

### source diff `main.py` (9 lines)
```diff
--- Bv1/main.py
+++ Bv2/main.py
@@ -26,5 +26,5 @@
 # measure 0.427 against it -- report/EVIDENCE.md 8f). Pinning the (net, flags)
 # PAIR here keeps both configurations shippable without a global default flip.
-AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}
+AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False, 'sym_k': 8}
 
 
```

### Runtime flags (exact source lines)
AGENT_KWARGS Bv1: `{'main.py:28': {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}}`
AGENT_KWARGS Bv2: `{'main.py:28': {'chip_targeting': False, 'energy_spread': False, 'counter_source': False, 'sym_k': 8}}`
```diff
--- Bv1
+++ Bv2
@@ -1 +1 @@
-main.py:28: AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}
+main.py:28: AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False, 'sym_k': 8}
```
### Deck multiset: identical
