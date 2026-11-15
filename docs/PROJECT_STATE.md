# PROJECT_STATE: Kaggle PTCG AI Battle Challenge Playground (as of 2026-11-15)

Goal: #1 on the ladder (top about 1577 on 11-05; your target 1500+). Deadline 2027-01-08. 5 submissions/day, only the latest 2 stay active.

## Status (ladder, your screenshot 11-14)
B2 1152.8 (best, 1 day old) | B 1131.1 | A2 982.7 | A3a 905.1 | A3b 837.8 | A 809.2 | C 753.7
A3b went 942 -> 838 as it played more, so early scores overstate. Treat B and B2 as about 1130-1150 until B2 has several days of games.

## What each submission is
- B  = 7th-place agent (Giovanni / Z3rox-dev, MIT) + Playground engine, Dragapult/Blaziken deck "F" (deck hash dc058243). Best family.
- B2 = B plus estimated lookup rows for the 164 new-engine cards (ids 1268-1431) and 199 new attacks. Only the four out/*.npy tables differ. Old-card rows are bit-identical; actions differ on 5/2313 old-card states (0.22%). Mirror B2 vs B: 65-65 over my runs (no regression).
- B2_licensed = B2 + LICENSE-7th-place + NOTICE.md. Same play.
- B3 = B2_licensed + in-band reset of recurrent state and trackers between the games of a match (docs/RESET_FIX.md). Fixed beat current B2 108-92 over 200 ladder-faithful matches (54%, not significant). Use B3 for any resubmission.
- A family = Dipam net (3rd place, MIT) with seat fix, UNK mapping, KO guard. Ceiling about 940-985 on this ladder.
- C = Slowking exploiter from the 7th-place repo: 753.7, not worth keeping.

## Decisions and evidence
- Playground engine: card ids <= 1267 and attack ids <= 1556 are unchanged vs round 1; new ids are 1268-1431 / 1557-1755.
- B's net has no card-id embeddings (id bags deleted). It reads cards through 4 lookup tables (text embedding 768-d, effect atoms 267-d, for cards and attacks) plus engine stats. New ids were zero rows in B.
- B relies heavily on those tables: with all tables zeroed it loses 175 of 177 games to itself.
- Estimated rows: text = TF-IDF -> ridge (cosine 0.74 in cross-validation), atoms = kNN over similar old cards. Reading the opponent's unfamiliar cards: 14.3% of decisions differ from the true-table decisions with estimates vs 16.1% with zeros. A small gain, which matches the small ladder gain.
- Deck swaps did not help (A3a/A3b). Local matches between decks do not predict ladder results.
- My hand-built new-card decks (N1-N4) piloted by B2 lose badly to B + deck F (0-77 for N3). Not shippable.
- The 7th-place energy search (FUNCNET4_ESEARCH) is switched OFF by its author. Official interpreter: time pool 600 s per match. Locally the search costs about 85 s per game and cannot be validated on 2 CPUs, so it stays off.
- The old local arena reset agents before every game; the ladder does not (deck asked once per match). Use tools/bo3_arena.py.

## Blockers
1. No real ladder meta: deck "E" (hash 1652a67e, about 34% share on 10-28) has an unknown list. Needs Kaggle's daily episode export.
2. No GPU here and Hugging Face downloads are blocked, so no training and no true text embeddings for new cards.

## Next
1. You run ladder_meta_extract.py in a Kaggle notebook on the daily episode export (or on B2's episodes) and send me ladder_meta_compact.json plus your team name.
2. With it: real deck shares and win rates, B2's matchups by opponent deck, the thinking-time pool per game.
3. Then: hand-annotate the most-played new cards' effect features (higher value than all 164), small deck tweaks for deck F against the real meta.

## Risks / rules
- Ladder noise is about +-25 after a day. Do not read a 20-point gap as real.
- Rules 2.4 / 3.6b: do not publish deck lists, card data or the simulator. Keep ladder_meta_compact.json private.
- MIT notices: ship the LICENSE file (done in B2_licensed; earlier tarballs lack it).

## Paths (sandbox, may be lost)
Sandbox paths were temporary; scripts now take paths from arguments or env (PTCG_V2_JSON, PTCG_B_OUT). Create a `logs/` dir before running the arena.
