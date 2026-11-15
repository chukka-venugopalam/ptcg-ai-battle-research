# PTCG AI Battle Challenge Playground: research and agents

Work toward a top ladder position in Kaggle's *The Pokémon Company - PTCG AI Battle Challenge Playground*
(best-of-3 Pokémon TCG agent ladder). Deadline 2027-01-08.

## Status (2026-11-14)

| Submission | What it is | Ladder score |
|---|---|---|
| B2 | 7th-place agent (round 1) + estimated lookup rows for the Playground's new cards | 1152.8 (best) |
| B3 | B2 + in-band reset between the games of a match (see docs/RESET_FIX.md) | not yet submitted |
| B | same agent without the new-card rows | 1131.1 |
| A2 | Dipam's 3rd-place net with seat fix, unknown-card mapping and KO guard | 982.7 |
| A3a / A3b / A / C | deck swaps, original net, exploiter | 905.1 / 837.8 / 809.2 / 753.7 |

Ladder top was about 1577 on 11-05. Scores drift for the first few days; treat differences under about 25 points as noise. Full history: `docs/LADDER_LOG.md`.

## What was learned

- The Playground engine keeps round-1 card ids (<= 1267) unchanged and adds 164 cards and 199 attacks.
- Agents built on the round-1 pool read new cards as blank or unknown. Filling in estimated rows (B2) was a small, safe gain.
- Deck swaps and a local deck matrix did not predict ladder results.
- Details, failed approaches and open items: `docs/DECISIONS.md`. Compact state for a new session: `docs/PROJECT_STATE.md`.

## Repository layout

```
docs/                      project state, ladder log, decisions
tools/
  ladder_meta_extract.py   Kaggle episode replays -> compact deck/result JSON
  bo3_arena.py, worker.py  ladder-faithful best-of-3 arena (deck asked once per match, no reset between games)
  proc_arena.py            older arena that resets agents before EVERY game (not ladder-faithful; use bo3_arena.py)
  shadow2.py               counts action disagreement between two agents on identical games
patch_new_card_rows/       estimates lookup rows for new cards (TF-IDF ridge for text, kNN for effect features)
```

Not in this repo, by competition rules (2.4 / 3.6b) and licensing: card data, deck lists, model weights, lookup tables, the engine, and the third-party agent code.

## How to use

**Ladder meta (needs Kaggle episode replays, no engine):**
```
python tools/ladder_meta_extract.py /path/to/episodes --team "YOUR TEAM" -o ladder_meta_compact.json
python tools/ladder_meta_extract.py --selftest
```
The output contains deck lists: keep it private.

**Local arena (needs the competition engine `cg/` and an agent folder with `main.py` + `deck.csv`):**
```
mkdir logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/bo3_arena.py <dir containing cg/> <agent A dir> <agent B dir> 100
```
Use absolute paths; MATCH_OFFSET=<even number> runs a second slice in parallel. Setting the thread
variables makes games about 15x faster. The arena mirrors the official cabt interpreter: the agent is
asked for its deck once per match and is NOT reset between games 2 and 3.

**New-card rows (needs the engine card dump and the agent's original tables):**
```
export PTCG_V2_JSON=path/to/engine_cards.json PTCG_B_OUT=path/to/agent/out/
python patch_new_card_rows/build_new_rows.py OUTDIR
```

## Submitting

A submission is `submission.tar.gz` with `main.py` and `deck.csv` at the top level, up to 197.7 MiB. Five per day; only the latest two stay active. Include the LICENSE files of any third-party code in the archive.

## Credits and licences

- Round-1 7th-place agent: Giovanni (Z3rox-dev), MIT.
- Policy net and encoder: Dipam Chakraborty (3rd place), MIT; Playground port by Cristobal Tudela.
- Replay format and ladder analysis ideas: Kelly Mao (MIT) and public Kaggle notebooks.
- Engine and card data: The Pokémon Company / Kaggle, competition use only.
- Code in this repo: yours to license as you choose; add a LICENSE file.
