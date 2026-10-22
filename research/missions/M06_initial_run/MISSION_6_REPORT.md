# MISSION_6_REPORT

## 1. Conclusion (observed)
**Decision B. Do not upload. No candidate archive exists.** The Mission 6 handoff bundle never reached the sandbox, so **0 real BO3 matches were played by any of your agents** (R2 compat candidate, Bv1, B2, Bv2) and none of their archives, decks or weights were inspected. No cause of the 826.0 → 780.4 movement is identified or claimed.
What was done instead, on a real engine (public PyPI `kaggle-environments==1.33.0`, hash-identical to Mission 3's recorded environment; equivalence to your official R2 package is UNKNOWN):
- 78 instrumented real BO3 matches with fixture agents (38 in-process, 40 through the tournament driver): all completed, 0 failures, canonical BO3 rule = engine reward in 78/78.
- Mission 3's full suite now gives **77 passed / 0 skipped** (it was 55 / 22 skipped without an engine).
- Built and validated: a tournament driver that hard-fails without the real engine, an exporter (one row per BO3 match + per-game diagnostics), a mirror-invariance seat test, a one-command Phase A/B script (dry-run on fixture archives). 32 tests pass in this package.

## 2. New verified facts that shape the next steps (local engine = `cabt.py` source + real runs)
1. **BO3 rule:** after every game, `remain = bo − games_played`; a side wins when its game wins exceed the other's plus `remain`; if `remain ≤ 0` and neither leads, the match is a draw (reward 0). So 1W-0L-2D is a match **win**; W,L,D is a **draw**.
2. **Seat structure:** agents keep their player index for the whole BO3. The go-first/second decision (`IS_FIRST`, context 41) is offered to seat 0 in games 1 and 3 and to seat 1 in game 2 (38/38 matches). Option types (1,2) = (YES, NO); YES makes the chooser go first (10/10), NO second (0/11). An agent whose first/second policy or any index handling is seat-dependent will behave differently by seat. `yourIndex`, `firstPlayer`, `players[]`, and every `playerIndex` are absolute.
3. **One illegal selection ends the whole match as a loss** (`battle_select` failure → INVALID, reward −1); timeouts/errors likewise. Candidate faults therefore cost matches, not games.
4. Engine ids: cards 1–1431, attacks 1–1755, 29 ACE SPEC, 679 Basic. Any embedding smaller than these needs an explicit id→slot map (Bv1 was reported with n_pool=172): the audit tool checks it, but it has not run on your archives.
5. Documented SDK (official docs v0.1.0) lists `api`/`utils` modules that the PyPI 1.33.0 package lacks (INFERRED: your R2 package may be packaged differently or be newer).

## 3. Corrections to Mission 5 (disclosed)
- Mission 5's game→match rule (win needs 2 game wins) was an unverified assumption and **wrong** for 1-0-2 sequences. Replaced by the cabt.py rule, with tests; the Monte Carlo coverage numbers are unaffected (they operate on match-level cell counts).
- `first_seat` renamed `agent_seat` (it is the player index, constant through a BO3; "who went first" is a per-game attribute).
- Vendored Mission 3 harness (unchanged, hashes in `VENDOR_SHA256.txt`) has two traps the new driver guards: `PTCG_HARNESS_FAKE=1` silently turns `--engine real` into the fake engine; a missing engine yields all-"worker_exception" records instead of aborting; and its built-in report treats a match draw as a non-win. This package analyses with the canonical module instead (no draws occurred in these runs, so no numbers were affected).

## 4. Phase C — public notebooks (none inspected)
| Item | Access result |
|---|---|
| "PTCG Bo3 agent, one-line seat fix \| LB 1132.4" · "Pokemon Challenge Ultra 900+" · "PTCG AI Battle Playground: Baseline + New Cards" · "PTCG Daily Meta" · "PTCG Bo3 Gauntlet" | **UNAVAILABLE.** Kaggle `/code` and `/overview/description` return only meta tags to the fetch tool (JS-rendered); a web search surfaced none of them. No URL, revision, license or code recorded; **what "seat fix" means is unknown** and is not guessed. |
| Official CABT docs (matsuoinstitute.github.io/cabt: index, api) | opened and used (VERIFIED). |
| Unvetted search hits, not opened or used: a third-party GitHub workspace (`TomBombadyl/kaggle_pokemon`) and a Hugging Face Space (`sayidm/pokemon-tcg-sim`) that bundles the competition engine under its own licence | leads only; check licence/permission before use; your standing rule is not to copy public submissions. |

## 5. Hypotheses (HYPOTHESIS_REGISTER.csv)
Order of testing: H1 seat / IS_FIRST / mirror symmetry → H2 id-range and embedding safety → H3 SDK/R2 semantic change → H4 rating noise and filename ambiguity (three rows share `submission.tar.gz`) → H5 deck → H6 policy. None has evidence about *your* agents yet; H1's tool and H2's audit are built and fixture-validated.

## 6. What unblocks the mission (exact)
Attach as separate files if one big zip fails (I cannot see why it did not arrive; this is the second round with no archives):
1. `latest_candidate_R2_compat.tar.gz` (expected sha256 `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`), the Bv1 baseline archive, B2 counter-source, Bv2 sym8.
2. The official R2 package (needed for engine identity; the audit uses whatever engine is installed).
3. `input_manifest.json`. Pasting `sha256sum` output as chat text also works for identity checks.
4. Optional, from an authenticated machine: the CLI outputs listed in LIVE_STATUS.md.
Then: `PY=python ./repro/run_phase_AB.sh BUNDLE OUT 20` → mirror check + IS_FIRST audit → Stage 0/1 per TOURNAMENT_PLAN.md. Statistics: fixed-N 1,600 matches per arm detects +5 pp at 80% power; 1,000 detects only ±6.3 pp.

## 7. Upload recommendation
**Do not upload anything.** Nothing new was evaluated, and an upload evicts one active agent whose identity is UNKNOWN without a live submissions listing.

## 8. Reproduce
`python3 -m venv venv && . venv/bin/activate && pip install kaggle-environments==1.33.0 pytest numpy && python -m pytest -q tests` (32 passed here; Python 3.12.3). Raw evidence: `raw/` (engine hashes, probe output, 5 local replays, 40-match smoke JSONL/CSV, fixture dry-run).
