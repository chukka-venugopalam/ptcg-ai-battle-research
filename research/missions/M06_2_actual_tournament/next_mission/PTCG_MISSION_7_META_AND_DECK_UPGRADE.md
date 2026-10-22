# Mission 7 — Build a materially stronger R2 agent (evidence-gated)

## Objective

Move beyond small `counter_source` / deck-flag experiments and seek a genuinely stronger agent. The target is to work toward a 1000+ public skill rating, but **never promise a score or upload a candidate without evidence**. Previous local runs found no statistically decisive improvement: (1) 128-game screening across near-identical agents, (2) 100-game confirmation of `counter_source`, and (3) 100-game test of two Shadowy Darkness Energy. Treat these results as the current baseline evidence.

## Required inputs

Use the attached Mission 6.2 evidence bundle and the previously supplied `PTCG_MISSION_6_1_RECOVERY_BUNDLE.zip`, including:
- exact R2 compatibility candidate and the original Bv1/B2/Bv2 archives;
- official R2 SDK / native engine / `EN_Card_Data_R2_full.csv`;
- write-once full game-level CSVs and the statistical report;
- prior four research packages / Mission 5 and 6 reports if present.

Before analysis, list every file received, verify SHA-256 hashes, and identify any missing inputs. Do not claim files were missing without showing a real input preflight. Do not silently substitute a different archive.

## Phase A — Establish the strongest attainable baseline

1. Re-read the Mission 6.2 report and reproduce its win counts, intervals and zero-failure checks from the raw CSVs.
2. Inspect the candidate `main.py`, feature builder, policy network, card/attack ID transforms, and complete SDK diff. Preserve the existing candidate exactly as the rollback baseline.
3. Separate safety from strength: the ID bounds guard is a valid robustness patch, but it does not train the model to recognize high IDs. The current card table reaches ID 1431; the model's card embedding has 1300 rows, so high IDs are generic/OOV unless a model/feature change is made.
4. Review the actual current competition specification: https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description. It describes rating updates after episodes/games and only the latest two submissions as active. Explain the mismatch between local one-game rows and any BO3/series test metric. Never equate local win rate mechanically to Kaggle rating points.

## Phase B — Inspect public approaches, not just their titles

Open the Kaggle competition Code listing: https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/code. Prioritize currently accessible notebook sources with high reported public scores and actionable artifacts, including titles such as:
- `Pokemon Challenge Ultra 900+`;
- `PTCG AI Battle Playground: Baseline + New Cards`;
- `PTCG Bo3 agent, one-line seat fix | LB 1132.4`;
- `PTCG Bo3 Gauntlet: Your Agent vs 6 Public Bots`;
- `PTCG Daily Meta: Matchups, Turn Order & Top Decks`;
- `PTCG Meta: What Changed & What to Test Next`.

For every notebook studied, record the actual author/URL/version, whether the source cells or downloadable artifact were obtained, its license, the deck list, policy/heuristic architecture, feature choices, and the exact ideas that transfer. If source code is inaccessible, explicitly mark it inaccessible. Do not state or imply that code was inspected just because a title or score appears in search results. Do not copy assets without recording license/provenance. Do not invent experiments or metrics.

## Phase C — Find a strategic gap in the current R2 pool

1. Build a card-pool inventory from the R2 CSV and classify cards by role, type, energy cost, search/setup, acceleration, damage scaling, bench protection, disruption, and synergy. Resolve duplicate rows for a card by `Card ID`; keep all attacks/effects when analyzing a Pokémon with multiple attack rows.
2. Compare the old 60-card deck to cards available in R2 and to any verified stronger public deck lists. The old deck has a maximum card ID of 1259 and most supplied agents share the same deck and same learned `sa/policy_net.npz`; past small flag changes are therefore poor evidence of true meta improvement.
3. Evaluate plausible new-card hypotheses from text in the supplied official data, including but not limited to Mega Darkrai ex (1389), Shadowy {D} Energy (1272), Risky Ruins (1260), and Dark Bell (1416). These are research leads only—not presumed upgrades. For each, document deck legality, energy acceleration, evolution/setup consistency, prize liability, attack costs, what the current policy can actually recognize, and whether the policy will select the relevant actions.
4. Do not put many high-ID cards into the old checkpoint without auditing all places where card/attack/target IDs are embedded. If new cards map to a single generic embedding, either provide a defensible feature/heuristic path, extend/train the embedding model using real data, or abandon the idea. Never silently map an entire new deck to row 0 and call it new-card intelligence.
5. Pick at most **one** primary strategic hypothesis for the first test. State a falsifiable hypothesis and predicted observable effect before running it.

## Phase D — Build diverse opponents and evaluate

1. Use real public agent code/decks as opponents if legitimately accessible. Record exact archive hashes. If unavailable, create 2–3 clearly labeled local archetype opponents from verified public deck evidence or well-defined documented alternatives; label them *synthetic test bots*, not public leaderboard agents.
2. Keep the current R2 compatibility candidate as baseline. Run a full deck-handshake and self-play validation for each new build. Verify 60-card count and legality with the current R2 engine.
3. First screen the candidate over at least 100 complete games against the reference candidate, with seats balanced 50/50. Also test against at least two materially different decks/policies. Save a complete CSV only after all games finish; ensure its row count matches the predeclared number. Include winner, both seats, agent version/hash, deck hash, fallbacks, invalid actions, engine errors and exceptions.
4. Use a test budget appropriate to the local runtime. Compute Wilson 95% intervals and exact two-sided binomial tests for each pair; report ties separately if any. Do not claim a statistical win if the interval includes 50% or p≥0.05. If the first screen is positive, run a **fresh independent confirmation** rather than reusing the same games. A prudent first-pass promotion bar is at least 61 wins out of 100 in a 100-game balanced confirmation (check the exact CI/p-value, don't use the count alone), plus evidence it doesn't collapse against the other opponent decks.
5. RNG seeding: the current native wrapper has not exposed a controllable seed. If this remains true, state that games are fresh stochastic trials and not same-seed paired. Never fabricate paired-seed results.
6. Analyze action logs and failure/turn-level behavior if available. Explain what caused wins/losses and which action-selection mechanism changed. A raw win count is not enough to diagnose strategy.

## Phase E — Candidate and upload gate

**Default decision is NO-GO.** Create a new `submission.tar.gz` only if all conditions pass:
- a clear, reproducible strategic change is implemented;
- R2 compatibility, no-`__file__` execution, 60-card handshake and current-engine legality pass;
- the full tournament CSV reconciles exactly with the predeclared game count;
- independent, seat-balanced confirmation is materially positive against baseline;
- results are not explained solely by one seat, a bug, fallback behavior or a single weak synthetic opponent;
- archive contents, top-level `main.py` and `deck.csv`, SHA-256 manifest and rollback archive are all verified.

If those gates fail, do not produce an upload-ready tarball. Deliver a NO-GO report and the next most promising experiment instead. Before any upload, remind the user that Kaggle only keeps the latest two submissions active, so a new upload can remove an existing active agent.

## Required deliverables

1. `MISSION_7_DECISION.md`: GO/NO-GO, exact evidence and caveats.
2. `PUBLIC_NOTEBOOK_AUDIT.csv`: title, author, URL, code access status, license, verified transferable ideas, no guessed fields.
3. `R2_CARD_POOL_INVENTORY.csv` and a concise deck-synergy memo.
4. `HYPOTHESIS_REGISTER.csv`: hypothesis, expected effect, confounders, status and exact evidence.
5. Raw write-once screen/confirmation game CSVs and summary tables.
6. Any actual candidate archive only when it passes the GO gate, with a SHA-256 manifest, validation report and top-level listing. Otherwise explicitly state no submission archive was made.
7. `MISSION_7_EVIDENCE_BUNDLE.zip` with the above files plus a README and a list of source hashes.

## Non-negotiable evidence rules

- Never invent or infer match outcomes, source code, opponent identities, experiments, seeds, skill ratings or metrics.
- Never say “wins” based on engine smoke tests or fixture agents.
- Keep previous archives unchanged and make candidate changes in separate directories.
- Keep test sets independent for screening and confirmation.
- Clearly label what was directly executed, what came from source material, what was web-researched, and what remains unknown.
- The objective is actual competitive improvement—not producing a new file at any cost.
