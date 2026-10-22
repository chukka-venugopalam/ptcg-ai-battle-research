# Playground competition audit — The Pokémon Company PTCG AI Battle Challenge Playground

Accessed 2026-10-10. Evidence tags: **[V]** read in primary text (official page excerpt or fetched page) · **[S]** secondhand (a third-party repo/PR/news item describing the primary) · **[I]** my inference · **[U]** not verified. IDs such as S01 point to `SOURCE_REGISTER.csv`.

## 0. Access log — what this audit is and is not

| Target | Result |
|---|---|
| Playground overview | `web_fetch` returned only HTML metadata (the page is JS-rendered). The text below comes from several mutually consistent search-index excerpts of the official page. [S01] |
| Playground leaderboard | Fetch attempted → metadata only. **No live ratings, ranks or episode counts were obtainable.** [S03] |
| Playground discussion | Fetch attempted → metadata only. Topics unread. [S04] |
| Playground code tab | Search excerpts only (notebook titles and scores). [S02, S35] |
| Playground rules page | Not retrieved. Anything about external data, model/code sharing or team limits *for this competition* is **[U]**. |
| Engine docs | Index page read in full; the `api` / `game` / `sim` sub-pages were not reachable with the tooling used. [S06] |
| Public repos | 5 of the 9 repos you listed had their pages fetched (SiamRahman29, TomBombadyl, wmh, Lawliet-ai, rahulsiiitm). **4 were not inspected** (Ducdata1808, pckhoa206, scha54, abhiiiesh). Extra repos found and read: hotorch, 4dalols (three PR pages), plus excerpts of others. README-level reading only — no source file was opened. |
| Execution | Nothing was run against the engine or SDK; none of your archives or logs were available. The only code executed is the statistics module and the harness's fake-environment plumbing. |

## 1. Mechanics

| # | Item | Finding | Tag / source |
|---|---|---|---|
| 1 | Format | Best of three. The page says "Episodes (games)". | [V] S01 |
| 2 | Local BO3 | `make("cabt", configuration={"bo": 3})` returns one result per match (statuses DONE/DONE, rewards like `[1, -1]`, "2-0"). Whether the live ladder updates ratings once per BO3 match or once per game is **[U]**. | [S] S12, S13 |
| 3 | Rating model | Gaussian N(μ, σ²); new submissions start at **μ0 = 600**; σ shrinks with evidence; wins raise, losses lower, ties even out; score margin is ignored; opponents have similar ratings. | [V] S01 |
| 4 | Submissions | ≤ 5 per day. **Only the latest 2 are tracked and used for the final.** Every agent keeps playing to the end, newer agents far more often. The leaderboard shows only your best-scoring agent. A new submission starts with a scheduled game against itself. | [V] S01 |
| 5 | Package | ≤ 197.7 MiB; files appear in `/kaggle_simulations/agent/`; `tar -czvf submission.tar.gz *`. | [V] S01 |
| 6 | Timeline | Start 2026-09-29 · entry and team-merger deadline 2027-01-01 · **final submission 2027-01-08** · games continue ~2 weeks (to 2027-01-22) · then a **Bradley–Terry tournament on those episodes among remaining active agents** sets the final board. Today is day 12; **90 days** remain. | [V] S01 |
| 7 | Field size | 242 teams in one snapshot, 436 submissions in an earlier one (Simulation: 6,807 teams). | [V] S01, S07 |
| 8 | Lineage | "A continuation of the Featured simulation competition." | [V] S01 |
| 9 | Card pool / SDK | Playground SDK = **R2 pool, 1,431 cards**. PyPI `kaggle-environments` 1.32.7 is a single-game build with 1,267 cards; BO3 and R2 need `kaggle-environments` master. Local tests on the wrong build test a different game. | [S] S11 (S33 confirms 1,267 for the old pool) |
| 10 | Agent contract | `main.py` at the archive root exposing `agent(observation: dict) -> list[int]`. First call has `observation["select"] is None` and must return the 60 card IDs; `deck.csv` = one ID per line, no header. | [V] S06, [S] S11 |
| 11 | Observation | `logs` (event log), `current` (board state; `None` at deck selection), `select` (options; the option **index** is the action; `maxCount` picks must be returned). Own hand visible; opponent hand count only; prizes face-down; discard visible. Multi-select is sequential in practice, with a legal "stop" once `minCount` picks are made. | [V] S06, [S] S27 |
| 12 | Game log handling | `logs` arrive incrementally each call, so trackers must consume every call; a tracker resets when the round index changes. | [S] S15 |
| 13 | Search API | `search_begin / search_step / search_end / search_release` give an in-engine forward model. Public agents ship rollout or PIMC search inside submissions. | [V] S06, [S] S12, S21 |
| 14 | Clock | A pool of **600 s `remainingOverageTime` per agent per BO3 match**. Observed use 56–200 s with a 1.5 s search budget; one 440-step, 563 s-wall match left ≥ 312 s. Simulation-era rules: 10 minutes per player per match, timeout = loss. Hard per-decision limit **[U]**. | [S] S12–S15, S08 |
| 15 | Compute | 2 vCPU and no GPU at inference were reported for the Simulation; Playground **[U]**. Engine/episode tooling needs Python ≥ 3.11. | [S] S22, S19 |
| 16 | Engine quirks | Native RNG is not seedable. Native memory grows across games, so benchmark workers need recycling. Some mirrors with search on stall past turn 1,000. Hand-size-scaled attacks report `damage = 0` in option data. | [S] S12, S13, S14, S21 |
| 17 | Data access | `kaggle competitions episodes <SUBMISSION_ID>`, `replay <EPISODE_ID> -p <PATH>`, `logs <EPISODE_ID> <AGENT_INDEX> -p <PATH>` (accept the rules on the website first). Kaggle publishes **daily "Playground Episodes" datasets**: JSON replays ranked by average agent rating, ≤ 20 GiB/day, with `manifest.csv`. | [S] S31, [V] S05 |
| 18 | Data licence | SDK zip, native libs and R2 CSVs are competition-use-only and must stay out of public repos. | [S] S11, S16 |

## 2. The rating system: known and unknown

**Unknown (must not be assumed):** σ0 and update constants; whether the displayed number is μ or a conservative bound; how a BO3 maps to an update; the episode rate of older agents; whether ties occur in BO3; and whether a team's final rank takes the better of its two agents.

**Noise anchors.** All come from the *Simulation* ladder, so treat them as order-of-magnitude:

* Byte-identical submissions sent a minute apart settled 12–18 rating points apart; the same bytes read 732 in the morning and 571 that afternoon; a day's rating swings 43 points; ≈ 35 games resolve in the first 8 hours. [S21]
* One package read 800.5 and 851.5 two days apart. [S18]
* A Lucario agent's recorded peak of 958.8 did not reproduce — an exact rerun settled near 600; another agent peaked at 753 and settled near 660. [S23]
* The Simulation winner's final rating was 1398.2; the 7th-place agent reached 1226.6. [S09, S10]

**Consequences for keep / replace / retire**

1. The first ~100 episodes (≈ 48 h) are burn-in. Early peaks are partly luck and regress.
2. Differences below ~50 points between two submissions cannot be attributed to code.
3. Under two plausible rating scales the standard error of a rating after *n* episodes (opponents near equal rating) is:

| episodes each | SE, Elo-400 scale | SE, probit-β100 scale | gap resolvable at 95% between two submissions (Elo / probit) |
|---:|---:|---:|---:|
| 35 | 58.7 | 30.0 | 163 / 83 |
| 100 | 34.7 | 17.7 | 96 / 49 |
| 300 | 20.1 | 10.2 | 56 / 28 |
| 1,000 | 11.0 | 5.6 | 30 / 16 |

   Resolving a 25-point difference needs **≈ 390 to 1,480 episodes per submission** depending on the true scale. The ladder cannot arbitrate small differences within weeks; local testing must, and the ladder serves as a veto for large regressions and a sanity check for large gains. Computed by `ptcg_eval_stats.py --tables`.
4. Decision rules are in `NEXT_SUBMISSION_PLAN.md` §4.

## 3. Playground vs Simulation — deltas that change decisions

* **R2 card pool and BO3** (rows 2, 9). Simulation-era decks, ID tables and local SDKs are suspect until re-validated against the R2 CSV.
* **Tiny pool** (242 teams vs 6,807). Ratings are not comparable across the two competitions; a similar-rating matchmaker has few distinct opponents to draw from.
* **Final board = Bradley–Terry over all remaining active agents' episodes** (row 6): a lucky ladder reading at the deadline does not decide the result; strength across the whole active pool does.
* **BO3-specific levers:** whether game 2 and 3 re-prompt for a deck (`select is None` again) and whether Python state persists across games are unknown and potentially valuable — experiment E03 in the matrix.
* Public notebooks currently score 370–732 [S02, S35], and the one published Playground-native search agent sat near 707 after 21 episodes [S14]. These are the only in-pool anchors I could read; your 813.5 is above all of them, but the leaderboard top is unknown.

## 4. Two-slot mechanics and final lock

"Latest 2 submissions" means chronological. Submitting anything evicts the **older** of your two active agents; the evicted agent stops accruing episodes and keeps its last rating, and it may still show as your "best" on the leaderboard.

* **Case A — Bv1 is the older active agent.** A new challenger would evict Bv1, leaving {Archaludon, challenger}. To retire Archaludon and keep Bv1 live you need two submissions: (1) a byte-identical re-upload of the Bv1 archive, then (2) the challenger. The re-upload restarts from μ0 = 600 with large σ, but doubles as an A/A measurement.
* **Case B — Bv1 is the newer active agent.** One submission (the challenger) evicts Archaludon and keeps Bv1.
* Read the timestamps first: `kaggle competitions submissions -c the-pokemon-company-ptcg-ai-battle-challenge-playground`.
* **Final lock:** the two most recent submissions at 2027-01-08 are final. Land the finalists by about 2026-12-25 so each has ≥ 2 weeks before the deadline plus the 2-week tail, and avoid any uploads after 12-26 except emergency rollbacks.

## 5. Open items and how to close them

| Open item | Command / check |
|---|---|
| Leaderboard top, your rank, pool size | `kaggle competitions leaderboard -c the-pokemon-company-ptcg-ai-battle-challenge-playground --show` (check `-h` for flags in your CLI version) |
| Which of your two active agents is older; statuses | `kaggle competitions submissions -c <slug>` |
| Episodes, opponents, rounds per episode | `kaggle competitions episodes <SUBMISSION_ID>` then `replay` / `logs` |
| Deck census of top agents | `kaggle datasets download -d kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-10-01` and later days; parse `manifest.csv` |
| Playground rules (external data, sharing, team size, AI-generated code) | Read the Rules tab; not retrievable here |
| Is an episode a BO3 match? | Count rounds per episode in replays |
| Final rank = better of two agents or both? | Rules tab / organisers; assume better-of-two only as a planning hypothesis |
| Deck re-selection and state persistence in BO3 | Experiment E03 |

## 6. Where "≈ 2000" sits

Win probability of a hypothetical agent at the target rating against today's 813.5 under both assumed scales (the true scale is unknown):

| target | gap | P(win), Elo-400 | P(win), probit-β100 |
|---:|---:|---:|---:|
| 1000 | +186.5 | 0.745 | 0.906 |
| 1200 | +386.5 | 0.902 | 0.997 |
| 1398.2 (highest found: Simulation winner) | +584.7 | 0.967 | 1.000 |
| 1500 | +686.5 | 0.981 | 1.000 |
| 2000 | +1186.5 | 0.999 | 1.000 |

The best result I could find in this environment family is 1398.2, achieved with a PPO-trained Transformer league. A rating near 2000 would imply beating Bv1-class opponents essentially every time. Nothing in the evidence supports it as a target; the ladder's scale also differs between a 242-team pool and a 6,807-team pool, so absolute numbers do not transfer. Define goals relative to a **concurrent Bv1 copy**.
