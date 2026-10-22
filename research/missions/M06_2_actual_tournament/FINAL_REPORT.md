# Mission 6.2 — Actual tournament evidence and decision

**Run date:** 2026-10-20  
**Decision:** **NO-GO for a new Kaggle submission.** Keep the R2 compatibility candidate as a compatibility-safe baseline, but do not claim its gameplay strength is higher based on these experiments.

## Executive summary

I recovered the actual candidate archives and R2 native engine and ran locally controlled, complete native-engine games. The logged evidence comprises **328 tournament games plus 2 smoke games**. All 330 rows finished without recorded engine errors, illegal actions, action-handler errors, fallbacks, or tracebacks. The tests establish that the agents execute cleanly under the tested local harness; they do **not** establish a meaningful win-rate improvement.

The only potentially interesting direction—using two copies of Shadowy {D} Energy—was not statistically decisive. Likewise, toggling `counter_source` was not decisive. Therefore no new deck or policy variant is being promoted, and no submission archive was created or uploaded.

## Method and limitations

- The games were run against the current R2 native engine in this workspace. Each agent's `main.py` was executed in a fresh worker without defining `__file__`, and the harness exercised the 60-card deck handshake.
- The matchup runner placed each side in seat 0 equally often for all 32-game comparisons and 100-game confirmations. The R2 wrapper did not expose a controllable RNG seed, so the tests were seat-balanced but not paired with identical seeded game randomness.
- Each CSV row represents one full native-engine game. The competition overview says skill updates happen after episodes/games and also describes the project as a BO3-format environment; these local game-level comparisons are **not** direct estimates of Kaggle's rating delta or of series performance. Official overview: <https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description>.
- The four original archives share the same learned `sa/policy_net.npz` weights and essentially the same deck. These tests isolate compatibility/flag changes rather than benchmarking a broad set of genuinely different competitive archetypes.
- Wilson 95% intervals and two-sided exact binomial tests compare each recorded win rate against 50%. These are exploratory comparisons, and no result below passes a 5% two-sided significance threshold. No ties occurred in these logged samples.

## Results

### Four-way screening (128 games)

Each comparison has 32 games and 16 games per agent in seat 0. The screening agent is the R2 compatibility candidate with `counter_source=False`.

| Comparison | Candidate win–loss | Win rate | Wilson 95% interval | Exact two-sided p |
|---|---:|---:|---:|---:|
| R2 compatibility candidate vs B2 (counter_source=True in B2 archive) | 21–11 | 65.6% | 48.3%–79.6% | 0.110 |
| R2 compatibility candidate vs Bv1 | 17–15 | 53.1% | 36.4%–69.1% | 0.860 |
| R2 compatibility candidate vs Bv2 (sym_k=8) | 16–16 | 50.0% | 33.6%–66.4% | 1.000 |
| counter_source=False vs counter_source=True (screening) | 14–18 | 43.8% | 28.2%–60.7% | 0.597 |

The candidate's 21–11 score against B2 is directional only: the 95% interval begins below 50% (48.3%), and the exact test returns p=0.110. It is not sufficient to declare the candidate stronger.

### `counter_source` confirmation (100 games)

The baseline candidate (`counter_source=False`) won **57–43** against an otherwise identical candidate with `counter_source=True`. The seats were split 50/50 for each version.

| Candidate | Wins–losses | Win rate | Wilson 95% interval | Exact two-sided p |
|---|---:|---:|---:|---:|
| `counter_source=False` | 57–43 | 57.0% | 47.2%–66.3% | 0.193 |
| `counter_source=True` | 43–57 | 43.0% | 33.7%–52.8% | 0.193 |

Interpretation: there is a directional lean toward `False` in this sample, but the uncertainty interval contains 50%; this does not prove the flag harms the agent.

### Shadowy {D} Energy 2-copy change (100 games)

The variant changed only the deck: 10 Basic Darkness Energy became 8 Basic Darkness Energy plus 2 Shadowy {D} Energy. Both versions use the same model weights and the same runtime. The 60-card total remains valid, with 8 Basic Darkness Energy retained for Grimmsnarl's energy-acceleration ability.

| Agent | Wins–losses | Win rate | Wilson 95% interval | Exact two-sided p |
|---|---:|---:|---:|---:|
| Baseline candidate | 44–56 | 44.0% | 34.7%–53.8% | 0.271 |
| Shadowy-energy variant | 56–44 | 56.0% | 46.2%–65.3% | 0.271 |

Interpretation: the result leans toward the Shadowy Energy variant, but remains inconclusive. This single self-play deck matchup does not show that the deck improves against a varied opponent pool, so the variant is not promoted.

## Compatibility finding remains useful but separate

A prior controlled synthetic option-ID probe showed that Bv1/B2/Bv2 can throw an out-of-range embedding `IndexError` at IDs such as 1300/1431, while the R2 compatibility candidate handled the probe with finite logits and no fallback. That is a real robustness improvement for unseen card IDs. However, **no natural fallback events were recorded in the 328 tournament games here**, so the compatibility patch has not been shown to explain the user's historical 780 rating or improve normal-game win rate.

The existing candidate embeds card IDs in a fixed 1,300-row table, while the official R2 card database goes through card ID 1431. Its bounds guard prevents crashes, but unknown high IDs still map to a generic embedding; this fixes robustness, not learned understanding of newer cards.

## Recommendation

1. **Do not upload a new submission on these results.** The two strategic variants are inconclusive; the compatibility patch has value for safety, not demonstrated strength.
2. Keep the existing R2 compatibility archive unchanged as the reference baseline.
3. Shift effort from changing individual flags to a materially different deck/policy based on the current R2 card pool and, where retrievable, public high-performing notebook code. The public Kaggle code listing currently advertises notebooks including “Pokemon Challenge Ultra 900+” and “PTCG AI Battle Playground: Baseline + New Cards”; its listing is visible here: <https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/code>. In this run, the listing was accessible but the full source for those notebooks was not independently retrieved; scores/titles alone are not evidence of their implementation.
4. Before any future upload, compare the genuinely new candidate against the baseline and at least two materially different opponent decks/policies, balance seats, save complete per-game CSVs, and require a meaningful independent confirmation. A local self-play result cannot guarantee a 1000+ ladder rating.

## Data integrity and files

The final CSV files were checked for exact row counts, and all failure/fallback columns sum to zero.

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `TOURNAMENT_SCREENING_FINAL.csv` | 128 | `e01a6a744aa92495770486f0c60fec97c2a124b7da37324a5e4117db3201563a` |
| `COUNTER_SOURCE_CONFIRMATION_FINAL.csv` | 100 | `62e975e075a731c0fe584efc5a859b0099ebcee3785455d84ed66593799a040e` |
| `SHADOWY2_CONFIRMATION.csv` | 100 | `d0bfe48001ba7ae769d2c3ac77e38a3649ab24216b546571698c215d198e1698` |
| `SHADOWY2_SMOKE.csv` | 2 | `65564513926f4eb561109f76acf1232dbb1fde08d939ad339085e3273c7e1c9a` |

Supporting scripts are included in the evidence ZIP. They were used against the extracted workspace setup and contain workspace-specific paths; the CSV files and summaries above are the durable results. Do not treat these scripts as drop-in portable tools without adapting the path configuration.
