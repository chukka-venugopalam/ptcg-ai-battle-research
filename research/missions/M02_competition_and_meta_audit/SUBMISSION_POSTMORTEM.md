# Submission post-mortem

Accessed 2026-10-10. Tags: **[V]** verified from a primary text · **[S]** secondhand · **[I]** my inference · **[U]** unavailable. Source IDs → `SOURCE_REGISTER.csv`.

## 1. What I had and did not have

I had only the table in your brief. **Not available:** the archives, SHA-256 hashes, deck lists, source, weights, episode counts, σ, opponent logs, replays, or your local evaluation logs. Everything below about *why* a score moved is therefore a ranked hypothesis with a test, not a finding. Nothing was executed against your files.

## 2. Score history

| Submission | Visible rating | Active? | What can be said |
|---|---:|---|---|
| `submission_Bv1_backup_826.tar.gz` | 813.5 | active (yours) | The file name implies the same bytes as the 826.0 submission. **Check SHA-256.** If equal, 826.0 vs 813.5 is a free A/A pair: Δ = 12.5 points. |
| `submission_archaludon_cleanroom_r12.tar.gz` | 528.2 | active (yours) | 285 points below Bv1. Below the 600 starting rating, so it has lost more than it has won. |
| `submission.tar.gz` (earlier) | 826.0 | inactive | Historical; **not** the current agent's score. |
| `submission_Bv2_sym8.tar.gz` | 781.6 | inactive | Variant of the Grimmsnarl family; what changed is **[U]**. |
| `submission_B2_counter_source_grimmsnarl.tar.gz` | 789.9 | inactive | Same. |

### 2.1 Reading the numbers

* **The four Grimmsnarl-family readings span 44.4 points (781.6 → 826.0).** Reported noise for *identical* code on the Simulation ladder is 12–18 points within minutes, ~50 points across two days, and 161 points (732 → 571) within one day [S18, S21]. With episodes and σ unknown, **no ordering among Bv1 / Bv2 / B2 is supported — treat them as a statistical tie.** In particular, there is no evidence that the sym8 or counter-source changes helped or hurt.
* **Peaks regress.** Two independent teams saw displayed peaks (958.8, 753) settle at ~600 and ~660 on exact reruns [S23]. If Bv1's 826.0 was a first-week reading, 813.5 may sit closer to its true level than 826.0 does, and its true level may still be lower.
* **Where 813.5 sits.** If entrants start at 600 and updates are roughly zero-sum, the pool mean stays near 600 **[I]** — Bv1 is then ≈ +213 above the mean. Public notebooks currently score 370–732 [S02, S35], so Bv1 is above every in-pool anchor I could read. The leaderboard top is **[U]**.
* **Is the Archaludon gap real?** Standard error of a rating after *n* episodes is 34.7 (n = 100) or 58.7 (n = 35) on an Elo-400-type scale, 17.7 or 30.0 on a probit-100 scale. The 285-point gap is ≥ 3.2σ even at n = 30 per agent, and ≥ 5.8σ at n = 100 (computed by `ptcg_eval_stats.py`). **Unless r12 has fewer than ~10 episodes, it is genuinely worse than Bv1 as implemented.** Its level is comparable to weak public baselines (Q-learning notebook 579.8; "PTCG AI Battle Challenge Playground" notebook 539.5) [S02].
* **It did not crash.** A policy that fell back to random or first-legal-option would sit far lower. 528 is a *functioning but weak* signature **[I]**, which points away from packaging failures (H6) and toward play quality or deck fit.

## 3. Active vs inactive, and what a new upload does

See `PLAYGROUND_COMPETITION_AUDIT.md` §4. Short form: the **older** of your two active agents is evicted by any upload. If Bv1 is older, one new upload evicts Bv1, not Archaludon. Read the timestamps before submitting anything.

## 4. Why did r12 score 528.2? Hypotheses, ranked

Priors are **[I]**, formed from public evidence; the tests decide.

| # | Hypothesis | Public evidence | What would confirm it |
|---|---|---|---|
| H1 | **Pilot defect on a combo deck** (setup, evolution timing, energy targeting, ability sequencing) | Rule-based pilots played a simple Bellibolt deck at 836 but complex combo decks at 532 on the same ladder [S20]; replay-driven loss taxonomies repeatedly found energy to non-ready or exposed targets and non-progress turns [S14, S23] | Low attack rate / many no-attack turns; Active not attacking by turn 3–4; energy in hand at end of turn; lower option-agreement with a reference pilot in the same contexts |
| H2 | **Deck IDs silently wrong in R2** (R1 → R2 ID drift; wrong card in a slot; illegal-but-accepted list) | Playground pool is R2 (1,431 cards) vs 1,267; third parties had to map lists to R2 IDs [S11, S12] | Printed names for each `deck.csv` ID differ from the intended list |
| H3 | **Runtime/clock** — time-pool exhaustion or fallbacks mid-BO3 | 600 s per BO3 match; timeout = match loss [S08, S13]; a search-heavy clean-room policy could burn it | `remainingOverageTime` near 0 at the end of losses; non-DONE statuses; fallback counters > 0 |
| H4 | **Deck is simply weak in this meta** | Archaludon is not among the 14 most common ladder lists (each < 3.1%) [S13]; no source shows it strong in the Playground | Reference pilot with the same deck is also poor (§5) |
| H5 | **Immature rating** | Early readings swing by 100+ points [S21, S23] | Episode count < ~100 and a rising trend; but 528 < 600 already signals net losses |
| H6 | **Packaging fallback** (weights/files not found; policy silently degraded) | — | Different results in extracted-dir cold start; fallback counter > 0 |

## 5. Protocol (cheapest first; each step has a stop/continue rule)

1. **Identity (minutes).** `sha256sum` each archive; `tar -tzf`; `diff -ru` extracted trees against Bv1. Answers: are the "different" archives different, which flags and rules differ from Bv1, which files carry the deck and weights.
2. **Deck audit (minutes).** Print `count, id, name` for every `deck.csv` line using the engine's card-data call and compare with the intended list. Check 60 cards, ≤ 4 copies per name (basic energy excepted), ≥ 1 Basic Pokémon, and that the engine's `battle_start` accepts it.
   ```python
   from collections import Counter
   from cg.api import all_card_data          # VERIFY import path / field names in your SDK build
   cards = {c.id: c for c in all_card_data()}
   deck = [int(x) for x in open("deck.csv") if x.strip()]
   for cid, n in sorted(Counter(deck).items(), key=lambda t: -t[1]):
       print(n, cid, cards[cid].name)
   ```
   *Stop here and fix* if any slot's name is not the intended card.
3. **Cold start (≤ 1 h).** Extract to a clean directory, with the same Python as Kaggle. First call with `select = None` must return 60 ints; recorded observations must return legal index lists respecting `minCount/maxCount`. Count fallbacks, repairs and exceptions; record per-decision latency and RSS.
4. **Ladder forensics (hours).** For r12 and for Bv1 pull ≥ 50 episodes each with `kaggle competitions episodes <SUBMISSION_ID>` and `replay`. Per game extract: opponent archetype, first/second, turn of first attack, number of turns without an attack, energy left in hand at end of turn, benched Basics after turn 1, prize race at each turn, terminal reason (prizes / no Pokémon / deck-out / timeout), final overage. Replays align as `steps[t-1][seat].observation → steps[t][seat].action` [S14]. Tools to adapt: Lawliet's `loop_autopsy.py` (MIT) and a replay viewer [S21, S30].
5. **Pilot-swap (CPU-day).** Run **the same Archaludon deck under two pilots** — r12's policy and a reference pilot (Bv1's generic logic, an official-sample-style heuristic, or any MIT/Apache agent that can pilot arbitrary lists) — against the frozen panel with `bo3_ab_harness.py`, seat-balanced, ≥ 300 matches per arm. Aggregate per-archetype differences with the Mantel–Haenszel routine in `ptcg_eval_stats.py`.
   * reference pilot ≥ +10 pp better field-weighted → **policy defect (H1)**; port the reference approach, not more patches.
   * both poor → **deck/meta (H4)**; stop.
6. **Decision-level audit (hours).** Replay 200 recorded decisions from r12's losses; compare r12's top-3 choices to the reference per select context (setup, energy target, evolve, ability, retreat, attack). The context with the lowest agreement and highest loss share is the first fix.

**Revisit rule for Archaludon:** continue only if step 1–3 or step 6 names a concrete defect *and* step 5 shows the deck reaching ≥ 50% field-weighted under a reference pilot. Otherwise retire it. This is experiment E80.

## 6. Matchup-loss vs policy-loss attribution (for Bv1 as well)

For each archetype *a* with ≥ 25 games, compare the candidate to the pilot-swap reference on the same deck. Policy loss in *a* = reference win rate − candidate win rate (Newcombe 95% interval); matchup loss = 50% − reference win rate. Aggregate strata (archetype × seat) with Mantel–Haenszel rather than pooling raw counts so that opponent mix does not masquerade as skill.

## 7. Files I need to turn this into findings

1. `submission_Bv1_backup_826.tar.gz`, `submission_archaludon_cleanroom_r12.tar.gz`, the earlier `submission.tar.gz` (and Bv2 / B2 if kept).
2. Output of `kaggle competitions submissions -c the-pokemon-company-ptcg-ai-battle-challenge-playground` (ids, timestamps, status, scores).
3. For each active submission id: `kaggle competitions episodes <id>` output plus ≥ 50 replays and logs, wins and losses.
4. The R2 card CSV / `all_card_data()` dump, your `kaggle-environments` commit, and Python version.
5. Your local evaluation harness, logs and any training scripts behind the weights.
