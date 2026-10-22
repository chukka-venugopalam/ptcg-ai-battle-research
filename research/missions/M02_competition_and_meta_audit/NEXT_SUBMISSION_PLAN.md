# Next-submission plan

Prepared 2026-10-10 · 90 days to the 2027-01-08 deadline. Tags: **[V]** verified from a primary text · **[S]** secondhand · **[I]** my inference · **[U]** unknown. Source IDs → `SOURCE_REGISTER.csv`; experiment IDs → `DECK_POLICY_EXPERIMENT_MATRIX.csv`. Nothing here was executed against your files or the engine. Nothing was submitted.

## 0. The plan in one paragraph

Do not submit anything experimental until a seat-balanced BO3 gate exists (about a week). Meanwhile run **one cheap live cohort**: a byte-identical re-upload of Bv1 plus the **unchanged official Dragapult sample** (§1, C3). That retires Archaludon, preserves Bv1's *behaviour*, gives a same-period A/A measurement, and calibrates how much pilot sophistication buys on this ladder. Then use the gate to validate **exactly one change to Bv1** (C1). Build the cloned-policy specialist (C2) in parallel as the upside bet. Land the final pair as a cohort by 2026-12-25.

## 1. Ranked candidates (at most three)

### C1 — best risk-adjusted: "Bv1-P1" = Bv1 + one forensic patch (E11)

| Field | Content |
|---|---|
| Public source | None to copy. Method precedents: loss-replay taxonomy → minimal fix [S14, S23]; knowledge injected as *candidates*, never hard overrides [S21]; behavioural-diff gate [S16]; tooling idea: Lawliet's `loop_autopsy.py`, MIT [S21]. |
| Policy / deck | Bv1 byte-for-byte except the patch. `deck.csv` hash-identical to Bv1. |
| The change | One patch for the most frequent, most fixable decisive error among ≥ 100 Bv1 losses. Check categories in this order: (a) attacks into immune or zero-damage targets and non-progress loops (a Grimmsnarl **ex** attacker is blanked by Crustle's ex-immunity and by Ogerpon's ability wall [S14]); (b) exposure-blind energy/bench placement; (c) deck-out guard; (d) missed lethal. Crustle was ~50% of the Simulation field at one point [S20] but 3–6% of the Playground panel on 2026-10-01 [S13]. |
| Artifacts | Bv1 source and weights; ≥ 100 loss replays; R2 CSV; frozen panel; harness. |
| Local gate | G0–G2 (§2) **plus** a behavioural diff: on ≥ 20,000 replayed decisions, every non-target decision is unchanged and ≥ 95% of target states move to the intended action. |
| Risks | Expected effect is small: +3 pp field win rate ≈ +11 to +21 rating points **[I]**, unresolvable on the ladder within days. A patch can open a new failure mode; Archaludon-style combo logic is not involved. |
| Why it may be better | It removes a verified misplay class while leaving every other decision untouched. |

### C2 — highest upside: deck-specialist cloned policy (E30), optional RL stage (E31)

| Field | Content |
|---|---|
| Public source | None to copy. Method: listwise behavioural cloning on **top-rated demonstrators only** (reported +115 Elo from the filter; 160× uncurated data cut win rate 0.72 → 0.44; no LICENSE visible, so reimplement) [S22]; exact-decklist specialists whose weights and deck are hash-gated, with a never-crash wrapper [S24]; REINFORCE fine-tune from Lawliet (MIT) [S21]. |
| Policy | Listwise scorer over options, numpy inference (~1 ms/decision), legality and clock wrapper, rule fallback beneath it. |
| Deck | Chosen by the census: the archetype with the most exact-list replays among top-rated agents in the Episodes datasets. Shortlist by frequency and by what the Simulation winners ran: Dragapult/Fezandipiti, Dragapult/Munkidori, Hydrapple/Teal Mask Ogerpon [S09, S13]. |
| Data | Daily Playground Episodes datasets (ranked by average agent rating) plus your own CLI pulls [S05, S31]. Check the dataset terms. |
| Local gate | G0–G2; beat Bv1 field-weighted by ≥ +5 pp with lower CI > 0 at n ≥ 1,000; ≥ 55% against the official Dragapult sample. |
| Risks | Demonstrators are other teams' agents; the 7th-place team measured ladder-replay cloning and dropped it [S10]; exact-list volume may be thin; 1–3 weeks of work; local-to-ladder transfer is weak [S13, S14, S17, S21, S23], so expect shrinkage. |
| Why it may be better | Every public result above ~1,000 came from learned policies or cloned-plus-guardrail policies [S09, S10, S22]. |

### C3 — distinct alternative / calibration: official Dragapult rule-based sample + its deck, unchanged (E20)

| Field | Content |
|---|---|
| Public source | Kaggle's official sample (Simulation-era notebook; check whether the Playground SDK zip ships an R2-valid version). Lucario sample is Apache-2.0 [S25]; **Dragapult sample licence is unverified**. |
| Evidence | One third-party catalog reports "official Dragapult pilot @ 880.9 μ", the only one of 21 submissions clearing 800 [S19, secondhand]. Dragapult-family lists are 16% of the panel [S13]; both Simulation winners ran Dragapult [S09, S10]. |
| Changes | None. If R2 ID validation forces a remap, record it and treat the result as "modified". |
| Gate | G0 plus "not pathological" (≥ 45% vs the panel). |
| Risks | R1 → R2 ID drift; many copies on the ladder; the sample may simply be weaker than tuned Bv1. |
| Why it matters | An independent archetype and pilot. It answers whether pilot sophistication or the deck is the binding constraint, at near-zero engineering cost. |

## 2. Local evaluation standard (pre-register before running)

**What local testing can establish:** safety (legality, clock, crashes), large effects (≳ 5–10 pp) against *known frozen pilots*, and behavioural diffs. **What it cannot:** a calibrated ladder rating, performance against unknown or adaptive pilots, or effects below ~3 pp. Evidence that local gates mislead: one team's gate ranked its eventual best ladder agent last because it only measured self-mirrors [S17]; the Playground's published search agent scored 0.78 on its own field benchmark and sat near 707 live after 21 episodes [S13, S14]; others call local play "a safety check only" [S23].

**Pool.** (1) Bv1 frozen. (2) Official sample pilots with their decks [S30]. (3) MIT/Apache agents that run locally: hotorch H036, rahulsiiitm Lucario/Alakazam, Lawliet's PIMC with a relaxed clock [S16, S21, S23]. (4) The 14-archetype panel as exact 60-card lists from the Episodes datasets, piloted by the pilots you trust. Weights = 50% share among top-rated episodes + 50% the opponents your agents actually faced (the ladder pairs similar ratings, not just top ones). (5) Unlicensed agents (4dalols, Scio, wmh, horno1337, TomBombadyl): optional local opponents only; do not copy or redistribute. (6) **Held-out:** 30% of archetypes/lists plus the newest 3 days of replays, used only at G3.

**Design.** Seat-balanced by construction; every match in its own spawned process killed on a wall-clock deadline; fresh worker per game (native memory grows [S12, S14]); same SDK build, deck and runtime across arms. **Seed-pairing is not available**: the native RNG exposes no seed [S12]. The substitute is independent samples stratified by archetype and seat, analysed with Newcombe/Mantel–Haenszel. Record first/second per game once the replay schema is known. Use the **same engine build as the ladder**: `kaggle-environments` master, R2 pool, `bo=3` [S11].

**Gates**

| Gate | Games | Rule |
|---|---|---|
| G0 safety | ≥ 300 BO3 + 10k-decision fuzz | 0 exceptions / illegal outputs / non-DONE / harness kills; p99 overage ≤ 300 s of 600; fallback rate ≤ 0.1% |
| G1 screen | ≥ 300 matches | **Exclusion only**: reject if field-weighted rate < Bv1 − 3 pp, or any archetype (≥ 25 matches) ≤ −15 pp. Passing proves nothing. |
| G2 confirm | ≥ 1,000 matches per arm | Newcombe 95% **lower bound** of the field-weighted difference vs Bv1 > 0; no archetype with ≥ 5% share and n ≥ 50 shows a regression whose CI upper bound < −5 pp |
| G3 finalist | 3,000 matches, fixed N, held-out panel | Used only to pick the final pair |

**Power (computed, 80% power, α = 0.05):**

| matches | detectable true rate vs fixed 50% | two independent arms |
|---:|---:|---:|
| 300 | 58.1% | 61.3% |
| 1,000 | 54.4% | 56.2% |
| 3,000 | 52.6% | 53.6% |

These are *match-level* (BO3) rates. A single-game edge is amplified: game 55% → BO3 57.5%; game 52% → 53.0%. So 300 matches resolve only a ≈ 11-point match edge between two arms; 1,000 resolve ≈ 6 points; a +3 pp edge needs ~4,400 per arm. For head-to-head against Bv1, a sequential test (H0 50% vs H1 55%, α = β = 0.05) averages ≈ 590 matches versus 1,294 fixed, but **only for that metric**. Peeking with a fixed-sample test every 50 games inflates false positives from 4.4% to 25.1% (simulated), so do not eyeball running totals. All numbers from `ptcg_eval_stats.py --tables`.

## 3. Package checks (every upload, including re-uploads)

1. `tar -tzf`: `main.py` and `deck.csv` at the archive root, no nested folder, no symlinks, no absolute paths; size ≪ 197.7 MiB. Record SHA-256 of the archive and of each file.
2. `deck.csv` printed as `count, id, name` against the R2 card data; 60 cards, ≤ 4 per name (basic energy excepted), ≥ 1 Basic Pokémon, accepted by `battle_start`. Names must equal the intended list.
3. Cold start in an extracted directory with Kaggle's Python: first call (`select = None`) returns 60 ints; later calls return in-range indices respecting `minCount/maxCount`.
4. ≥ 10,000 recorded decisions replayed: 0 illegal outputs, 0 exceptions; fallback counters logged.
5. `make("cabt", configuration={"bo": 3})` from the extracted directory, both seats, vs a second copy and vs a baseline: all statuses DONE, overage used ≤ 300 s, RSS flat across three games.
6. Two matches back-to-back in one process and in separate processes behave identically (no hidden global state leaking).
7. Weights/deck hash-gate loads, and the agent still plays legally if the weights file is deleted (fallback path exercised).
8. Log volume bounded.

## 4. Live protocol

**Cohort design.** "Latest two" is chronological, so uploading anything evicts the older active agent; keeping a given agent live across several challengers would force a re-upload each time anyway (§4 of the audit). Make that a feature: **always upload a pair on the same day — a byte-identical Bv1 re-upload, then the challenger.** Both start at μ0 = 600 together, get the same new-agent episode rate and the same opponent pool, and the concurrent difference ΔR = R(challenger) − R(Bv1 copy) cancels early-peak and pool-drift effects. Each cohort also re-measures the A/A noise. Budget: 2 of the 5 daily uploads; a cohort needs ≥ 5 days.

**Decision rules** (n = episodes of each agent; SE_diff from `ptcg_eval_stats.py`: 63/32 at n = 60, 40/21 at 150, 28/14.5 at 300 on the two assumed scales):

| Situation | Action |
|---|---|
| n < 100 or age < 48 h | Burn-in. Do not read ratings. |
| n ≥ 60 and ΔR ≤ −150, **or** ≥ 2 non-DONE episodes among the first 60, **or** time pool < 60 s left in ≥ 3% of episodes | Retire the challenger now. |
| n ≥ 150 and ΔR ≤ −100 | Retire. |
| n ≥ 300 and ΔR ≥ +60 **and** G2 passed **and** live per-archetype results are not worse than Bv1 copy in the top-3 archetypes | Promote: the challenger becomes the control in the next cohort. |
| n ≥ 300 and \|ΔR\| < 60 | Unresolved. Keep Bv1 as control; do not infer. Break ties on local gate evidence only if G2 passed. |
| Local gate and live disagree | Trust live only when beyond 2 SE_diff; otherwise treat as tie. |

**Rollback conditions:** any Error/Invalid status or ≥ 2% non-DONE episodes → re-upload the last validated pair immediately; after 2026-12-25 only rollbacks, no new architecture.

**Final lock.** Choose the pair that maximises the *estimated better-of-two* **[I]** (aggregation rule **[U]**): the best-evidenced agent plus the highest-upside decorrelated one (different archetype). Upload both together by 2026-12-25.

## 5. Calendar

| Dates | Work |
|---|---|
| Oct 10–13 | Read-only audit: submissions list and timestamps, leaderboard top, episodes for both active agents, hashes, deck audit, E03 probe (does `select is None` recur in games 2–3; does state persist), Rules tab, download every Episodes dataset so far |
| Oct 13–14 | **Cohort 1:** Bv1 re-upload + C3 (official Dragapult) |
| Oct 14–20 | Build panel + harness; E00 baseline matrix; E01 A/A; label ≥ 100 Bv1 losses; choose the C1 patch |
| Oct 20–26 | C1 through G0–G2; cohort 2 [Bv1 copy, C1] if it passes; C2 census and dataset |
| Oct 27–Nov 16 | C2 cloning and gate; cohort 3 if G2 passes; E50/E65 only if forensics or E03 justify |
| Nov 17–Dec 14 | Second wave (E31 RL stage only if C2 shows signal); every cohort ≥ 5 days |
| Dec 15–24 | G3 on held-out panel; freeze code |
| by Dec 25 | Upload final pair |
| Dec 26–Jan 8 | No uploads except rollbacks |
| Jan 8 / Jan 22 | Deadline / end of post-deadline games and Bradley–Terry standings |

## 6. Realistic goals and what would make bigger ones defensible

Ratings are pool-dependent. Define goals as ΔR over a **concurrent Bv1 copy**; absolute anchors below are from other ladders.

| Tier | Target | Evidence for plausibility | What must be shown first |
|---|---|---|---|
| Floor | hold ≈ 800–830 | Bv1 now | G0 clean |
| Plausible | 850–950 (ΔR ≈ +40 to +140) | Replay-patched and cloned-plus-guardrail agents peaked 910–990 in the Simulation [S17, S22]; official Dragapult 880.9 [S19]; all subject to regression [S23] | G2 pass **and** live ΔR ≥ +60 at n ≥ 300 |
| Stretch | 1,000–1,100 | Peak 990.7 [S22] | Head-to-head vs Bv1 ≥ 70% over ≥ 1,000 matches (Elo-400 mapping: +186.5 ≈ 74.5%; probit mapping ≈ 90.6%) |
| Exceptional | 1,200+ | 7th place 1,226.6 with a 110-exploiter league [S10] | Head-to-head ≥ 85%, and ≥ 55% against the strongest public pilots; days of self-play compute |
| Not supported | 1,500+ | Highest found: 1,398.2 [S09] | Would exceed every known result; needs demonstrated dominance over the top replays' agents |
| Not defensible | ≈ 2,000 | None | ≈ 99.9% win probability over Bv1 under the Elo-400 mapping |

## 7. Single recommendation

**Build first: the replay-derived panel and the seat-balanced BO3 gate (`bo3_ab_harness.py` + `ptcg_eval_stats.py`), run Cohort 1 now as the only live experiment, then use the gate to validate exactly one change to Bv1 (C1).** Keep C2 as the funded upside project from day 7.

**Evidence that would disprove it**
1. The A/A run (E01) shows local noise or bias larger than the effects of interest → the gate is invalid; repair before any decision.
2. Forensics on ≥ 100 losses show no repeated fixable pattern (top category < 10% of losses), or the pilot-swap shows reference pilots do no better with the same deck → patching Bv1 is the wrong lever; go to C2, C3 or a deck change (E10).
3. Cohort 1: the unchanged official Dragapult is ≥ Bv1 copy + 2·SE_diff at n ≥ 300 → pilot sophistication is not the constraint; redirect to the Dragapult archetype.
4. C1 live: ΔR ≤ −100 at n ≥ 150, or ≤ 0 at n ≥ 300 → stop patching; the remaining time goes to C2.
5. E03 shows decks can be re-selected per game and state persists → BO3-aware play (E65) outranks everything else.

## 8. What I need from you

The five items in `SUBMISSION_POSTMORTEM.md` §7, plus the top of the live leaderboard and the Rules tab text on external data and sharing.
