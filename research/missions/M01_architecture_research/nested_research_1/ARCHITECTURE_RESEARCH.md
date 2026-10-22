# ARCHITECTURE_RESEARCH — PTCG AI Battle: what the strongest agents actually did

Prepared 2026-10-09. Companion files: `REPO_CATALOG.csv`, `WINNING_AGENT_MATRIX.csv`, `TRANSFER_AND_GAPS.md`, `IMPLEMENTATION_ROADMAP.md`, `repro/`.

## 0. Scope, evidence tags, and what was NOT done

**Tags.** **[V] Verified** = I opened the file/code/figure/CSV myself (commit SHAs pinned in `repro/SOURCE_MANIFEST.csv`). **[R] Reported** = an author's or third party's claim I could not reproduce; this includes every Kaggle page, because Kaggle pages are JavaScript-rendered and `web_fetch` returned only metadata — I read Kaggle writeups only as excerpts surfaced by a search engine. **[H] Hypothesis** = my inference; needs an experiment.

**Done.** ~25 searches; shallow clones of 45 public repos (14 you listed + 31 found via GitHub search; 48 more discovered but not inspected, see catalog); read READMEs/docs/writeups in the high-value repos; rendered the 1st-place author's SVG tables to images and read them; listed array shapes inside `.npz` files and members of archives; analysed five real Simulation leaderboard CSV snapshots found inside one repo.

**Not done.** I did not run any third-party code, the engine, any benchmark or any training. I did not load any `.pt`/`.pkl` (pickle) file. I did not download Kaggle data, and could not read Kaggle leaderboards, discussions or the full text of any Kaggle writeup. Writeups for 2nd, 3rd (title only), 4th–6th, 8th, 10th, 11th, 13th–15th, 17th, 20th, 27th were **not located or not retrievable**. An aggregator list showing "1st/2nd/7th/14th/25th" belongs to a different competition (Playground Series S6E8) and was discarded.

## 1. Ten findings that drive decisions

1. **Nothing near 2000 exists in the evidence.** Highest Simulation figure: **1398.2** (1st place, post-deadline final Elo, author-reported [R]). Highest in-period leaderboard score in the snapshots I could read: **1322.6** (2026-08-02) [V]. Current Playground: the only figure I have is the ~1135.7 leader reading you supplied; I could not verify it.
2. **Calibration of your ~800 [V].** In the 2026-08-16 snapshot (6,878 teams): ≥800 = top 14.1% (969 teams), ≥900 = 5.8%, ≥1000 = 1.6% (110 teams), ≥1100 = 25 teams, ≥1200 = 5 teams, median 611.2. This is the *Simulation* population, not the Playground.
3. **Top agents are small-to-mid Transformers trained by RL at scale, with BC as a warm start or not at all.** 2.2M (1st), ~3.1M (3rd), 7.8M (9th), 10.7M (7th), 20M (12th) parameters; 0.1B to ~59B training decisions (§3, §7).
4. **Per-deck specialization is the most repeated lever** (12th: the "ship lever"; 1st: single-deck Stage 4; 9th: per-archetype fine-tunes; 18th: 15 specialists per deck) [R].
5. **Tree search lost or tied everywhere it was isolated** (1st, 7th, 9th, Lawliet, Scio, TomBombadyl) [R]. The one search-positive case (18th place) did not isolate search. Search as *input features* (9th) is the only variant with a plausible success story.
6. **Hidden-information modelling is unresolved.** 1st place deliberately avoided Bayesian deck priors; 9th/18th/7th use belief mechanisms; **no source isolates a win-rate gain** → UNKNOWN.
7. **Offline accuracy is a poor selector.** Six logged inversions of held-out top-k vs game strength (Jun-Morita); data volume up, accuracy up, win rate down (Scio); offline gains that regressed online (XERO47) [R].
8. **The ladder is very noisy.** ~35 games in 8 h ⇒ ±24 pp detectable; identical bytes 12–18 points apart minutes later, 732→571 within a day (Lawliet); bit-identical relaunches up to 216 points apart (XERO47); one agent read 958.8 then ~600 on rerun (rahulsiiitm) [R]. **Your 826.0 / 813.5 / 789.9 / 781.6 readings are statistically indistinguishable; do not attribute differences to code.**
9. **Deck choice is first-order.** Dragapult ex in 16 of the 30 final-top-15 submissions, Mega Kangaskhan ex 8, Hydrapple ex 3, Alakazam 2, Munkidori 1 [V, my transcription of the 1st-place table].
10. **A Playground replay feed exists** (daily "Top episode replays… ranked by average agent rating, capped at 20 GiB per day") [R, dataset page excerpt] — the enabling data source for replay cloning and for building opponent pools.

## 2. Competition facts and the rating scale

| Item | Fact | Tier |
|---|---|---|
| Simulation | 2026-06-16 → 2026-08-17 (JST); 6,807 teams; 5 submissions/team/day; μ₀ = 600; only best agent shown; official ranking after ~2-week post-deadline evaluation of each team's final two submissions | R (official site, Kaggle excerpts); kento-umeda README [V] confirms re-convergence |
| Match rules | Standard-format-based, organizer card list, 10-minute total time per player per match (timeout = loss) | R (official site) |
| Strategy | 939 teams; $240,000 (8 × $30,000); scoring Model 70% / Deck 20% / Report 10%; top 8 → finals | R |
| Playground | Separate Kaggle competition ("Build an AI Training Agent…"); daily Playground episode datasets exist (e.g. 2026-10-01). Rules, card pool, rating settings: **not retrievable** | R / UNKNOWN |
| Engine | `cabt` (Matsuo Institute); agents receive only legal options; engine ships under a competition-use-only license per the 9th-place NOTICE | R |

### 2.1 Real leaderboard snapshots (Simulation) [V]
Source: `SiamRahman29/pokemon-tcg-agent`, `out/lb*/…zip` (CSV inside each). Reproduce with `repro/leaderboard_snapshots.py`.

| Snapshot (UTC) | Teams | Top | ≥800 | ≥900 | ≥1000 | ≥1100 | ≥1200 | Median |
|---|---|---|---|---|---|---|---|---|
| 08-01 06:02 | 6,073 | 1274.8 | 932 (15.4%) | 334 (5.5%) | 120 (2.0%) | 23 | 1 | 640.0 |
| 08-01 07:03 | 6,075 | 1300.6 | 927 (15.3%) | 334 (5.5%) | 119 (2.0%) | 25 | 2 | 639.4 |
| 08-02 10:34 | 6,136 | 1322.6 | 902 (14.7%) | 329 (5.4%) | 95 (1.5%) | 14 | 1 | 636.8 |
| 08-12 14:54 | 6,771 | 1217.2 | 978 (14.4%) | 371 (5.5%) | 92 (1.4%) | 19 | 3 | 621.5 |
| 08-16 20:20 | 6,878 | 1286.4 | 969 (14.1%) | 402 (5.8%) | 110 (1.6%) | 25 | 5 | 611.2 |

### 2.2 Final top-15 (Simulation), transcribed from the 1st-place author's Table 17 [V figure; R values]
File: `strategy-track/figs/tab17_evaluation_win_rates.svg` in `yijieyuan/kaggle-pokemon-tcg` (SHA in `ARTIFACT_HASHES.csv`). The figure's own caption: "30 active submissions from the final top-15 teams". I read it by rendering to PNG; **transcription errors are possible**.

| Rank | Team | Entries (deck → final Elo) |
|---|---|---|
| 1 | Luca | Dragapult ex f38fb5 → **1398.2**; Dragapult ex 917977 → 1291.1 |
| 2 | palsystem | Mega Kangaskhan ex → 1297.8; Dragapult ex 1ec8f2 → 1160.5 |
| 3 | Unown Gradiant | Hydrapple ex b86657 → 1280.5; Mega Kangaskhan ex → 1226.4 |
| 4 | flg | Dragapult ex acbe04 → 1266.1 / 1208.9 |
| 5 | Petit Canard | Mega Kangaskhan ex → 1257.3 / 1223.4 |
| 6 | KawattaTaido | Mega Kangaskhan ex → 1229.5; Dragapult ex 2858c9 → 1146.3 |
| 7 | LumenLiquidity | Dragapult ex dc0582 → 1226.6 / 1211.8 |
| 8 | やる気元気ミワハルキ | Dragapult ex 0a67ba → 1214.2 / 1137.7 |
| 9 | Rmy | Hydrapple ex b86657 → 1200.0 / 1133.8 |
| 10 | Azat Akhtyamov | Dragapult ex 7c605f → 1195.3 / 1194.8 |
| 11 | e-toppo + kurupical | Munkidori ff5f6c → 1194.8; Mega Kangaskhan ex → 1166.0 |
| 12 | Majkel1337 | Alakazam 5ad1c9 → **1186.4**; Dragapult ex c47f4c → 1107.5 |
| 13 | LiamK | Dragapult ex 528bab → 1186.3; Alakazam bff6f8 → 1154.9 |
| 14 | Preferred 213tubo | Dragapult ex 42f645 → 1182.7; Mega Kangaskhan ex → 1145.8 |
| 15 | 李秉叡 (ntumlnoob) | Mega Kangaskhan ex → 1178.0; Dragapult ex e63804 → 1098.5 |

Cross-check [V]: `Michal1337/pkmn-kaggle` README says rank 12, score 1186.4, Alakazam + Dragapult/Dusknoir — matches "Majkel1337". Mapping 9th-place repo ↔ "Rmy" is **inferred, not verified**.

### 2.3 Noise, regression to the mean, and final-vs-peak [R unless stated]
- Lawliet-ai: a submission resolves ≈35 games in its first 8 hours → minimum detectable difference ±24 pp; byte-identical submissions sent a minute apart settle 12–18 points apart; same bytes read 732 in the morning and 571 that afternoon; a single day's rating swings 43 points ≈ 600 places. House rule: p > 0.05 is a tie. [V text in README]
- XERO47: bit-identical archive relaunches differed by up to 216 rating points; common-seed engine evaluations repeatedly favoured models that later regressed online.
- rahulsiiitm: "958.8 is the historical maximum… a fresh exact rerun saturated near 600."
- 7th place: the same frozen agent's rank on the final scale fluctuates around a median of 8th (extremes 2nd and 22nd).
- 12th place: held rank 1 for days mid-competition; Alakazam hovered >1200 at rank 5–13, then fell in the final two days.
- kento-umeda: in-period best 993.3, final 896.9 — final re-converges all teams' last two submissions and is harsher.
- Jun-Morita: with two eligible submissions, the leaderboard shows the max of two independent ratings; after convergence both approach the agent's true level (~920 in their case).
- 1st place: "around 100 games per opponent are too few… at least 500 are needed."

**Implication for your readings [H].** Four readings within 44 points (826.0, 813.5, 789.9, 781.6) are inside the documented noise band, especially if their ages/episode counts differ. The 528.2 reading is plausibly a real regression but is also unverified. Any ladder comparison needs concurrent, same-age, bit-identical control copies.

## 3. Evidence dossiers

### 3.1 1st place — Luca (Simulation) [R; figures V]
Sources: [Kaggle writeup](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/new-writeup-1784257142000) (excerpts only), [repo](https://github.com/yijieyuan/kaggle-pokemon-tcg) (figures, spreadsheets; **no code, no weights**; README: "Code will be available after the second round").
- Result: 1st of 6,807; f38fb5 final Elo 1398.2, 917977 1291.1; f38fb5 ranked above every other team for 97.3% of tracked time since day 2.
- Model [V from Fig. 2/Table 6]: PTCGNet, **2,235,649 parameters**; 3-layer pre-LN Transformer (d256, 8 heads, FF 512); up to 141 tokens; object MLP input 494 = 366 features + 64 card embedding + 64 evolution-history embedding; global input 191; pointer-attention action head (query from global token, keys from options, /√256); value head from global ‖ mean token; 567 MFLOPs per forward pass; argmax at inference, sampling in training; autoregressive picks for multi-select.
- Information: model sees "facts and bounds established from observable game history", **not** Bayesian estimates from familiar decks (rationale: priors can under-cover unseen decks).
- Training [V from Table 7/15]: PPO, terminal ±1, γ 0.999, λ 0.95, clip 0.2, entropy 1e-4, AdamW lr 1e-4 (500-step warm-up), 4,096 concurrent games, horizon 32, 131,072 samples/rollout, minibatch 8,192 × 16, FP32. JAX model; C++ official engine on CPU through a custom batched interface; DDP. Throughput on 4× RTX 5090: ~40–60k steps/s ≈ 220–460 games/s.
- Curriculum: S1 broad self-play on a daily-refreshed top-50 deck pool (archetypes uniform; families/decks by win-rate-based weights); S2 archetype champions from a local arena; S3 Dragapult/Hydrapple pool to create stronger opponents; S4 single-deck training initialised from S2 checkpoints.
- Scale [V Table 15]: f38fb5 lineage = 445,700 updates, **364.2M games, 58.84B policy decisions**; ≈ 180 decisions/game.
- Decks [V]: final Dragapult list = 18 Pokémon/32 Trainers/10 Energy; derived from public parent 7c605f by one card swap (Dawn → Chi-Yu). 646 candidate decks (537 active) in 38 archetypes.
- Negative [R]: search during training and inference "neither sufficiently benefited"; learning the deck end-to-end produced a no-Energy deck.

### 3.2 3rd place — Unown Gradiant (Simulation) [V design; R rank]
Sources: [repo, MIT](https://github.com/dipamc/kaggle-ptcg-ai-battle); writeup title "Team Unown Gradiant solution: all decks on hand" (content unread). Rank 3 from the 1st-place table.
- **Training stack in C/CUDA** (patched PufferLib trainer; C game environment; policy network in CUDA); no PyTorch in the hot path. One CUDA process per GPU, NCCL multi-GPU.
- Model [V docs/model.md]: d256, 4 layers, 8 heads, FFN 512; ≤160 tokens (typically 40–90 live); pointer scoring over option slots with picked-token feedback and STOP; **≈3.1M params** (trunk 2.11M, critic top 0.53M, auxiliary opponent-hand head 0.49M). Blind critic for deployment; **oracle critic with the true opponent hand used in training only**.
- Training: terminal ±1, γ 1.0, λ 0.95, Muon optimizer, 616-deck pool re-weightable live, 87 coverage decks, league banks of frozen opponents, deck-vs-deck matrix, a "deck lab" that A/B-tests deck proposals.
- Throughput [V docs/training.md]: ~16K steps/s on 8× RTX 4090 (d256), ~25K (d128); ~790 SPS on a 12 GB RTX 3060 (with an open hang issue on 12 GB cards).
- Artifact: `submission/model.pt` (19.5 MB, not loaded), exact final submission 55559155 (entry-to-rank mapping not stated).

### 3.3 7th place — "Reading the Cards" (Simulation) [R, excerpts]
[Writeup](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/7th-place-solution-for-the-ptcg-ai-battle-challeng).
- ~10.7M-param encoder-decoder Transformer with a GRU (256) belief core; conditioned on card text; no card-ID parameters (one network pilots any list); pointer head; 51-bin HL-Gauss distributional value head.
- IMPALA-style V-trace + UPGO inside a PPO clip (0.2), entropy 0.01, AdamW 6e-5, batches of 6 trajectories. League of 110 trained exploiters (~72 live) frozen to beat the main agent.
- Scale: ~3.6B decisions (~35M games) on **one RTX 4090 learner** plus game generators; vectorised generation was the biggest lever.
- Inference: greedy, plus a deterministic lethal check and a mask on dominated actions.
- Negative [R]: hand-written rules and mechanical teachers each cost win rate (up to 36 points); search to generate or label training games gave nothing over greedy; energy-only search +5 points, within noise.
- Wiping the belief's past changes 5.9% of 10,109 decisions (an activity measure, not a win-rate effect).

### 3.4 9th place — Ogerpon–Hydrapple (Simulation) [R; repo V]
[Repo](https://github.com/YoInaOwO/pokemon-tcg-ai-battle-9th) (`reports/writeup_en.md`, weights zip `net.npz` 28.96 MB, no LICENSE file; NOTICE restricts engine/card-DB/replay redistribution).
- 7.8M params; 5-layer Transformer (d384, 6 heads), 64 state tokens; each legal option reads the state by cross-attention; count head (24 classes) for multi-select; two critics (blind + oracle with opponent hand), auxiliary future-prize head.
- **Engine lookahead as features:** for eligible main-phase single-select decisions, up to 64 candidate actions are tried in temporary engine states (following forced continuations and coin flips) and summarised into 28 features per candidate; no tree search at play time.
- Belief: replay-derived archetype posterior over 14 archetypes + "other" as an input (dropout to "unknown" during BC).
- Pipeline: BC on 3–12 Aug replays (winners 1.0, losers 0.3 weight) → per-archetype fine-tunes → value fine-tune → PPO. Opponent pool: 75% top-40 exact lists by arena frequency, 10% mutated lists, 10% organizer starter scripts, 5% mirrors. Rollout 8.39M decisions/update (16× larger raised the plateau). 8× RTX 4090, 57 updates, 56.3 h, 403M decisions to the peak training win rate.
- Results: training win rate vs BC pool 52.4% → 83.3%, but ladder 57.3% (2,000 games) and **53.5%** when rating gap ≤200 [R] — training gains do **not** transfer one-to-one. Mirror vs identical list: 71.6% (53/74).
- Negative [R]: determinized MCTS (policy priors + critic leaves) gave no useful improvement; larger network no help.

### 3.5 12th place — self-play RL at scale (Simulation) [R; repo V]
[Repo](https://github.com/Michal1337/pkmn-kaggle) (README + `WRITEUP.md`; no weights; no LICENSE file). Score 1186.4 matches the 1st-place table.
- 20M-param token Transformer, pointer policy plus SUBMIT logit, value from CLS; >300 tokens per state (author's own criticism).
- 12M-decision BC warm start from public replays (~2× step compression); two-sided self-play PPO; teacher-KL (0.005) with gated promotion at 0.53 win rate; asymmetric truncation terminals (anti-stall); γ 0.997, λ 0.95, batch 196,608, fixed LR (annealing "crawled").
- Pool: 4,941 real decks (3,517 ladder, 1,424 Limitless top-cuts); inverse-similarity (TF-IDF) sampling cut Dragapult/Alakazam share roughly in half.
- Engineering: engine-native environment, C observation encoder byte-identical to Python, CUDA-graph collection: 16.5k steps/s on 4× H200; ~11B steps generalist.
- **Two-net per-deck finetune** (agent net vs live counter-adapting opponent net; alternate on a +2 pp gate; 6 rounds/3.06B steps Alakazam, ≥8 rounds/2.67B Dragapult). "Unbalanced self-play destroys the loser and then the signal."
- Lessons [R]: evaluation over 1,553 decks diluted the signal; scaled model size before tuning; reward changes are incentive changes on both sides of a zero-sum mirror.

### 3.6 18th place — small PPO + specialists + router + search (Simulation) [R; repo V]
[Repo, MIT](https://github.com/KleinHoumani/ptcg-selfplay-agent). Sylveon 1162.4 (18th); Dragapult variant 1039.1.
- Set Transformer, **1,001,987 params**; PPO from random weights; 16 archetype pools of real ladder lists; 512 games/iteration, 20% against frozen past versions; ~1,000 iterations (≈85 s/iter on 1× RTX 4090 + 16 CPU workers ⇒ base ≈ 24 h, derived).
- 15 specialists per deck (400-iteration fine-tunes each); naive-Bayes posterior over pool decklists swaps a specialist in at 0.8 confidence and out below 0.7.
- Determinized PUCT every decision within the 600 s time bank (raw policy below 40 s). Inference: CPU only, 2 vCPUs. **No ablation separates search, router and specialists.**

### 3.7 24th place — GBDT teacher → Transformer → population PPO [R, excerpts]
[Writeup](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/writeups/24th-solution-a-population-based-rl-ecosystem) (2026-09-12). ~1M-param Transformer; GBDT imitates 100–1,000 replays, plays 15,000–30,000 internal games, which are distilled into the Transformer, then population PPO/GAE against 20–40 opponent types; C++/OpenBLAS inference ≈1 ms for 32 tokens on one thread. A Transformer trained directly on the small replay set "fit the recorded moves yet performed poorly." The GitHub repo named `ptcg-population-rl` is **not** a credible code release (see catalog: UNSAFE).

### 3.8 The BC / imitation line (≈880–1000) [R]
Scio (peak 990.7, rank 129 of 6,483), CAB314 (993.8, 138th; 864k-param NumPy policies, BC → PPO league), squidistaken (980, 168th), tangaii (358th; grouped legal-option ranker), Jun-Morita (911.1), kento-umeda (896.9 final / 993.3 peak), XERO47 (880.1 final / 997.3 peak), horno1337. Consistent ceiling around 900–1000 on the Simulation scale with ≤1M-param models and no large-scale RL.

### 3.9 Negative-result and measurement line [R]
Lawliet-ai (50.1% over 1,999 episodes; PIMC+PUCT tree ≈ 3-day REINFORCE policy, 55.6% vs 52.9%, p = 0.81; tree 6 s vs policy 0.3 ms; a static damage constant fed to the lethality checker dropped win rate 60% → 38%); TomBombadyl (June: RL+MCTS v5 580.6; only an official-sample Dragapult pilot at 880.9 exceeded 800); rahulsiiitm (best RL/PPO 342.0); oscar-chw (value head worse than a constant in one run; PPO not submitted); marcmaldonadolorca (best local candidate never submitted — process failure); scha54 (88.5% against greedy **inside its own simulator** — not ladder evidence).

## 4. The ten approaches

| | Approach | Verdict | Strongest evidence |
|---|---|---|---|
| A | Replay BC from strong demos | Do first; ceiling ≈ demonstrators | Scio +115 Elo from filtering; 12th BC warm start; mid-tier cluster 880–1000 |
| B | Listwise ranking over legal options | Adopt (de-facto standard) | Pointer/option-scoring heads in 1st, 3rd, 7th, 9th, 12th, 18th |
| C | Generalist → deck specialization | Strongest repeated lever | 12th two-net finetune; 1st Stage 4; 9th/18th per-archetype |
| D | Self-play PPO vs frozen/real-deck pools | Core of every top-12 solution; compute-bound | 1st, 3rd, 7th, 9th, 12th, 18th |
| E | Dynamics/representation pretraining | No direct evidence | BC warm start, auxiliary heads only |
| F | Bayesian deck/hand belief | Unproven; contested | 18th router vs 1st's explicit rejection |
| G | Recurrent / memory | Unproven | 7th GRU; event-history features elsewhere |
| H | Learned policy + narrow sound overrides | Narrow only | 7th lethal check; Scio guard; Lawliet rule-override losses |
| I | MCTS / determinization / lookahead | Do not lead with it | Negative in 1st, 7th, 9th, Lawliet, Scio, TomBombadyl |
| J | Ensembles / averaging / portfolios | Little evidence | taichiiiiiiii ensemble (unmeasured); no averaging found |

**A. Replay BC.** *For:* Scio (Elo ≥1120 demonstrator filtering +115 Elo [R]); 12M-decision warm start gave ~2× RL step compression (12th); BC + per-deck fine-tune is the whole method of 468th and ≈990-class agents. *Against:* 160× more uncurated rows → win rate fell to 0.440 while training accuracy rose (Scio); small-replay Transformer fit but played poorly (24th); BC models trailed Kaggle opponents (9th: "imitation learning struggles to surpass those it imitates"); BC pilot lost badly (Jun-Morita exp048a). *Transfer:* the Playground feed is ranked by average agent rating, so demonstrator quality may exceed what Simulation-era agents saw [H]. Confidence: Medium.

**B. Listwise ranking.** Every top solution scores a dynamic option list with a pointer/attention head and masks illegal options; differences are in multi-select (autoregressive step in 1st; STOP in 3rd/12th; count head in 9th) and in merging equivalent options (1st, ≤58 options). No negative result found. Confidence: High.

**C. Specialization.** 12th: the generalist spread 20M parameters over ~5,000 decks (~20k games/deck) — "far too little to fight for the top"; finetuning was "the ship lever." 1st: Stage 4 single-deck models from Stage 2 checkpoints; three fixed-deck specialists. Caveats [R]: 1st saw a Stage-4 matchup edge (70.3%) vanish as opponents improved; 18th's runtime router has no ablation; on a ladder that shows the max of independent ratings, "complementary" matchup coverage across two slots has no ladder value (Jun-Morita). Confidence: Medium-High for per-deck finetuning, Low for routing.

**D. Self-play PPO vs historical/real-deck pools.** All of 1st/3rd/7th/9th/12th/18th. Sampling variants actually used: win-rate-weighted families (1st), arena-frequency + mutations + starter scripts + mirrors (9th), inverse deck-similarity (12th), league of trained exploiters (7th), frozen banks + deck matrix (3rd). I found no source that uses the term "PFSP" in its writeup (keyword hits exist in three repos I did not read); the closest measured analogue is win-rate-based sampling (1st). Failure mode [R]: unbalanced two-net self-play collapses (12th); training win rate vs a fixed BC pool overstates ladder strength (9th). Small-scale RL without these safeguards did badly (rahulsiiitm 342.0; TomBombadyl 580.6). Confidence: High that it works at scale; Low that it works on a Kaggle-notebook budget.

**E. Dynamics/representation pretraining.** NOT FOUND in the sources read ("contrastive" appears only in your own repo). Closest analogues: BC warm start (12th), GBDT-teacher distillation (24th), auxiliary heads — future prize gain (9th), opponent hand (3rd), diagnostics (18th). Caution: oscar-chw's imitation-trained value head scored worse than a constant. [H] Not worth building before the BC/PPO baseline passes its gates.

**F. Bayesian opponent belief.** *For:* 18th router, lamanbo22 (posterior over 160 exact decks), 9th (14-archetype posterior input), 7th (belief core). *Against:* 1st deliberately excluded it; 9th still scored 12.5% (4/32) against Espeon-Sylveon; 7th's agent "cannot change strategy against an unseen deck." No win-rate ablation anywhere → UNKNOWN. A facts-and-bounds tracker (seen cards, deduced prize cards, known hand cards — used by 1st/3rd/18th) is the low-risk subset.

**G. Recurrent policies.** 7th and lamanbo22 use GRUs; 1st, 3rd, 9th, 12th rely on explicit history features. No ablation. [H] Event-history inputs are cheaper than recurrence on CPU.

**H. Learned policy + conservative overrides.** *For:* 7th ships a deterministic lethal check and a dominated-action mask; Scio's lethal-KO guard recovered +0.104 win rate against a lean policy. *Against:* Lawliet's hard-coded rule overrides scored 582/604/676 vs 709 for injecting the same knowledge as candidates; 7th: rules and mechanical teachers cost up to 36 points; a constant leaked into the lethality checker dropped 60% → 38%. Rule: override only when the engine can *prove* the outcome; A/B every guard.

**I. Search hybrids.** *Against:* see §3. Mechanisms cited [R]: each determinized tree plans as if its sampled state were certain, and a fixed time budget is split across samples (9th); credit-assignment variance (Scio); 6 s vs 0.3 ms per decision (Lawliet). *Possibly for:* 18th uses PUCT but did not isolate it; Jun-Morita used targeted search priors in specific contexts; 9th's lookahead **features** are the only variant with a stated success. **Does search help only in specific circumstances?** The sources that isolated it found no general gain; a gain, if any, is limited to bounded one-turn consequences (damage/KO/prize/newly-legal-attack) used as inputs [H].

**J. Ensembles/averaging/portfolios.** taichiiiiiiii shipped 4-model offline-imitation ensembles (no ablation); checkpoint averaging NOT FOUND; weights-averaging claims could not be tested. Portfolio value is bounded by the ladder showing the max of independent readings (selection bias, not skill).

**Conflicts, not averaged.** Search: negative (1st, 7th, 9th) vs used (18th). Belief: rejected (1st) vs adopted (7th, 9th, 18th). BC vs RL: "RL from scratch lost; imitation won" (horno1337, small compute) vs RL-from-scratch winners (1st, 7th, 18th, 3rd). Small RL: 1.0M params reached 1162 (18th) while other small RL runs scored 342–580 — training quality, opponent pools and specialization differ, so size and algorithm labels do not predict outcome.

## 5. The three cross-cutting questions

1. **Do reported training gains survive deterministic evaluation?** Partly, with a large discount. 9th: 83.3% in training vs BC opponents, 53.5–57.3% on the ladder. 12th: relative instruments "cannot see mutual degradation" — always include a frozen external reference. 1st trains with sampling and ships argmax, but no source measured the sampled-vs-argmax gap → UNKNOWN. Common-seed engine gates favoured models that later regressed (XERO47); a gate built on a wrong opponent mix mispredicted the ladder (Jun-Morita).
2. **Does offline action accuracy predict strength?** No. Held-out top-k ordered builds in the exact reverse of game-gate strength six times (Jun-Morita: "fewer data → higher held-out, weaker play"); Scio's 160× data; XERO47. Select checkpoints by games.
3. **Training-only infrastructure vs shippable artifacts.** Shippable (CPU, 2 vCPUs, 600 s bank): NumPy/small-Transformer inference (CAB314, KleinHoumani, 24th's C++ OpenBLAS ≈1 ms), optional bounded engine calls. Training-only: CUDA env + CUDA policy (3rd), JAX model + batched C++ engine (1st), native C encoder (12th), a pure-JAX re-implementation of the rules (muran169633, parity with the official engine not independently checked), a ~110M steps/s GPU prototype of a *simplified* game (oscar-chw). Never ship the engine binaries in a public repo (see `TRANSFER_AND_GAPS.md`).

## 6. Is ~2000 plausible?

- **Highest verified scores.** Simulation: 1398.2 final (author-reported); 1322.6 in-period snapshot [V]. Playground: not verifiable; your ~1135.7 leader reading is the only datum.
- **Three different numbers.** *In-period (historical) rating*: live, noisy, drifts daily. *Final Simulation rating*: re-converged for ~2 weeks from each team's last two submissions; usually below in-period peaks (kento 993 → 897; Jun-Morita expected ~920). *Playground rating*: a different, continuing population; rating settings unknown. *Tournament result*: top-8 Strategy teams play a live finals; Strategy ranking weights stability and explanation (70/20/10), so it is not a ladder rating.
- **Verdict.** No public evidence for ≥1400 anywhere, none for ~2000. Treat 2000 as unsupported. The most credible route past the field is the 1st-place pattern: a small Transformer, PPO against pools of real decks, single-deck specialization, ≥10⁸–10¹⁰ decisions, strict game-based gating — plus careful deck choice. Rating scales are not shown to be comparable between competitions, so **1000+, 1200+ cannot be predicted defensibly** either (see §8.5).

## 7. Resource estimates (all **estimates**; derived from reported setups)

| Plan | Time | Compute | Data | Expected effect [H] |
|---|---|---|---|---|
| 1. Short-term: measure, re-deck, patch the incumbent; use public agents only as sparring/reference | 3–7 person-days | CPU only. Eval throughput reference: 18th ≈ 6 games/s (512 games / 85 s, 16 CPU workers, derived); 1st 220–460 games/s on a batched-C++ box. A 500-game × 8-opponent × 2-seat gate ≈ 8,000 games ≈ 0.4–1.5 h at 2–6 games/s | Playground daily replays for deck/opponent frequencies | 0 to +150 (mostly from deck–pilot match and targeted fixes); no promise |
| 2. Medium-term: rating-filtered, winner-weighted replay cloning + per-deck fine-tune (+ optional lookahead features) | 2–4 weeks | 1 GPU, ~10–40 GPU-hours for 1–8M-param models; plus CPU games for gates | ≥5–12M decisions (12th mined 12M; 9th used 10 days of replays) | Mid-tier cluster 880–1000 on the Simulation scale; relative to ~800 baseline: +100 to +200 plausible, not established |
| 3. High-upside: generalist + league PPO + deck specialists (+ router only if proven) | 6–12 weeks | Reported anchors: 1st ≈ 365M games ≈ 9–19 days on 4× RTX 5090 (≈ 880–1,840 GPU-h, derived); 12th ≈ 11B steps ≈ 7.7 days on 4× H200 + ~3B per deck; 9th 8× RTX 4090 × 56.3 h ≈ 450 GPU-h for PPO after BC; 7th 3.6B decisions on one RTX 4090 learner; **minimum viable (18th-style): ~24 h base + ~9 h per specialist on 1× RTX 4090 + 16 CPU cores** | Real-deck opponent pool (16–616+ lists) mined from replays | 1,100–1,250 if scales transfer; ≥1,400 needs 1st-place-class compute and luck |

If training is limited to Kaggle notebook GPUs, plan 3 reduces to the 18th-style minimum or plan 2.

## 8. Required closing items

**1. Top three technically credible routes**
1. Rating-filtered, winner-weighted replay BC of a pointer policy over legal options, with deck-specific fine-tuning, optionally with bounded engine-lookahead features (A+B+C, 9th-style features).
2. PPO from that BC initialization against a real-deck opponent pool (frozen checkpoints, mutations, starter scripts, mirrors), oracle critic for training only, large rollouts (D; 9th/12th/18th pattern) — sized to available compute.
3. Per-deck two-net finetune / specialists (C), adding a runtime router only if a cross-deck evaluation shows a gain (F).
Not recommended first: MCTS-style search, larger networks, end-to-end deck building, unfiltered data volume, broad hand-written rule overrides.

**2. Single highest-value task first.** Build the Playground replay corpus and the paired-seat gauntlet, then train the rating-filtered BC pointer policy (route 1) and judge it only by games against frozen B-v1.

**3. Data, weights and code required.** *Data:* Playground daily episode datasets (top replays by average agent rating) and the competition SDK + card sheet from the Playground data page; decklists mined from replays. *Weights:* only your frozen B-v1 as the baseline; public checkpoints (CAB314, Scio, 9th-place `net.npz`) as **sparring opponents/reference only** (your rule "don't copy public submissions" and license limits apply). *Code:* replay parser → (state, legal options, chosen, outcome, rating); pointer-policy model; winner-weighted BC loop; deck fine-tune; the gauntlet harness and statistics (`repro/stats_gates.py`).

**4. Falsification experiment.** Arms: (A) frozen B-v1; (B) filtered BC (top-rated agents' games, winners 1.0 / losers 0.3) + deck FT; (C) unfiltered BC with the same number of rows; (D) B-v1 on a different deck (deck-effect control). ≥3 decks, ≥8 opponent pilots drawn by replay frequency, ≥500 seat-balanced paired-world games per cell, Wilson intervals. **Recommendation is falsified if** pooled B vs A < 55% or its Wilson lower bound ≤ 50%, or B ≈ C (filtering adds nothing), or the held-out top-k ranking of checkpoints is negatively correlated with game results (then top-k is invalid and only games may select). Then confirm on the ladder with two bit-identical A copies as noise controls: B must exceed both by more than their spread after ≥400 completed episodes; failure means the local gauntlet does not transfer and must be rebuilt from replays before any scaling. Sample size: ≈620 games detect a 5-point edge at 80% power (see `stats_gates.py`).

**5. Most important unanswered question.** *How does a Playground rating map onto a Simulation-final rating for the same agent* (card pool, opponent population, matchmaking, rating re-convergence)? Without a calibration anchor no defensible prediction of 1000+, 1200+ or 2000 is possible.

## Appendix A — Source index (URL · who · date · track · evidence note)
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/new-writeup-1784257142000 · Luca · slug ~2026-07-17, content final · Simulation · [R] excerpts via search.
- https://github.com/yijieyuan/kaggle-pokemon-tcg · commit 5b6d99c · 2026-09-13 · figures/spreadsheets inspected [V].
- https://github.com/dipamc/kaggle-ptcg-ai-battle · commit a176778 · 2026-09-13 · MIT · docs inspected [V].
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/7th-place-solution-for-the-ptcg-ai-battle-challeng · ~Sep 2026 · [R] excerpts.
- https://github.com/YoInaOwO/pokemon-tcg-ai-battle-9th · 2026-09-14 · [V] repo; [R] results.
- https://github.com/Michal1337/pkmn-kaggle · 2026-09-13 · [V] repo; [R] results.
- https://github.com/KleinHoumani/ptcg-selfplay-agent · 2026-09-13 · MIT.
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/writeups/24th-solution-a-population-based-rl-ecosystem · 2026-09-12 · [R] excerpts.
- https://github.com/SiamRahman29/pokemon-tcg-agent · 2026-08-26 · leaderboard snapshots [V].
- https://github.com/Lawliet-ai/ptcg-ai-battle-agent · 2026-09-04 · MIT.
- https://github.com/Jun-Morita/kaggle-ptcg-ai-battle · 2026-09-09 · MIT.
- https://github.com/XERO47/ptcg-ai-battle-temporal-il · https://github.com/rahulsiiitm/ptcg-rl-agent · https://github.com/TomBombadyl/kaggle_pokemon · https://github.com/kento-umeda-biz/kaggle-ptcg-solution · https://github.com/CAB314/ptcg-ai-battle-agent · https://github.com/horno1337/ptcg-agent.
- https://www.kaggle.com/datasets/kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-10-01 · [R] dataset page excerpt.
- https://ptcg-abc.pokemon.co.jp/ · https://www.pocketmonsters.net/news/9296 · official rules text via search excerpts [R].
Full per-repo detail: `REPO_CATALOG.csv`; per-claim ledger: `repro/claims_ledger.csv`.

## Appendix B — Not found / unverified
2nd, 4th–6th, 8th, 10th, 11th, 13th–15th, 17th, 20th, 27th writeups (15th has a known URL, content not retrievable); 3rd-place writeup text; Playground leaderboard, rules and card pool; any Playground score above ~1135.7; the exact engine search-API semantics (check the current SDK); any measured gain from belief tracking, recurrence, checkpoint averaging or contrastive/dynamics pretraining.
