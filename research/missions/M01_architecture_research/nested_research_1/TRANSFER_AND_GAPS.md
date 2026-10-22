# TRANSFER_AND_GAPS — from the Simulation track to today's Playground

Tags as in `ARCHITECTURE_RESEARCH.md`: **[V]** inspected, **[R]** reported, **[H]** hypothesis. Date: 2026-10-09.

## 1. Bottom line
Methods transfer; numbers, decks, thresholds and ratings do not. Everything below that says "verify" is a concrete test you can run against the current SDK and Playground data before relying on a Simulation-era finding.

## 2. What transfers (with the condition that makes it true)

| Item | Why it should transfer | Condition / verification |
|---|---|---|
| Pointer/listwise scoring over the legal-option list with masking | Same engine family (`cabt`) presents only legal options; used by 1st, 3rd, 7th, 9th, 12th, 18th [R/V] | Confirm option fields and multi-select semantics (min/max count, STOP) in the current SDK; unit-test mask equals engine-legal set on ≥10⁵ recorded decisions |
| ~1–20M-parameter Transformer over card/zone tokens | Inference fits CPU (≈1 ms for 32 tokens in C++/OpenBLAS, 24th [R]); training works on 1 GPU at the small end (18th, 1.0M [R]) | Measure your latency on 2 vCPUs, no GPU (18th's reported evaluation hardware) |
| PPO with terminal ±1 reward against pools of *real* decklists (frozen checkpoints, mutated lists, starter scripts, mirrors) | Common to all top-12 solutions [R]; the opponent mix is built from replays, which the Playground also publishes daily [R] | Check the Playground replay schema contains both seats' decks and results |
| Oracle critic (opponent hand visible to the critic only) | 3rd (documented, V) and 9th (R) use it; policy input stays legal | Ensure no oracle field leaks into policy inputs or exported weights; test with a shuffled-hand invariance check |
| Rating-filtered, winner-weighted BC as warm start | Scio +115 Elo, 12th ~2× step compression, 9th winner 1.0 / loser 0.3 weights [R] | The *threshold* (Scio used Elo ≥1120) is population-specific; re-derive from Playground rating percentiles |
| Per-deck fine-tune after a generalist | 12th "ship lever", 1st Stage 4, 9th, 18th [R] | Evaluate on ≥3 decks; keep a deck-agnostic control |
| Game-based gating with ≥500 games per opponent, Wilson CIs, "p > 0.05 is a tie" | 1st, Lawliet, 12th [R] | `repro/stats_gates.py` |
| Engine lookahead as *features* (not tree search) | 9th [R]; engine search/probe API referenced by several repos (e.g. TomBombadyl: SearchBegin ≈200 ms on specific contexts) | Confirm the search API exists in the Playground SDK, its cost per call, and that a probe never advances real state |

## 3. What is outdated or population-specific

| Item | Why | Action |
|---|---|---|
| All Simulation ratings (1398.2, 1322.6, 990.7 …) | Different population, scale re-convergence, time | Use only as relative calibration, never as predictions |
| Final top-15 decks (Dragapult ex 16/30, Mega Kangaskhan ex 8/30, …) and matchup tables | Aug 2026 meta; Playground ladder is later and different | Rebuild deck frequencies from Playground episodes before choosing a deck |
| Opponent priors and deck pools (9th's 148-list arena, 12th's 4,941 lists, 18th's 16 pools) | Built from Simulation replays | Rebuild from Playground replays |
| Elo ≥ 1120 demonstrator threshold (Scio) | Specific to Simulation rating distribution | Pick a percentile (e.g. top 1–3% of agents) in the Playground feed |
| TomBombadyl's June results (880.9, 580.6, 1196.1 leader) | Early ladder, weak field | Do not compare to later ratings |
| "Search never helps" | Established under the Simulation's time/hardware limits and for tree search specifically | Re-test only the narrow forms in §2 |

## 4. What is unavailable

| Missing | Consequence |
|---|---|
| 1st-place code and weights ("after the second round") | Only specs: feature lists, hyper-parameters, token limits, deck lists. Re-implementation needed |
| 12th-place weights; 7th-place code; 9th-place decklist and opponent prior (excluded from the zip) | Recipes only |
| Writeups for 2nd, 4th–6th, 8th, 10th, 11th, 13th–15th, 17th, 20th, 27th; 3rd-place writeup text | Method of ≥10 of the top-15 teams is unknown |
| Engine source | Cannot be modified or vectorised by you; a re-implemented engine (JAX/CUDA, muran169633, oscar-chw prototype) carries parity risk |
| Kaggle leaderboard, discussion and writeup pages | I read them only through search excerpts; the Playground leaderboard figure (~1135.7) is yours, unverified by me |
| Playground rules, card pool and rating settings | See §5 |

## 5. Verify against the current SDK / Playground (checklist)

| # | Check | Why it matters | Test |
|---|---|---|---|
| 1 | **Card-id range vs embedding tables.** Reported card-DB sizes: 1,267 post-rotation (12th, README), 1,284-row card array (3rd's `embed_pca.npz`, V), 1,300 rows (Scio, V), 1,536 (CAB314, V); the official site says the second round adds new cards | Your incumbent is Scio-derived with 1,300-row embeddings; an id ≥ table size would clip, wrap or crash silently | Compare max card id in the Playground card sheet to every table size; fail the build on any clip; log out-of-range ids at inference |
| 2 | Observation/option schema (`obs["select"]["option"]`, selection min/max, STOP) | Pointer heads and masks depend on it | Replay-to-tensor parity test against live engine on a seed set |
| 3 | Time limit: official 10 minutes per player per match [R] | At ≈180 decisions/game (1st-place ratio, derived) ≈ 3.3 s average per decision; heavy search is unaffordable | Measure p99 decision latency on 2 vCPUs; enforce a hard fallback |
| 4 | Submission packaging: whether `cg/` is bundled (sources disagree) | A packaging error costs a submission slot (5/day) | Smoke-test the exact archive via the official self-play start |
| 5 | Replay schema in Playground episode datasets | BC needs per-step acting-player observation, chosen indices, outcome, both ratings | Parse one day; check coverage of options and chosen indices |
| 6 | Whether replays leak hidden information about the non-acting player | Oracle-style features must not enter the policy | Inspect fields; assert policy features are computable from acting-player view |
| 7 | Rating system parameters (μ₀, σ, matchmaking) | Noise estimates (±24 pp at ~35 games) are Simulation-measured | Submit two bit-identical copies; measure their spread at fixed ages |
| 8 | Engine search API semantics (SearchBegin/Step/End/Release names, determinization, cost) | 9th-style probe features need it | Microbenchmark 64 probes/decision on the 2-vCPU target |
| 9 | Simulator deviations from official rules (organizers say some differ) | Hand-written tactical rules and lethal checks must follow the *engine*, not the rulebook | Differential tests from recorded games |
| 10 | Static constants for stochastic or hand-size-scaled effects | Lawliet: engine reported damage 0 for a hand-size attack; a constant estimate poisoned the lethality checker (60% → 38%) | Property-test lethal logic on all such cards |

## 6. Licensing, rules and safety flags

1. **Your public repo currently tracks the compiled engine** (`ptcg_claude_agent_builder_pack/competition_sdk/cg/`: `cg.dll`, `libcg.so`, `libcg.dylib`, `libcg-arm64.so`) [V]. The 9th-place NOTICE states the engine ships under a competition-use-only license and may not be reposted [R]. **VERIFY** the Playground data license and consider removing these files from the public repo and its history. (Several other public repos do the same; that is not a defence.)
2. **`Paschal-titre4015/ptcg-population-rl` is a lure-style repository** (consumer "download the Windows app" README; bundled zip listing `cli.exe`, `Application.bat`, `skey.txt`; I never extracted or ran it). Do not download or execute it. The 24th-place writeup on Kaggle is the legitimate source for that method.
3. **Pickle files** (`.pt`, `.pkl`, `.joblib`) from third parties can execute code on load. I loaded none. Prefer `.npz` (loaded with `allow_pickle=False` in `repro/inspect_npz.py`) or sandbox the load.
4. **Reuse policy conflict.** The mission asks to improve "using existing public code and weights"; your standing rule is "don't copy public submissions." This package resolves it as: public agents are *sparring opponents, teachers' behaviour references and implementation references*, not ship candidates, unless you decide otherwise explicitly after checking each license and Kaggle's rules. Licenses: 3rd place (MIT, weights included), CAB314 (MIT, weights included), KleinHoumani (MIT, no weights); 9th place and Michal1337 have no LICENSE file.
5. The 1st-place README says code will follow after the second round; do not treat the spec as a license to reconstruct weights.

## 7. Gaps that would change the plan if filled
- A calibration anchor between Playground and Simulation ratings (the single most important unknown).
- The 2nd-, 4th–6th-place methods (several use Mega Kangaskhan ex; nothing is known about their pilots).
- A win-rate ablation for belief tracking, recurrence, search-as-features and specialist routing.
- The actual Playground opponent population: who is on the ladder, which decks, and whether Simulation top teams resubmitted.
