# IMPLEMENTATION_ROADMAP

Ordered, gated, evidence-linked. All compute/time/benefit figures are **estimates [H]**; sources in `ARCHITECTURE_RESEARCH.md`. Nothing here promises a rating.

## Ground rules (from the evidence)
1. Judge by games, never by held-out top-k or training win rate.
2. ≥500 seat-balanced, paired-world games per opponent cell; Wilson intervals; p > 0.05 = tie.
3. Ladder readings only against concurrent, same-age, bit-identical control copies.
4. Frozen B-v1 is the baseline for every experiment (your standing decision). Never submit an archive identical to B-v1.
5. Do not lead with tree search, a bigger network, end-to-end deck building, unfiltered data volume or broad rule overrides.

## Phase 0 — Hygiene and measurement (days 0–3)
| | |
|---|---|
| Work | (a) Resolve the public-repo engine-binary issue (see `TRANSFER_AND_GAPS.md` §6). (b) Run the 10-point SDK checklist (`TRANSFER_AND_GAPS.md` §5), especially card-id range vs 1,300-row embeddings. (c) Parse one day of Playground episodes; build deck/archetype frequency table and the rating distribution of agents. (d) Build the paired-seat gauntlet harness using `repro/stats_gates.py`; opponents = frozen B-v1, public reference agents (sparring only), top decks by replay frequency. (e) Submit two bit-identical copies of the current best archive to measure ladder noise. |
| Dependencies | SDK + card sheet; Playground episode download access |
| Compute | CPU only; ≈8,000 games per full gate ≈ 0.4–1.5 h at 2–6 games/s (estimate) |
| Expected benefit | No rating change; removes false positives. |
| Risks | Replay schema lacks per-step options or chosen indices → Phase 1 must fall back to self-generated data. |
| **Gate G0** | Harness reproduces B-v1 vs itself at 50% ± Wilson CI over ≥500 games per seat; embedding/id audit passes; noise spread between the two control copies recorded. |

## Phase 1 — Pilot × deck matrix (days 2–5, parallel with Phase 0 tail)
| | |
|---|---|
| Work | Evaluate the incumbent pilot (B-v1) on 3–5 candidate decks (the replay-frequency top decks, including Dragapult-ex variants) against the gauntlet. Evidence that deck matters: 16/30 final top-15 entries were Dragapult ex; an official-sample Dragapult pilot beat all home-grown pilots in June (TomBombadyl, early ladder). |
| Dependencies | G0 |
| Compute | CPU, ~5 decks × 8 opponents × 500 games ≈ 20,000 games ≈ 1–3.5 h |
| Expected benefit | 0 to +150 if the current deck is weak in today's meta [H]. |
| Risks | The incumbent's features may be deck-specific (state_in/opt_in tied to the Grimmsnarl pool) → a re-deck needs retraining, not just a swap. |
| **Gate G1** | A deck is adopted only if it beats the current deck ≥55% with Wilson lower bound >50% over ≥500 games against the frozen gauntlet, and no control matchup falls below 45% (point estimate). |

## Phase 2 — Rating-filtered BC pointer policy (weeks 1–3)
| | |
|---|---|
| Work | Replay parser → (state, legal options, chosen, outcome, agent rating). Filter to top-percentile agents; winner weight 1.0 / loser 0.3 (9th place); deck-balanced sampling; pointer policy over options (1.0–8M params; 1st-place layout is a spec: 3 layers, d256, 8 heads, ≤141 tokens); optional bounded engine-lookahead features (9th: ≤64 probes) if the SDK supports them within latency; deck-specific fine-tune. Select checkpoints by games only. Include an unfiltered same-size control. |
| Dependencies | G0, G1 (target deck), replay schema check |
| Compute | 1 GPU, ~10–40 GPU-hours; plus gate games (CPU) |
| Expected benefit | +100 to +200 over ~800 plausible on the strength of the 880–1000 BC cluster (Simulation scale); not established for the Playground. |
| Risks | BC ceiling ≈ demonstrators; top-k/gate inversion; overfitting to a deck mix; hidden-info leakage from replay fields. |
| **Gate G2 (the falsification test)** | Arm B (filtered BC + deck FT) vs frozen B-v1: ≥55% pooled, Wilson lower bound >50%, ≥500 games per cell on ≥3 decks and ≥8 opponent pilots; B must beat arm C (unfiltered, same rows) by a Wilson-separated margin; Spearman correlation between held-out top-k and game win rate across checkpoints must be ≥0 or top-k is dropped as a selector. Ladder: after ≥400 completed episodes B exceeds both bit-identical A copies by more than their spread. **Failure ⇒ stop scaling; rebuild gauntlet from replays.** |

## Phase 3 — PPO from BC initialization with a real-deck pool (weeks 3–6)
| | |
|---|---|
| Work | 18th-style minimum first: ~1M params, 512 games/iteration, opponents = pool of real lists mined from replays (16 archetype pools), 20% frozen past versions, plus mutations/starter-script/mirror mix (9th); oracle critic for value only; teacher-KL and gated promotion (12th: 0.53); anti-stall truncation; terminal ±1. Light conservative override only for engine-provable lethal (A/B-gated). |
| Dependencies | G2 passed; CPU engine throughput measured in Phase 0 |
| Compute | Minimum viable ≈ 24 h base + ~9 h per specialist on 1× 4090-class GPU + 16 CPU cores (derived from 18th's reported 85 s/iter); Kaggle-notebook GPUs make this a multi-session job (estimate). Scale-up anchors: 9th 8× RTX 4090 × 56.3 h; 12th 11B steps on 4× H200; 1st ~365M games on 4× RTX 5090 |
| Expected benefit | If scales transfer: 1,100–1,250 on a Simulation-like scale [H]; improvement over BC must be shown, not assumed. |
| Risks | Training win rate vs a fixed pool overstates ladder strength (9th: 83% vs 53–57%); unbalanced self-play collapse (12th); CPU-bound engine throughput (6 games/s class vs 220–460). |
| **Gate G3** | PPO model beats G2's BC model ≥55% (Wilson lower >50%) on the frozen gauntlet **and** a frozen external reference not used in training; no regression on any control deck >5 pp. |

## Phase 4 — Specialization and decision (weeks 6–10)
| | |
|---|---|
| Work | Two-net per-deck finetune (12th) or single-deck Stage-4-style training (1st) for the one or two best decks. Runtime router only if an ablation shows a gain over the single best specialist. |
| Dependencies | G3 |
| Compute | ~2–3 days per deck on a multi-GPU box (12th: ~3B steps) or ~9 h on one 4090 in the minimal 18th form (estimates) |
| Expected benefit | Largest repeated lever in the evidence [R], size unknown here. |
| Risks | Counter-adaptation erases matchup edges (1st: 70.3% edge vanished); router adds failure surface; two ladder slots are independent readings (max-of-two selection bias). |
| **Gate G4** | Specialist beats generalist ≥55% on its deck (≥500 games per opponent cell) without dropping >5 pp on control decks; router ablation shows ≥ +2 pp pooled with Wilson-separated CI or it is removed. |

## Kill / continue rules
- G2 fails → keep B-v1; spend effort on deck selection and gauntlet fidelity only.
- G3 fails but G2 passes → ship the BC model; revisit PPO only with ≥10× more games.
- Compute limited to Kaggle notebook GPUs → stop after Phase 3's minimum viable form unless a faster batched engine path is available.
- Ladder result contradicts local gates → treat as gauntlet-fidelity failure, not as noise, once control copies agree.

## Dependency map
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4. Replays (Phase 0) feed everything; the harness (Phase 0) gates everything.

## Effort summary (estimates)
| Plan | Calendar | GPU | Notes |
|---|---|---|---|
| 1 Short-term (Phases 0–1) | ≤1 week | none | CPU gates only |
| 2 Medium-term (Phase 2) | 2–4 weeks | 1 GPU, 10–40 h | BC + deck FT |
| 3 High-upside (Phases 3–4) | 6–12 weeks | 1 GPU minimum; 4–8 GPUs for 1st/12th-class runs | Gated at each phase |
