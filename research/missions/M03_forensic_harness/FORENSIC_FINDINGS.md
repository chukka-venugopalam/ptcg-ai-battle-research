# FORENSIC_FINDINGS.md — Mission 3 (Pokémon TCG AI Battle Challenge Playground)

Sandbox date 2026-10-15. Every point is tagged **verified** (I ran it / read the source here), **reported** (a document says so; not re-checked), **inferred** or **unknown**.

## 0. Result

* **No forensic statement about Bv1 or R12 can be made yet.** *[verified]* The uploads contain only the two research ZIPs. None of the five submission archives, no Kaggle CLI output, no replays, no episode dataset/`manifest.csv`, and no official SDK were attached, and kaggle.com is unreachable from the sandbox. Phase 0.1–0.6 on your real archives and Phase 1 forensics are **NOT RUN**.
* **The reports' biggest engine gap is closed.** *[verified on the local build]* PyPI `kaggle-environments==1.33.0` (wheel sha256 `d22b4fd6…`, files dated 2026-10-02) ships the `cabt` engine with **BO3 as the default (`bo = 3`) and a 1,431-card table (ids 1–1431)**. Both reports say BO3/R2 needed `kaggle-environments` master (PyPI 1.32.7 = single game / 1,267 cards). I did not test 1.32.7. That 1.33.0 equals the live ladder engine is **inferred** (pool size and BO3 match the reports' description of R2/BO3). Compare fingerprints on your machine — every harness run header records them (libcg sha256 `c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22`, card-table sha256 `7fd6c45fcdc2913a1fe19ddb67c01aa9b204718cb760be7582cf4fcc3513de9c`).
* **R12 = 528.2: cause undetermined.** Five hypotheses, an ordered decision procedure and the tool that runs it are in §6; none can be decided without the archive and replays.
* **What now exists and is tested (77/77 tests pass, `evidence/unit_test_log.txt`):** a repaired statistics module, a repaired real-engine BO3 harness, an archive audit/diff tool, a replay census tool, and a read-only Kaggle evidence script. See `HARNESS_AUDIT_AND_TESTS.md`.

## 1. Inputs received

| Input | Status | sha256 (first 16) |
|---|---|---|
| `pokemon_reserch1.zip` (72,746 B) → `reserch1.zip`, `repro_package.zip` | received, extracted recursively | `aa1a5b65a7ebade9`; nested `8b1f8c966b4cf5d6`, `e9d724b85bb659d0` |
| `reserch2.zip` (43,182 B) → `files.zip` | received, extracted recursively | `fedc6d85e27ac7c9`; nested `43bd03087be7ae0a` |
| `submission_Bv1_backup_826.tar.gz` (813.5) | **MISSING** | — |
| `submission.tar.gz` (826.0, historical) | **MISSING** | — |
| `submission_archaludon_cleanroom_r12.tar.gz` (528.2) | **MISSING** | — |
| `submission_Bv2_sym8…`, `submission_B2_counter_source_grimmsnarl…` | **MISSING** | — |
| Official R2 SDK / card data / exact `kaggle-environments` commit | **MISSING** (PyPI 1.33.0 used instead, see §3) | wheel `d22b4fd6d4ce3e2e…` |
| `kaggle competitions submissions` (ids, order, timestamps, episodes) | **MISSING** | — |
| ≥ 50 replays/logs per active agent; official episodes + `manifest.csv` | **MISSING** | — |

Full per-file hashes of all 30 extracted files: `evidence/INVENTORY_SHA256.csv`. The reports' own `ARTIFACT_HASHES.csv` (135 rows) hashes **third-party repository artifacts** that are not in the uploads, so it cannot be checked here *[verified]*.

## 2. Artifact identity (Phase 0.1–0.2) — NOT RUN

Nothing to hash, extract or diff. Ready to run once the archives exist:

```bash
sha256sum submission*.tar.gz
python tools/submission_audit.py diff submission_Bv1_backup_826.tar.gz submission.tar.gz submission_archaludon_cleanroom_r12.tar.gz \
       submission_Bv2_sym8.tar.gz submission_B2_counter_source_grimmsnarl.tar.gz --out diff.json
```
It prints tree/hash differences, unified line diffs, **top-level constant ("rule-flag") changes**, `deck.csv` multiset changes with card names and per-array `.npz` differences. *Why this matters (inferred):* if `submission.tar.gz` (826.0) and `submission_Bv1_backup_826.tar.gz` (813.5) are **byte-identical**, you already have a free A/A pair: the same bytes read 12.5 rating points apart. Check the hashes before reading any ≤ 20-point gap among 826.0 / 813.5 / 789.9 / 781.6 as signal.

## 3. What the real SDK does — verified on `kaggle-environments 1.33.0`

| # | Fact | How verified |
|---|---|---|
| 1 | `bo` default **3**; `actTimeout` 0, `runTimeout` 2000, `remainingOverageTime` starts at 600 s | `cabt.json`; `make("cabt").configuration` |
| 2 | Engine card table: **1,431 cards, ids 1…1431**. Copy limit is by **card name** across ids (4 = ok, 5 = rejected); no Basic Pokémon / 2 ACE SPEC / unknown id / id 0 are rejected | native `AllCard`; `tests/test_audit.py::test_rule_checker_agrees_with_the_engine` (8 deck cases, rule checker vs engine) |
| 3 | Engine deck-rejection codes observed: `errorType` 1 = unknown id, 2 = copy limit, 4 = other rule (no Basic / ACE SPEC). The code→meaning map is observed, not documented | probes |
| 4 | File-path agents are loaded by `get_last_callable`: `exec` in a **fresh dict `{}` → no `__file__`**; the **last callable in module globals is the entrypoint** (a helper defined or imported after `def agent` hijacks it); **no `chdir`**; `sys.path` gets the agent dir only *during* the exec | SDK source; runtime `has___file__ = False`; fixture `trap` |
| 5 | An exception in the agent is caught → status `ERROR` → that seat loses; `duration − actTimeout > remainingOverageTime` → `TIMEOUT` | SDK source; `crash5` fixture |
| 6 | Decks are submitted **once** (step 0); games 2–3 reuse them. `select is None` never recurs: 16 deck prompts at step 0, **0 after**, over 8 real BO3 (19 games). Answers report-2 experiment **E03** for this build | interpreter source + 8 real matches |
| 7 | First/second player is an **in-game choice**; `reverse_player` alternates which seat chooses. `current.firstPlayer` is −1 until chosen, then 0/1; `current.round`, `players[j].win`, `draw` give game boundaries | source + observed; used by `parse_games` |
| 8 | Rewards +1 / −1 / 0; a **draw is possible** at match level (all 3 games played and tied). None drawn in 19 local games, so frequency unknown | source |
| 9 | **The engine RNG is not seedable**: `BattleStart` takes only the two decks; six identical deterministic matches (same python seed) produced six different traces (57–467 steps). **No paired-seed designs** | API + experiment |

Equivalence of this build to the live ladder: **inferred** (see §0).

## 4. R2 deck-ID audit (Phase 0.4) — NOT RUN on your decks

Needs each archive's `deck.csv` **and** the deck its code actually returns on the first call (they can differ; the first-call deck is the one that is played). The audit tool prints count / id / canonical name / multiplicity, checks 60 cards, ids ∈ 1…1431, copies by name, ≥ 1 Basic, ACE SPEC ≤ 1, **engine acceptance** via the real `BattleStart`, `.npz` embedding-table row counts against the maximum deck id (fails loudly on rows ≤ max id; hashed pools need the id→slot map read in source), and flags `clip` / `% pool` / `min(id, …)` / `dict.get(card, 0)` idioms that silently clip or wrap a new id.

## 5. Entrypoint, model-load and fallback path (Phase 0.3, 0.5, 0.6) — NOT RUN on your archives

`python tools/submission_audit.py audit <archive> --matches 20` performs the Kaggle-style cold start (no `__file__`, SDK last-callable rule, **first call must return the 60-card deck**, equal to `deck.csv`), then real BO3 matches with per-decision p50/p95/p99, RSS, overage used, illegal/exception counts, agent-exposed fallback counters (globals whose names match fallback/illegal/repair/exception/error/missing — *heuristic*) and a stderr scan for `Traceback`/`fallback`. Gate: zero failures of any kind, p99 ≤ 1 s, ≤ 50 % overage — **a pass proves safety only, never strength**. Hazards to look for first in Bv1/R12 *[inferred from SDK semantics]*:

1. `__file__` used unguarded → `NameError` on the first call.
2. Weights/deck located only via `/kaggle_simulations/agent` or cwd; a miss followed by a broad `except` that returns a default → **silent fallback** (the harness bind-mounts each agent's own dir there when run as root).
3. A callable defined/imported after `agent` (SDK picks it).
4. Card id clipped/wrapped into a smaller embedding table.
5. Pickle/torch weights whose dependency is not available in the runner.

## 6. R12 (528.2) — ordered diagnosis; none decided

| Order | Hypothesis | Decisive evidence (not available now) |
|---|---|---|
| 1 | Archive / load / fallback / clock failure | `audit` safety gate on the R12 archive; per-episode status/logs from the CLI |
| 2 | R1→R2 id or name drift; deck not what was intended | `deck_audit` names vs the intended list; engine acceptance; `.npz` bounds |
| 3 | Combo/pilot failure (setup, evolution, energy, targeting, no-progress decisions) | ≥ 50 R12 replays: turn of first attack, attacks available vs used, energy attach rate, bench/evolution timing |
| 4 | Deck/meta matchup weakness | per-archetype record vs a competent reference pilot with the same deck (needs ≥ 50 % field-weighted to keep the deck, per your rule) |
| 5 | Immature/noisy rating | episode count and rating uncertainty from `kaggle competitions submissions -v` |

Report 2's reading that 528 is "a functioning-but-weak signature, pointing away from packaging failures" is a **weak inference** *[inferred]*: a fallback-heavy agent on a young rating can sit near 528 as well. Do not choose among 1–5 on that basis. Do not repeat the R12 implementation unless 1–2 name a concrete defect **and** a reference pilot shows the deck can hold ≥ 50 % on the current panel.

## 7. Bv1 — single most frequent *fixable* loss class: UNKNOWN

No replays. Pre-registered qualification for patch family A (so the criteria are fixed before looking): a class qualifies only if **(a)** it appears in ≥ 10 % of Bv1's losses (≥ 5 of the first 50 loss replays), **(b)** a single-condition detector exists, **(c)** the fix changes **< 1 %** of recorded decisions (behavioural-diff on ≥ 20,000 decisions: non-target states byte-identical, target states changed as intended). Otherwise: no patch, keep Bv1. Loss-tagging fields: opponent list/archetype, seat, first/second per game, turn of first attack, attacks available vs used, immunity/walls, energy allocation, bench/setup quality, supporter/item sequencing, prize race, deck-out/no-active/timeout, overage left. **Not built:** the replay `logs` schema of the official dataset is unverified; writing the tagger first would be guessing. Build it against the first 5 real replays.

## 8. Claim ledger for the two research reports

| Claim (source) | Status | Evidence |
|---|---|---|
| PyPI is single-game/1,267 cards; BO3/R2 need master (reserch2 plan/audit/harness header) | **contradicted for 1.33.0** | §0, §3 |
| 14-archetype third-party panel: shares and Wilson CIs | arithmetic **verified** (shares = entries/1,729 within 0.05 pp; 14/14 Wilson CIs within 0.04 pp) | `CURRENT_META_REBUILT.csv` |
| …but the 14 rows sum to exactly 1,729 (100 %) | **inconsistent** with "other, rarer lists exist"; cause **unknown** | panel restricted to 14, or counts normalised |
| Best Simulation result 1,398.2 (Dragapult ex, ~2.24 M-param Transformer, PPO) | **reported** (author-reported; highest in-period snapshot 1,322.6) | not testable; historical method evidence only |
| Replay filtering helps; uncurated expansion hurts; offline accuracy is a poor selector | **reported** (Scio +115 Elo; six logged inversions) | secondhand |
| Tree search gave no repeatable gain | **reported** (e.g. 55.6 % vs 52.9 %, p = 0.81) | secondhand; keep as a prior |
| Only the latest 2 submissions stay active | **reported**: reserch2 quotes official-page excerpts; news on the *Simulation* competition says the same. The Playground Rules page itself was not retrievable | **verify live before any upload** |
| "SPRT averages ≈ 590 matches vs 1,294 fixed" | **not reproduced**: Wald 529, simulation 533; matched fixed-N (one-sided α = β = .05) is 1,077; 1,294 is two-sided α with 95 % power | `tests/test_stats.py` |
| Power table (300/1,000/3,000; +3 pp ≈ 4,400 per arm) | **verified** (0.581/0.613, 0.544/0.562, 0.526/0.536; 4,356) | `ptcg_eval_stats.py --tables` |
| Roadmap gate "≥ 55 % pooled and Wilson lower > 50 %" | **flawed**: passes a true-55 % agent only ≈ 51–52 % of the time at any N | exact binomial, `gate_power` |
| `sprt()` reverses sign for p1 < p0 | **confirmed and fixed** | reproduced by execution |
| Two stats modules agree | **no**: 617 (one-sided) vs 783 (two-sided) games for +5 pp | reconciled, `HARNESS_AUDIT_AND_TESTS.md` §3 |
| Experiment E03 (BO3 deck re-prompt) | **answered**: no re-prompt on 1.33.0 | §3 #6 |
| 18-row experiment matrix | all rows are `NOT RUN` plans, no results | CSV status column |
| Agent matrix evidence | 25 of 26 rows tiered "Reported"; 1 "Verified" | CSV |
| Playground leader ≈ 1,135.7 | **reported by you**, unverified here | report 1 |

## 9. Unknowns and exact blockers

1. SHA-256 and diffs of Bv1 / 826.0 / R12 / Bv2 / B2 — attach the five archives.
2. Which active agent is older (what a new upload evicts) — `kaggle competitions submissions -c <slug> -v`.
3. Episode counts and rating uncertainty of the two active agents.
4. ≥ 50 replays + logs per active agent; the official episode dataset + `manifest.csv` (meta census).
5. Whether the ladder engine equals PyPI 1.33.0 — compare the two fingerprints in §0 with your official SDK.
6. Official Playground Rules text (latest-N semantics, data/code-sharing limits) and the competition slug.
7. Exact official source and licence of the "official Dragapult sample" (family B).

Run `tools/fetch_kaggle_evidence.sh` (read-only; **NOT RUN here**) to collect 2–4.
