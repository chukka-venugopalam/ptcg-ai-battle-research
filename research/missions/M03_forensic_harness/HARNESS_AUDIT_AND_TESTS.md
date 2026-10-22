# HARNESS_AUDIT_AND_TESTS.md

Sandbox date 2026-10-15 · engine under test: PyPI `kaggle-environments 1.33.0`, `cabt`, `bo = 3`, 1,431 cards · host: 1 vCPU, root (bind mounts allowed).
All agents used in tests are **fixtures** (tiny first-option / random / fault-injecting agents generated at run time). **Nothing here measures strength of Bv1, R12 or any candidate.**

## 0. What is and is not validated

| Item | Status | Evidence |
|---|---|---|
| Statistics module: 31 unit tests | **PASS** | `evidence/unit_test_log.txt` |
| Harness, fake engine + pure helpers: 14 tests (race stubs, kills, accounting, CSV) | **PASS** | same |
| Harness, **real `cabt` engine**: 12 tests | **PASS** | same |
| Audit/diff tool on fixture archives: 18 tests (incl. rule checker vs engine) | **PASS** | same |
| Census tool on locally generated real-engine replays: 2 tests | **PASS** | same |
| Fake-mode self-test (plumbing only) | **PASS** | `evidence/fake_selftest_output.txt` |
| Real-engine CLI smoke (`run --control --h2h --gate screen`, 20 BO3, exit 0) | **PASS (plumbing only)** | `evidence/real_engine_smoke_output.txt` |
| Any run on Bv1 / R12 / a real candidate | **NOT RUN** (archives not provided) | — |
| Equivalence of PyPI 1.33.0 to the live ladder engine | **NOT RUN** (inferred only) | compare fingerprints (FORENSIC §0) |
| Kaggle CLI collection, official replays, official episode schema | **NOT RUN** | `tools/fetch_kaggle_evidence.sh` |
| Concurrency > 1 worker on the real engine; scale ≥ 300 matches | **NOT RUN** (1-vCPU sandbox; largest real run = 20 matches) | — |
| `--kaggle-path mount` without root / `CAP_SYS_ADMIN` | **NOT RUN**; falls back to `off` with a stderr warning, and an agent that hard-codes `/kaggle_simulations/agent` then **fails loudly** (tested) | `test_cwd_relative_and_kaggle_path_agents` |

## 1. Harness defects (legacy `bo3_ab_harness.py`, sha256 `e5ca1a10…`) → v2

| ID | Legacy behaviour (line) | Proven by | Fix in v2 | Test |
|---|---|---|---|---|
| D1 | `if not q.empty(): … elif not p.is_alive(): "exited without result"` (l.110–112): a result that lands between the two checks is recorded as a crash | stub reproduces it; legacy decision logic returns `died` | `poll_child`: blocking `get(timeout)`, never `empty()`; if the child is gone, a **bounded drain** | `PollRaceTests` (5) |
| D2 | `paths` built, then `os.chdir(dirname(cand_main))` (l.63–68) | `evidence/demo_legacy_defects_output.txt` on the real engine: the **opponent submitted the candidate's deck**; with relative paths `env.run` raises `FileNotFoundError` | absolute paths only; **each agent in its own subprocess**, cwd = its own dir, loaded by the SDK's own `get_last_callable` (no `__file__`); each agent's dir bind-mounted at `/kaggle_simulations/agent` when privileges allow | `test_path_isolation_and_module_name_collision`, `test_cwd_relative_and_kaggle_path_agents`, `test_relative_paths_are_absolutised`, `test_cold_start_without___file__` |
| D3 | `p.terminate()` kills only the match process (l.116); agent/native children can leak | by inspection | worker = own session; `killpg(SIGKILL)` on deadline and a sweep after every match | `test_hang_is_killed_and_nothing_leaks`, `test_hang_and_crash_are_contained_and_cleaned_up` |
| D4 | `win = r["rewards"][c] > r["rewards"][1-c]` (l.138): a **draw** (0/0) counts as a loss; `None` rewards raise | by inspection + SDK source (draw is possible) | W / L / D / **failed** per match | `test_outcome_draw_is_not_a_loss_and_none_rewards_fail` |
| D5 | failed matches dropped, weights renormalised over archetypes that happen to have data (`weighted_rate`) | by inspection | failed cells stay in the table; a positively weighted cell with no completed match raises `MissingCellError` and **fails the gate** | `test_failed_cells_are_not_renormalised_away`, `StrictCellTests` |
| D6 | no game-level data, per-decision timing, illegal/exception/fallback accounting; exit code always 0 | by inspection | per-game `first_player`/`winner`; per-seat p50/p95/p99, RSS, overage min/end, illegal (type/range/count/unknown deck id), exceptions, hang, type coercion, agent-exposed counters, stderr scan; **exit 3 on any candidate/control/h2h/unattributed failure** | `test_game_level_parse_matches_engine_totals`, `test_instrumentation_fields`, `test_exception_is_attributed_to_the_right_agent`, `test_illegal_index_is_detected`, `test_deck_with_id_outside_card_table_fails_loudly`, `test_internal_fallback_counters_are_collected`, `test_runtime_failures_fail_the_gate` |
| D7 | no candidate-vs-control comparison; no CI for the difference | by inspection | `--control`, `--h2h`, `compare`, stratified-MOVER difference CI, executable G1/G2 | `IntervalTests`, `GateTests` |
| D8 | `open(out, "a")` silently mixes runs (l.98) | by inspection | `x` mode unless `--append` | `test_no_lost_jobs_and_clean_exit_code` |

Found while building (not in the legacy file): the first decision's latency included interpreter + SDK start-up (a 0.5 s artefact) → fixed with a readiness handshake. The SDK takes the **last callable** in `main.py` as the entrypoint; the harness reports which one it resolved (`host.entry`), and the audit fails if it is not `agent` (`test_sdk_last_callable_trap_is_visible`, `test_last_callable_trap_fails_cold_start`).

## 2. Statistics

**SPRT sign bug (confirmed by execution).** Legacy `sprt(10,100,0.50,0.45)` → `accept H0` and `sprt(70,100,0.50,0.45)` → `accept H1` — both wrong. 10/100 wins has LLR = +7.52 in favour of a 45 % regression; the legacy code negated it for `p1 < p0`. v2 returns `accept H1` and `accept H0` respectively; the LLR is the plain log-likelihood ratio for both directions (verified against direct computation on 200 random cases; operating characteristics simulated: truth 45 % → accepts the regression > 90 %, truth 50 % → < 8 %).

**One- vs two-sided reconciliation.**

| Module | Convention | Games for +5 pp vs 50 % (80 % power) |
|---|---|---|
| `ptcg_eval_stats.py` (reserch2) | two-sided α = .05 | 783 (1,565 per arm for two independent arms) |
| `stats_gates.required_games` (reserch1) | one-sided α = .05 | 617 |
| **v2 canonical `PREREG`** | total α = .05 over two tails ≡ one-sided .025 ≡ "lower end of the 95 % CI > 0"; `sides=` explicit | 783 / 1,565 (617 reproduced with `sides=1`) |

Its own gate (`gate_decision`: pooled rate ≥ .55 **and** Wilson lower > .5) mixed the two: exact binomial pass probability at a **true 55 %** is 0.43 (n = 300), 0.52 (500), 0.52 (800), 0.51 (1,000) — about a coin flip at any N; at true 60 % it passes > 94 %.

**Pre-registered test (frozen before any candidate data).**

* **Unit:** the BO3 match; a draw is not a win. Games are not independent within a match; CIs use matches.
* **Estimand:** Δ = Σ w_a (p_cand,a − p_Bv1,a) over (archetype × seat) strata, weights frozen from the census before running, seat-balanced.
* **Estimator / CI:** stratified MOVER (Newcombe's hybrid-score interval generalised to a weighted sum of independent proportions; one stratum ⇒ exactly Newcombe). **Empirical coverage** (3,000 sims, 12 strata, nominal 95 %): difference 0.937 / 0.946 / 0.950 and single-arm field rate 0.940 / 0.949 / 0.953 for 360 / 840 / 1,680 matches per arm. Weights are treated as fixed, so uncertainty in the panel itself is **not** included.
* **Not paired:** the engine RNG cannot be seeded (FORENSIC §3 #9), so candidate and control are independent samples stratified by archetype × seat. The CI for **one candidate's field rate** (`weighted_rate_ci`) and the CI for **the difference from Bv1** (`weighted_diff_ci`) are different functions; the second is wider.
* **G0 safety:** all matches complete; zero illegal responses, exceptions, type coercions, non-DONE statuses, unexpected fallbacks, timeouts; p99 ≤ 1 s and ≤ 50 % overage (starting values — tune to the real agent).
* **G1 screen (≥ 300 BO3 per arm):** exclusion only. Reject if field-weighted Δ < −3 pp or any archetype with ≥ 25 matches in both arms is ≤ −15 pp.
* **G2 confirm (≥ 1,000 per arm):** pass only if N ≥ 1,000/arm, **lower end of the 95 % CI of Δ > 0**, and no archetype with share ≥ 5 % and ≥ 50 matches per arm has a Newcombe upper bound below −5 pp. Finalist: 3,000/arm on held-out archetypes/exact lists/recent episodes; log configuration hashes.
* **No peeking:** fixed N; the only early stop is a safety failure. SPRT (corrected) is permitted for head-to-head *early rejection* only. With k > 1 candidates reaching G2, use α/k.

**Power consequence (important).** G2 at 1,000 matches per arm is **under-powered for small improvements**: power is 14 % at +2 pp, **27 % at +3 pp**, 61 % at +5 pp, 80 % at +6.3 pp. To confirm +3 pp you need ≈ 4,356 per arm (+4 pp: 2,448; a 3,000-per-arm finalist detects +3.6 pp). A one-defect patch (family A) is therefore unlikely to be *confirmable* on the field at 1,000: use head-to-head (single-sample vs 50 %: 2,178 BO3 for +3 pp) for the effect, and the field panel for no-regression. Report 2's SPRT figure (≈ 590 vs 1,294) does not reproduce (Wald 529, simulated 533, matched fixed-N 1,077).

## 3. Test inventory (`python -W ignore -m unittest discover -s tests -t . -v`, all 77 passed in 103.9 s; log in `evidence/`)

| File | Tests | Needs SDK |
|---|---|---|
| `tests/test_stats.py` | 31 | no |
| `tests/test_harness.py` | 14 fake/pure + 12 real-engine | real-engine part only (skipped loudly otherwise) |
| `tests/test_audit.py` | 18 | engine-backed part |
| `tests/test_census.py` | 2 | yes |

Install recipe used here (heavy optional dependencies of the wheel are not needed): `python -m venv v && v/bin/pip install --no-deps kaggle-environments==1.33.0 && v/bin/pip install jsonschema requests pyjson5 termcolor numpy`.

## 4. Limitations to keep in mind

* Only the PyPI 1.33.0 engine and fixture agents were exercised. Real agents (network calls, large weights) will be slower: measured 3–5 s per BO3 for trivial agents on 1 vCPU; measure before planning compute.
* **Fallback detection is only as good as what the agent exposes**: globals whose names match `fallback|illegal|repair|exception|error|missing|degrad` and a stderr scan. A silent `except: return default` with no counter is invisible — add a counter to a *copy* of the agent before auditing it.
* Latency is measured inside the agent host (plus ~1 ms IPC in the SDK-visible round trip); the ladder runner's own overhead is unknown.
* `parse_games` and the census parser are validated on locally generated replays; the **official episode schema is unverified** and both fail loudly rather than guess.
* Local win rates never calibrate a live rating; the rating model's constants are unknown.
* Legacy files kept in `legacy/` (hashes in FORENSIC §1 inventory) so the defect demonstrations can be re-run.
