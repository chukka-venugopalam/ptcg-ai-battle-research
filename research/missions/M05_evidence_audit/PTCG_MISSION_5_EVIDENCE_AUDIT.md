# MISSION 5 — Evidence acquisition, live-slot audit, archive forensics, and evaluator reconciliation

## Mission type
READ-ONLY FORENSICS. Do not train a model, modify the playing policy, package a candidate, or upload to Kaggle in this mission. The purpose is to obtain the missing evidence needed to make one informed candidate decision. No candidate is eligible until this mission returns its artifacts.

## Context
Two prior research outputs have been produced:

- `pokemon_mission_3.zip`: a forensic/decision report and a nested `GIT_SAFE_RESEARCH.zip` containing `tools/submission_audit.py`, `tools/bo3_ab_harness.py`, `tools/build_meta_census.py`, `tools/ptcg_eval_stats.py`, statistics/harness/audit/census tests, an evidence script, legacy code, and fixture-only evidence.
- `pokemon_mission_4.zip`: a separate `eval_stats.py` and `test_eval_stats.py`. Its nine test functions pass under Python 3.13 in a separate inspection. This does not validate agent strength or CABT integration.

Mission 3's decision was to preserve Bv1 and upload nothing because the actual submitted tarballs and official replay data were absent. Mission 3 records 77 passing tests for its harness and helpers; however, its real-engine smoke used fixture agents only. It did not run Bv1 or R12. Its local CABT engine identity is inferred to match the live engine, not proven.

The last screenshot we had showed Bv1 backup at 813.5, R12 at 528.2, and earlier submissions at 826.0, 781.6 and 789.9. Treat these as historical visible readings only. Refresh current live status and episode counts.

## Required input artifacts
Look for these exact files in attached uploads, the authorized project workspace, and any permitted local repository before asking for them:

1. `submission_Bv1_backup_826.tar.gz` (the reported 813.5 submission).
2. `submission.tar.gz` (the historical 826.0 submission).
3. `submission_archaludon_cleanroom_r12.tar.gz` (the reported 528.2 submission).
4. `submission_Bv2_sym8.tar.gz` and `submission_B2_counter_source_grimmsnarl.tar.gz`, if available.
5. The current official CABT SDK/card table or the exact package/commit used for submission.
6. Current Kaggle submissions listing and episode counts, plus at least 50 completed replays/logs per active candidate if the CLI makes them accessible.
7. The official Episodes dataset and matching `manifest.csv`, if accessible and permitted.

Do not treat a filename, screenshot, past assistant statement, or research-report claim as proof that the corresponding archive exists or was inspected. Never invent missing files. If actual submission archives are not available, stop archive-specific phases and report the exact files that need to be attached.

## Phase A — establish the live submission situation
1. Read current `kaggle --version` and `kaggle competitions -h`; then validate all competition CLI subcommands rather than assuming the research script is correct.
2. Identify the precise current Playground competition slug from the official URL and verify current rules, deadline, episode/match semantics, and the exact latest-N submission behavior.
3. Use read-only CLI calls to collect submissions in order with IDs, names, timestamps, scores, status and episode counts. Inspect the leaderboard and available uncertainty/deviation fields. Save raw command output with timestamps and exit codes.
4. Determine exactly which submission is currently older and which a new upload would evict. Do not upload anything. Keep Bv1 safe.

If Kaggle auth or network access is unavailable, say so explicitly and produce a small, exact upload/command checklist for the user. Do not spend the mission substituting old search excerpts for live evidence.

## Phase B — exact archive identity and package audit
For every actual submission archive found:
1. Compute SHA-256 of the tar and every critical member; create a manifest.
2. Extract into a unique clean directory, with safe path handling. Record the complete member tree, sizes, root layout, `main.py`, `deck.csv`, weight paths and runtime dependencies.
3. Compare the archives pairwise. Report byte-identical pairs, file-level hashes, source diffs, deck multiset differences, model-array differences, rule/flag changes and entrypoint differences. Do not infer that `submission.tar.gz` and the Bv1 backup are identical until their hashes match.
4. Execute the exact Kaggle loading contract: source executed without `__file__`, SDK `get_last_callable` behavior, first call returning the intended 60-card deck, then legal action output.
5. Use the *current* R2 card table and actual engine to audit each deck returned by the policy's first call, not only the visible CSV. Verify count, canonical card names, per-name multiplicities, Basic requirement, ACE SPEC rule, valid ID range, actual engine acceptance, and network embedding/vocabulary safety. Fail loudly on silent clipping, modulo wrapping, unknown IDs, wrong deck path, late-callable entrypoint traps or missing weights.
6. Run 20 real BO3 games per actual candidate for safety/instrumentation only, after a successful cold start. These are NOT strength measurements. Report match completion, draws, candidate/opponent failures, illegal outputs, exceptions, fallbacks, overage used, RSS and p50/p95/p99 latency. If 20-game testing is too slow, report actual N and why.

## Phase C — engine identity comparison
The Mission 3 local harness used `kaggle-environments==1.33.0`, which it found with `bo=3` and 1,431 card IDs. It only inferred, rather than proved, equivalence to the live ladder.

Record and compare SHA-256 fingerprints of `libcg`, card table and rules/configuration from the user's official SDK/current environment and Mission 3's environment. Report each fingerprint and the precise source/version. If they differ, do not treat harness results as portable until the difference has been understood and tested. Do not accept only matching card counts as engine equivalence.

## Phase D — acquire and inspect actual match evidence
1. From the validated current CLI, collect at least 50 completed replays for each active submission if available, plus the corresponding agent/game logs. Archive raw responses immutably and hash them.
2. Open and manually inspect five representative replays first, documenting the true schema of states, `current`, `select`, logs, game boundaries, `players.win`, results and seat/first-player fields. Then build/adjust parsers using the schema actually observed; do not guess field names from a local synthetic replay.
3. Parse one row per **BO3 match** for strength statistics. Preserve individual-game rows separately for forensic details; never treat the three correlated games inside one BO3 match as three independent matches for confidence intervals.
4. Build the current meta census only from the official dataset/manifest when available. The previous 14-archetype table is a carried-over third-party panel and sums to exactly 1,729 entries despite the prior report's note about rarer lists; it is not an adequate substitute for a live census. Keep exact deck-list counts separate from hand-assigned archetype labels and preserve unknown/unmatched lists.
5. For every agent loss, extract fields only when the schema supports them: opponent exact list/archetype, game-level seat and first-player choice, turn of first attack, attack choices that were available vs used, damage/immunity/walls, energy allocation, bench/setup and evolution timing, draw/supporter/item decisions, prize race, deck-out/no-active/timeout, and time/overage left.

Do not infer strategic weakness from low rating alone. Separate load/runtime failure, invalid deck/ID, policy decisions, matchup performance and rating uncertainty.

## Phase E — reconcile the statistics before promotion decisions
Do not maintain two contradictory gate implementations.

1. Compare Mission 3's `tools/ptcg_eval_stats.py` with Mission 4's `eval_stats.py`; select one canonical module or formally integrate them.
2. The canonical inferential unit must match the acceptance estimand. For this Playground, use one BO3 match outcome per observation of candidate/control strength, unless direct official documentation establishes a different rated unit. Game-level records are diagnostic observations nested within matches, not independent match trials.
3. Keep first-player/seat stratification as match attributes and game-level diagnostics, without pseudo-replication.
4. Re-test draws, missing rewards, failed cells, seat imbalance, opponent weights, confidence intervals for the candidate's field score and confidence intervals for candidate-minus-control.
5. Reconcile one-sided/two-sided alpha, minimum sample sizes and power. The reports disagree about per-arm totals versus minimum games per opponent; define these unambiguously. If 300/500/1,000 means per opponent, say so. If it means total per arm, say so.
6. Retain tests for the previously confirmed SPRT sign bug, add adversarial synthetic cases, and run Monte Carlo coverage/false-positive tests under the actual BO3 unit. Verify that the Python version required by the tests is stated and that the scripts run under that version.
7. Clearly separate sequential head-to-head early-rejection rules from fixed-N field-weighted confirmation. Do not inspect results repeatedly and then stop when they look good.
8. In the statistical notes, distinguish confidence intervals on average match score from exact BO3 win probability. If draw points are half credit, explain and validate the chosen interval rather than blindly applying a binomial Wilson interval as if draws were ordinary binary trials.

## Deliverables
Produce all of these, with raw evidence and hashes where possible:

1. `LIVE_SUBMISSION_STATUS.md` — exact current submission order/status, which upload would evict which agent, current rules and timestamps.
2. `ARCHIVE_FORENSICS.csv` and `ARCHIVE_DIFFS.md` — SHA-256 hashes, tree/member differences, actual entrypoint, deck audit and exact differences among every archive actually supplied.
3. `R2_ENGINE_FINGERPRINTS.md` — exact current SDK/engine fingerprints and equivalence result; explicit UNKNOWN if it cannot be verified.
4. `AGENT_SAFETY_AUDITS.csv` plus raw logs — actual Bv1/R12 cold-start/smoke results, failure counts, latency and resource measures. Never confuse safety with playing strength.
5. `RAW_REPLAY_INDEX.csv` and `REPLAY_SCHEMA.md` — replay IDs/metadata/hashes and confirmed official schema.
6. `Bv1_R12_LOSS_ANALYSIS.csv` plus `LOSS_ROOT_CAUSE_REPORT.md` — only evidence-backed tags, sample sizes, matchup strata and a clear uncertainty statement.
7. `CURRENT_META.csv` — official date range/manifest, exact-list and archetype frequencies, counts and unresolved IDs. If unavailable, preserve the old panel only as `REPORTED_NOT_CURRENT`.
8. `STATISTICS_RECONCILIATION.md` plus a single canonical module and tests — define BO3 match unit, point metric, seat strata, fixed-N gate, power, draws and failure treatment consistently.
9. `MISSION_5_DECISION.md` — exactly one outcome: (A) enough evidence to start a narrowly defined candidate, (B) input blocked with exact missing artifacts, or (C) a safety bug demands a minimal fix before any policy experiment.

## Hard constraints
- READ-ONLY: no Kaggle uploads, no candidate policy changes, no new submission tarballs in this mission.
- No invented archive inspection, Kaggle CLI output, replays, model tests or scores.
- Label facts `VERIFIED`, `REPORTED`, `INFERRED` or `UNKNOWN`.
- Cite official rules and command outputs for current claims.
- Respect Kaggle rules, repository licenses and restrictions on competition data. Do not put restricted binaries, current episode data, or third-party models into public Git packages without explicit permission.
- If credentials/egress are unavailable, make the blocker immediately clear and generate reproducible commands plus an exact list of files the user must attach. Continue with code/schema reconciliation that can be done safely.
- Never promise a 1000+ rating. The output must let the next mission make an evidence-based choice rather than produce another speculative agent.

At the end, propose **Mission 6 only after reviewing Mission 5's actual output**. Do not jump ahead to a model/training change before the necessary data exists.
