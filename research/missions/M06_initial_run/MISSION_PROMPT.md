# MISSION 6 — PTCG Rating Recovery and Evidence-First Agent Improvement

**Created:** 2026-10-16  
**Goal:** Stop speculative submissions, identify the largest verified cause of weakness, and produce at most one measurable stronger candidate. Long-term target: **1000+ skill rating**, with a first gate of demonstrably beating the best available baseline. This is a target, not a promise.

## Mission order

You are given a handoff ZIP containing the recent research packages, Mission 5 output, the current rating screenshot, several actual agent archives, the official current R2 package, and the R2 compatibility smoke-test report. Unzip it first and work from the included files. Do not ask for files already present in the handoff bundle.

**Do not upload to Kaggle in this mission.** Do not replace the current active agents. Return an offline-evaluated candidate archive only if the gates below pass. If a gate cannot be executed, label it blocked and continue with other executable work; do not fabricate it.

## 1. What the evidence currently says

### Screenshot snapshot — 2026-10-16

The attached Kaggle Submissions screenshot visibly lists:

| Displayed filename | Rating shown | Approx. age shown |
|---|---:|---:|
| `submission.tar.gz` | **780.4** | 15h |
| `submission.tar.gz` | 737.2 | 7d |
| `submission_Bv1_backup_826.tar.gz` | 769.4 | 8d |
| `submission_archaludon_cleanroom_r12.tar.gz` | 530.7 | 8d |
| `submission_Bv2_sym8.tar.gz` | 781.6 | 9d |
| `submission_B2_counter_source_grimmsnarl.tar.gz` | 789.9 | 10d |
| `submission.tar.gz` | **826.0** | 13d |

The user reports previously seeing a score around 850, but 850 is not shown in this screenshot. Preserve that as a user-reported earlier reading, not a value verified by this image.

These are skill-rating readings at different ages, not a controlled head-to-head experiment. They do not prove which source change caused a rating movement. The current latest row at 780.4 is the immediate concern; the older 826.0 row is a reference, not proof that its exact bytes are the available local Bv1 archive.

Kaggle's official Playground description says the rating is a Gaussian skill estimate, episodes are matched by similar rating, new submissions receive more frequent play, and only the latest two submissions are active. Thus another upload can change which agents remain active. Refresh the live page/CLI before any future upload decision. Do not assume a submission's displayed filename identifies the exact local archive.

### Mission 5 result — the key limitation

`MISSION_5_OUTPUT.zip` returned **Outcome B: input blocked** in that execution environment:
- **0 real matches** were analyzed; `LOSS_ROOT_CAUSE_REPORT.md` states `n = 0`.
- No submission archives, official live replays, Kaggle CLI/authentication, or engine package were available in that sandbox.
- Mission 3's recorded test result was 55 passed and 22 engine-backed tests skipped; “77 passing” was not reproduced there.
- Mission 4's nine statistics tests and Mission 5's 21 canonical-statistics tests were synthetic/statistical validation only. They do not show that any policy is strong.
- Mission 5 corrected an important evaluation issue: **BO3 match**, not each game within it, is the inferential unit for candidate strength. Seat and game records are diagnostics/strata.

Mission 5 did **not** find the cause of the rating decline. Do not describe runtime, deck quality, model weakness, the SDK update, or seat handling as the root cause until evidence supports it.

## 2. Inputs supplied in this handoff

Use `input_manifest.json` to verify every file's SHA-256. The bundle includes:
- the latest locally built R2 compatibility candidate (same bytes as local `submission.tar.gz`);
- the local Bv1 baseline archive;
- B2 counter-source and Bv2 symmetry-k=8 archives;
- the current official Playground/R2 package;
- research packages 1–4, Mission 5's output, the current screenshot, and the R2 validation note.

**Important identity rule:** the local archive called `latest_candidate_R2_compat.tar.gz` has SHA-256 `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`. It is the October 15 compatibility candidate, but the screenshot row named `submission.tar.gz` at 780.4 is not proven byte-identical unless Kaggle's exact submitted artifact can be downloaded and hashed. Likewise, do not assert a screenshot row and local archive are the same binary solely from filenames.

The exact R12 submission archive and actual live episode/replay data are **not included**. Do not reconstruct R12 from a directory or guess its code. If the exact R12 archive cannot be recovered through authorized local/Kaggle access, explicitly leave R12 archive comparisons blocked.

## 3. Required execution stages

### Phase A — inventory and exact identity (read-only)

1. Verify all handoff hashes and archive roots; safely extract each archive into its own directory.
2. Build an archive/member hash manifest and source diffs for latest R2 candidate, Bv1 baseline, B2, and Bv2. Compare `main.py`, deck multiset, SDK/native library, policy weights, feature mapping, embedding guards, flags, dependencies, and entrypoint behavior.
3. If Kaggle CLI/auth is available, first read `kaggle --version` and `kaggle competitions -h`, confirm subcommands, and gather the live submission order/IDs/timestamps/status/rating/episode counts **without uploading**. Download exact submitted archives only if the interface permits it. Save raw output, timestamps, exit codes and SHA-256. If auth is absent, retain screenshot facts and continue offline.
4. Identify which two submissions are currently active from fresh evidence. Never infer eviction order from the screenshot alone if timestamps/status have changed.

Deliver: `LIVE_STATUS.md`, `ARCHIVE_FORENSICS.csv`, `ARCHIVE_DIFFS.md`, raw command output where available.

### Phase B — establish that the actual agents work correctly

Use the **included official R2 package**, not a fixture-only engine. Fingerprint the native engine libraries, card table and rules/config by SHA-256. Compare to any environment used by prior harnesses. Card counts alone do not prove engine equivalence.

For every supplied archive:
1. Verify `main.py` is at the archive root and `deck.csv` exists.
2. Run it under the real loading contract (including execution without `__file__` if that is the Kaggle contract), cold-start it, and verify the intended first-call 60-card deck is returned and accepted by `BattleStart`.
3. Validate deck legality against the current R2 card table and engine, including total 60 cards, per-name limits, Basic Pokémon requirement, ACE SPEC rules and valid IDs.
4. Exercise legal actions over complete games. Record crashes, illegal output, fallbacks, missing weights, high card/attack IDs, latency, memory and completion status.
5. Keep **safety** results separate from **strength** results. A successful self-play smoke is not a win-rate result.

Deliver: `R2_ENGINE_FINGERPRINTS.md`, `AGENT_SAFETY_AUDITS.csv`, raw logs, and a one-command reproduction script.

### Phase C — investigate high-value strategic failure modes

Inspect the current public Playground notebooks/results, prioritizing:
- “PTCG Bo3 agent, one-line seat fix | LB 1132.4”
- “Pokemon Challenge Ultra 900+”
- “PTCG AI Battle Playground: Baseline + New Cards”
- the recent “PTCG Daily Meta” and “PTCG Bo3 Gauntlet” notebooks.

Official listing: https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/code

Do not use titles/scores as proof that you have inspected source. Open the notebooks and examine code/data when accessible. Record URL, revision/date, license/permission constraints, exact reproducible idea, and what was or was not accessible. If code is unavailable, label it unavailable; do not guess what “seat fix” means.

Audit these hypotheses in the supplied agents, in this order:
1. **Seat/turn-order handling and BO3 symmetry.** Verify player indices, current-player perspective, active/bench ownership, first-player logic and game/seat changes. Use equivalent mirrored states to check whether equivalent decisions remain equivalent. Treat the public notebook title as a lead, not evidence of our bug.
2. **Action/embedding correctness.** Confirm current R2 card, attack, target and slot IDs cannot cause out-of-range indexing, stale ID semantics, or silent clipping that changes the selected move.
3. **Deck performance and setup plan.** Compare the actual supplied decks and exact card lists. Use the current card data and accessible recent meta evidence; avoid card changes based solely on intuition.
4. **Policy quality.** Only after protocol, SDK, seat and deck issues are ruled out, test one policy change at a time.

Create `HYPOTHESIS_REGISTER.csv` with columns: hypothesis, evidence_for, evidence_against, test, result, decision, confidence. Rank by measured effect and confidence. Do not mix multiple strategic changes into one candidate.

### Phase D — run a real local head-to-head tournament

Build or repair the evaluation harness by reusing the Mission 3/Mission 5 tools where sound, but verify their assumptions against the included current R2 engine. Do not rely on engine-backed tests that silently skip. Add a hard test that fails if the engine is unavailable when the tournament is requested.

**Match design:**
- Compare the exact available agents first: latest R2 compatibility candidate vs Bv1, B2 and Bv2; then compare a single controlled candidate against the best-performing available control.
- Use real BO3 matches, record a single match outcome per BO3 and retain individual-game logs only for diagnostics.
- Balance first-seat/player-side conditions and opponent assignments. Use reproducible, recorded seeds; if paired seeds or swapped-side controls are valid under the engine, use them.
- Record every match in `TOURNAMENT_MATCHES.csv`: candidate hash, control hash, seed, BO3 result, games, seat/first-player strata, opponent, completion, error/fallback tags and replay/log path.
- Inspect at least five completed local replays manually before trusting the parser. The official live replay schema is still unknown unless actual official replays are obtained.
- Start with a small correctness smoke (at least 20 complete BO3 matches per compared pair where runtime permits), then run a screening stage of at least 200 BO3 matches per candidate-control pairing with seat balance. Screening is only for identifying large effects and safety issues; **it cannot reliably confirm a small +5 percentage-point advantage**.
- Pre-register the final sample size and metric before looking at confirmation results. For a fixed-N confirmatory comparison, use the canonical Mission 5 statistics module and one-sided alpha 0.025/power 0.80; Mission 5 estimated roughly 1,100–1,600 BO3 matches per arm to detect +5pp depending on draw rate/variance assumptions. If that is unaffordable, state the achievable detectable effect and report the confidence interval; do not claim a small improvement was proven.
- No repeated peeking to stop when the result looks good. Sequential early rejection may only use the tested rule; fixed-N confirmation must remain fixed-N.
- Exclude infrastructure/opponent failures only under the canonical rules and report them. Candidate-caused failures count against the candidate, never disappear from results.
- Report match points (win=1, draw=0.5, loss=0), match win probability separately, seat/opponent strata, confidence intervals and failure rates. Do not treat the three games in a BO3 as independent samples.

Deliver: `TOURNAMENT_PLAN.md`, raw match CSV, per-game diagnostic CSV, raw logs/replays, `TOURNAMENT_RESULTS.md`, canonical stats code/tests, and seed/config manifest. Any synthetic-only test results must be clearly labeled synthetic.

### Phase E — one controlled improvement and a promotion decision

Only if Phases A–D identify a concrete issue or a credible strategic improvement:
1. Create a branch/copy from the best verified local archive; preserve each baseline and record hashes.
2. Change **one** thing, with the hypothesis and expected failure mode stated before running.
3. Re-run all safety/deck/entrypoint tests plus head-to-head matches against the best available control. Confirm the candidate is not merely benefiting from seat imbalance, an opponent mismatch, more episodes, or runtime instability.
4. Produce a submission tarball only when the archive is valid and local results meet a predeclared promotion gate. Never overwrite the baseline archive.
5. The promotion gate is: no protocol/deck/runtime regression; zero unexplained candidate faults; balanced reproducible BO3 results; an improvement supported by its match-level uncertainty interval or clearly labelled as only a screening lead; and exact archive hash documented. If evidence is weak or results are mixed, **do not package a new submission**—keep the best baseline and state the next test needed.

Target is to exceed the best available baseline first, then work toward 1000+. Never claim the live rating will equal the local tournament score.

## 4. Deliverables checklist

Required final files:
- `MISSION_6_DECISION.md` — choose exactly one:
  - **A: Candidate passes the predeclared safety/strength gate** (attach archive + checksum; still do not upload);
  - **B: No candidate proven stronger** (recommend preserving the best verified archive; give the strongest evidence and next discriminating test);
  - **C: A critical correctness issue was identified/fixed, but strength is not yet proven** (attach the fix only if safe; no upload).
- `LIVE_STATUS.md`
- `ARCHIVE_FORENSICS.csv`, `ARCHIVE_DIFFS.md`
- `R2_ENGINE_FINGERPRINTS.md`
- `AGENT_SAFETY_AUDITS.csv`
- `HYPOTHESIS_REGISTER.csv`
- `TOURNAMENT_PLAN.md`, `TOURNAMENT_MATCHES.csv`, per-game diagnostics, logs and `TOURNAMENT_RESULTS.md`
- `MISSION_6_REPORT.md`
- If and only if the gate passes: one candidate tarball + `candidate_manifest.json` + reproducible packaging/test script.

## 5. Non-negotiable rules

- No Kaggle upload or other irreversible action.
- Do not claim live ratings, API output, replays or engine tests not actually observed.
- Distinguish `VERIFIED`, `REPORTED`, `INFERRED` and `UNKNOWN`.
- The current local R2 candidate is a compatibility fix, not a proven strategic upgrade. Its successful smoke test establishes safety for the tested paths, not superior win rate.
- If a public notebook or engine artifact cannot be accessed, record the blocker and move on to offline tests with the included inputs.
- Preserve the user's best-known archive and never let a speculative candidate replace it.
- Final answer must lead with the observed conclusion, not a long progress narrative: what was tested, how many real BO3 matches, result and uncertainty, known failure modes, exact output hashes, and whether a new candidate should or should not be uploaded.

## Official source references

- Playground overview/rules and rating mechanics: https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description
- Public Playground notebooks listing: https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/code
- CABT simulator API docs: https://matsuoinstitute.github.io/cabt/
