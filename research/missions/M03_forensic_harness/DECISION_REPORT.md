# DECISION_REPORT.md

Sandbox date 2026-10-15. Tags: **verified** / **reported** / **inferred** / **unknown**.

## Decision: **PRESERVE Bv1. Upload nothing today.**

The available evidence does **not** support submitting any candidate. No candidate passed a gate; none has even been built, because the inputs they need were not attached. No `submission.tar.gz` is provided by design (your rule: only if a candidate passes the gates).

## Why

1. **There is no candidate to defend.** *[verified]* Family A needs the Bv1 archive and ≥ 50 loss replays; B needs the exact official source, its licence and a verified 60-card list; C needs an official replay corpus. None was provided. `CANDIDATE_COMPARISON.csv` therefore has only `NOT RUN` rows.
2. **No safety evidence for anything you could upload.** *[verified]* The safety gate (cold start without `__file__`, first call returns 60 valid ids, zero illegal/exception/fallback/timeout) has never been run on Bv1, R12 or any candidate — the archives were not available.
3. **An upload probably evicts Bv1.** Only the latest two submissions stay active *[reported — reserch2 quotes official-page excerpts; news on the Simulation competition says the same; the Playground Rules page was not retrievable here]*. A new upload evicts the **older** of {Bv1_backup 813.5, R12 528.2}. Which is older is **unknown**; R12 looks like the newer experiment *[inferred from naming only]*, in which case any upload removes your best agent. Read the timestamps first (step 1 below).
4. **The 781–826 readings cannot be ranked.** *[inferred, assumed rating scale]* Episode counts and uncertainty are missing. Under the reports' own noise model (Elo-400 scale; SE ≈ 347/√n per agent) an agent with 100 episodes has SE ≈ 35 points, so 826.0 / 813.5 / 789.9 / 781.6 are mutually indistinguishable. If `submission.tar.gz` (826.0) and `submission_Bv1_backup_826.tar.gz` (813.5) are byte-identical, the 12.5-point gap is already a free A/A measurement of rating noise.
5. **Even a good candidate would be hard to confirm at the gate sizes.** *[verified: exact power]* G2 at 1,000 matches per arm has 27 % power at +3 pp, 61 % at +5 pp, 80 % at +6.3 pp (`HARNESS_AUDIT_AND_TESTS.md` §2). Small patches need head-to-head testing or ≈ 4,400 matches per arm.
6. **Waiting is cheap.** The reports give the deadline as 2027-01-01 (final submissions 2027-01-08) *[reported]*: ≈ 78 days remain. A day spent on a read-only audit costs a day of ladder episodes for a new upload; a wrong eviction costs Bv1's accrued rating.

## What I advise against

* **Report 2's "Cohort 1"** (byte-identical Bv1 re-upload + the unchanged official Dragapult sample): two uploads evict **both** active agents and restart Bv1's rating from scratch to learn what the existing 826.0 / 813.5 pair may already show. A lone Bv1 re-upload is only defensible to retire R12's slot, and it buys no new information. Do not spend two slots on an A/A pair.
* Re-uploading an R12 variant or any speculative rule patch. The R12 reading is a reason to investigate, not a forensic explanation.

## What would change this decision (falsifiers)

| Evidence | Effect |
|---|---|
| CLI shows **R12 is the older** active agent | One upload would evict R12 only; a validated candidate may then be uploaded without touching Bv1 (it must still pass G0 → G2) |
| Bv1 fails the safety gate (silent fallback, exception, wrong entrypoint, id clipped) | A safety-only fix becomes the candidate; must show a behavioural-diff and head-to-head no-regression |
| ≥ 10 % of Bv1 losses share one single-condition class and the fix changes < 1 % of 20,000 recorded decisions | Family A becomes a candidate |
| R12 audit names a concrete defect **and** a reference pilot holds ≥ 50 % field-weighted with the same deck on the current panel | R12 fix becomes a candidate |
| Official Rules say something other than "latest 2" | Re-plan slot use |
| Engine fingerprints (libcg / card-table sha256 in FORENSIC §0) differ from your official SDK | Local results do not transfer; re-run on the official SDK |

## Ordered next actions (≈ 1–2 h of your time)

1. `kaggle competitions submissions -c <slug> -v` (read-only; `tools/fetch_kaggle_evidence.sh` wraps it): ids, **order/timestamps**, scores, episode counts. This answers "what would an upload evict".
2. `sha256sum` the five archives; keep a read-only copy of `submission_Bv1_backup_826.tar.gz`.
3. Install the engine (recipe in `HARNESS_AUDIT_AND_TESTS.md` §3) and run, on Linux as root so the per-agent `/kaggle_simulations/agent` mount works:
   `python tools/submission_audit.py audit submission_Bv1_backup_826.tar.gz --matches 20 --out bv1_audit.json` (same for R12), then `python tools/submission_audit.py diff …`.
4. Collect ≥ 50 replays + logs per active agent and the latest official Episodes dataset + `manifest.csv`; run `tools/build_meta_census.py`.
5. Compare your official SDK's `libcg` and card-table hashes with FORENSIC §0.
6. Bring back `bv1_audit.json`, `r12_audit.json`, `diff.json`, a handful of replays. The loss-tagger gets written against the real replay `logs` schema, then family A can be judged against the §7 criteria.

## Candidate triage once unblocked

| Family | Verdict | Notes |
|---|---|---|
| **A** Bv1 + one forensic patch | first, **only if** the §7 criteria are met | cheapest and lowest-risk, but the effect is small; test head-to-head (≈ 2,178 BO3 for +3 pp) and use the field panel for no-regression |
| **B** public official pilot + matching deck | use as a **reference pilot / panel opponent** first | old 880.9 is Simulation-era history; needs licence/Rules clearance, the full 60-card list verified against the 1,431-card table, and the pilot run with its own deck. Do not mix its policy with another deck |
| **C** rating-filtered replay-cloned specialist | start collecting data now; build only after A/B are judged | percentile filter from **current** ratings, same-size unfiltered control arm, listwise legal-option head, checkpoints chosen by held-out games, no hidden-information inputs; PPO only if BC beats Bv1 |

Retire the R12 deck unless the evidence in the table above says otherwise.

## On 1000+

Nothing in the two reports shows a 1000+ agent was built for the current Playground; 1,398.2 is a historical Simulation result *[reported]* and no local win rate can be mapped to a live rating (the rating constants are unknown). I make no score prediction. The strongest defensible step is evidence collection, then at most one validated change in one free slot.

## Slot protocol for a future candidate

Read the live submission list immediately before uploading; name the agent that will be evicted; keep a hashed copy of Bv1; upload **one** candidate; archive root must contain `main.py` and `deck.csv` only (plus required weights), no notebooks; record sha256 of the tar and its critical members in `submission_manifest.json`; assert it is not byte-identical to Bv1 unless it is an approved A/A control.
