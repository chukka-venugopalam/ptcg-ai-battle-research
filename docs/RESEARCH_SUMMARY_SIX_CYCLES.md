# Six Research Cycles — Consolidated Findings

Date: 2026-10-22. This is a synthesis of the actual research packages, Mission 6.2 results, and the available agent archive evidence. It deliberately distinguishes tested facts from proposed work.

## Cycle 1 — Architecture / public research

Reviewed public approaches including behavioral cloning, high-quality action demonstrations, deck specialization, reinforcement learning, and meta-weighted evaluation. The work establishes candidate directions, not a reproduced winning model. No new model was trained from this research.

## Cycle 2 — Competition / meta / submission audit

Collected card and source inventories, experiment ideas, and the submission postmortem. Historical ratings show that previous variants have been around the high-700s to low-800s in reported snapshots, while a clean-room Archaludon/R12 variant was around 530. The exact archive-to-rating mapping is incomplete, and those are historical observations rather than current ratings.

## Cycle 3 — Forensics / harness

Added archive inspection, source diffing, engine fingerprinting and local harness plans. The initial run lacked real agent archives and included `NOT RUN` candidate rows; its fixture tests are harness checks, not strategic comparisons.

## Cycle 4 — Evaluation statistics

Added a standalone stats implementation and unit tests. Later audits highlighted the need to keep match-level and game-level evaluation units distinct, balance seats, record failures, and use predeclared stopping rules.

## Cycle 5 — Evidence/statistics audit

Mission 5 reported zero historical matches analysed and no replay-backed root cause. It strengthened the canonical evaluation and evidence protocol but did not demonstrate a new winner.

## Cycle 6 — Actual archive investigation / local A/B tests

The available Bv1, B2, Bv2 and R2-compat archives share the same learned policy weights and essentially the same deck; differences are chiefly runtime compatibility and minor flags. A fixed-size embedding-table boundary issue was reproduced with synthetic IDs and guarded in the R2 compatibility candidate. Mission 6.2 recorded 328 local tournament games plus two smoke games without reported engine errors, illegal actions or fallback events.

Results from Mission 6.2:

| Experiment | Observed outcome | Conclusion |
|---|---:|---|
| Compatibility candidate vs B2 | 21–11 (32 games) | Screening-only, not decisive |
| Compatibility candidate vs Bv1 | 17–15 (32 games) | Not decisive |
| Compatibility candidate vs Bv2 | 16–16 (32 games) | No observed difference |
| `counter_source=False` vs `True` | 57–43 (100 games) for False | Not decisive (reported p=0.193) |
| Two-copy Shadowy Darkness Energy variant vs baseline | 56–44 (100 games) for variant | Not decisive (reported p=0.271) |

The 128-game screening results and the two 100-game confirmations total 328 games. These samples screen variants; they do not establish the modest improvement needed to justify promotion. Also, most of the variants use the same weights/deck and therefore do not test a genuinely different strategy family.

## Earlier single-rule screen preserved in the records

The October 9 `poffin_force=True` candidate was reported at 102–97 over 199 decided games, which was inconclusive. Its recorded archive SHA-256 was `5dc1f5dbaefc63c0e410052363d94d0934076ba1920b63a4b23e13c6736ffb6e`, but the archive bytes and row-level result log were not available in the build workspace, so it is registered as a historical note rather than an included archive.

## What has actually improved

1. Archive provenance and variant differences are easier to audit.
2. An out-of-range embedding guard makes the tested R2 candidate more robust to card/attack IDs beyond the legacy table sizes.
3. Native-engine smoke execution and legal-action checks are reproducible.
4. The local evaluation protocol now distinguishes runtime success from playing strength.

## What is still missing for a 1500+ ambition

1. A current meta dataset tied to permitted live episode/replay data.
2. Real loss-by-loss analysis from the exact submitted archive(s).
3. A genuinely different deck-policy combination.
4. Training on high-quality current competitive decisions; no such new validated model has been produced in these six cycles.
5. Generalization testing against a diverse opponent panel.
6. A sufficiently powered, pre-registered tournament plus live validation.

The next useful work is a meta/deck/policy upgrade, not another upload of the current compatibility archive. 1500+ remains a target, not a result supported by current evidence.
