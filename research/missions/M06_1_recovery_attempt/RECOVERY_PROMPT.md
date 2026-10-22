# MISSION 6.1 — Recover the Handoff and Test the Actual Agents

## Goal
Finish the agent-level work that Mission 6 could not perform. Do not make another gameplay change or upload a submission unless a real-agent head-to-head test supports it. The target is better competitive performance; a 1000+ rating is an aspiration, not a promise.

## Mandatory preflight — do this before interpreting Mission 6
This recovery bundle contains TWO nested archives:
- `PTCG_MISSION_6_INPUTS_BUNDLE.zip` — the actual input bundle, agents, official R2 package and hash manifest.
- `MISSION_6_OUTPUT.zip` — Mission 6's report, which says those inputs did not reach its execution sandbox.

Extract both, inspect the complete member lists, and actually open `PTCG_MISSION_6_INPUTS_BUNDLE.zip`. Do not repeat “archives not received” unless you show a concrete open/extraction error against this attached recovery bundle. If any path is missing, print the exact missing path and stop only that step; continue all other available static audits.

Verify `input_manifest.json` against every file it lists. At minimum verify these agent archives:
- `inputs/agents/latest_candidate_R2_compat.tar.gz` — SHA-256 `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`
- `inputs/agents/Bv1_baseline_local_archive.tar.gz` — SHA-256 `7ea11c6181f1b68ff23a97311998901b66b3ab5e7bcd1371e3d46b7d0c9d4feb`
- `inputs/agents/B2_counter_source_grimmsnarl.tar.gz` — SHA-256 `b61ed901d778ac23647dc2be97a13770e632c2e62b7a6d289508ddf6910baea8`
- `inputs/agents/Bv2_sym8_ready.tar.gz` — SHA-256 `169e1ac85e9edba28ccf19ff99e244c2016712096e7cdb0aab69cf9e0574318c`
- `inputs/engine/official_playground_bundle_R2.zip` — SHA-256 `6fe557e77ca21b86dea9056d7f4531cbeb328abc1b25aecf9c3a4e2b7d5b2c87`

The official R2 package's `cg/libcg.so` hash is `c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22`, which matches the native library hash recorded by Mission 6 for PyPI `kaggle-environments==1.33.0`. This establishes native-library hash identity only: Mission 6 reported different hashes for Python wrapper files `cg/game.py` and `cg/sim.py`. Compare their exact source diffs; do not claim the entire SDKs are identical without evidence.

## Known static facts to verify against extracted bytes
These are review leads, not strength claims:
- The four archives have matching `deck.csv` and `sa/policy_net.npz` hashes in the input bundle.
- Candidate R2 compatibility vs Bv1 differs in seven paths: `cg/cg.dll`, `cg/game.py`, `cg/libcg-arm64.so`, `cg/libcg.dylib`, `cg/libcg.so`, `cg/sim.py`, and `sa/policynet.py`.
- B2 changes `AGENT_KWARGS['counter_source']` from `False` to `True` relative to Bv1.
- Bv2 adds `sym_k: 8` relative to Bv1 and uses the current R2 SDK files, but does not contain the candidate's `sa/policynet.py` guard.
- The mission archive does not include the exact R12 tarball or official ladder episode/replay evidence. Do not reconstruct either or map a local archive to a Kaggle rating from its filename alone.

## Execution plan
### Phase A — actual byte-level audit
1. Verify the manifest and SHA-256 values first.
2. Safely list and extract the four tarballs; reject unsafe paths, duplicate members, unexpected executables, and malformed archives.
3. Produce a source-level file diff for candidate-vs-Bv1, B2-vs-Bv1 and Bv2-vs-Bv1. Record exact source lines for runtime flags and every SDK/runtime difference.
4. Inspect `sa/policynet.py`, `sa/features.py`, all embedding array dimensions, ID maps, and option/target/bag/slot indexing. Test boundary IDs including 1299, 1300, 1431 for cards and 1599, 1600, 1755 for attacks as applicable to each table. Distinguish valid remapped IDs from raw IDs. A finite-output synthetic test is a safety check, not a strength result.
5. Compare official R2 and local engine fingerprints, including native `libcg` and Python wrapper files. Mark every mismatch and its likely behavioral relevance.
6. Test deck legality against the official R2 card table; cold-start every agent under the intended SDK exec mode (especially with `__file__` absent); record illegal choices, exceptions, fallbacks, timeouts and latency.

### Phase B — true agent tests, not fixture tests
1. If the official R2-compatible environment can be run, make at least 20 complete BO3 matches for each safety comparison, balanced by agent seat. Run candidate against Bv1, B2 and Bv2.
2. Export one row per BO3 match and separate per-game diagnostics. Count a BO3 match—not a game—as the independent observation. Cross-check match results against the actual engine reward. Count candidate-caused invalid actions/exceptions/timeouts as losses and flag them.
3. If the environment is missing or is not sufficiently equivalent to R2, do not fabricate results. Still complete Phase A and give exact dependency/environment blockers and runnable commands.
4. If Phase B suggests a promising one-change candidate, pre-register a Stage 1 screen of 200 matches per arm/pairing with both seats. Stage 1 is regression screening only. Do not call a candidate stronger based on fixture matches, safety smoke tests, or an underpowered screen.
5. Only proceed to the 1,600-match-per-arm confirmation in the Mission 6 tournament plan if the compute budget and environment permit it. Report confidence intervals, raw counts, seat-stratified diagnostics, failures, exact archive hashes and all exclusions.

## Required output files
- `INPUT_PREFLIGHT.md`: member presence, all manifest/hash results, plus any access problem.
- `ARCHIVE_DIFFS.md`: exact file lists and source diffs.
- `ARCHIVE_FORENSICS.csv`: archive hash, safety findings, deck legality, model/embedding dimensions.
- `SDK_FINGERPRINT_COMPARISON.md`: official R2 vs local engine hashes and semantic differences.
- `AGENT_SAFETY_AUDITS.csv`: actual per-archive safety metrics.
- `TOURNAMENT_MATCHES.csv` and `TOURNAMENT_GAMES.csv`: populated only with actual agent games; leave header-only if no real-agent tournament ran.
- `MISSION_6_1_DECISION.md`: choose (A) a candidate supported by the tests, (B) preserve baseline/no upload, or (C) blocked by a precisely named environment issue.

## Non-negotiable rules
- Keep fixture-agent outputs in a separate section and label them **HARNESS VALIDATION ONLY**.
- Never present fixture scores as evidence about our policy's strength.
- Never state that the rating decline is explained unless a controlled test supports that conclusion.
- Do not alter weights, deck, strategy flags or the baseline during forensic comparison.
- Preserve all source archives; every candidate must be a separate archive with an exact SHA-256.
- Do not upload or invoke live Kaggle submission actions. No upload is authorized by this mission.
- Do not stop after a generic “inputs unavailable” statement: inspect the attached recovery bundle first and report the concrete result.
