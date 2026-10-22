# Mission 6.1 Output Review — 18 October 2026

## Decision

**Mission 6.1's own run is correctly labelled blocked; it did not run a real-agent tournament.** The reason it reports is that its separate runner could not see the recovery/input ZIPs in that runner's `/mnt/user-data/uploads` path. That is an execution-environment/file-handoff failure, not evidence that the files never existed.

I independently checked the input bundle available in the current workspace, then performed a narrow, direct native-engine smoke test. This recovers meaningful compatibility evidence, but it still does **not** establish which strategy wins more BO3 matches or explain the live ladder change causally.

## What was independently verified

- Input bundle SHA-256: `a9d063e95803a4eb3e9e92f8fbd3e80204fcef0519c65f5161e22cdc162f9082`.
- The candidate, Bv1, B2, Bv2 and official R2 SDK archives all match the five expected SHA-256 values in the Mission 6.1 tool.
- All four agents have the same 60-card deck multiset and the same `sa/policy_net.npz` arrays. The learned embedding tables have 1,300 card rows and 1,600 attack rows.
- Exact source changes vs Bv1:
  - **R2 compatibility candidate:** R2 `cg/*` files plus the `sa/policynet.py` embedding-row guard.
  - **B2:** only the `main.py` runtime flag changes `counter_source` from `False` to `True`.
  - **Bv2:** R2 `cg/*` files plus `sym_k: 8`; it does not include the candidate's `policynet.py` guard.
- R2 native library `cg/libcg.so` SHA-256: `c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22`.
- After calling the native library's required `GameInitialize()`, `AllCard()` returns 1,431 cards (IDs 1–1431), card-table SHA-256 `7fd6c45fcdc2913a1fe19ddb67c01aa9b204718cb760be7582cf4fcc3513de9c`, and `AllAttack()` returns 1,755 attacks (IDs 1–1755).
- One direct game on the current R2 native engine accepted the supplied deck and completed 223 selections. Both Bv1 and the R2 compatibility candidate returned legal actions, with zero fallback and zero missing-network events in that game. It was Bv1 as seat 0 vs candidate as seat 1. **This is a single-game smoke test only, not an A/B win-rate result.**

## Controlled embedding-boundary probe

I captured real observations during that native-engine game. The observed skill-card selection auto-selected all options and bypassed neural scoring, so I changed its synthetic test bounds to a one-of selection to exercise the neural forward path. I then changed the option card IDs to 1300 and 1431. These are deliberately mutated test observations, not claims that those exact observations occurred naturally in the game.

| Agent archive | Card ID 1300 | Card ID 1431 | Result |
|---|---:|---:|---|
| R2 compatibility candidate | 0 fallbacks | 0 fallbacks | Legal action; model path completed |
| Bv1 | 1 fallback | 1 fallback | `IndexError` at `card_emb[1300]` |
| B2 | 1 fallback | 1 fallback | `IndexError` at `card_emb[1300]` |
| Bv2 | 1 fallback | 1 fallback | `IndexError` at `card_emb[1300]` |

The attack-ID variants 1600 and 1755 did not trigger the same fault in this test because `sa/optfeat.py` already maps attack IDs greater than or equal to 1,600 to row 0. The demonstrated gap is the **out-of-range option-card embedding lookup**. This probe does not prove that any live match actually exposed one of these new card IDs to the old agents, nor does it quantify the win-rate cost of falling back.

## What Mission 6.1 did not establish

1. No real BO3 head-to-heads were run. `TOURNAMENT_MATCHES.csv` and `TOURNAMENT_GAMES.csv` are header-only.
2. No direct R2 tournament comparison among candidate, Bv1, B2 and Bv2 exists in this output.
3. The displayed Kaggle scores are historical snapshots; the exact downloaded submission bytes were not matched against the live rows. They cannot isolate a code change's effect.
4. The reported 47 passing tests are test-suite/harness checks. They are not 47 agent-vs-agent matches.

## Tooling issues found in Mission 6.1

- The report's `manifest_pairs()` parser does not recognize the actual manifest key `path_in_bundle`; its first manifest table consequently says valid files are missing and claims most packaged files are not in the manifest. The dedicated five-hash check later in preflight does correctly verify all five key archives. Treat the first table as a parser defect, not evidence of corrupt files.
- The preflight path logic prints missing sibling ZIP warnings when passed the inner input bundle directly. It still extracts and checks the embedded agent/engine files, so those two warnings are misleading in this invocation.
- `tools/m61.py _tables` calls `AllCard()` without first calling `GameInitialize()`. On the supplied R2 library this returns an empty table and the helper errors at `min(ids)`. The native table fingerprint above was computed with initialization performed first. Fix this helper before trusting its automatic SDK table-fingerprint stage.
- The SDK/tournament pipeline requires `kaggle-environments==1.33.0`; it is not installed in this runtime, and package installation could not reach PyPI due unavailable network access. Therefore no official-interpreter BO3 tournament is reported here.

## Recommendation

**Do not upload based on Mission 6.1 alone.** The concrete result is that the R2 embedding guard prevents a tested silent-fallback path. It is a reliability fix, not a demonstrated strength increase.

The most informative next local experiment is an R2-compatible pair:

- Control: current R2 compatibility candidate (`counter_source=False`).
- Treatment: same R2 compatibility candidate and weights, but `counter_source=True` (the one policy flag that distinguishes B2 from Bv1).

Run paired BO3s against each other and against Bv1/B2 with both agent seats, record match-level outcomes and faults, and use a fixed-N confirmation after any screening phase. Do not infer success from one game or from historical Elo snapshots. The exact R12 agent and official live replays remain unavailable in the provided inputs.
