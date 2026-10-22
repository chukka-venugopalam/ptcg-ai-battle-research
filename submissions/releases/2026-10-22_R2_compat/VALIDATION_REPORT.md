# Submission Validation Report — 2026-10-22

## Decision

This is the preserved R2 compatibility candidate, copied byte-for-byte from the archive last validated on 2026-10-15. No gameplay flag, deck, or network-weight change has been promoted today because the later Mission 6.2 tests did not establish a statistically decisive improvement.

**Do not upload if the same SHA-256 is already active.** A renamed duplicate does not create a new agent or new evidence.

## Artifact identity

- File: `submission.tar.gz`
- SHA-256: `0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a`
- Size: 4,839,805 bytes
- Tar members: 39
- Top-level `main.py` / `deck.csv`: PASS
- Deck entries: 60 (unique card IDs: 19)
- `.pyc` / `__pycache__`: none
- `sa/policy_net.npz` SHA-256: `3c79fb1a85bf27f5a931ce459914dbc2d1dae402763cebe9ce39d3e6ca29f1fd`
- `cg/libcg.so` SHA-256: `c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22` (matches the native library in the available official R2 package)

## Tests actually run today

1. **Structural validation:** `python tools/validate_submission.py submissions/current/submission.tar.gz` passed. Python files compiled for syntax; root paths were correct, exactly 60 deck IDs were found, and no Python cache artifacts were present.
2. **Exec-style startup:** `tools/smoke_test_submission.py` loaded `main.py` through `exec()` without providing `__file__`. The deck handshake returned 60 card IDs and the R2 native engine accepted the decks.
3. **Native-engine smoke:** one self-play game completed in 158 action selections in 4.070 seconds. Seat 0 took 66 actions and seat 1 took 92; both agent processes reported zero fallback calls, zero missing-network calls, and no exception. The structured output is preserved in `SMOKE_TEST_OUTPUT.json`. The smoke-runner checked each selected action's type, cardinality and option-index range. Earlier independent smoke runs of this same archive completed in 181, 182 and 239 selections, also without reported fallbacks/missing-network calls.
4. **Out-of-range embedding checks:** card IDs 1300 and 1431 and attack ID 1755 safely mapped to row 0 for this legacy identity-mapped checkpoint; in-range IDs 1299 and 1599 remained identity-mapped. The guard behavior is confirmed, but this does not give the model learned semantics for unseen cards.
5. **Strategic performance:** NOT ESTABLISHED. Mission 6.2 screening did not prove any variant superior. Self-play smoke tests are not win-rate tests.

## Practical recommendation

Keep this archive as the safe reference baseline. Continue with current-meta analysis and a materially different deck/policy candidate. Run a pre-registered balanced tournament before promotion. Record the exact hash and active submission before any Kaggle upload.
