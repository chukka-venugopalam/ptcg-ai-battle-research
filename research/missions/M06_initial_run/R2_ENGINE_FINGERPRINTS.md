# R2_ENGINE_FINGERPRINTS — Phase B
## Engine used here (VERIFIED): public PyPI `kaggle-environments==1.33.0`, `envs/cabt`, Python 3.12.3, installed 2026-10-17
```
0945d0abd0d57f8a8ae0b6d9f188efb2d9c36ea9c5d8afa2b495eb7502abd2c4  cabt.py
e5f4afd03bfb02d6a2c1fc6704b3214c456d663ab3b1e1e9489c46e7527da96c  cabt.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  cg/__init__.py
eeb7f9c18a21710087a875bdc4e6ae1fbffe83e46d8b8fe571876ed8f45ccd41  cg/game.py
d39837f069cc7fee54d5f5bd8d10e80bf977155aea32dbf6be5e1430f44fb1e3  cg/sim.py
c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22  cg/libcg.so
95628f0bd20ac5e6f1e142c88cd52c9dc7361f9c371b5599f63b27c110892545  cg/libcg-arm64.so
da53cb331277cc8aa0a6181da28afeea28ae2f205372c234eca4a5c9cf3f4672  cg/libcg.dylib
e084185a17fa1487ee5ffec5cff828d120278f03547838031e2d70bb2a8811fe  cg/cg.dll
```
Harness fingerprint (`raw/engine_fingerprint.json`): libcg_sha256 `c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22`; card table (native `AllCard` raw JSON) sha256 `7fd6c45fcdc2913a1fe19ddb67c01aa9b204718cb760be7582cf4fcc3513de9c`; 1,431 cards, ids 1–1431 contiguous; 1,755 distinct attack ids 1–1755; 29 ACE SPEC cards; 679 Basic Pokémon; `cabt.json` bo default 3, actTimeout 0, runTimeout 2000, overage 600 s.
Native exports include `SearchBegin/SearchStep/SearchEnd/SearchRelease/AllAttack/AgentStart`.

## Equivalence results
| Comparison | Result | Tag |
|---|---|---|
| This sandbox vs Mission 3's environment | libcg `c699020e…de22` and card table `7fd6c45f…de9c` are **identical** to the hashes Mission 3 recorded in FORENSIC_FINDINGS.md (asserted in code) | VERIFIED (hash match; M3's values are as REPORTED in its file) |
| This engine vs the official R2 package / live ladder | **UNKNOWN** — the R2 package was not received | UNKNOWN |
| Docs vs installed package | the official docs (matsuoinstitute.github.io/cabt, v0.1.0, accessed 2026-10-17) list `api`, `game`, `sim`, `utils` modules with search/card/attack helpers; installed `cg/` contains only `game.py`, `sim.py` | INFERRED: the documented SDK is packaged differently from (possibly newer than) PyPI 1.33.0 |
Matching card counts or versions is not equivalence. To close this: run `repro/run_phase_AB.sh` with the official R2 package installed in the interpreter (it writes `engine_fingerprint.json`) and compare libcg/card-table/config hashes with the table above. Any difference means every local result here is non-portable until re-run.
