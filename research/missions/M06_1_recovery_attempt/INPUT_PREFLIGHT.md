# INPUT_PREFLIGHT — Mission 6.1 (2026-10-18)

## Concrete result: the attached recovery bundle is not in the execution sandbox. This is an observed open error, not an assumption.
| Path I tried to open | Result |
|---|---|
| `/mnt/user-data/uploads/PTCG_MISSION_6_INPUTS_BUNDLE.zip` | `ls: cannot access ...: No such file or directory` |
| `/mnt/user-data/uploads/MISSION_6_OUTPUT.zip` | `ls: cannot access ...: No such file or directory` |
| any `PTCG_MISSION_6_INPUTS_BUNDLE*`, `input_manifest.json`, `official_playground*`, `latest_candidate*`, `Bv1_baseline*`, `B2_counter*`, `Bv2_sym8*` anywhere on the filesystem (outside my own work dirs) | no match |
| any `*.zip *.tar.gz *.tgz *.tar` anywhere else (outside venv/proc/usr/tmp/my work dirs) | none |
| files >3 MB modified in the last 24 h | none |

`/mnt/user-data/uploads` contains exactly six files, all pre-existing from Oct 15/16 (directory mtime 2026-10-18 04:17 UTC = the start of this turn, so the directory was refreshed but gained nothing): `PTCG_MISSION_5_EVIDENCE_AUDIT.md` and `_1.md` (identical), `pokemon_mission_3.zip` and `_1.zip` (identical, sha256 `aeaf7281…0417`), `pokemon_mission_4.zip` and `_1.zip` (identical, sha256 `697d7b98…57ef`). No `<uploaded_files>` entry for a recovery bundle was delivered with this turn either. The only Mission 6 zip in the sandbox is my own earlier output (`/mnt/user-data/outputs/MISSION_6_OUTPUT.zip`, sha256 `928263f4…ec8a`), which is not your wrapper. A guess at the cause (UNVERIFIED): the upload failed or was dropped before reaching the sandbox, probably on size; I cannot see the reason.

## Preflight tool output on the real uploads directory (`python tools/m61.py preflight --root /mnt/user-data/uploads`)
# INPUT_PREFLIGHT
Root searched: `/mnt/user-data/uploads`
- **MISSING PATH (exact): `/mnt/user-data/uploads/**/PTCG_MISSION_6_INPUTS_BUNDLE.zip`** (recursive search, no match)
- **MISSING PATH (exact): `/mnt/user-data/uploads/**/MISSION_6_OUTPUT.zip`** (recursive search, no match)

## Result
Inner bundle not present: every agent/engine/manifest step is BLOCKED at this path. Nothing below was inferred.

## Consequence
Manifest verification, the five SHA-256 checks from the brief, member lists, safe extraction, diffs, SDK comparison, deck/ID/cold-start audits and all real-agent matches are BLOCKED at the single missing path above. Per the brief, only those steps stopped; everything that does not need your bytes was completed and is listed in `HARNESS_VALIDATION_ONLY.md`.

## Review leads from the brief — all UNVERIFIED until the bytes arrive
| Lead (stated in the brief, not verified here) | How the pipeline will verify it |
|---|---|
| the four archives share `deck.csv` and `sa/policy_net.npz` hashes | `ARCHIVE_FORENSICS.csv` columns `deck_sha`, `npz_arrays`; per-array hashes in `ARCHIVE_DIFFS.md` |
| candidate vs Bv1 differs in 7 paths (`cg/cg.dll`, `cg/game.py`, `cg/libcg-arm64.so`, `cg/libcg.dylib`, `cg/libcg.so`, `cg/sim.py`, `sa/policynet.py`) | "Changed" table + source diffs in `ARCHIVE_DIFFS.md` |
| B2 changes `AGENT_KWARGS['counter_source']` False → True | exact source-line diff of runtime flags in `ARCHIVE_DIFFS.md` |
| Bv2 adds `sym_k: 8`, ships R2 SDK files, lacks the candidate's `policynet.py` guard | same, plus the "which SDK files each agent ships" table in `SDK_FINGERPRINT_COMPARISON.md` |
| official R2 `cg/libcg.so` = `c699020e…de22` = PyPI 1.33.0's | hash of the supplied file; wrapper files `game.py`/`sim.py`/`cabt.py` get exact diffs, no claim of whole-SDK identity |
| ID boundaries (cards 1299/1300/1431, attacks 1599/1600/1755) | `ID_BOUNDARY_PROBE.md` (black box: exceptions/illegal output only; clipping or modulo wrapping is invisible to it, so `sa/policynet.py` and `sa/features.py` must still be read) |

## Raw evidence
```
# raw evidence 2026-10-18T04:28:45Z
## ls -la /mnt/user-data/uploads
total 268
drwxr-xr-x 1 claude ubuntu      0 2026-10-18 04:17:20.733317504 +0000 .
drwxr-xr-x 5 root   root     4096 2026-10-16 02:54:31.152261823 +0000 ..
-rw-r--r-- 1 claude ubuntu  12335 2026-10-15 12:06:14.828933000 +0000 PTCG_MISSION_5_EVIDENCE_AUDIT.md
-rw-r--r-- 1 claude ubuntu  12335 2026-10-16 02:54:13.685583000 +0000 PTCG_MISSION_5_EVIDENCE_AUDIT_1.md
-rw-r--r-- 1 claude ubuntu 110272 2026-10-15 12:06:15.297327000 +0000 pokemon_mission_3.zip
-rw-r--r-- 1 claude ubuntu 110272 2026-10-16 02:54:17.335239000 +0000 pokemon_mission_3_1.zip
-rw-r--r-- 1 claude ubuntu   7105 2026-10-15 12:06:14.983043000 +0000 pokemon_mission_4.zip
-rw-r--r-- 1 claude ubuntu   7105 2026-10-16 02:54:15.891416000 +0000 pokemon_mission_4_1.zip
## exact expected paths
ls: cannot access '/mnt/user-data/uploads/PTCG_MISSION_6_INPUTS_BUNDLE.zip': No such file or directory
ls: cannot access '/mnt/user-data/uploads/MISSION_6_OUTPUT.zip': No such file or directory
## sha256 of every file present
63544fa6681fbf11fd560d676b117c8f1d48306ab4574fdb044d9c394d4abb8f  /mnt/user-data/uploads/PTCG_MISSION_5_EVIDENCE_AUDIT.md
63544fa6681fbf11fd560d676b117c8f1d48306ab4574fdb044d9c394d4abb8f  /mnt/user-data/uploads/PTCG_MISSION_5_EVIDENCE_AUDIT_1.md
aeaf7281d83f5c0017bf89709252e6c86f0754306d9d588f2ff7d0972ea60417  /mnt/user-data/uploads/pokemon_mission_3.zip
aeaf7281d83f5c0017bf89709252e6c86f0754306d9d588f2ff7d0972ea60417  /mnt/user-data/uploads/pokemon_mission_3_1.zip
697d7b9897d99d6cdc3b3ec005b6620547dc8ad507f240a060fee084e5a157ef  /mnt/user-data/uploads/pokemon_mission_4.zip
697d7b9897d99d6cdc3b3ec005b6620547dc8ad507f240a060fee084e5a157ef  /mnt/user-data/uploads/pokemon_mission_4_1.zip
## find / -iname (zip|tar.gz|tgz|tar) outside venv/proc/usr/tmp/my work dirs
(none other than the above)
## find by name
(end)
## files >3MB modified in last 24h outside venv/usr/var/proc
(end)
```
