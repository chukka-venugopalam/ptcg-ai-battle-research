# INPUT_PREFLIGHT
Root searched: `/mnt/data/PTCG_MISSION_6_INPUTS_BUNDLE.zip`
- wrapper archive: 22315753 bytes sha256 `a9d063e95803a4eb3e9e92f8fbd3e80204fcef0519c65f5161e22cdc162f9082`, 16 members, problems: none
- **MISSING PATH (exact): `/mnt/data/m61_actual_out/_root/**/PTCG_MISSION_6_INPUTS_BUNDLE.zip`** (recursive search, no match)
- **MISSING PATH (exact): `/mnt/data/m61_actual_out/_root/**/MISSION_6_OUTPUT.zip`** (recursive search, no match)
- inputs bundle: 22315753 bytes sha256 `a9d063e95803a4eb3e9e92f8fbd3e80204fcef0519c65f5161e22cdc162f9082`, 16 members, problems: none, findings: 0

## Manifest `input_manifest.json`: 2 (path, sha256) entries parsed
| manifest path | status | expected sha256 | actual sha256 |
|---|---|---|---|
| latest_candidate_R2_compat.tar.gz | MISSING FILE | 0f91f896ef703761 | |
| Bv1_baseline_local_archive.tar.gz | MISSING FILE | 7ea11c6181f1b68f | |

Files in the bundle but not in the manifest: 15
- README.txt
- MISSION_6_EXECUTE_AND_WIN.md
- inputs/mission5/MISSION_5_OUTPUT.zip
- inputs/mission5/PTCG_MISSION_5_EVIDENCE_AUDIT.md
- inputs/research/research_package_1.zip
- inputs/research/research_package_2.zip
- inputs/research/pokemon_mission_4.zip
- inputs/research/pokemon_mission_3.zip
- inputs/scores/submissions_screenshot_2026-10-16.png
- inputs/agents/Bv1_baseline_local_archive.tar.gz
- inputs/agents/latest_candidate_R2_compat.tar.gz
- inputs/agents/B2_counter_source_grimmsnarl.tar.gz
- inputs/agents/Bv2_sym8_ready.tar.gz
- inputs/validation/R2_COMPAT_VALIDATION_2026-10-15.md
- inputs/engine/official_playground_bundle_R2.zip

## Hashes required by the Mission 6.1 brief
| path | status | expected | actual |
|---|---|---|---|
| inputs/agents/latest_candidate_R2_compat.tar.gz | OK | 0f91f896ef703761 | 0f91f896ef703761 |
|  ↳ extraction candidate | extracted | problems: none; findings: none | |
| inputs/agents/Bv1_baseline_local_archive.tar.gz | OK | 7ea11c6181f1b68f | 7ea11c6181f1b68f |
|  ↳ extraction Bv1 | extracted | problems: none; findings: none | |
| inputs/agents/B2_counter_source_grimmsnarl.tar.gz | OK | b61ed901d778ac23 | b61ed901d778ac23 |
|  ↳ extraction B2 | extracted | problems: none; findings: none | |
| inputs/agents/Bv2_sym8_ready.tar.gz | OK | 169e1ac85e9edba2 | 169e1ac85e9edba2 |
|  ↳ extraction Bv2 | extracted | problems: none; findings: none | |
| inputs/engine/official_playground_bundle_R2.zip | OK | 6fe557e77ca21b86 | 6fe557e77ca21b86 |
