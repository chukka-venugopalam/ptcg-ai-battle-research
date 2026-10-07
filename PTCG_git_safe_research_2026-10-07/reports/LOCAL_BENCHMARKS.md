# Local benchmarks — 2026-10-07

## Current known ladder anchors
- Exact existing `submission.tar.gz`: 826.0 on the user's Playground submission page.
- Existing `submission_B2_counter_source_grimmsnarl.tar.gz`: 798.6.
- Existing `submission_Bv2_sym8.tar.gz`: 788.4.

## New local tests
### Archaludon clean-room R12 vs Bv1
- 200 games total, alternating seat ownership.
- Bv1: 100 wins.
- Archaludon R12: 100 wins.
- Invalid/timeout: 0.
- Both agents stayed legal; Bv1 reported `fallbacks=0` and `net_missing=0` during the run.
- Interpretation: non-inferior on this single local instrument, but NOT evidence of >1000 ladder strength.

### Bv1 + Poffin-force probe vs Bv1
- 199 decided games out of 200 (1 invalid during the final run).
- Poffin probe: 102 wins.
- Bv1: 97 wins.
- Estimated probe WR: 51.3% on decided games.
- The interval includes 50%; this is a weak local canary, not a promotion-quality result.

## Decision
Primary live pair recommendation:
1. `submission_archaludon_cleanroom_r12.tar.gz` — high-upside specialist candidate.
2. `submission_Bv1_backup_826.tar.gz` — rollback / stability anchor.

Hold `submission_Bv1_poffin_probe.tar.gz` unless both primary candidates disappoint.
