# MISSION_5_DECISION — Outcome **(B): input blocked**

## Result
Phases A–D could not run. VERIFIED: no submission archive, no Kaggle CLI/credentials, no kaggle.com egress, no engine (`libcg`/`kaggle-environments`). Phase E was completed and verified on synthetic data.

## Exact missing artifacts (attach or run on an authenticated machine)
1. `submission_Bv1_backup_826.tar.gz`, `submission.tar.gz`, `submission_archaludon_cleanroom_r12.tar.gz` (+ `submission_Bv2_sym8.tar.gz`, `submission_B2_counter_source_grimmsnarl.tar.gz` if available).
2. Output of `kaggle --version`, `kaggle competitions -h`, then `submissions -c <SLUG> -v`, `leaderboard`, `episodes <ID> -v` for both active submissions; the Playground slug and saved rules page.
3. SHA-256 of official `libcg`, card table, rules/config, and the SDK version/commit used for submission.
4. ≥50 completed replays + logs per active agent, and the official Episodes dataset + `manifest.csv` if permitted.

## Decision
Preserve Bv1. Upload nothing. No candidate is eligible. The single canonical stats module replaces both prior gates.

## Immediate flags
- Mission 3's 1,000-match confirm gate is underpowered (~60%) for +5pp; set N from `n_per_arm`, not by habit.
- "77 passing tests" is 55 passing + 22 engine-skipped here.
- Mission 6 will be proposed only after Mission 5 evidence exists; none is proposed now.
