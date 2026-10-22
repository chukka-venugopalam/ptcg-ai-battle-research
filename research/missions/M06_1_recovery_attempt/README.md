Mission 6.1 package. Start with MISSION_6_1_DECISION.md, then INPUT_PREFLIGHT.md (the concrete open result), then RUN_ON_YOUR_MACHINE.md.
tools/m61.py = the pipeline (preflight, audit, sdk, idprobe, safety, pairs, all). tools/ptcg_eval_canonical.py = the single BO3-match statistics module. vendor_m3/ = unmodified Mission 3 tools (VENDOR_SHA256.txt).
All synthetic outputs are labelled HARNESS VALIDATION ONLY; no real-agent result exists in this package. Tests: `python -m pytest -q tests` (engine-backed tests fail, never skip, without kaggle-environments==1.33.0).
