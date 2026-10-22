# Run Mission 6.1 without attaching anything big
Three attachments have now failed to arrive, so the lowest-friction path is to run the audit where the files already are and send back the small reports.
```bash
# Linux x86_64 or WSL (libcg.so is the Linux library); Python 3.12 was used for all validation here
python3 -m venv venv && . venv/bin/activate
pip install kaggle-environments==1.33.0 numpy          # supplies the engine the tools import; the pipeline overlays the official R2 files on top
python tools/m61.py all --root PTCG_MISSION_6_INPUTS_BUNDLE.zip --out out61 --matches 20
#   --root accepts the inner bundle zip, the outer recovery zip, or an already-extracted directory.  Try --matches 4 first for a quick check.
```
Stages (each stops only itself, results in `out61/RUN_LOG.txt`): preflight → audit → sdk (builds `out61/r2_overlay`) → safety (submission_audit: tar safety, deck vs R2 card table, cold start without `__file__`, N real BO3) → idprobe → pairs (candidate vs Bv1, B2, Bv2, both seats, one row per BO3 match).
**Send back** `out61/*.md`, `out61/*.csv`, `out61/*.json`, `RUN_LOG.txt` (small text; not the extracted agents). Run `python -m pytest -q tests` first if you want to confirm the tools work on your machine (engine-backed tests fail, never skip, without the engine).

## If you would still rather attach to the chat
```bash
ls -l inputs/agents/*.tar.gz inputs/engine/*.zip                       # note any file over the upload limit
sha256sum PTCG_MISSION_6_INPUTS_BUNDLE.zip inputs/agents/* inputs/engine/*   # paste this TEXT into the chat: it lets me verify what arrives
split -b 20M -d inputs/agents/Bv1_baseline_local_archive.tar.gz Bv1.part_      # for any single file that is too large; attach all parts, I `cat` and verify the sha256
# the official R2 package, text files only (enough to build the overlay; PyPI natives are kept, native hash equality is checked from your sha256sum output)
unzip -o inputs/engine/official_playground_bundle_R2.zip -d r2 && (cd r2 && find . -type f \( -name '*.py' -o -name '*.json' \) | zip -@ ../r2_text_files.zip && sha256sum $(find . -type f) > ../r2_all_files.sha256)
```
Attach the four tarballs as separate files rather than one nested zip; they must stay byte-exact (hashes are checked against the brief).
