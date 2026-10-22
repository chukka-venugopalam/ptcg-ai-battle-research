#!/usr/bin/env bash
# One-command Phase A/B reproduction.  usage: PY=python ./run_phase_AB.sh BUNDLE_ZIP_OR_DIR OUT_DIR [MATCHES=20]
# Needs the OFFICIAL R2 package installed into $PY first (the audit uses whatever kaggle-environments/cabt is installed;
# engine_fingerprint.json records which).  Read-only: never contacts Kaggle, never uploads.
set -u
BUNDLE="${1:?bundle zip or dir}"; OUT="${2:?out dir}"; N="${3:-20}"; PY="${PY:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$OUT/bundle" "$OUT/audits"
if [ -f "$BUNDLE" ]; then unzip -qo "$BUNDLE" -d "$OUT/bundle"; else cp -r "$BUNDLE"/. "$OUT/bundle"/; fi
# 1. hashes of every handoff file + check each against input_manifest.json (shape-agnostic: hash must appear in its text)
( cd "$OUT/bundle" && find . -type f ! -name input_manifest.json -print0 | sort -z | xargs -0 sha256sum ) > "$OUT/BUNDLE_SHA256.txt"
MAN="$(find "$OUT/bundle" -name input_manifest.json | head -1)"
while read -r H F; do
  if [ -z "$MAN" ]; then S="NO_MANIFEST"; elif grep -q "$H" "$MAN"; then S="VERIFIED_IN_MANIFEST"; else S="NOT_IN_MANIFEST"; fi
  echo "$S  $H  $F"
done < "$OUT/BUNDLE_SHA256.txt" > "$OUT/MANIFEST_CHECK.txt"; grep -c -v VERIFIED_IN_MANIFEST "$OUT/MANIFEST_CHECK.txt" | sed 's/^/files NOT verified against manifest: /'
# 2. engine identity (libcg, card table, rules/config) of the engine this python will use
"$PY" - > "$OUT/engine_fingerprint.json" <<PYEOF
import sys, json; sys.path.insert(0, "$ROOT/vendor_m3")
import bo3_ab_harness as h; print(json.dumps(h.engine_fingerprint(), indent=1))
PYEOF
# 3. per-archive audit (identity, tar safety, deck vs engine card table, cold start without __file__, N real BO3, safety gate)
: > "$OUT/audits/EXIT_CODES.txt"; ARCH=()
while IFS= read -r A; do
  B="$(basename "$A")"; ARCH+=("$A")
  "$PY" "$ROOT/vendor_m3/submission_audit.py" audit "$A" --workdir "$OUT/audits/$B.work" --matches "$N" --out "$OUT/audits/$B.json" > "$OUT/audits/$B.log" 2>&1
  echo "$B exit=$?" >> "$OUT/audits/EXIT_CODES.txt"
done < <(find "$OUT/bundle" -name '*.tar.gz' | sort)
# 4. pairwise diffs (first archive is the reference), 5. flatten to the CSV deliverables
[ "${#ARCH[@]}" -ge 2 ] && "$PY" "$ROOT/vendor_m3/submission_audit.py" diff "${ARCH[@]}" > "$OUT/ARCHIVE_DIFFS_raw.txt" 2>&1
"$PY" "$ROOT/tools/flatten_audits.py" --audits "$OUT/audits" --out "$OUT"
echo "done -> $OUT   (review EXIT_CODES.txt, MANIFEST_CHECK.txt, engine_fingerprint.json, ARCHIVE_DIFFS_raw.txt)"
