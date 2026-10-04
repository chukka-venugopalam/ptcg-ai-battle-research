#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/submission"
OUT="$ROOT/submission.tar.gz"
MAX_BYTES=207093043   # 197.7 MiB in decimal-ish approximation; competition page is authoritative.

cd "$SUB"
python - <<'PY'
from pathlib import Path
import tarfile

required = ['main.py', 'deck.csv']
for name in required:
    p = Path(name)
    if not p.exists():
        raise SystemExit(f'Missing required submission file: {name}')

lines = [x.strip() for x in Path('deck.csv').read_text().splitlines() if x.strip()]
if len(lines) != 60:
    raise SystemExit(f'deck.csv must contain exactly 60 card IDs; found {len(lines)}')
for x in lines:
    int(x)

with tarfile.open('../submission.tar.gz', 'w:gz') as tf:
    for p in Path('.').rglob('*'):
        if p.is_file():
            tf.add(p, arcname=str(p))

with tarfile.open('../submission.tar.gz', 'r:gz') as tf:
    names = tf.getnames()
    print('Archive members:')
    for n in names:
        print(' ', n)
    assert 'main.py' in names and 'deck.csv' in names

print('submission.tar.gz created')
PY

SIZE=$(stat -c%s "$OUT")
echo "submission.tar.gz size: $SIZE bytes"
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
  echo "WARNING: archive exceeds the local guard threshold; official Kaggle limit is authoritative."
  exit 2
fi
