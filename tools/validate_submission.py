#!/usr/bin/env python3
"""Validate a PTCG submission tar.gz structurally; does not assess playing strength."""
from __future__ import annotations
import argparse, hashlib, io, pathlib, tarfile, sys

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('archive', type=pathlib.Path)
    ap.add_argument('--max-bytes', type=int, default=197_700_000)
    args=ap.parse_args()
    p=args.archive
    if not p.is_file(): print(f'ERROR: file not found: {p}',file=sys.stderr); return 2
    raw=p.read_bytes()
    errors=[]
    try:
        tf=tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')
    except Exception as e:
        print(f'ERROR: invalid gzip/tar: {e}',file=sys.stderr); return 2
    with tf:
        members=tf.getmembers(); names=[m.name for m in members]
        if 'main.py' not in names: errors.append('main.py is not at archive root')
        if 'deck.csv' not in names: errors.append('deck.csv is not at archive root')
        for m in members:
            pp=pathlib.PurePosixPath(m.name)
            if pp.is_absolute() or '..' in pp.parts: errors.append(f'unsafe path: {m.name}')
            if '__pycache__' in pp.parts or pp.name.endswith('.pyc'): errors.append(f'cache artifact: {m.name}')
        try: deck=[int(x) for x in tf.extractfile('deck.csv').read().decode('utf-8-sig').split()]
        except Exception as e: deck=[]; errors.append(f'deck.csv parse failed: {e}')
        if len(deck)!=60: errors.append(f'deck must contain 60 IDs; got {len(deck)}')
        # Compile top-level Python sources without executing imports/native code.
        for m in members:
            if m.isfile() and m.name.endswith('.py'):
                try:
                    b=tf.extractfile(m).read(); compile(b,m.name,'exec')
                except Exception as e: errors.append(f'python syntax failed for {m.name}: {e}')
        unpacked=sum(m.size for m in members if m.isfile())
        if len(raw)>args.max_bytes: errors.append(f'compressed archive exceeds configured limit {args.max_bytes} bytes')
    print(f'archive={p}')
    print(f'sha256={sha256(raw)}')
    print(f'compressed_bytes={len(raw)}')
    print(f'members={len(members)}')
    print(f'deck_cards={len(deck)} unique_card_ids={len(set(deck)) if deck else 0}')
    print(f'uncompressed_file_bytes={unpacked}')
    if errors:
        for e in errors: print('FAIL:',e)
        return 1
    print('STRUCTURAL_VALIDATION=PASS')
    print('NOTE: structural validation does not establish win rate or strategic quality.')
    return 0
if __name__=='__main__': raise SystemExit(main())
