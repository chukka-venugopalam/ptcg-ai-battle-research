#!/usr/bin/env python3
"""Validate, copy and register a candidate archive without overwriting past releases."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, pathlib, re, shutil, tarfile, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def h(data): return hashlib.sha256(data).hexdigest()
def slug(s): return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_')
def inspect(p):
 raw=p.read_bytes(); tf=tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')
 with tf:
  names=tf.getnames()
  for root in ['main.py','deck.csv']:
   if root not in names: raise ValueError(f'{root} must be at archive root')
  if any('__pycache__' in n.split('/') or n.endswith('.pyc') for n in names): raise ValueError('archive contains Python cache artifacts')
  deck=[int(x) for x in tf.extractfile('deck.csv').read().decode('utf-8-sig').split()]
  if len(deck)!=60: raise ValueError(f'expected 60 cards, got {len(deck)}')
  model=h(tf.extractfile('sa/policy_net.npz').read()) if 'sa/policy_net.npz' in names else ''
  deck_hash=h(tf.extractfile('deck.csv').read())
 return {'sha256':h(raw),'size_bytes':len(raw),'deck_cards':len(deck),'unique_card_ids':len(set(deck)),'model_weights_sha256':model,'deck_sha256':deck_hash,'members':len(names)}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--archive',required=True,type=pathlib.Path); ap.add_argument('--date',required=True)
 ap.add_argument('--name',required=True); ap.add_argument('--notes',default='')
 ap.add_argument('--score',default=''); ap.add_argument('--score-source',default=''); ap.add_argument('--allow-same-hash',action='store_true')
 args=ap.parse_args()
 if not args.archive.is_file(): print('ERROR: archive not found',file=sys.stderr);return 2
 try: meta=inspect(args.archive)
 except Exception as e: print(f'ERROR: invalid submission: {e}',file=sys.stderr);return 2
 csvp=ROOT/'submissions'/'registry.csv'
 if csvp.exists():
  with csvp.open(newline='',encoding='utf-8') as f: existing=list(csv.DictReader(f))
  dup=next((r for r in existing if r.get('sha256')==meta['sha256']),None)
  if dup and not args.allow_same_hash:
   print(f"ERROR: same SHA-256 already registered as {dup.get('name')} at {dup.get('path')}; refusing duplicate. Use --allow-same-hash only to record provenance intentionally.",file=sys.stderr); return 2
 destdir=ROOT/'submissions'/'archive';destdir.mkdir(parents=True,exist_ok=True)
 dest=destdir/f'{args.date}_{slug(args.name)}.tar.gz'
 if dest.exists(): print(f'ERROR: refusing to overwrite {dest}',file=sys.stderr);return 2
 shutil.copy2(args.archive,dest)
 row={'artifact_date':args.date,'name':args.name,'path':str(dest.relative_to(ROOT)),'sha256':meta['sha256'],'size_bytes':meta['size_bytes'],'model_weights_sha256':meta['model_weights_sha256'],'deck_sha256':meta['deck_sha256'],'historical_score':args.score,'score_provenance':args.score_source,'status':'candidate_not_promoted','notes':args.notes}
 csvp=ROOT/'submissions'/'registry.csv'
 with csvp.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f)); cols=list(rows[0].keys()) if rows else list(row.keys())
 with csvp.open('a',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=cols);w.writerow({k:row.get(k,'') for k in cols})
 (dest.parent / (dest.name + '.manifest.json')).write_text(json.dumps({**row,**meta},indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'archive':str(dest.relative_to(ROOT)),**meta},indent=2));print('status=candidate_not_promoted');return 0
if __name__=='__main__': raise SystemExit(main())
