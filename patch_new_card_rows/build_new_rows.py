import sys, re, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import Ridge
from knn_lib import *
from atoms_lib import *

cards, atks = load()
SRC = __import__('os').environ.get('PTCG_B_OUT','bundleB/out/')
OUT = sys.argv[1]
Ec = np.load(SRC + 'card_text_emb.npy').astype(np.float64)
Ea = np.load(SRC + 'attack_text_emb.npy').astype(np.float64)
Ac = np.load(SRC + 'card_atom_feats.npy').astype(np.float64)
Aa = np.load(SRC + 'attack_atom_feats.npy').astype(np.float64)
NC, NA = max(cards) + 1, max(atks) + 1          # 1432, 1756
old_c = list(range(1, 1268)); new_c = [i for i in sorted(cards) if i > 1267]
old_a = list(range(1, 1557)); new_a = [a for a in sorted(atks) if a > 1556]
print("rows", NC, NA, "new cards", len(new_c), "new attacks", len(new_a))
print("old emb norms mean", np.linalg.norm(Ec[1:1268], axis=1).mean(), np.linalg.norm(Ea[1:1557], axis=1).mean())

def name_toks(c): return ' '.join('NAME_' + w for w in re.findall(r"[\w']+", norm_text(c['name'])))
def unit(P): return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

# ---------------- text embeddings (ridge on TF-IDF) ----------------
cdocs_all = {i: card_doc(cards[i], atks) + ' ' + name_toks(cards[i]) for i in cards}
Xc = build_vec([cdocs_all[i] for i in sorted(cards)]); cidx = {i: k for k, i in enumerate(sorted(cards))}
rc = Ridge(alpha=1.0).fit(Xc[[cidx[i] for i in old_c]], Ec[1:1268])
Pc = unit(rc.predict(Xc[[cidx[i] for i in new_c]]))
adocs_all = {a: attack_doc(atks[a]) for a in atks}
Xa = build_vec([adocs_all[a] for a in sorted(atks)]); aidx = {a: k for k, a in enumerate(sorted(atks))}
ra = Ridge(alpha=1.0).fit(Xa[[aidx[a] for a in old_a]], Ea[1:1557])
Pa = unit(ra.predict(Xa[[aidx[a] for a in new_a]]))

# ---------------- atoms ----------------
par = {}
for i in cards:
    for a in cards[i]['attacks']: par[a] = i
Sa = (Xa @ Xa.T).tocsr()
ref_a = [a for a in old_a if Aa[a].any()]
new_att_rows = {}
for a in new_a:
    c = cards[par[a]]; dmg = int(atks[a]['damage'] or 0)
    if parent_textless(c, atks):
        new_att_rows[a] = np.zeros(DIM); continue
    s = Sa[aidx[a], [aidx[r] for r in ref_a]].toarray()
    p, idx = knn_predict(s, Aa[ref_a], 5, 2)
    new_att_rows[a] = fix_attack_scalars(p[0], dmg)

Sc = (Xc @ Xc.T).tocsr()
def ref_cards(pred):
    return [i for i in old_c if Ac[i].any() and pred(cards[i])]
new_card_rows = {}; notes = {}
for i in new_c:
    c = cards[i]; dm = [int(atks[a]['damage'] or 0) for a in c['attacks']]
    if parent_textless(c, atks) and c['cardType'] == 0:
        new_card_rows[i] = np.zeros(DIM); notes[i] = 'vanilla->zero'; continue
    if c['cardType'] == 0 and not c['skills']:
        new_card_rows[i] = pool([new_att_rows[a] if a in new_att_rows else Aa[a] for a in c['attacks']], dm); notes[i] = 'pooled attacks'; continue
    if c['cardType'] == 0:
        refs = ref_cards(lambda x: x['cardType'] == 0 and x['skills']); k = 5
    else:
        refs = ref_cards(lambda x: x['cardType'] == c['cardType']); k = 3
    s = Sc[cidx[i], [cidx[r] for r in refs]].toarray()
    p, idx = knn_predict(s, Ac[refs], k, 2); p = p[0]
    if c['cardType'] == 0:
        p[SC + 2] = 1.0 if c['attacks'] else 0.0; p[SC + 3] = 1.0; p[SC + 4] = min(1.0, max(dm) / 300.0) if dm else 0.0
        # blend in the attack part exactly where it is certain
        notes[i] = 'knn(ability pokemon): ' + ', '.join(cards[refs[j]]['name'] for j in idx[0][:3])
    else:
        notes[i] = 'knn(%s): ' % c['cardType'] + ', '.join(f"{cards[refs[j]]['name']}({s[0][j]:.2f})" for j in idx[0][:3])
    new_card_rows[i] = p

# ---------------- write extended tables ----------------
def ext(old, n, rows):
    out = np.zeros((n, old.shape[1]), dtype=np.float32); out[:old.shape[0]] = old[:old.shape[0]]
    for k, v in rows.items(): out[k] = v
    return out
import os; os.makedirs(OUT, exist_ok=True)
np.save(OUT + '/card_text_emb.npy', ext(Ec, NC, {i: Pc[k] for k, i in enumerate(new_c)}))
np.save(OUT + '/attack_text_emb.npy', ext(Ea, NA, {a: Pa[k] for k, a in enumerate(new_a)}))
np.save(OUT + '/card_atom_feats.npy', ext(Ac, NC, new_card_rows))
np.save(OUT + '/attack_atom_feats.npy', ext(Aa, NA, new_att_rows))
json.dump({str(k): v for k, v in notes.items()}, open(OUT + '/notes.json', 'w'), indent=1)
for i in new_c:
    c = cards[i]
    if c['cardType'] != 0 or c['skills']:
        print(i, c['name'], '|', clean(' / '.join(s['text'] for s in c['skills']))[:130], '=>', notes[i])
print("done; nonzero rows: card_text", int((np.abs(np.load(OUT + '/card_text_emb.npy')).sum(1) > 0).sum()),
      "card_atom", int((np.abs(np.load(OUT + '/card_atom_feats.npy')).sum(1) > 0).sum()),
      "atk_atom", int((np.abs(np.load(OUT + '/attack_atom_feats.npy')).sum(1) > 0).sum()))
