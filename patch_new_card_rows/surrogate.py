import sys, re, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import Ridge
from knn_lib import *
from atoms_lib import *

SRC = __import__('os').environ.get('PTCG_B_OUT','bundleB/out/')
cards, atks = load()
T = {k: np.load(SRC + k + '.npy').astype(np.float64) for k in ['card_text_emb', 'attack_text_emb', 'card_atom_feats', 'attack_atom_feats']}
par = {}
for i in cards:
    for a in cards[i]['attacks']: par[a] = i

def name_toks(c): return ' '.join('NAME_' + w for w in re.findall(r"[\w']+", norm_text(c['name'])))
def unit(P): return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

_cache = {}
def _vecs():
    if 'X' not in _cache:
        ids = sorted(cards); aids = sorted(atks)
        Xc = build_vec([card_doc(cards[i], atks) + ' ' + name_toks(cards[i]) for i in ids])
        Xa = build_vec([attack_doc(atks[a]) for a in aids])
        _cache.update(X=(Xc, Xa), cidx={i: k for k, i in enumerate(ids)}, aidx={a: k for k, a in enumerate(aids)})
    return _cache['X'][0], _cache['X'][1], _cache['cidx'], _cache['aidx']

def build(target_cards, ref_cards, text='ridge', atoms=True, alpha=1.0, k_atom=5):
    """Estimate (text emb, atom feats) rows for target cards and their attacks from reference OLD cards only.
    Returns dict with 'card_text','card_atom','atk_text','atk_atom' : {id: row}."""
    Xc, Xa, cidx, aidx = _vecs()
    ref_cards = [i for i in ref_cards if i <= 1267]
    ref_atk = [a for i in ref_cards for a in cards[i]['attacks'] if a <= 1556]
    tgt_atk = sorted({a for i in target_cards for a in cards[i]['attacks']})
    out = {'card_text': {}, 'card_atom': {}, 'atk_text': {}, 'atk_atom': {}}
    if text == 'ridge':
        rc = Ridge(alpha=alpha).fit(Xc[[cidx[i] for i in ref_cards]], T['card_text_emb'][ref_cards])
        P = unit(rc.predict(Xc[[cidx[i] for i in target_cards]]))
        for j, i in enumerate(target_cards): out['card_text'][i] = P[j]
        ra = Ridge(alpha=alpha).fit(Xa[[aidx[a] for a in ref_atk]], T['attack_text_emb'][ref_atk])
        P = unit(ra.predict(Xa[[aidx[a] for a in tgt_atk]]))
        for j, a in enumerate(tgt_atk): out['atk_text'][a] = P[j]
    elif text == 'typemean':
        for i in target_cards:
            same = [r for r in ref_cards if cards[r]['cardType'] == cards[i]['cardType']]
            out['card_text'][i] = unit(T['card_text_emb'][same].mean(0, keepdims=True))[0]
        m = unit(T['attack_text_emb'][ref_atk].mean(0, keepdims=True))[0]
        for a in tgt_atk: out['atk_text'][a] = m
    if atoms:
        ref_a = [a for a in ref_atk if T['attack_atom_feats'][a].any()]
        Sa = (Xa @ Xa.T).tocsr() if 'Sa' not in _cache else _cache['Sa']; _cache['Sa'] = Sa
        att_rows = {}
        for a in tgt_atk:
            c = cards[par[a]]; dmg = int(atks[a]['damage'] or 0)
            if parent_textless(c, atks): att_rows[a] = np.zeros(DIM); continue
            s = Sa[aidx[a], [aidx[r] for r in ref_a]].toarray()
            p, _ = knn_predict(s, T['attack_atom_feats'][ref_a], k_atom, 2); att_rows[a] = fix_attack_scalars(p[0], dmg)
        out['atk_atom'] = att_rows
        Sc = (Xc @ Xc.T).tocsr() if 'Sc' not in _cache else _cache['Sc']; _cache['Sc'] = Sc
        for i in target_cards:
            c = cards[i]; dm = [int(atks[a]['damage'] or 0) for a in c['attacks']]
            if parent_textless(c, atks) and c['cardType'] == 0: out['card_atom'][i] = np.zeros(DIM); continue
            if c['cardType'] == 0 and not c['skills']:
                out['card_atom'][i] = pool([att_rows[a] for a in c['attacks']], dm); continue
            if c['cardType'] == 0:
                refs = [r for r in ref_cards if T['card_atom_feats'][r].any() and cards[r]['cardType'] == 0 and cards[r]['skills']]; k = 5
            else:
                refs = [r for r in ref_cards if T['card_atom_feats'][r].any() and cards[r]['cardType'] == c['cardType']]; k = 3
            s = Sc[cidx[i], [cidx[r] for r in refs]].toarray()
            p, _ = knn_predict(s, T['card_atom_feats'][refs], k, 2); p = p[0]
            if c['cardType'] == 0:
                p[SC + 2] = 1.0 if c['attacks'] else 0.0; p[SC + 3] = 1.0; p[SC + 4] = min(1.0, max(dm) / 300.0) if dm else 0.0
            out['card_atom'][i] = p
    return out

def write_tables(outdir, rows, n_cards=1268, n_atk=1557, zero_targets=None, base=None):
    """Write 4 arrays: start from base (old tables); apply rows; optionally zero given card ids/attack ids."""
    import os; os.makedirs(outdir, exist_ok=True)
    base = base or T
    arr = {k: np.zeros((n, base[k].shape[1]), dtype=np.float32) for k, n in
           [('card_text_emb', n_cards), ('card_atom_feats', n_cards), ('attack_text_emb', n_atk), ('attack_atom_feats', n_atk)]}
    for k in arr:
        m = min(n_cards if k.startswith('card') else n_atk, base[k].shape[0]); arr[k][:m] = base[k][:m]
    for name, key in [('card_text', 'card_text_emb'), ('card_atom', 'card_atom_feats'), ('atk_text', 'attack_text_emb'), ('atk_atom', 'attack_atom_feats')]:
        for i, v in rows.get(name, {}).items(): arr[key][i] = v
    if zero_targets:
        zc, za = zero_targets
        for i in zc: arr['card_text_emb'][i] = 0; arr['card_atom_feats'][i] = 0
        for a in za: arr['attack_text_emb'][a] = 0; arr['attack_atom_feats'][a] = 0
    for k, v in arr.items(): np.save(f'{outdir}/{k}.npy', v)
