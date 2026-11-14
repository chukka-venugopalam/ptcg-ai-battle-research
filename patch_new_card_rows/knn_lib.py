import json, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def load():
    v2 = json.load(open('' + __import__('os').environ.get('PTCG_V2_JSON','cmp/v2.json') + ''))
    cards = {c['cardId']: c for c in v2['cards']}
    atks = {a['attackId']: a for a in v2['attacks']}
    return cards, atks

def clean(t):
    return re.sub(r'\s+', ' ', str(t or '')).strip()

def norm_text(t):
    t = clean(t).lower()
    t = t.replace('’', "'")
    return t

def attack_doc(a):
    # no attack name (unique tokens carry no function); numbers + cost + effect text
    dmg = int(a.get('damage') or 0)
    toks = [f"dmg_{dmg//10*10}"] if dmg else ["dmg_0"]
    en = a.get('energies') or []
    toks.append(f"cost_{len(en)}")
    toks += [f"ecost_{e}" for e in sorted(set(en))]
    return ' '.join(toks) + ' ' + norm_text(a.get('text'))

def card_doc(c, atks):
    ct = c['cardType']
    toks = [f"kind_{ct}", f"etype_{c.get('energyType')}"]
    if ct == 0:
        stage = 'basic' if c['basic'] else ('s1' if c['stage1'] else ('s2' if c['stage2'] else 'x'))
        toks += [f"stage_{stage}", f"hp_{(c['hp'] or 0)//30*30}", f"retreat_{c.get('retreatCost')}"]
        if c['ex']: toks.append('rule_ex')
        if c['megaEx']: toks.append('rule_mega')
        if c['tera']: toks.append('rule_tera')
        toks.append(f"natk_{len(c['attacks'])}")
    doc = ' '.join(toks)
    for s in c['skills']:
        doc += ' ABILITYTEXT ' + norm_text(s['text'])
    for aid in c['attacks']:
        doc += ' ATTACKTEXT ' + attack_doc(atks[aid])
    return doc

def build_vec(docs):
    vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w{}]+\b", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs)
    return normalize(X)

def knn_predict(S, Y, k, power, exclude_self=False):
    # S: (n_query, n_ref) similarity; Y: (n_ref, d)
    S = S.copy()
    if exclude_self:
        np.fill_diagonal(S, -1)
    idx = np.argsort(-S, axis=1)[:, :k]
    out = np.zeros((S.shape[0], Y.shape[1]), dtype=np.float64)
    for i in range(S.shape[0]):
        w = np.clip(S[i, idx[i]], 1e-6, None) ** power
        out[i] = (w[:, None] * Y[idx[i]]).sum(0) / w.sum()
    return out, idx
