import numpy as np, re
from knn_lib import *
A_EFF=0;A_TGT=72;A_UNIT=104;A_MOD=119;A_TRIG=191;A_VSUM=231;A_VMAX=246;SC=261;DIM=267

def parent_textless(c, atks):
    return (not c['skills']) and all(not clean(atks[a]['text']) for a in c['attacks'])

def pool(rows, base_dmg, has_abil=0.0):
    v = np.zeros(DIM)
    if len(rows) == 0:
        return v
    rows = np.array(rows)
    v[0:A_VSUM] = np.minimum(1.0, rows[:, 0:A_VSUM].sum(0))
    v[A_VSUM:A_VMAX] = np.minimum(3.0, rows[:, A_VSUM:A_VMAX].sum(0))
    v[A_VMAX:SC] = rows[:, A_VMAX:SC].max(0)
    v[SC] = min(1.0, rows[:, SC].sum()); v[SC + 1] = min(1.0, rows[:, SC + 1].sum())
    v[SC + 2] = 1.0; v[SC + 3] = has_abil
    v[SC + 4] = min(1.0, max(base_dmg) / 300.0) if base_dmg else 0.0
    v[SC + 5] = rows[:, SC + 5].max()
    return v

def fix_attack_scalars(v, dmg):
    v = v.copy(); v[SC + 2] = 1.0; v[SC + 3] = 0.0; v[SC + 4] = min(1.0, dmg / 300.0)
    return v
