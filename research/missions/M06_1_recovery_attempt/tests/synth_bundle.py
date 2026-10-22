"""Builds a SYNTHETIC recovery bundle (HARNESS VALIDATION ONLY) with the layout described in the Mission 6.1 brief."""
import hashlib, io, json, os, tarfile, zipfile
from pathlib import Path
import numpy as np

MAIN = '''import os, sys
try:
    _H = os.path.dirname(__file__)
except NameError:
    _H = os.getcwd()
sys.path.insert(0, _H)
DECK = [int(x) for x in open(os.path.join(_H, "deck.csv")).read().split()]
AGENT_KWARGS = {kwargs}
from sa import policynet as _P
def agent(obs):
    sel = obs["select"]
    if sel is None:
        return DECK
    cur = obs["current"]; me = cur["players"][cur["yourIndex"]]
    for c in (me.get("hand") or []):
        _P.card_vec(c["id"])
    for op in sel["option"]:
        if "attackId" in op: _P.attack_vec(op["attackId"])
        if "cardId" in op: _P.card_vec(op["cardId"])
    n = len(sel["option"]); k = max(sel["minCount"], min(sel["maxCount"], n))
    return list(range(k))
'''
POLICY_RAW = "import numpy as np\n_C = np.zeros((1300, 4))\n_A = np.zeros((1600, 4))\ndef card_vec(i):\n    return _C[i]\ndef attack_vec(i):\n    return _A[i]\n"
POLICY_GUARD = "import numpy as np\n_C = np.zeros((1300, 4))\n_A = np.zeros((1600, 4))\ndef card_vec(i):\n    return _C[min(i, 1299)]\ndef attack_vec(i):\n    return _A[min(i, 1599)]\n"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(root: Path) -> Path:
    import kaggle_environments
    pyp = Path(kaggle_environments.__file__).parent / "envs" / "cabt"
    sys_deck = [int(x) for x in (root / "_deck.txt").read_text().split()] if (root / "_deck.txt").exists() else None
    from kaggle_environments.envs.cabt.cabt import deck as DECK
    root.mkdir(parents=True, exist_ok=True); npz = root / "policy_net.npz"
    np.savez(npz, card_emb=np.zeros((1300, 4)), atk_emb=np.zeros((1600, 4)), w=np.arange(12.0).reshape(3, 4))
    cg_old = {n: (pyp / "cg" / n).read_bytes() for n in ("__init__.py", "game.py", "sim.py")}
    cg_new = dict(cg_old); cg_new["game.py"] = cg_old["game.py"] + b"\n# R2 wrapper revision (synthetic)\n"
    variants = {"latest_candidate_R2_compat": ("{'counter_source': False}", POLICY_GUARD, cg_new), "Bv1_baseline_local_archive": ("{'counter_source': False}", POLICY_RAW, cg_old),
                "B2_counter_source_grimmsnarl": ("{'counter_source': True}", POLICY_RAW, cg_old), "Bv2_sym8_ready": ("{'counter_source': False, 'sym_k': 8}", POLICY_RAW, cg_new)}
    ag = root / "inputs" / "agents"; ag.mkdir(parents=True); files = {}
    for name, (kw, pol, cg) in variants.items():
        members = {"main.py": MAIN.format(kwargs=kw).encode(), "deck.csv": ("\n".join(map(str, DECK)) + "\n").encode(), "sa/__init__.py": b"", "sa/policynet.py": pol.encode(), "sa/policy_net.npz": npz.read_bytes(),
                   **{f"cg/{n}": b for n, b in cg.items()}}
        p = ag / f"{name}.tar.gz"
        with tarfile.open(p, "w:gz") as tf:
            for n in sorted(members):
                ti = tarfile.TarInfo(n); ti.size = len(members[n]); ti.mtime = 1_700_000_000; ti.mode = 0o644; tf.addfile(ti, io.BytesIO(members[n]))
        files[f"inputs/agents/{name}.tar.gz"] = sha(p)
    eng = root / "inputs" / "engine"; eng.mkdir(parents=True); ez = eng / "official_playground_bundle_R2.zip"
    with zipfile.ZipFile(ez, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ("cabt.py", "cabt.json"): z.write(pyp / n, f"official_playground_bundle_R2/cabt/{n}")
        for n, b in cg_new.items(): z.writestr(f"official_playground_bundle_R2/cabt/cg/{n}", b)
        z.write(pyp / "cg" / "libcg.so", "official_playground_bundle_R2/cabt/cg/libcg.so")
    files["inputs/engine/official_playground_bundle_R2.zip"] = sha(ez)
    (root / "input_manifest.json").write_text(json.dumps({"files": [{"path": k, "sha256": v} for k, v in files.items()]}, indent=1))
    inner = root / "PTCG_MISSION_6_INPUTS_BUNDLE.zip"
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(root / "input_manifest.json", "input_manifest.json")
        for k in files: z.write(root / k, k)
    outz = root / "MISSION_6_OUTPUT.zip"
    with zipfile.ZipFile(outz, "w") as z: z.writestr("README.md", "synthetic stand-in for Mission 6 output\n")
    wrap = root.parent / (root.name + "_RECOVERY.zip")
    with zipfile.ZipFile(wrap, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(inner, inner.name); z.write(outz, outz.name)
    return wrap
