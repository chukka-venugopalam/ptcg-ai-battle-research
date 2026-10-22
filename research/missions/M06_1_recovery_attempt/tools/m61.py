#!/usr/bin/env python3
"""m61.py -- Mission 6.1 pipeline: recovery-bundle preflight, byte-level archive audit, SDK comparison, R2 overlay engine,
ID-boundary probe, safety audits and candidate-vs-baseline BO3 pairs.  READ-ONLY: never contacts Kaggle, never uploads, never
modifies a source archive.  Every stage is wrapped so a failure stops only that stage.  Stdlib + numpy (+ kaggle-environments for
engine stages).  Fixture/synthetic outputs are labelled HARNESS VALIDATION ONLY by the caller.
  m61.py all --root RECOVERY.zip|DIR --out OUT [--matches 20]     (or run stages: preflight audit sdk idprobe safety pairs)"""
import argparse, ast, collections, csv, difflib, hashlib, json, os, re, shutil, stat, subprocess, sys, tarfile, traceback, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEX64 = re.compile(r"[0-9a-f]{64}")
EXPECTED = {   # from the Mission 6.1 brief
    "inputs/agents/latest_candidate_R2_compat.tar.gz": "0f91f896ef703761d8601215f743177c28e929f49251f02022b936610982f28a",
    "inputs/agents/Bv1_baseline_local_archive.tar.gz": "7ea11c6181f1b68ff23a97311998901b66b3ab5e7bcd1371e3d46b7d0c9d4feb",
    "inputs/agents/B2_counter_source_grimmsnarl.tar.gz": "b61ed901d778ac23647dc2be97a13770e632c2e62b7a6d289508ddf6910baea8",
    "inputs/agents/Bv2_sym8_ready.tar.gz": "169e1ac85e9edba28ccf19ff99e244c2016712096e7cdb0aab69cf9e0574318c",
    "inputs/engine/official_playground_bundle_R2.zip": "6fe557e77ca21b86dea9056d7f4531cbeb328abc1b25aecf9c3a4e2b7d5b2c87"}
R2_LIBCG = "c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22"
AGENT_KEYS = (("candidate", "latest_candidate"), ("candidate", "r2_compat"), ("Bv1", "bv1"), ("B2", "b2_"), ("Bv2", "bv2"))
NATIVE_OK = re.compile(r"(^|/)cg/(libcg\.so|libcg-arm64\.so|libcg\.dylib|cg\.dll)$")
MAGIC = ((b"\x7fELF", "ELF"), (b"MZ", "PE"), (b"\xcf\xfa\xed\xfe", "MachO"), (b"\xfe\xed\xfa\xcf", "MachO"), (b"\xca\xfe\xba\xbe", "MachO-fat"))
TEXT_EXT = {".py", ".csv", ".json", ".txt", ".md", ".cfg", ".yaml", ".yml", ".toml", ".ini"}
FLAG_RE = re.compile(r"AGENT_KWARGS|counter_source|sym_k")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _bad(n):
    p = n.replace("\\", "/")
    return p.startswith("/") or bool(re.match(r"^[A-Za-z]:", p)) or ".." in p.split("/") or "\x00" in p


# ------------------------------------------------------------------------------------------ safe listing + extraction
def extract_archive(path, dest=None, native_ok=NATIVE_OK):
    """List every member, flag problems, and (only when none is reject-class) extract manually (never extractall, modes dropped).
    Reject-class: unsafe path, duplicate member (exact or case-folded), link/device/fifo, native-code magic outside cg/*, malformed.
    Findings (not rejected): exec bit on a non-native file."""
    path = Path(path); res = dict(archive=str(path), sha256=sha_file(path), bytes=path.stat().st_size, members=[], problems=[], findings=[], extracted=None)
    try:
        if zipfile.is_zipfile(path): kind, opener = "zip", zipfile.ZipFile(path)
        else: kind, opener = "tar", tarfile.open(path, "r:*")
    except Exception as e:
        res["problems"].append(f"MALFORMED: {type(e).__name__}: {e}"); return res
    res["kind"] = kind; seen = {}; fold = {}; reads = []
    with opener as ar:
        infos = ar.infolist() if kind == "zip" else ar.getmembers()
        for m in infos:
            name = m.filename if kind == "zip" else m.name
            if kind == "zip":
                mode = (m.external_attr >> 16) & 0xFFFF; isdir = m.is_dir(); isreg = not isdir and not stat.S_ISLNK(mode); typ = "dir" if isdir else "link" if stat.S_ISLNK(mode) else "file"; size = m.file_size
            else:
                mode = m.mode; isdir = m.isdir(); isreg = m.isreg(); size = m.size
                typ = "dir" if isdir else "file" if isreg else "link" if (m.issym() or m.islnk()) else "special"
            ent = dict(name=name, type=typ, size=size, mode=oct(mode & 0o7777), sha256=None, flags=[])
            if _bad(name): res["problems"].append(f"UNSAFE_PATH {name!r}"); ent["flags"].append("unsafe")
            key = name.rstrip("/")
            if key in seen: res["problems"].append(f"DUPLICATE_MEMBER {name!r}")
            elif key.lower() in fold and fold[key.lower()] != key: res["problems"].append(f"CASE_DUPLICATE {name!r} vs {fold[key.lower()]!r}")
            seen[key] = 1; fold.setdefault(key.lower(), key)
            if typ in ("link", "special"): res["problems"].append(f"{typ.upper()}_MEMBER {name!r}"); ent["flags"].append(typ)
            if isreg:
                data = ar.read(m) if kind == "zip" else ar.extractfile(m).read()
                ent["sha256"] = hashlib.sha256(data).hexdigest(); reads.append((name, data))
                magic = next((n for b, n in MAGIC if data[:len(b)] == b), None)
                if magic and not native_ok.search(name): res["problems"].append(f"UNEXPECTED_EXECUTABLE {magic} {name!r}"); ent["flags"].append("native-magic")
                elif mode & 0o111 and not native_ok.search(name): res["findings"].append(f"exec-bit on non-native file {name!r}"); ent["flags"].append("exec-bit")
            res["members"].append(ent)
    if dest is not None and not res["problems"]:
        dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
        for name, data in reads:
            t = dest / name; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(data)
        res["extracted"] = str(dest)
    return res


# ------------------------------------------------------------------------------------------ manifest
def manifest_pairs(obj):
    out = []
    def hv(o, ks):
        return next((o[k].lower() for k in ks if isinstance(o.get(k), str) and HEX64.fullmatch(o[k].lower())), None)
    def walk(o):
        if isinstance(o, dict):
            h = hv(o, ("sha256", "sha-256", "hash", "digest", "sha")); p = next((o[k] for k in ("path", "file", "filename", "name", "relpath") if isinstance(o.get(k), str)), None)
            if h and p: out.append((p, h)); return
            for k, v in o.items():
                if isinstance(v, str) and HEX64.fullmatch(v.lower()) and ("/" in k or "." in k): out.append((k, v.lower()))
                elif isinstance(v, dict) and hv(v, ("sha256", "hash")) and ("/" in k or "." in k): out.append((k, hv(v, ("sha256", "hash"))))
                else: walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(obj); return out


def resolve(rel, roots):
    for r in roots:
        for cand in (Path(r) / rel, Path(r) / "inputs" / rel):
            if cand.is_file(): return cand
    base = Path(rel).name
    hits = [p for r in roots for p in Path(r).rglob(base) if p.is_file()]
    return hits[0] if len(hits) == 1 else None


def find_files(root, basename):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.name.lower() == basename.lower())


# ------------------------------------------------------------------------------------------ stage: preflight
def stage_preflight(root, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); L = []; pf = dict(root=str(root), problems=[], agents={}, engine=None, bundle_dir=None)
    members_md = ["# MEMBER_LISTS (complete)\n"]
    def log(s): L.append(s); print(s)
    def listing(res, label):
        members_md.append(f"\n## {label}\n`{res['archive']}` sha256 `{res['sha256']}` {res['bytes']} bytes; problems: {res['problems'] or 'none'}; findings: {res['findings'] or 'none'}\n")
        members_md.append("| member | type | size | mode | sha256 |\n|---|---|---|---|---|")
        members_md.extend(f"| {m['name']} | {m['type']} | {m['size']} | {m['mode']} | {(m['sha256'] or '')[:16]} |" for m in res["members"])
    root = Path(root); search = root
    log(f"# INPUT_PREFLIGHT\nRoot searched: `{root}`")
    if not root.exists():
        log(f"- **MISSING PATH (exact): `{root}`**"); pf["problems"].append(f"missing root {root}")
    elif root.is_file():
        r = extract_archive(root, out / "_root"); listing(r, f"recovery wrapper {root.name}"); log(f"- wrapper archive: {r['bytes']} bytes sha256 `{r['sha256']}`, {len(r['members'])} members, problems: {r['problems'] or 'none'}")
        if r["extracted"]: search = Path(r["extracted"])
        else: pf["problems"].append("wrapper rejected/unreadable")
    inner = find_files(search, "PTCG_MISSION_6_INPUTS_BUNDLE.zip") if search.exists() and search.is_dir() else []
    outz = find_files(search, "MISSION_6_OUTPUT.zip") if search.exists() and search.is_dir() else []
    for nm, hits in (("PTCG_MISSION_6_INPUTS_BUNDLE.zip", inner), ("MISSION_6_OUTPUT.zip", outz)):
        if hits: log(f"- found `{nm}` at `{hits[0]}` ({hits[0].stat().st_size} bytes)")
        else: log(f"- **MISSING PATH (exact): `{search}/**/{nm}`** (recursive search, no match)"); pf["problems"].append(f"missing {nm}")
    if outz:
        r = extract_archive(outz[0], out / "_mission6_output"); listing(r, "MISSION_6_OUTPUT.zip"); log(f"- MISSION_6_OUTPUT.zip: {len(r['members'])} members, problems: {r['problems'] or 'none'}")
    if root.is_file() and root.name.lower() == "ptcg_mission_6_inputs_bundle.zip": inner = [root]          # the inner bundle was given directly
    pre = find_files(search, "input_manifest.json") if (not inner and search.exists() and search.is_dir()) else []
    if not inner and not pre:
        log("\n## Result\nInner bundle not present: every agent/engine/manifest step is BLOCKED at this path. Nothing below was inferred.")
        (out / "INPUT_PREFLIGHT.md").write_text("\n".join(L) + "\n"); (out / "MEMBER_LISTS.md").write_text("\n".join(members_md) + "\n"); json.dump(pf, open(out / "preflight.json", "w"), indent=1); return pf
    if inner:
        rb = extract_archive(inner[0], out / "bundle"); listing(rb, "PTCG_MISSION_6_INPUTS_BUNDLE.zip")
        log(f"- inputs bundle: {rb['bytes']} bytes sha256 `{rb['sha256']}`, {len(rb['members'])} members, problems: {rb['problems'] or 'none'}, findings: {len(rb['findings'])}")
    else:
        rb = dict(archive=str(pre[0].parent), sha256="(pre-extracted directory)", bytes=0, members=[], problems=[], findings=[], extracted=str(pre[0].parent))
        log(f"- inner zip absent, but an already-extracted bundle (input_manifest.json) was found at `{pre[0].parent}`; using it as is")
    if not rb["extracted"]: pf["problems"].append("inputs bundle rejected/unreadable"); log("- **inputs bundle NOT extracted (see problems)**")
    else:
        bd = Path(rb["extracted"]); pf["bundle_dir"] = str(bd)
        mans = find_files(bd, "input_manifest.json")
        if not mans: log(f"- **MISSING PATH (exact): `{bd}/**/input_manifest.json`**")
        else:
            man = json.load(open(mans[0])); pairs = manifest_pairs(man)
            log(f"\n## Manifest `{mans[0].relative_to(bd)}`: {len(pairs)} (path, sha256) entries parsed" + ("" if pairs else f" -- UNRECOGNISED SHAPE, top-level keys {list(man)[:10] if isinstance(man, dict) else type(man).__name__}"))
            log("| manifest path | status | expected sha256 | actual sha256 |\n|---|---|---|---|")
            listed = set()
            for rel, h in pairs:
                f = resolve(rel, [bd, mans[0].parent])
                if f is None: log(f"| {rel} | MISSING FILE | {h[:16]} | |"); continue
                listed.add(f.resolve()); a = sha_file(f); log(f"| {rel} | {'OK' if a == h else 'MISMATCH'} | {h[:16]} | {a[:16]} |")
            extra = [str(p.relative_to(bd)) for p in bd.rglob("*") if p.is_file() and p.resolve() not in listed and p.name != "input_manifest.json"]
            log(f"\nFiles in the bundle but not in the manifest: {len(extra)}" + ("".join(f"\n- {e}" for e in extra[:40])))
        log("\n## Hashes required by the Mission 6.1 brief\n| path | status | expected | actual |\n|---|---|---|---|")
        for rel, h in EXPECTED.items():
            f = resolve(rel, [bd])
            if f is None: log(f"| {rel} | **MISSING PATH** | {h[:16]} | |"); continue
            a = sha_file(f); log(f"| {rel} | {'OK' if a == h else 'MISMATCH'} | {h[:16]} | {a[:16]} |")
            if "agents" in rel:
                key = next(k for k, pat in AGENT_KEYS if pat in f.name.lower())
                ra = extract_archive(f, out / "agents" / key); listing(ra, f"agent {key}: {f.name}")
                pf["agents"][key] = dict(archive=str(f), sha256=a, extracted=ra["extracted"], problems=ra["problems"], findings=ra["findings"], expected_ok=(a == h))
                log(f"|  ↳ extraction {key} | {'extracted' if ra['extracted'] else 'REJECTED'} | problems: {ra['problems'] or 'none'}; findings: {ra['findings'] or 'none'} | |")
            else:
                re_ = extract_archive(f, out / "engine_r2", native_ok=re.compile(r".*")); listing(re_, f"engine {f.name}")
                pf["engine"] = dict(archive=str(f), sha256=a, extracted=re_["extracted"], problems=re_["problems"], expected_ok=(a == h))
    (out / "INPUT_PREFLIGHT.md").write_text("\n".join(L) + "\n"); (out / "MEMBER_LISTS.md").write_text("\n".join(members_md) + "\n"); json.dump(pf, open(out / "preflight.json", "w"), indent=1)
    return pf


# ------------------------------------------------------------------------------------------ stage: byte-level audit
def tree(d):
    d = Path(d); return {str(p.relative_to(d)).replace(os.sep, "/"): (p.stat().st_size, sha_file(p)) for p in sorted(d.rglob("*")) if p.is_file()}


def is_text(rel, size): return Path(rel).suffix.lower() in TEXT_EXT and size < 3_000_000


def npz_info(path):
    import numpy as np
    out = {}
    try:
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            a = z[k]; i = dict(shape=list(a.shape), dtype=str(a.dtype), sha=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16])
            if a.dtype.kind in "iu" and a.size: i.update(min=int(a.min()), max=int(a.max()))
            if a.dtype.kind == "f" and a.size: i.update(nonfinite=int((~np.isfinite(a)).sum()))
            out[k] = i
    except Exception as e: out["_error"] = f"{type(e).__name__}: {e}"
    return out


def npz_maxdiff(a, b):
    import numpy as np
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False); out = {}
    for k in sorted(set(za.files) | set(zb.files)):
        if k not in za.files or k not in zb.files: out[k] = "only in " + ("base" if k in za.files else "other"); continue
        x, y = za[k], zb[k]
        out[k] = "identical" if x.shape == y.shape and x.dtype == y.dtype and (x == y).all() else (f"differs max|d|={float(abs(x.astype('f8') - y.astype('f8')).max()):.6g}" if x.shape == y.shape else f"shape {list(x.shape)} vs {list(y.shape)}")
    return out


def flag_info(d):
    d = Path(d); lines = []; kw = {}
    for p in sorted(d.rglob("*.py")):
        rel = str(p.relative_to(d)).replace(os.sep, "/")
        if rel.startswith("cg/"): continue
        src = p.read_text(errors="replace")
        lines += [f"{rel}:{i}: {l.strip()}" for i, l in enumerate(src.splitlines(), 1) if FLAG_RE.search(l)]
        try: t = ast.parse(src)
        except SyntaxError: continue
        for n in ast.walk(t):
            val = None
            if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "AGENT_KWARGS" for x in n.targets): val = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "AGENT_KWARGS": val = n.value
            if val is not None:
                try: kw[f"{rel}:{n.lineno}"] = ast.literal_eval(val)
                except Exception: kw[f"{rel}:{n.lineno}"] = ast.get_source_segment(src, val)
    return kw, lines


def deck_of(d):
    p = Path(d) / "deck.csv"
    return collections.Counter(int(x) for x in p.read_text().replace(",", " ").split()) if p.exists() else None


def stage_audit(pf, out, cap=400):
    out = Path(out); ag = {k: v for k, v in pf.get("agents", {}).items() if v.get("extracted")}
    if "Bv1" not in ag: raise RuntimeError("Bv1 baseline not extracted: no diffs possible")
    T = {k: tree(v["extracted"]) for k, v in ag.items()}; rows = []
    for k, v in ag.items():
        d = Path(v["extracted"]); kw, lines = flag_info(d); dk = deck_of(d); npz = {}
        for rel in T[k]:
            if rel.endswith(".npz"): npz[rel] = npz_info(d / rel)
        rows.append(dict(archive=k, file=Path(v["archive"]).name, sha256=v["sha256"], expected_hash_ok=v["expected_ok"], n_files=len(T[k]), problems=";".join(v["problems"]), findings=";".join(v["findings"]),
                         main_py_sha=T[k].get("main.py", (0, ""))[1][:16], deck_sha=T[k].get("deck.csv", (0, ""))[1][:16], deck_n=sum(dk.values()) if dk else None, deck_distinct=len(dk) if dk else None,
                         npz={r: {n: f"{i['shape']}:{i['dtype']}" if "shape" in i else i for n, i in a.items()} for r, a in npz.items()}, agent_kwargs=kw, flag_lines=lines,
                         cg_files=sorted(r for r in T[k] if r.startswith("cg/")), sa_files=sorted(r for r in T[k] if r.startswith("sa/"))))
    json.dump(rows, open(out / "forensics.json", "w"), indent=1, default=str)
    md = ["# ARCHIVE_DIFFS", "Base = Bv1 baseline. Exact bytes extracted from the supplied tarballs; nothing modified.\n"]
    for k in ag:
        if k == "Bv1": continue
        a, b = T["Bv1"], T[k]; da, db = Path(ag["Bv1"]["extracted"]), Path(ag[k]["extracted"])
        only_a = sorted(set(a) - set(b)); only_b = sorted(set(b) - set(a)); chg = sorted(r for r in set(a) & set(b) if a[r][1] != b[r][1]); same = len(set(a) & set(b)) - len(chg)
        md += [f"\n## {k} vs Bv1\nIdentical files: {same}. Only in Bv1: {only_a or 'none'}. Only in {k}: {only_b or 'none'}. **Changed ({len(chg)}):**", "| path | size Bv1→" + k + " | sha256 Bv1→" + k + " |", "|---|---|---|"]
        md += [f"| {r} | {a[r][0]}→{b[r][0]} | {a[r][1][:12]}→{b[r][1][:12]} |" for r in chg]
        for r in chg:
            if is_text(r, max(a[r][0], b[r][0])):
                ud = list(difflib.unified_diff((da / r).read_text(errors="replace").splitlines(), (db / r).read_text(errors="replace").splitlines(), f"Bv1/{r}", f"{k}/{r}", n=2, lineterm=""))
                md += [f"\n### source diff `{r}` ({len(ud)} lines{', truncated' if len(ud) > cap else ''})", "```diff"] + ud[:cap] + ["```"]
            elif r.endswith(".npz"): md += [f"\n### npz `{r}`", "```", *(f"{n}: {s}" for n, s in npz_maxdiff(da / r, db / r).items()), "```"]
            else: md.append(f"\n### binary `{r}`: sizes {a[r][0]}→{b[r][0]} (hash only; native/SDK file)")
        ka, la = flag_info(da); kb, lb = flag_info(db); dka, dkb = deck_of(da), deck_of(db)
        md += ["\n### Runtime flags (exact source lines)", f"AGENT_KWARGS Bv1: `{ka}`", f"AGENT_KWARGS {k}: `{kb}`", "```diff", *difflib.unified_diff(la, lb, "Bv1", k, n=0, lineterm=""), "```",
               f"### Deck multiset: {'identical' if dka == dkb else 'DIFFERS ' + str((dkb - dka, dka - dkb))}"]
    (out / "ARCHIVE_DIFFS.md").write_text("\n".join(md) + "\n")
    write_forensics(out); return rows


def write_forensics(out):
    out = Path(out); rows = json.load(open(out / "forensics.json")); audits = {}
    for p in (out / "audits").glob("*.json") if (out / "audits").exists() else []:
        try: r = json.load(open(p)); audits[r.get("archive_report", {}).get("sha256")] = r
        except Exception: pass
    cols = ["archive", "file", "sha256", "expected_hash_ok", "n_files", "problems", "findings", "main_py_sha", "deck_sha", "deck_n", "deck_distinct", "npz_arrays", "agent_kwargs", "cg_files",
            "deck_legal_vs_engine", "cold_start_ok", "first_call_returns_60", "has___file__", "safety_gate_passed"]
    with open(out / "ARCHIVE_FORENSICS.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            a = audits.get(r["sha256"], {}); d = a.get("deck_audit", {}); c = a.get("cold_start", {})
            w.writerow([r["archive"], r["file"], r["sha256"], r["expected_hash_ok"], r["n_files"], r["problems"], r["findings"], r["main_py_sha"], r["deck_sha"], r["deck_n"], r["deck_distinct"],
                        json.dumps(r["npz"]), json.dumps(r["agent_kwargs"]), ";".join(r["cg_files"]),
                        (not d.get("problems") and d.get("engine_accepts")) if d else "NOT RUN", c.get("ok", "NOT RUN"), c.get("returns_60", "NOT RUN"), c.get("main_globals_have___file__", "NOT RUN"), a.get("safety_gate_passed", "NOT RUN")])


# ------------------------------------------------------------------------------------------ stage: SDK comparison + overlay
def _tables(lib_path):
    import ctypes
    lib = ctypes.CDLL(lib_path); lib.AllCard.restype = ctypes.c_char_p; lib.AllCard.argtypes = []
    raw = lib.AllCard(); cards = json.loads(raw.decode()); ids = [c["cardId"] for c in cards]; att = {a for c in cards for a in c.get("attacks", [])}
    return dict(n_cards=len(cards), id_min=min(ids), id_max=max(ids), card_table_sha256=hashlib.sha256(raw).hexdigest(), n_attack_ids=len(att), attack_min=min(att) if att else None,
                attack_max=max(att) if att else None, n_ace_spec=sum(bool(c.get("aceSpec")) for c in cards), n_basic=sum(bool(c.get("basic")) for c in cards))


def pypi_cabt():
    import kaggle_environments
    return Path(kaggle_environments.__file__).parent / "envs" / "cabt"


def r2_layout(engine_dir):
    libs = find_files(engine_dir, "libcg.so") or [p for p in find_files(engine_dir, "game.py") if p.parent.name == "cg"]   # text-only R2 package: wrapper files suffice, PyPI natives are kept
    if not libs: return None
    cg = libs[0].parent; cabt_dir = cg.parent
    cabt_py = (cabt_dir / "cabt.py") if (cabt_dir / "cabt.py").exists() else next(iter(find_files(engine_dir, "cabt.py")), None)
    cabt_json = (cabt_dir / "cabt.json") if (cabt_dir / "cabt.json").exists() else next(iter(find_files(engine_dir, "cabt.json")), None)
    return dict(cg=cg, cabt_py=cabt_py, cabt_json=cabt_json)


def _syms(p):
    try: return set(l.split()[-1] for l in subprocess.run(["nm", "-D", "--defined-only", str(p)], capture_output=True, text=True, timeout=60).stdout.splitlines() if l.strip())
    except Exception: return None


def stage_sdk(pf, out, cap=300):
    out = Path(out); L = ["# SDK_FINGERPRINT_COMPARISON", "Official R2 package (from the bundle) vs the local engine (PyPI `kaggle-environments`). Hash identity of one file does not imply identity of the rest.\n"]
    eng = pf.get("engine") or {}; lay = r2_layout(eng["extracted"]) if eng.get("extracted") else None
    if not lay: raise RuntimeError("official R2 package not extracted or no cg/libcg.so inside: SDK comparison blocked")
    pyp = pypi_cabt(); pairs = {}
    for nm, p in (("cabt.py", lay["cabt_py"]), ("cabt.json", lay["cabt_json"])):
        pairs[nm] = (p, pyp / nm)
    for p in sorted(lay["cg"].iterdir()):
        if p.is_file(): pairs[f"cg/{p.name}"] = (p, pyp / "cg" / p.name)
    for p in sorted((pyp / "cg").iterdir()):
        if p.is_file() and f"cg/{p.name}" not in pairs and p.suffix != ".pyc": pairs[f"cg/{p.name}"] = (None, p)
    L += ["| file | R2 sha256 | PyPI 1.33.0 sha256 | status |", "|---|---|---|---|"]; diffs = []; res = {}
    for nm, (r, l) in pairs.items():
        hr = sha_file(r) if r and Path(r).exists() else None; hl = sha_file(l) if Path(l).exists() else None
        st = "IDENTICAL" if hr and hr == hl else "ONLY IN PyPI" if not hr else "ONLY IN R2" if not hl else "DIFFERS"
        res[nm] = dict(r2=hr, pypi=hl, status=st); L.append(f"| {nm} | {(hr or '')[:16]} | {(hl or '')[:16]} | **{st}** |")
        if st == "DIFFERS" and Path(nm).suffix in TEXT_EXT:
            ud = list(difflib.unified_diff(Path(l).read_text(errors="replace").splitlines(), Path(r).read_text(errors="replace").splitlines(), f"PyPI/{nm}", f"R2/{nm}", n=2, lineterm=""))
            hits = sorted({k for k in ("reverse", "search", "timeout", "deck", "card", "attack", "result", "reward", "argtypes", "restype", "select", "first") for l_ in ud if l_[:1] in "+-" and k in l_.lower()})
            diffs.append((nm, ud, hits))
    r2h = res.get('cg/libcg.so', {}).get('r2')
    L.append(f"\nR2 libcg.so sha256 {r2h or 'NOT SUPPLIED (text-only package; PyPI native kept in the overlay)'}; brief says `{R2_LIBCG}`" + (f": **{'MATCH' if r2h == R2_LIBCG else 'MISMATCH'}**; PyPI libcg identical to R2: {res['cg/libcg.so']['status'] == 'IDENTICAL'}" if r2h else ""))
    for nm, ud, hits in diffs:
        L += [f"\n### diff `{nm}` ({len(ud)} lines; changed lines mention: {hits or 'none of the watched keywords'})", "```diff", *ud[:cap], "```"]
    for nm in [n for n in res if n.endswith((".so", ".dylib", ".dll")) and res[n]["status"] == "DIFFERS"]: L.append(f"- native file {nm} DIFFERS (hash only)")
    sa, sb = _syms(lay["cg"] / "libcg.so"), _syms(pyp / "cg" / "libcg.so")
    L.append(f"\nExported native symbols: R2 {len(sa) if sa else 'n/a'} / PyPI {len(sb) if sb else 'n/a'}; only in R2: {sorted(sa - sb) if sa and sb else 'n/a'}; only in PyPI: {sorted(sb - sa) if sa and sb else 'n/a'}")
    for lab, lib in (("R2", lay["cg"] / "libcg.so"), ("PyPI", pyp / "cg" / "libcg.so")):
        r = subprocess.run([sys.executable, str(HERE / "m61.py"), "_tables", str(lib)], capture_output=True, text=True, timeout=120)
        L.append(f"- {lab} tables: " + (r.stdout.strip() if r.returncode == 0 else f"UNKNOWN ({r.stderr.strip()[-200:]})")); res[f"tables_{lab}"] = r.stdout.strip()
    L.append("\n## Which SDK files each agent archive ships (cg/*)\n| agent | file | == R2 | == PyPI 1.33.0 |\n|---|---|---|---|")
    for k, v in pf.get("agents", {}).items():
        if not v.get("extracted"): continue
        for rel, (sz, h) in tree(v["extracted"]).items():
            if rel.startswith("cg/") and not rel.endswith(".pyc"):
                L.append(f"| {k} | {rel} | {h == (res.get(rel) or {}).get('r2')} | {h == (res.get(rel) or {}).get('pypi')} |")
    (out / "SDK_FINGERPRINT_COMPARISON.md").write_text("\n".join(L) + "\n"); json.dump(res, open(out / "sdk_compare.json", "w"), indent=1)
    build_overlay(lay, out / "r2_overlay"); return res


def build_overlay(lay, dst):
    """kaggle_environments with envs/cabt replaced by the R2 files (cabt.py, cabt.json, cg/*); everything else symlinked.  Use via PYTHONPATH."""
    import kaggle_environments
    src = Path(kaggle_environments.__file__).parent; dst = Path(dst) / "kaggle_environments"
    if dst.exists(): shutil.rmtree(dst.parent)
    dst.mkdir(parents=True)
    for e in src.iterdir():
        if e.name != "envs": os.symlink(e, dst / e.name)
    (dst / "envs").mkdir()
    for e in (src / "envs").iterdir():
        if e.name != "cabt": os.symlink(e, dst / "envs" / e.name)
    shutil.copytree(src / "envs" / "cabt", dst / "envs" / "cabt", symlinks=False, ignore=shutil.ignore_patterns("__pycache__"))
    for nm in ("cabt_py", "cabt_json"):
        if lay[nm]: shutil.copy(lay[nm], dst / "envs" / "cabt" / Path(lay[nm]).name)
    for p in lay["cg"].iterdir():
        if p.is_file(): shutil.copy(p, dst / "envs" / "cabt" / "cg" / p.name)
    return dst.parent


def env_for(out):
    ov = Path(out) / "r2_overlay"; e = dict(os.environ)
    if ov.exists(): e["PYTHONPATH"] = str(ov) + os.pathsep + e.get("PYTHONPATH", "")
    return e


# ------------------------------------------------------------------------------------------ stage: ID-boundary probe (end-to-end, black box)
def rewrite_ids(o, card=None, attack=None):
    o = json.loads(json.dumps(o))
    def w(x):
        if isinstance(x, dict):
            if card is not None:
                if "serial" in x and "id" in x: x["id"] = card
                if "cardId" in x: x["cardId"] = card
            if attack is not None and "attackId" in x: x["attackId"] = attack
            for v in x.values(): w(v)
        elif isinstance(x, list):
            for v in x: w(v)
    w(o); return o


def _idworker(agent_dir, obs_file, out_file, k=40):
    agent_dir = os.path.abspath(agent_dir); sys.path.insert(0, agent_dir); os.chdir(agent_dir)
    from kaggle_environments.agent import get_last_callable
    ag = get_last_callable(open("main.py").read(), path=os.path.join(agent_dir, "main.py"))     # SDK exec mode: no __file__, LAST callable
    allo = [o for o in json.load(open(obs_file)) if o.get("select")]; sample = allo[::max(1, len(allo) // k)][:k]
    V = [("raw (unmodified)", None, None)] + [(f"card id={c}", c, None) for c in (1, 1299, 1300, 1431, 1432)] + [(f"attack id={a}", None, a) for a in (1, 1599, 1600, 1755, 1756)] + [("card 1431 + attack 1755", 1431, 1755)]
    rows = []
    for name, c, a in V:
        exc = collections.Counter(); bad = 0; tb = None
        for o in sample:
            o2 = rewrite_ids(o, c, a)
            try:
                act = ag(o2); s = o2["select"]; n = len(s["option"])
                bad += not (isinstance(act, (list, tuple)) and all(isinstance(x, int) and 0 <= x < n for x in act) and len(set(act)) == len(act) and s["minCount"] <= len(act) <= s["maxCount"])
            except Exception as e:
                exc[type(e).__name__] += 1; tb = tb or traceback.format_exc(limit=3)
        rows.append(dict(variant=name, n_obs=len(sample), n_exceptions=sum(exc.values()), exception_types=dict(exc), n_illegal=bad, first_traceback=tb,
                         note="valid R2 id" if name.split("=")[-1].split()[0] not in ("1432", "1756") else "id beyond the engine table (robustness only)"))
    json.dump(rows, open(out_file, "w"))


def stage_idprobe(pf, out, games=4):
    out = Path(out); env = env_for(out); obs = out / "idprobe_observations.json"
    dk = next((Path(v["extracted"]) / "deck.csv" for v in pf.get("agents", {}).values() if v.get("extracted") and (Path(v["extracted"]) / "deck.csv").exists()), None)
    r = subprocess.run([sys.executable, str(HERE / "mirror_check.py"), "capture", "--out", str(obs), "--games", str(games)] + (["--deck", str(dk)] if dk else []), capture_output=True, text=True, env=env, timeout=900)
    if r.returncode: raise RuntimeError("observation capture failed (engine required): " + r.stderr[-300:])
    L = ["# ID_BOUNDARY_PROBE (black box: real engine observations with every card id / attack id rewritten)",
         "A PASS shows no exception or illegal output at that id. It does NOT show the id is embedded correctly (silent clipping or modulo wrapping is invisible here); read sa/policynet.py and sa/features.py for that.\n",
         "| agent | variant | n_obs | exceptions | illegal | note |", "|---|---|---|---|---|---|"]; res = {}
    for k, v in pf.get("agents", {}).items():
        if not v.get("extracted"): continue
        f = out / f"idprobe_{k}.json"
        r = subprocess.run([sys.executable, str(HERE / "m61.py"), "_idworker", v["extracted"], str(obs), str(f)], capture_output=True, text=True, env=env, timeout=1800)
        if r.returncode or not f.exists(): L.append(f"| {k} | WORKER FAILED | | | | {r.stderr.strip()[-200:]} |"); continue
        rows = json.load(open(f)); res[k] = rows
        L += [f"| {k} | {x['variant']} | {x['n_obs']} | {x['n_exceptions']} {x['exception_types'] or ''} | {x['n_illegal']} | {x['note']} |" for x in rows]
        L += [f"\n`{k}` first traceback at `{x['variant']}`:\n```\n{x['first_traceback']}\n```" for x in rows if x["first_traceback"]][:1]
    (out / "ID_BOUNDARY_PROBE.md").write_text("\n".join(L) + "\n"); return res


# ------------------------------------------------------------------------------------------ stages: safety audits and candidate-vs-baseline pairs
def stage_safety(pf, out, matches):
    out = Path(out); env = env_for(out); (out / "audits").mkdir(exist_ok=True)
    for k, v in pf.get("agents", {}).items():
        if not v.get("extracted"): continue
        a = Path(v["archive"]); rep = out / "audits" / f"{k}.json"
        r = subprocess.run([sys.executable, str(HERE.parent / "vendor_m3" / "submission_audit.py"), "audit", str(a), "--workdir", str(out / "audits" / f"{k}.work"), "--matches", str(matches), "--out", str(rep)],
                           capture_output=True, text=True, env=env, timeout=7200)
        (out / "audits" / f"{k}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr); (out / "audits" / "EXIT_CODES.txt").open("a").write(f"{k} exit={r.returncode}\n")
    subprocess.run([sys.executable, str(HERE / "flatten_audits.py"), "--audits", str(out / "audits"), "--out", str(out)], check=False)
    if (out / "forensics.json").exists(): write_forensics(out)


def stage_pairs(pf, out, matches, prefix="", deadline=3000):
    out = Path(out); env = env_for(out); ag = pf.get("agents", {}); cand = (ag.get("candidate") or {}).get("extracted")
    if not cand: raise RuntimeError("candidate not extracted")
    (out / "pairs").mkdir(exist_ok=True); mrows = []; grows = []; S = [f"# PAIR RESULTS ({matches} BO3 per pair, both seats) -- SAFETY COMPARISON ONLY, NOT A STRENGTH RESULT\n", "| pair | matches | W-D-L (candidate view) | mean match score [95% CI, conservative] | P(win) [CI] | cand faults | excluded | by seat |", "|---|---|---|---|---|---|---|---|"]
    sys.path.insert(0, str(HERE)); import ptcg_eval_canonical as C
    for other in ("Bv1", "B2", "Bv2"):
        o = (ag.get(other) or {}).get("extracted")
        if not o: S.append(f"| candidate vs {other} | NOT RUN (agent not extracted) | | | | | | |"); continue
        name = f"candidate_vs_{other}"; od = out / "pairs" / name
        r = subprocess.run([sys.executable, str(HERE / "tournament.py"), "run", "--cand", cand, "--control", o, "--h2h", str(matches), "--matches", "0", "--opp", f"x={o}:1", "--workers", "1",
                            "--deadline", str(deadline), "--out", str(out / "pairs" / f"{name}.jsonl"), "--outdir", str(od), "--stage", "none"], capture_output=True, text=True, env=env, timeout=deadline + 600)
        (out / "pairs" / f"{name}.log").write_text(r.stdout[-4000:] + "\n--- stderr ---\n" + r.stderr[-4000:])
        if r.returncode or not (od / "tournament_analysis.json").exists(): S.append(f"| candidate vs {other} | FAILED exit={r.returncode} (see pairs/{name}.log) | | | | | | |"); continue
        for src, dst in (("TOURNAMENT_MATCHES.csv", mrows), ("TOURNAMENT_GAMES.csv", grows)):
            rows = list(csv.reader(open(od / src))); dst.append((name, rows[0], rows[1:]))
        a = json.load(open(od / "tournament_analysis.json")); h = a["arms"]["h2h"]; f = h["field"]; cells = h["cells"]
        w = sum(c[0] for c in cells.values()); d = sum(c[1] for c in cells.values()); l = sum(c[2] for c in cells.values()); n = w + d + l
        wp = C.win_prob_ci(C.Cell(w, d, l)) if n else (0, 0, 0)
        S.append(f"| candidate vs {other} | {n} | {w}-{d}-{l} | {f.get('score', float('nan')):.3f} [{f.get('lo', float('nan')):.3f}, {f.get('hi', float('nan')):.3f}] | {wp[0]:.3f} [{wp[1]:.3f}, {wp[2]:.3f}] | {h['n_cand_fail']} | {h['n_excluded']} | " + "; ".join(f"{k}: {v[0]}-{v[1]}-{v[2]}" for k, v in cells.items()) + " |")
    for fn, data in (("TOURNAMENT_MATCHES.csv", mrows), ("TOURNAMENT_GAMES.csv", grows)):
        with open(out / (prefix + fn), "w", newline="") as f:
            w_ = csv.writer(f)
            if data:
                w_.writerow(["pair"] + data[0][1])
                for name, _, rows in data: w_.writerows([[name] + r for r in rows])
    (out / (prefix + "PAIR_RESULTS.md")).write_text("\n".join(S) + "\n"); return S


# ------------------------------------------------------------------------------------------ orchestration
def run_all(root, out, matches, synthetic=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); log = []; prefix = "SYNTHETIC_" if synthetic else ""
    if synthetic: (out / "SYNTHETIC_RUN.txt").write_text("HARNESS VALIDATION ONLY: this run used a synthetic bundle built by the test-suite; nothing here is evidence about any real agent.\n")
    def step(name, fn):
        try: r = fn(); log.append(f"{name}: OK"); return r
        except Exception as e: log.append(f"{name}: STOPPED -- {type(e).__name__}: {e}"); (out / f"{name}_ERROR.txt").write_text(traceback.format_exc()); return None
    pf = step("preflight", lambda: stage_preflight(root, out))
    if pf and pf.get("agents"):
        step("audit", lambda: stage_audit(pf, out)); step("sdk", lambda: stage_sdk(pf, out))
        step("safety", lambda: stage_safety(pf, out, matches)); step("idprobe", lambda: stage_idprobe(pf, out)); step("pairs", lambda: stage_pairs(pf, out, matches, prefix))
    (out / "RUN_LOG.txt").write_text("\n".join(log) + "\n"); print("\n".join(log)); return log


def main(argv=None):
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("all", "preflight"):
        p = sp.add_parser(n); p.add_argument("--root", required=True); p.add_argument("--out", required=True); p.add_argument("--matches", type=int, default=20); p.add_argument("--synthetic", action="store_true")
    for n in ("audit", "sdk", "idprobe", "safety", "pairs"):
        p = sp.add_parser(n); p.add_argument("--out", required=True); p.add_argument("--matches", type=int, default=20)
    p = sp.add_parser("_tables"); p.add_argument("lib")
    p = sp.add_parser("_idworker"); p.add_argument("agent"); p.add_argument("obs"); p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "_tables": print(json.dumps(_tables(a.lib))); return 0
    if a.cmd == "_idworker": _idworker(a.agent, a.obs, a.out); return 0
    if a.cmd == "all": run_all(a.root, a.out, a.matches, a.synthetic); return 0
    if a.cmd == "preflight": stage_preflight(a.root, a.out); return 0
    pf = json.load(open(Path(a.out) / "preflight.json"))
    {"audit": lambda: stage_audit(pf, a.out), "sdk": lambda: stage_sdk(pf, a.out), "idprobe": lambda: stage_idprobe(pf, a.out), "safety": lambda: stage_safety(pf, a.out, a.matches), "pairs": lambda: stage_pairs(pf, a.out, a.matches)}[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
