#!/usr/bin/env python3
"""submission_audit.py -- Phase 0 tooling for REAL submission archives (Bv1, R12, ...).  Nothing here has been run on your
actual archives yet: they were not attached to Mission 3.  It is validated on FIXTURE archives only (tests/test_audit.py).

  audit ARCHIVE.tar.gz [--workdir DIR] [--matches N] [--baseline DIR] [--out report.json]
        identity (sha256/size) -> tar safety + layout -> safe clean extraction + per-file sha256 -> deck audit against the
        engine's OWN card table (count, id, name, multiplicity, rules, engine acceptance) -> static risk scan (clip / modulo
        wrap / broad except / hard-coded paths / __file__) -> npz embedding-table bounds -> imports -> Kaggle-style cold
        start (exec without __file__, SDK last-callable rule) -> N real BO3 matches with latency / RAM / illegal / exception /
        fallback-counter / overage instrumentation -> SAFETY GATE verdict (exit 0 pass, 2 fail)
  diff  A B [C ...]   (archives or directories; first is the reference) tree + hash diff, unified line diffs, top-level constant
        ("rule flag") diffs, deck.csv multiset diff, .npz array diffs.
"""
from __future__ import annotations

import argparse
import ast
import collections
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bo3_ab_harness as H  # noqa: E402

MAX_MIB_DEFAULT = 197.7  # reported by the research package (PLAYGROUND_COMPETITION_AUDIT, S01); verify on the live Rules page
OBSERVED_DECK_ERRORS = {1: "unknown card id", 2: "copy limit exceeded", 4: "other deck-rule violation (no Basic Pokemon / ACE SPEC...)"}

RISK_PATTERNS = [
    (r"np\.clip\(|numpy\.clip\(", "clip(): can hide out-of-range card ids"),
    (r"%\s*(n_pool|N_POOL|pool\w*|POOL\w*|vocab\w*|VOCAB\w*|n_cards|NUM_CARDS|num_cards|n_emb\w*)\b", "modulo wrap of an id/index"),
    (r"\bmin\(\s*\w*(card|cid|idx|id)\w*\s*,", "min() clamp of an id/index"),
    (r"except\s*(\(?\s*(Base)?Exception\s*\)?)?\s*(as\s+\w+)?\s*:", "broad except (possible silent fallback)"),
    (r"\.get\(\s*\w*(card|cid)\w*\s*,\s*0\s*\)", "dict.get(card, 0): unknown ids silently map to 0"),
    (r"random\.(choice|sample|randrange|shuffle)|np\.random|numpy\.random", "random choice (possible fallback policy)"),
    (r"__file__", "uses __file__ (ABSENT when the SDK exec()s main.py)"),
    (r"/kaggle_simulations/agent", "hard-coded Kaggle agent directory"),
    (r"os\.getcwd\(\)|os\.chdir\(", "cwd dependence"),
    (r"torch\.load|np\.load|numpy\.load|pickle\.load|joblib\.load", "weights load (check path, allow_pickle, dependency)"),
    (r"allow_pickle\s*=\s*True", "allow_pickle=True"),
]


# ------------------------------------------------------------------ identity / archive
def sha256_file(p: str) -> str:
    return H.sha256_file(p)


def tar_report(archive: str, max_mib: float) -> dict:
    rep: dict = dict(path=os.path.abspath(archive), sha256=sha256_file(archive), bytes=os.path.getsize(archive), problems=[], warnings=[], members=[])
    try:
        tf = tarfile.open(archive, "r:*")
    except tarfile.TarError as e:
        rep["problems"].append(f"not a readable tar archive: {e}")
        return rep
    total = 0
    roots = set()
    with tf:
        for m in tf.getmembers():
            norm = os.path.normpath(m.name)
            flags = []
            if os.path.isabs(m.name):
                flags.append("absolute path")
            if norm.startswith("..") or ".." in norm.split(os.sep):
                flags.append("path traversal")
            if m.issym() or m.islnk():
                flags.append("symlink/hardlink")
            if m.isdev() or m.isfifo():
                flags.append("device/fifo")
            if flags:
                rep["problems"].append(f"{m.name}: {', '.join(flags)}")
            if m.isfile():
                total += m.size
                roots.add(norm.split(os.sep)[0] if os.sep in norm else norm)
            rep["members"].append(dict(name=m.name, size=m.size, type="dir" if m.isdir() else "file" if m.isfile() else "other"))
    names = {os.path.normpath(x["name"]) for x in rep["members"] if x["type"] == "file"}
    rep["uncompressed_bytes"] = total
    for need in ("main.py", "deck.csv"):
        if need not in names:
            nested = sorted(n for n in names if n.endswith(os.sep + need))
            rep["problems"].append(f"{need} not at archive root" + (f" (found nested: {nested[:3]} -> archive was probably built from the parent folder)" if nested else ""))
    if total / 2 ** 20 > max_mib:
        rep["problems"].append(f"uncompressed size {total / 2 ** 20:.1f} MiB > limit {max_mib} MiB")
    if rep["bytes"] / 2 ** 20 > max_mib:
        rep["problems"].append(f"archive size {rep['bytes'] / 2 ** 20:.1f} MiB > limit {max_mib} MiB")
    extra = sorted(n for n in names if n.endswith((".ipynb", ".ipynb_checkpoints")) or "__pycache__" in n or n.startswith("."))
    if extra:
        rep["warnings"].append(f"unrelated/hidden files in archive: {extra[:6]}")
    return rep


def safe_extract(archive: str, dest: str) -> List[dict]:
    if os.path.exists(dest):
        raise FileExistsError(f"{dest} exists: extraction must target a clean directory")
    os.makedirs(dest)
    base = os.path.abspath(dest)
    files = []
    with tarfile.open(archive, "r:*") as tf:
        for m in tf.getmembers():
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                continue
            out = os.path.normpath(os.path.join(base, m.name))
            if not (out == base or out.startswith(base + os.sep)):
                raise ValueError(f"path traversal blocked: {m.name}")
            if m.isdir():
                os.makedirs(out, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with tf.extractfile(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            files.append(dict(path=os.path.relpath(out, base), bytes=os.path.getsize(out), sha256=sha256_file(out)))
    return sorted(files, key=lambda f: f["path"])


# ------------------------------------------------------------------ deck audit
def read_deck_csv(path: str) -> Tuple[List[int], List[str]]:
    problems: List[str] = []
    ids: List[int] = []
    for ln, line in enumerate(open(path, encoding="utf-8-sig"), 1):
        for tok in re.split(r"[,\s;]+", line.strip()):
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                problems.append(f"line {ln}: non-integer token {tok!r} (header row or names?)")
    if len(ids) != 60:
        problems.append(f"deck has {len(ids)} entries (need 60)")
    return ids, problems


def deck_audit(ids: List[int], with_engine: bool = True) -> dict:
    rep: dict = dict(n=len(ids), table=[], problems=[], rules_checked=False, engine_accepts=None)
    if not with_engine:
        rep["problems"].append("NOT RUN: engine card table unavailable")
        return rep
    try:
        cards, sha = H.r2_card_table()
    except Exception as e:  # noqa: BLE001
        rep["problems"].append(f"NOT RUN: cannot load engine card table ({type(e).__name__}: {e})")
        return rep
    by_id = {c["cardId"]: c for c in cards}
    rep["card_table"] = dict(count=len(cards), id_range=[min(by_id), max(by_id)], sha256=sha)
    cnt = collections.Counter(ids)
    for cid, k in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        c = by_id.get(cid)
        rep["table"].append(dict(count=k, id=cid, name=c["name"] if c else None))
        if c is None:
            rep["problems"].append(f"card id {cid} x{k} is NOT in the engine card table (ids {min(by_id)}..{max(by_id)})")
    if any(by_id.get(c) is None for c in cnt):
        return rep
    rep["rules_checked"] = True
    by_name = collections.Counter()
    for cid, k in cnt.items():
        by_name[by_id[cid]["name"]] += k
    for name, k in by_name.items():
        is_basic_energy = any(by_id[c]["cardType"] == 5 and by_id[c]["name"].startswith("Basic ") for c in cnt if by_id[c]["name"] == name)
        if k > 4 and not is_basic_energy:
            rep["problems"].append(f"{k} copies of '{name}' (max 4 by NAME across ids)")
    if not any(by_id[c]["cardType"] == 0 and by_id[c]["basic"] for c in cnt):
        rep["problems"].append("no Basic Pokemon in the deck")
    ace = sum(k for c, k in cnt.items() if by_id[c]["aceSpec"])
    if ace > 1:
        rep["problems"].append(f"{ace} ACE SPEC cards (max 1)")
    if len(ids) == 60:
        try:
            from kaggle_environments.envs.cabt.cg.game import battle_start, battle_finish
            obs, sd = battle_start(ids, ids)
            rep["engine_accepts"] = obs is not None
            if obs is not None:
                battle_finish()
            else:
                rep["problems"].append(f"ENGINE REJECTS the deck: errorPlayer={sd.errorPlayer} errorType={sd.errorType} "
                                       f"({OBSERVED_DECK_ERRORS.get(sd.errorType, 'unknown code')})")
        except Exception as e:  # noqa: BLE001
            rep["problems"].append(f"NOT RUN: engine acceptance check failed to execute ({type(e).__name__}: {e})")
    return rep


# ------------------------------------------------------------------ static / weights scans
def static_scan(root: str, max_hits: int = 6) -> dict:
    hits: Dict[str, List[str]] = collections.defaultdict(list)
    imports = set()
    local = {os.path.splitext(f)[0] for _r, _d, fs in os.walk(root) for f in fs if f.endswith(".py")} | \
            {d for _r, ds, _f in os.walk(root) for d in ds}
    for r, _d, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(r, f)
            rel = os.path.relpath(fp, root)
            try:
                src = open(fp, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for i, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for rx, why in RISK_PATTERNS:
                    if re.search(rx, line) and len(hits[why]) < max_hits:
                        hits[why].append(f"{rel}:{i}: {stripped[:110]}")
            try:
                for node in ast.walk(ast.parse(src)):
                    if isinstance(node, ast.Import):
                        imports.update(a.name.split(".")[0] for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        imports.add(node.module.split(".")[0])
            except SyntaxError as e:
                hits["SYNTAX ERROR"].append(f"{rel}: {e}")
    std = getattr(sys, "stdlib_module_names", set())
    third = sorted(m for m in imports if m not in std and m not in local)
    return dict(risk_hints=dict(hits), third_party_imports=third, note="heuristic hints for human review, not verdicts")


def entry_static(main_py: str) -> dict:
    src = open(main_py, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    agent_line = next((n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "agent"), None)
    after = []
    if agent_line is not None:
        for n in tree.body:
            if n.lineno > agent_line and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                after.append(f"{n.name}@{n.lineno}")
            elif n.lineno > agent_line and isinstance(n, ast.ImportFrom):
                after.append("from " + (n.module or ".") + " import ... @" + str(n.lineno))
    return dict(agent_defined=agent_line is not None, agent_line=agent_line, callable_bindings_after_agent=after,
                warning=("the SDK takes the LAST callable in main.py's globals; these can hijack the entrypoint" if after else None))


def npz_scan(root: str, deck_ids: List[int]) -> dict:
    try:
        import numpy as np
    except ImportError:
        return dict(status="NOT RUN: numpy unavailable")
    out = []
    mx = max(deck_ids) if deck_ids else 0
    for r, _d, fs in os.walk(root):
        for f in fs:
            fp = os.path.join(r, f)
            rel = os.path.relpath(fp, root)
            if f.endswith((".pkl", ".pickle", ".pt", ".pth", ".joblib")):
                out.append(dict(file=rel, kind="pickle-based weights", hint="executes code on load; confirm dependency + Rules allow it"))
            elif f.endswith((".npz", ".npy")):
                try:
                    z = np.load(fp, allow_pickle=False)
                    arrays = {k: z[k] for k in (z.files if hasattr(z, "files") else [])} if f.endswith(".npz") else {"": z}
                except Exception as e:  # noqa: BLE001
                    out.append(dict(file=rel, error=f"{type(e).__name__}: {e}"[:120]))
                    continue
                for k, a in arrays.items():
                    ent = dict(file=rel, array=k, shape=list(a.shape), dtype=str(a.dtype), flags=[])
                    if re.search(r"emb|card|vocab|pool|table", k, re.I) or (a.ndim == 2 and 100 <= a.shape[0] <= 4096):
                        ent["flags"].append("possible card/id embedding table")
                        if a.ndim >= 1 and a.shape[0] <= mx:
                            ent["flags"].append(f"rows={a.shape[0]} <= max deck id {mx}: ids may index out of range or be clipped/wrapped")
                    out.append(ent)
    return dict(arrays=out, max_deck_id=mx, note="direct-index tables need rows > max id; hashed/pooled tables need the id->slot map checked in source")


# ------------------------------------------------------------------ cold start + matches
def _baseline_dir(workdir: str) -> str:
    d = os.path.join(workdir, "baseline_first_option")
    os.makedirs(d, exist_ok=True)
    from kaggle_environments.envs.cabt.cabt import deck
    with open(os.path.join(d, "deck.csv"), "w") as fh:
        fh.write("\n".join(map(str, deck)) + "\n")
    with open(os.path.join(d, "main.py"), "w") as fh:
        fh.write('DECK = [int(x) for x in open("deck.csv").read().split()]\n'
                 'def agent(obs):\n    sel = obs["select"]\n    if sel is None:\n        return DECK\n'
                 '    k = max(sel["minCount"], min(sel["maxCount"], len(sel["option"])))\n    return list(range(k))\n')
    return d


def cold_start(main_py: str, deck_ids: List[int], kpath: str, log_dir: str) -> dict:
    from kaggle_environments import make
    env = make("cabt", configuration={"bo": 3})
    env.reset()
    obs0 = dict(env.state[0].observation)
    cfg = dict(env.configuration)
    rep: dict = dict(first_observation_keys=sorted(obs0), ok=False)
    host = None
    try:
        host = H.AgentHost(main_py, "coldstart", kaggle_path_mode=kpath, log_dir=log_dir)
        t0 = time.perf_counter()
        reply = host.act(obs0, cfg, hard_timeout=120.0)
        rep["first_call_wall_s"] = round(time.perf_counter() - t0, 3)
        rep["load_s"], rep["agent_s"] = reply.get("load_s"), reply.get("agent_s")
        rep["call_ok"] = bool(reply.get("ok"))
        if not reply.get("ok"):
            rep["error"] = reply.get("error", "")[-600:]
        else:
            act = reply["action"]
            rep["returns_60"] = len(act) == 60
            rep["strict_int_list"] = bool(reply.get("strict_types"))
            rep["equals_deck_csv"] = act == deck_ids
            bad, _ = H.validate_action(obs0, act, {c["cardId"] for c in H.r2_card_table()[0]})
            rep["validation"] = bad
            rep["ok"] = rep["returns_60"] and rep["strict_int_list"] and not bad
        st = host.stats()
        rep["entrypoint_resolved_by_sdk_rule"] = st.get("entry")
        rep["main_globals_have___file__"] = st.get("has___file__")
        rep["entry_is_agent"] = st.get("entry") == "agent"
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"[:600]
    finally:
        if host:
            host.close()
    return rep


def _pool(xs: List[float], q: float) -> Optional[float]:
    return H._pct(xs, q)


def play_matches(extracted: str, baseline: str, n: int, workers: int, workdir: str, kpath: str, deadline: float) -> dict:
    out = os.path.join(workdir, "audit_matches.jsonl")
    log_dir = os.path.join(workdir, "agentlogs")
    os.makedirs(log_dir, exist_ok=True)
    jobs = H.build_jobs(extracted, None, {"baseline": (baseline, 1.0)}, n, 1, "real", kpath, log_dir, 0, keep_logs=True, keep_lat=True)
    res = H.run(jobs, workers, deadline, out, header={"audit": True, "engine": H.engine_fingerprint(), "kaggle_path_mode": kpath})
    agg = dict(planned=len(jobs), completed=len(res), failed=0, wins=0, losses=0, draws=0, decisions=0, illegal=0, exceptions=0,
               type_coerced=0, suspect_duplicates=0, nondone_statuses=0, tracebacks=0, fallback_counters={}, fallback_log_lines=0,
               hosts_rss_mb_max=0.0, worker_rss_mb_max=0.0, overage_used_max_s=0.0, opponent_failures=0, failure_examples=[])
    lat: List[float] = []
    rtt: List[float] = []
    for r in res:
        a = str(r["a_seat"])
        if r.get("failed"):
            agg["failed"] += 1
            if H.classify(r) == "fail_opp":
                agg["opponent_failures"] += 1
            if len(agg["failure_examples"]) < 3:
                agg["failure_examples"].append(dict(jid=r["jid"], kind=r.get("failure_kind"), fail_seats=r.get("fail_seats"), error=r.get("error"),
                                                    seat_exc=(r.get("per_seat", {}).get(a, {}) or {}).get("exception_examples")))
        else:
            agg[{"win": "wins", "loss": "losses", "draw": "draws"}[r["a_outcome"]]] += 1
        ps = (r.get("per_seat") or {}).get(a, {})
        if not ps:
            continue
        agg["decisions"] += ps.get("decisions", 0)
        agg["illegal"] += ps.get("illegal", 0)
        agg["exceptions"] += ps.get("exceptions", 0)
        agg["type_coerced"] += ps.get("type_coerced", 0)
        agg["suspect_duplicates"] += ps.get("suspect_duplicates", 0)
        if (r.get("statuses") or ["?", "?"])[r["a_seat"]] != "DONE":
            agg["nondone_statuses"] += 1
        sc = ps.get("stderr_scan") or {}
        agg["tracebacks"] += sc.get("traceback", 0)
        agg["fallback_log_lines"] += sc.get("fallback", 0)
        for k, v in ((ps.get("host") or {}).get("counters") or {}).items():
            if isinstance(v, (int, float)):
                agg["fallback_counters"][k] = max(agg["fallback_counters"].get(k, 0), v)
        agg["hosts_rss_mb_max"] = max(agg["hosts_rss_mb_max"], (ps.get("host") or {}).get("rss_mb") or 0)
        agg["worker_rss_mb_max"] = max(agg["worker_rss_mb_max"], r.get("worker_rss_mb") or 0)
        oe = (r.get("overage_end") or [None, None])[r["a_seat"]]
        if oe is not None:
            agg["overage_used_max_s"] = max(agg["overage_used_max_s"], round(600.0 - oe, 3))
        lat += ps.get("agent_s_raw", [])
        rtt += ps.get("rtt_s_raw", [])
    agg["agent_s"] = dict(p50=_pool(lat, .5), p95=_pool(lat, .95), p99=_pool(lat, .99), max=max(lat) if lat else None)
    agg["rtt_s"] = dict(p50=_pool(rtt, .5), p95=_pool(rtt, .95), p99=_pool(rtt, .99), max=max(rtt) if rtt else None)
    agg["results_file"] = out
    return agg


def cmd_audit(a) -> int:
    work = os.path.abspath(a.workdir or tempfile.mkdtemp(prefix="ptcg_audit_"))
    os.makedirs(work, exist_ok=True)
    rep: dict = dict(archive=os.path.abspath(a.archive), audit_started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     tool="submission_audit.py", note="engine-run sections use whatever kaggle-environments is installed here")
    gate: List[Tuple[str, bool, str]] = []
    tr = tar_report(a.archive, a.max_mib)
    rep["archive_report"] = tr
    gate.append(("archive layout/safety", not tr["problems"], "; ".join(tr["problems"]) or "ok"))
    if any(k in p for p in tr["problems"] for k in ("not a readable", "path traversal", "absolute path")):
        rep["extraction"] = "REFUSED: unreadable archive or unsafe member paths"
        return _finish(rep, gate, a)
    ex = os.path.join(work, "extracted")
    if os.path.exists(ex) and a.force:
        shutil.rmtree(ex)
    files = safe_extract(a.archive, ex)
    rep["files"] = files
    main_py, deck_csv = os.path.join(ex, "main.py"), os.path.join(ex, "deck.csv")
    ids: List[int] = []
    if os.path.isfile(deck_csv):
        ids, dprob = read_deck_csv(deck_csv)
        d = deck_audit(ids, with_engine=not a.no_engine)
        d["problems"] = dprob + d["problems"]
        rep["deck_audit"] = d
        gate.append(("deck: 60 valid ids, legal, accepted by engine", not d["problems"] and (a.no_engine or d.get("engine_accepts") is True),
                     "; ".join(d["problems"]) or "ok"))
    if os.path.isfile(main_py):
        rep["entry_static"] = entry_static(main_py)
    rep["static_scan"] = static_scan(ex)
    rep["weights_scan"] = npz_scan(ex, ids)
    if a.no_engine or not os.path.isfile(main_py):
        rep["cold_start"] = rep["matches"] = "NOT RUN (--no-engine or main.py missing)"
        return _finish(rep, gate, a)
    kpath = H.probe_kaggle_path_mode() if a.kaggle_path == "auto" else a.kaggle_path
    rep["kaggle_path_mode"] = kpath
    cs = cold_start(main_py, ids, kpath, os.path.join(work))
    rep["cold_start"] = cs
    gate.append(("cold start (no __file__, SDK entrypoint rule): first call returns the 60-card deck", bool(cs.get("ok")) and cs.get("entry_is_agent") is True,
                 json.dumps({k: cs.get(k) for k in ("error", "equals_deck_csv", "entrypoint_resolved_by_sdk_rule", "validation") if cs.get(k) not in (None, [], True)})[:300] or "ok"))
    if cs.get("ok") and cs.get("equals_deck_csv") is False:
        gate.append(("first-call deck equals deck.csv", False, "agent returns a different deck than deck.csv"))
    m = play_matches(ex, a.baseline or _baseline_dir(work), a.matches, a.workers, work, kpath, a.deadline)
    rep["matches"] = m
    fb = {k: v for k, v in m["fallback_counters"].items() if v}
    gate += [
        (f"all {m['planned']} BO3 matches completed with status DONE", m["completed"] == m["planned"] and m["failed"] == 0 and m["nondone_statuses"] == 0,
         f"completed={m['completed']} failed={m['failed']} non-DONE={m['nondone_statuses']} examples={m['failure_examples'][:1]}"),
        ("zero illegal responses / exceptions / type coercions", m["illegal"] == 0 and m["exceptions"] == 0 and m["type_coerced"] == 0,
         f"illegal={m['illegal']} exceptions={m['exceptions']} type_coerced={m['type_coerced']}"),
        ("zero unexpected fallbacks (agent-exposed counters + log scan)", not fb and m["fallback_log_lines"] == 0 and m["tracebacks"] == 0,
         f"counters>0={fb} fallback_log_lines={m['fallback_log_lines']} tracebacks={m['tracebacks']}  (counter names are heuristic: review)"),
        (f"p99 per-decision agent time <= {a.max_p99_s}s and worst-match overage use <= {a.max_overage_frac:.0%} of 600s",
         (m["agent_s"]["p99"] or 0) <= a.max_p99_s and m["overage_used_max_s"] <= a.max_overage_frac * 600,
         f"p99={m['agent_s']['p99']} max={m['agent_s']['max']} overage_used_max={m['overage_used_max_s']}s"),
    ]
    return _finish(rep, gate, a)


def _finish(rep: dict, gate: List[Tuple[str, bool, str]], a) -> int:
    rep["safety_gate"] = [dict(check=c, passed=p, detail=d) for c, p, d in gate]
    rep["safety_gate_passed"] = all(p for _c, p, _d in gate)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
    print(f"archive sha256: {rep['archive_report']['sha256']}  ({rep['archive_report']['bytes']} bytes)")
    d = rep.get("deck_audit")
    if d and d.get("table"):
        print(f"\ndeck ({d['n']} cards; engine card table: {d.get('card_table', {}).get('count')} cards)")
        for row in d["table"]:
            print(f"  {row['count']:>2} x {row['id']:>5}  {row['name']}")
    hints = (rep.get("static_scan") or {}).get("risk_hints", {})
    if hints:
        print("\nstatic risk hints (heuristic, for human review):")
        for why, lst in hints.items():
            print(f"  [{why}] {len(lst)}+ hit(s), e.g. {lst[0]}")
    print("\nSAFETY GATE")
    for g in rep["safety_gate"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['check']}  -- {g['detail'][:240]}")
    print("VERDICT:", "PASS" if rep["safety_gate_passed"] else "FAIL (a pass proves safety only, never strength)")
    return 0 if rep["safety_gate_passed"] else 2


# ------------------------------------------------------------------ diff
def _materialise(path: str, tmp: str, tag: str) -> str:
    if os.path.isdir(path):
        return os.path.abspath(path)
    dest = os.path.join(tmp, tag)
    safe_extract(path, dest)
    return dest


def _tree(root: str) -> Dict[str, str]:
    out = {}
    for r, _d, fs in os.walk(root):
        for f in fs:
            fp = os.path.join(r, f)
            out[os.path.relpath(fp, root)] = sha256_file(fp)
    return out


def py_constants(path: str) -> Dict[str, str]:
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return {}
    out = {}
    for n in tree.body:
        targets, value = [], None
        if isinstance(n, ast.Assign):
            targets, value = [t for t in n.targets if isinstance(t, ast.Name)], n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            targets, value = [n.target], n.value
        for t in targets:
            try:
                out[t.id] = repr(ast.literal_eval(value))[:160]
            except Exception:  # noqa: BLE001
                pass
    return out


def diff_pair(ra: str, rb: str, max_lines: int = 200) -> dict:
    ta, tb = _tree(ra), _tree(rb)
    res: dict = dict(only_in_a=sorted(set(ta) - set(tb)), only_in_b=sorted(set(tb) - set(ta)),
                     identical=sorted(k for k in ta if k in tb and ta[k] == tb[k]), changed=[])
    for k in sorted(k for k in ta if k in tb and ta[k] != tb[k]):
        ent: dict = dict(file=k, sha256_a=ta[k], sha256_b=tb[k])
        fa, fb = os.path.join(ra, k), os.path.join(rb, k)
        if k.endswith(".py"):
            la = open(fa, encoding="utf-8", errors="replace").read().splitlines()
            lb = open(fb, encoding="utf-8", errors="replace").read().splitlines()
            ud = list(difflib.unified_diff(la, lb, f"a/{k}", f"b/{k}", lineterm="", n=2))
            ent["added_lines"] = sum(1 for x in ud if x.startswith("+") and not x.startswith("+++"))
            ent["removed_lines"] = sum(1 for x in ud if x.startswith("-") and not x.startswith("---"))
            ent["unified_diff"] = ud[:max_lines] + ([f"... ({len(ud) - max_lines} more lines)"] if len(ud) > max_lines else [])
            ca, cb = py_constants(fa), py_constants(fb)
            ent["constant_changes"] = {n: (ca.get(n), cb.get(n)) for n in sorted(set(ca) | set(cb)) if ca.get(n) != cb.get(n)}
        elif k == "deck.csv":
            da, _ = read_deck_csv(fa)
            db, _ = read_deck_csv(fb)
            ca, cb = collections.Counter(da), collections.Counter(db)
            names = {}
            try:
                names = {c["cardId"]: c["name"] for c in H.r2_card_table()[0]}
            except Exception:  # noqa: BLE001
                pass
            ent["deck_multiset_diff"] = {f"{i} {names.get(i, '?')}": f"{ca.get(i, 0)} -> {cb.get(i, 0)}" for i in sorted(set(ca) | set(cb)) if ca.get(i, 0) != cb.get(i, 0)}
        elif k.endswith(".npz"):
            try:
                import numpy as np
                za, zb = np.load(fa, allow_pickle=False), np.load(fb, allow_pickle=False)
                arr = {}
                for name in sorted(set(za.files) | set(zb.files)):
                    if name not in za.files or name not in zb.files:
                        arr[name] = "only in " + ("a" if name in za.files else "b")
                    elif za[name].shape != zb[name].shape:
                        arr[name] = f"shape {za[name].shape} -> {zb[name].shape}"
                    else:
                        arr[name] = f"max|diff|={float(np.max(np.abs(za[name].astype('float64') - zb[name].astype('float64')))) if za[name].size else 0.0:.3g}"
                ent["npz_arrays"] = arr
            except Exception as e:  # noqa: BLE001
                ent["npz_arrays"] = f"NOT COMPARED: {type(e).__name__}"
        res["changed"].append(ent)
    return res


def cmd_diff(a) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        roots = [_materialise(p, tmp, f"t{i}") for i, p in enumerate(a.paths)]
        report = dict(reference=a.paths[0], comparisons=[])
        for p, r in zip(a.paths[1:], roots[1:]):
            d = diff_pair(roots[0], r, a.max_lines)
            d["vs"] = p
            report["comparisons"].append(d)
            print(f"\n=== {a.paths[0]}  vs  {p}\n  only in A: {d['only_in_a']}\n  only in B: {d['only_in_b']}\n  identical files: {len(d['identical'])}   changed: {[c['file'] for c in d['changed']]}")
            for c in d["changed"]:
                if "constant_changes" in c:
                    print(f"  [{c['file']}] +{c['added_lines']}/-{c['removed_lines']} lines; constants changed: {c['constant_changes'] or 'none'}")
                if "deck_multiset_diff" in c:
                    print(f"  [deck.csv] {c['deck_multiset_diff'] or 'same multiset (order only)'}")
                if "npz_arrays" in c:
                    print(f"  [{c['file']}] {c['npz_arrays']}")
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("audit")
    pa.add_argument("archive")
    pa.add_argument("--workdir")
    pa.add_argument("--force", action="store_true")
    pa.add_argument("--matches", type=int, default=8)
    pa.add_argument("--workers", type=int, default=1)
    pa.add_argument("--baseline")
    pa.add_argument("--deadline", type=float, default=1500.0)
    pa.add_argument("--max-mib", type=float, default=MAX_MIB_DEFAULT)
    pa.add_argument("--max-p99-s", type=float, default=1.0)
    pa.add_argument("--max-overage-frac", type=float, default=0.5)
    pa.add_argument("--kaggle-path", choices=("auto", "mount", "off"), default="auto")
    pa.add_argument("--no-engine", action="store_true", help="skip every step that needs the SDK")
    pa.add_argument("--out")
    pd = sub.add_parser("diff")
    pd.add_argument("paths", nargs="+")
    pd.add_argument("--max-lines", type=int, default=200)
    pd.add_argument("--out")
    a = ap.parse_args()
    return cmd_audit(a) if a.cmd == "audit" else cmd_diff(a)


if __name__ == "__main__":
    sys.exit(main())
