#!/usr/bin/env python3
"""tournament.py -- Mission 6 driver around Mission 3's bo3_ab_harness (vendor_m3/, byte-identical copy, hashes in
VENDOR_SHA256.txt).  Adds: (1) HARD failure when the real engine is unavailable or the harness fake-engine switch is set,
(2) export of one row per BO3 MATCH and a separate per-GAME diagnostic file, (3) analysis with the canonical BO3-match
statistics (ptcg_eval_canonical), never the legacy game-level gates, (4) BO3-rule cross-check against engine rewards.
    tournament.py run --cand DIR [--control DIR] --opp label=DIR:weight ... --matches N --out RUN.jsonl --outdir DIR
    tournament.py export --results RUN.jsonl [...] --weights label=w ... --outdir DIR [--stage none|screen|confirm --fresh]"""
import argparse, csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "vendor_m3"))
import ptcg_eval_canonical as C


class EngineUnavailable(RuntimeError):
    pass


def require_engine() -> dict:
    """Raise EngineUnavailable unless the REAL cabt engine loads.  Never returns a fallback."""
    if os.environ.get("PTCG_HARNESS_FAKE") == "1":
        raise EngineUnavailable("PTCG_HARNESS_FAKE=1: the harness would silently switch to its FAKE engine")
    try:
        import kaggle_environments  # noqa: F401
        import bo3_ab_harness as h
    except Exception as e:
        raise EngineUnavailable(f"engine import failed: {type(e).__name__}: {e}") from e
    fp = h.engine_fingerprint()
    if "engine_error" in fp or not fp.get("libcg_sha256") or not fp.get("card_count"):
        raise EngineUnavailable(f"engine fingerprint failed: {fp}")
    return fp


def to_match(rec: dict, h) -> C.MatchRecord:
    cls = h.classify(rec)
    seat = rec["a_seat"]
    if cls == "ok":
        return C.MatchRecord(rec["jid"], rec["label"], seat, {"win": "W", "loss": "L", "draw": "D"}[rec["a_outcome"]], "ok")
    if cls == "fail_self":
        return C.MatchRecord(rec["jid"], rec["label"], seat, "L", "cand_fail")      # counted as a LOSS
    if cls == "fail_opp":
        return C.MatchRecord(rec["jid"], rec["label"], seat, "W", "opp_fail")       # excluded, reported
    return C.MatchRecord(rec["jid"], rec["label"], seat, "L", "infra_fail")         # unattributed: excluded, reported


def a_view_games(rec: dict):
    """Game winners (absolute seat 0/1/'draw') -> W/L/D from the evaluated agent's seat."""
    a = rec["a_seat"]
    return [("D" if g["winner"] == "draw" else "W" if g["winner"] == a else "L") for g in rec.get("games") or [] if g.get("winner") is not None]


def export(paths, weights, outdir, stage="none", fresh=False):
    import bo3_ab_harness as h
    import zlib
    recs, headers = h.load_results(paths)
    os.makedirs(outdir, exist_ok=True)
    argv = (headers[0].get("argv") if headers else None) or []
    seed0 = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0   # harness job seed = seed0 + crc32(jid); NOT passed to the native engine (not seedable)
    mism = 0
    with open(os.path.join(outdir, "TOURNAMENT_MATCHES.csv"), "w", newline="") as fm, open(os.path.join(outdir, "TOURNAMENT_GAMES.csv"), "w", newline="") as fg:
        wm = csv.writer(fm); wg = csv.writer(fg)
        wm.writerow(["jid", "arm", "opponent", "agent_seat", "agent_hash", "opponent_hash", "harness_job_seed_NOT_ENGINE_SEED", "bo3_outcome", "match_points", "games_agent_view",
                     "first_player_by_game", "completion", "failure_kind", "fail_seats", "bo3_rule_ok", "per_seat_json", "wall_s", "log_paths"])
        wg.writerow(["jid", "game_index", "first_player_abs", "winner_abs", "agent_view", "first_step"])
        for r in recs:
            cls = h.classify(r); gv = a_view_games(r)
            ok = None
            if cls == "ok" and gv:
                try: ok = C.match_result_from_games(gv) == {"win": "W", "loss": "L", "draw": "D"}[r["a_outcome"]]
                except ValueError: ok = False
                mism += ok is False
            pts = {"win": 1, "draw": 0.5, "loss": 0}.get(r.get("a_outcome"), "")
            logs = [v.get("stderr_log") for v in (r.get("per_seat") or {}).values() if v.get("stderr_log")]
            wm.writerow([r["jid"], r["arm"], r["label"], r["a_seat"], r.get("a_cfg"), r.get("b_cfg"), seed0 + zlib.crc32(r["jid"].encode()) % 10 ** 9, r.get("a_outcome"), pts, "".join(gv),
                         "".join(str(g.get("first_player")) for g in r.get("games") or []), cls, r.get("failure_kind") or "",
                         json.dumps(r.get("fail_seats") or {}), ok, json.dumps(r.get("per_seat") or {}, default=str), r.get("wall_s"), ";".join(logs)])
            for i, g in enumerate(r.get("games") or []):
                wg.writerow([r["jid"], i + 1, g.get("first_player"), g.get("winner"), (gv[i] if i < len(gv) else ""), g.get("first_step")])
    out = {"n_records": len(recs), "bo3_rule_mismatches": mism, "arms": {}}
    tallies = {}
    for arm in sorted({r["arm"] for r in recs}):
        t = C.tally(to_match(r, h) for r in recs if r["arm"] == arm); tallies[arm] = t
        w = weights if arm != "h2h" else {"control": 1.0}
        try: fs = C.field_score_ci(t.cells, w)
        except C.MissingCellError as e: fs = {"incomplete": str(e)}
        out["arms"][arm] = dict(n_ok=t.n_ok, n_cand_fail=t.n_cand_fail, n_excluded=t.n_excluded, field=fs,
                                cells={f"{o}|seat{s}": [c.w, c.d, c.l] for (o, s), c in sorted(t.cells.items())})
    if "cand" in tallies and "ctrl" in tallies:
        try:
            d = C.diff_ci(tallies["cand"].cells, tallies["ctrl"].cells, weights); d.pop("per_cell"); out["cand_minus_ctrl"] = d
            if stage == "screen": g = C.screen(tallies["cand"], tallies["ctrl"], weights)
            elif stage == "confirm": g = C.confirm(tallies["cand"], tallies["ctrl"], weights, fresh_matches=fresh)
            else: g = None
            if g: out["gate"] = dict(stage=stage, passed=g.passed, reasons=list(g.reasons))
        except (C.MissingCellError, C.DesignError) as e:
            out["gate"] = dict(stage=stage, passed=False, reasons=[f"{type(e).__name__}: {e}"])
    json.dump(out, open(os.path.join(outdir, "tournament_analysis.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--cand", required=True); r.add_argument("--control"); r.add_argument("--h2h", type=int, default=0)
    r.add_argument("--opp", action="append", required=True); r.add_argument("--matches", type=int, default=100); r.add_argument("--workers", type=int, default=2)
    r.add_argument("--deadline", type=float, default=1500.0); r.add_argument("--seed", type=int, default=0); r.add_argument("--out", required=True)
    r.add_argument("--outdir", required=True); r.add_argument("--stage", choices=("none", "screen", "confirm"), default="none"); r.add_argument("--fresh", action="store_true")
    e = sp.add_parser("export"); e.add_argument("--results", nargs="+", required=True); e.add_argument("--weights", nargs="+", required=True)
    e.add_argument("--outdir", required=True); e.add_argument("--stage", choices=("none", "screen", "confirm"), default="none"); e.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        try: fp = require_engine()
        except EngineUnavailable as ex: print(f"HARD FAIL: tournament requested but engine unavailable: {ex}", file=sys.stderr); return 5
        import bo3_ab_harness as h
        opps = h._parse_opps(a.opp); kp = h.probe_kaggle_path_mode()
        os.makedirs(a.outdir, exist_ok=True); logd = a.out + ".agentlogs"; os.makedirs(logd, exist_ok=True)
        jobs = h.build_jobs(a.cand, a.control, opps, a.matches, a.seed, "real", kp, logd, a.h2h, False, {})
        hdr = dict(harness="vendor_m3/bo3_ab_harness.py", engine=fp, kaggle_path_mode=kp, n_jobs=len(jobs), argv=sys.argv, python=sys.version.split()[0])
        res = h.run(jobs, a.workers, a.deadline, a.out, header=hdr)
        if len(res) != len(jobs): print(f"LOST JOBS {len(jobs)} -> {len(res)}", file=sys.stderr); return 4
        export([a.out], {k: v[1] for k, v in opps.items()}, a.outdir, a.stage, a.fresh); return 0
    weights = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.weights}
    export(a.results, weights, a.outdir, a.stage, a.fresh); return 0


if __name__ == "__main__":
    sys.exit(main())
