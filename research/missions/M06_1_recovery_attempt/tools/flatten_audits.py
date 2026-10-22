#!/usr/bin/env python3
"""Flatten submission_audit.py JSON reports into ARCHIVE_FORENSICS.csv and AGENT_SAFETY_AUDITS.csv (no new measurements)."""
import argparse, csv, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--audits", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
F = ["archive", "sha256", "bytes", "n_members", "deck_n", "deck_problems", "engine_accepts_deck", "card_table_sha256", "entrypoint", "cold_start_ok",
     "first_call_returns_60", "equals_deck_csv", "has___file__", "weights_max_deck_id", "n_weight_arrays", "safety_gate_passed"]
S = ["archive", "sha256", "matches_planned", "completed", "failed", "wins", "losses", "draws", "decisions", "illegal", "exceptions", "type_coerced",
     "fallback_counters", "fallback_log_lines", "tracebacks", "agent_p50_s", "agent_p95_s", "agent_p99_s", "agent_max_s", "overage_used_max_s",
     "hosts_rss_mb_max", "opponent_failures", "NOTE"]
rows_f, rows_s = [], []
for p in sorted(glob.glob(os.path.join(a.audits, "*.json"))):
    r = json.load(open(p)); ar = r.get("archive_report", {}); d = r.get("deck_audit", {}); c = r.get("cold_start", {}); m = r.get("matches", {}); w = r.get("weights_scan", {})
    name = os.path.basename(r.get("archive", p))
    rows_f.append([name, ar.get("sha256"), ar.get("bytes"), len(ar.get("members", [])), d.get("n"), ";".join(d.get("problems", [])), d.get("engine_accepts"),
                   d.get("card_table", {}).get("sha256"), c.get("entrypoint_resolved_by_sdk_rule"), c.get("ok"), c.get("returns_60"), c.get("equals_deck_csv"),
                   c.get("main_globals_have___file__"), w.get("max_deck_id"), len(w.get("arrays", [])), r.get("safety_gate_passed")])
    q = m.get("agent_s", {}) or {}
    rows_s.append([name, ar.get("sha256"), m.get("planned"), m.get("completed"), m.get("failed"), m.get("wins"), m.get("losses"), m.get("draws"), m.get("decisions"),
                   m.get("illegal"), m.get("exceptions"), m.get("type_coerced"), json.dumps(m.get("fallback_counters")), m.get("fallback_log_lines"), m.get("tracebacks"),
                   q.get("p50"), q.get("p95"), q.get("p99"), q.get("max"), m.get("overage_used_max_s"), m.get("hosts_rss_mb_max"), m.get("opponent_failures"),
                   "SAFETY/INSTRUMENTATION ONLY - self-play-vs-fixture results are NOT strength"])
for fn, hdr, rows in (("ARCHIVE_FORENSICS.csv", F, rows_f), ("AGENT_SAFETY_AUDITS.csv", S, rows_s)):
    with open(os.path.join(a.out, fn), "w", newline="") as f: csv.writer(f).writerows([hdr] + rows)
print(f"flattened {len(rows_f)} audit report(s)")
