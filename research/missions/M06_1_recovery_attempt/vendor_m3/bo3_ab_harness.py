#!/usr/bin/env python3
"""bo3_ab_harness.py v2 -- seat-swapped, process-isolated BO3 evaluation for the cabt engine (repaired).

Replaces the legacy template (reserch2.zip files.zip).  Defects fixed (details + tests: HARNESS_AUDIT_AND_TESTS.md):
  D1  Queue.empty()-then-is_alive() race could record a finished match as "child exited without result"  -> poll_child()
  D2  os.chdir(<candidate dir>) AFTER building relative agent paths broke relative --cand paths and made the OPPONENT
      resolve deck/weights in the candidate's directory  -> every agent runs in its own subprocess, cwd = its own dir,
      absolute paths only, optional per-agent bind mount of /kaggle_simulations/agent
  D3  terminate() killed only the match process; native/agent child processes could leak -> process groups + SIGKILL
  D4  draws (reward 0/0) were silently counted as losses; rewards None crashed summarise()       -> W/L/D + failed
  D5  failed matches were dropped and weights renormalised around missing cells                  -> strict cells, gate fails
  D6  no game-level data, no per-decision latency, no illegal/exception accounting, no exit code   -> full records
  D7  no candidate-vs-control comparison and no CI for the DIFFERENCE                              -> compare / stratified MOVER
  D8  output file silently appended across runs                                                  -> refuses unless --append

Agents are loaded EXACTLY like the SDK loads file-path agents (kaggle_environments.agent.get_last_callable: exec in a
fresh dict => NO __file__ in globals, no chdir, last callable in the module wins).  The match itself runs on the real
`cabt` environment with configuration {"bo": 3}.  `--engine fake` exercises plumbing only (never evidence of strength).

    python bo3_ab_harness.py run --cand ./cand --control ./bv1 --opp hydra=./pilots/hydra:0.14 --matches 100 \
           --workers 2 --out run1.jsonl
    python bo3_ab_harness.py compare --results run1.jsonl --weights hydra=0.14 --csv CANDIDATE_COMPARISON.csv
    python bo3_ab_harness.py selftest
Exit codes: 0 ok | 3 runtime failure(s) of the candidate/control or incomplete cells | 4 lost jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue as _queue
import random
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ptcg_eval_stats as S  # noqa: E402

HARNESS_VERSION = "2.0"
KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
HARD_MARGIN_S = 5.0
COUNTER_RE = re.compile(r"(fallback|illegal|repair|exception|error|missing|degrad)", re.I)
STDERR_PATTERNS = {"traceback": re.compile(r"Traceback \(most recent call last\)"),
                   "fallback": re.compile(r"fall.?back", re.I), "error": re.compile(r"\berror\b", re.I),
                   "warning": re.compile(r"\bwarn", re.I)}


# =============================================================================== small utilities
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


_FP_CACHE: Dict[str, str] = {}


def dir_fingerprint(d: str) -> str:
    """sha256 over sorted (relative path, file sha256) of every file under d (ignores __pycache__/.pyc/.git)."""
    d = os.path.abspath(d)
    if d in _FP_CACHE:
        return _FP_CACHE[d]
    items = []
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", ".git"))
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            fp = os.path.join(root, f)
            items.append((os.path.relpath(fp, d), sha256_file(fp)))
    h = hashlib.sha256()
    for rel, s in items:
        h.update(rel.encode() + b"\0" + s.encode() + b"\n")
    _FP_CACHE[d] = h.hexdigest()
    return _FP_CACHE[d]


_CARDS: Optional[Tuple[list, str]] = None


def r2_card_table() -> Tuple[list, str]:
    """The engine's own card table (native AllCard).  Returns (cards, sha256 of raw json).  Never shipped in
    research packages: it is read at run time from the installed SDK."""
    global _CARDS
    if _CARDS is None:
        import ctypes
        from kaggle_environments.envs.cabt.cg.sim import lib
        lib.AllCard.restype = ctypes.c_char_p
        lib.AllCard.argtypes = []
        raw = lib.AllCard()
        _CARDS = (json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())
    return _CARDS


def engine_fingerprint() -> dict:
    out: dict = {}
    try:
        import importlib.metadata as md
        out["kaggle_environments"] = md.version("kaggle-environments")
    except Exception as e:  # noqa: BLE001
        out["kaggle_environments"] = f"unknown ({type(e).__name__})"
    try:
        from kaggle_environments.envs.cabt.cg import sim
        out["libcg_sha256"] = sha256_file(sim.lib_path)
        cards, sha = r2_card_table()
        out["card_count"] = len(cards)
        out["card_id_range"] = [min(c["cardId"] for c in cards), max(c["cardId"] for c in cards)]
        out["card_table_sha256"] = sha
        spec = json.load(open(os.path.join(os.path.dirname(sim.__file__), "..", "cabt.json")))
        out["cabt_bo_default"] = spec["configuration"]["bo"]["default"]
    except Exception as e:  # noqa: BLE001
        out["engine_error"] = f"{type(e).__name__}: {e}"[:300]
    return out


# =============================================================================== agent host (child side)
def _json_default(o):
    try:
        return int(o)
    except Exception:  # noqa: BLE001
        return repr(o)


def _vmhwm_mb() -> Optional[float]:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _agent_host_main(main_py: str) -> int:
    """Child process: load main.py the way the SDK does and serve JSON-line requests on stdin/stdout.
    The agent's own prints go to stderr (a log file in the parent)."""
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    from kaggle_environments.agent import get_last_callable
    from kaggle_environments.utils import structify
    raw = open(main_py, "r").read()
    st = {"fn": None, "load_s": None}

    def send(obj: dict) -> None:
        proto.write((json.dumps(obj, default=_json_default) + "\n").encode())

    send({"ready": True})  # SDK imported; from here on, latency is the agent's own (load + decisions)
    for line in sys.stdin.buffer:
        try:
            msg = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        cmd = msg.get("cmd")
        if cmd == "quit":
            break
        if cmd == "stats":
            fn = st["fn"]
            g = getattr(fn, "__globals__", {}) if fn else {}
            counters = {}
            for k, v in g.items():
                if k.startswith("__") or not COUNTER_RE.search(k):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    counters[k] = v
                elif isinstance(v, dict) and len(v) < 50 and all(isinstance(x, (int, float)) for x in v.values()):
                    counters[k] = dict(v)
            send({"ok": True, "counters": counters, "rss_mb": _vmhwm_mb(), "pid": os.getpid(),
                  "entry": getattr(fn, "__name__", None), "load_s": st["load_s"],
                  "has___file__": ("__file__" in g) if g else None})
            continue
        # cmd == "act"
        t0 = time.perf_counter()
        ok, err, action, strict = True, None, None, True
        try:
            if st["fn"] is None:
                tl = time.perf_counter()
                st["fn"] = get_last_callable(raw, path=main_py)
                st["load_s"] = time.perf_counter() - tl
            fn = st["fn"]
            args = [structify(msg["obs"]), structify(msg["cfg"])]
            if hasattr(fn, "__code__") and hasattr(fn.__code__, "co_argcount"):
                args = args[: fn.__code__.co_argcount]
            action = fn(*args)
        except BaseException:  # noqa: BLE001 -- a crash is data
            ok, err = False, traceback.format_exc()[-1500:]
        dt = time.perf_counter() - t0
        reply = {"ok": ok, "agent_s": dt, "load_s": st["load_s"]}
        if ok:
            try:
                strict = isinstance(action, list) and all(type(x) is int for x in action)
                reply["action"] = [int(x) for x in action]
                reply["strict_types"] = strict
                reply["action_type"] = type(action).__name__
            except Exception:  # noqa: BLE001
                reply["ok"], err = False, f"action is not a list of ints: {type(action).__name__}"
        if not reply["ok"]:
            reply["error"] = err
        send(reply)
    return 0


# =============================================================================== agent host (parent side)
class AgentHostError(RuntimeError):
    pass


class AgentHang(AgentHostError):
    pass


def _mount_wrap(agent_dir: str) -> List[str]:
    return ["unshare", "--mount", "--propagation", "private", "sh", "-c",
            f'mkdir -p {KAGGLE_AGENT_DIR} && mount --bind "$1" {KAGGLE_AGENT_DIR} && shift && exec "$@"', "sh", agent_dir]


def probe_kaggle_path_mode() -> str:
    """'mount' if a private bind mount of /kaggle_simulations/agent works here (root or CAP_SYS_ADMIN), else 'off'."""
    try:
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "marker"), "w").write("ok")
            r = subprocess.run(_mount_wrap(td) + ["cat", f"{KAGGLE_AGENT_DIR}/marker"], capture_output=True, timeout=20)
            return "mount" if (r.returncode == 0 and r.stdout.strip() == b"ok") else "off"
    except Exception:  # noqa: BLE001
        return "off"


class AgentHost:
    def __init__(self, main_py: str, name: str, kaggle_path_mode: str = "off", log_dir: Optional[str] = None):
        self.main = os.path.abspath(main_py)
        if not os.path.isfile(self.main):
            raise FileNotFoundError(self.main)
        self.dir = os.path.dirname(self.main)
        self.name = name
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--agent-host", self.main]
        if kaggle_path_mode == "mount":
            cmd = _mount_wrap(self.dir) + cmd
        self.err_path = os.path.join(log_dir or tempfile.gettempdir(), f"agent_{name}_{os.getpid()}_{int(time.time()*1e3)%10**8}.stderr")
        self._err = open(self.err_path, "wb")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PTCG_AGENT_DIR=self.dir)
        self.p = subprocess.Popen(cmd, cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                  bufsize=0, env=env)
        self._buf = b""
        try:
            hello = json.loads(self._read_line(90.0))
        except Exception as e:  # noqa: BLE001
            self.close()
            raise AgentHostError(f"{name}: host failed to start: {type(e).__name__}: {e}") from e
        if not hello.get("ready"):
            self.close()
            raise AgentHostError(f"{name}: unexpected hello {hello!r}")

    def _read_line(self, timeout: float) -> bytes:
        end = time.monotonic() + timeout
        fd = self.p.stdout.fileno()
        while b"\n" not in self._buf:
            left = end - time.monotonic()
            if left <= 0:
                raise AgentHang(f"{self.name}: no reply within {timeout:.1f}s")
            r, _, _ = select.select([fd], [], [], min(left, 0.5))
            if r:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    raise AgentHostError(f"{self.name}: host exited (rc={self.p.wait(timeout=2)})")
                self._buf += chunk
            elif self.p.poll() is not None:
                raise AgentHostError(f"{self.name}: host died (rc={self.p.returncode})")
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _send(self, obj: dict) -> None:
        try:
            self.p.stdin.write((json.dumps(obj, default=_json_default) + "\n").encode())
        except (BrokenPipeError, OSError) as e:
            raise AgentHostError(f"{self.name}: cannot write to host: {e}") from e

    def act(self, obs: dict, cfg: dict, hard_timeout: float) -> dict:
        self._send({"cmd": "act", "obs": obs, "cfg": cfg})
        return json.loads(self._read_line(hard_timeout))

    def stats(self) -> dict:
        try:
            self._send({"cmd": "stats"})
            return json.loads(self._read_line(10.0))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}

    def close(self) -> None:
        try:
            if self.p.poll() is None:
                try:
                    self._send({"cmd": "quit"})
                except AgentHostError:
                    pass
                try:
                    self.p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait(timeout=5)
        finally:
            for fh in (self.p.stdin, self.p.stdout, self._err):
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass


def scan_stderr(path: str, limit: int = 2_000_000) -> dict:
    try:
        txt = open(path, "rb").read(limit).decode("utf-8", "replace")
    except OSError:
        return {}
    return {k: len(rx.findall(txt)) for k, rx in STDERR_PATTERNS.items()}


# =============================================================================== action validation + seat recorder
def validate_action(obs: dict, action: list, r2_ids: Optional[set]) -> Tuple[List[str], List[str]]:
    """(violations, suspects).  violations = engine-illegal for sure; suspects = e.g. duplicate indices."""
    bad: List[str] = []
    sus: List[str] = []
    sel = obs.get("select")
    if not isinstance(action, list) or not all(type(i) is int for i in action):
        return ["action is not list[int]"], sus
    if sel is None:  # deck selection
        if len(action) != 60:
            bad.append(f"deck has {len(action)} cards (need 60)")
        if r2_ids is not None:
            unknown = sorted({i for i in action if i not in r2_ids})
            if unknown:
                bad.append(f"deck has ids not in the engine card table: {unknown[:8]}")
        return bad, sus
    n_opt = len(sel["option"])
    lo, hi = sel["minCount"], sel["maxCount"]
    if not (lo <= len(action) <= hi):
        bad.append(f"count {len(action)} outside [{lo},{hi}]")
    if any(i < 0 or i >= n_opt for i in action):
        bad.append(f"index out of range 0..{n_opt - 1}: {[i for i in action if i < 0 or i >= n_opt][:5]}")
    if len(set(action)) != len(action):
        sus.append("duplicate indices")
    return bad, sus


def _pct(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 6)


class SeatRecorder:
    def __init__(self, name: str, host: AgentHost, r2_ids: Optional[set]):
        self.name, self.host, self.r2_ids = name, host, r2_ids
        self.rtt: List[float] = []
        self.agent_s: List[float] = []
        self.illegal: List[str] = []
        self.suspect = 0
        self.exceptions: List[str] = []
        self.hang = False
        self.type_coerced = 0
        self.first_reply: Optional[dict] = None
        self.deck: Optional[List[int]] = None

    def make_agent(self):
        def agent(observation, configuration):  # two args -> the SDK passes (observation, configuration)
            t0 = time.perf_counter()
            obs = dict(observation)
            remaining = float(obs.get("remainingOverageTime", 600.0))
            try:
                reply = self.host.act(obs, dict(configuration), hard_timeout=max(1.0, remaining) + HARD_MARGIN_S)
            except AgentHang as e:
                self.hang = True
                self.exceptions.append(str(e))
                raise
            except AgentHostError as e:
                self.exceptions.append(str(e))
                raise
            self.rtt.append(time.perf_counter() - t0)
            self.agent_s.append(reply.get("agent_s") or 0.0)
            if self.first_reply is None:
                self.first_reply = {k: reply.get(k) for k in ("load_s", "agent_s")}
            if not reply.get("ok"):
                self.exceptions.append(reply.get("error", "unknown error")[-600:])
                raise RuntimeError(f"agent {self.name} raised: {reply.get('error', '')[-300:]}")
            action = reply["action"]
            if not reply.get("strict_types", True):
                self.type_coerced += 1
            bad, sus = validate_action(obs, action, self.r2_ids)
            if obs.get("select") is None and not bad:
                self.deck = list(action)
            if bad:
                self.illegal.append("; ".join(bad)[:300])
            self.suspect += len(sus)
            return action
        return agent

    def summary(self, host_stats: dict, raw: bool = False) -> dict:
        out = dict(
            decisions=len(self.rtt),
            rtt_s=dict(p50=_pct(self.rtt, .5), p95=_pct(self.rtt, .95), p99=_pct(self.rtt, .99),
                       max=round(max(self.rtt), 6) if self.rtt else None, sum=round(sum(self.rtt), 3)),
            agent_s=dict(p50=_pct(self.agent_s, .5), p95=_pct(self.agent_s, .95), p99=_pct(self.agent_s, .99),
                         max=round(max(self.agent_s), 6) if self.agent_s else None, sum=round(sum(self.agent_s), 3)),
            illegal=len(self.illegal), illegal_examples=self.illegal[:3], suspect_duplicates=self.suspect,
            exceptions=len(self.exceptions), exception_examples=[e[-300:] for e in self.exceptions[:2]],
            hang=self.hang, type_coerced=self.type_coerced, first_call=self.first_reply,
            host={k: host_stats.get(k) for k in ("ok", "counters", "rss_mb", "entry", "load_s", "has___file__", "error")})
        if raw:
            out["agent_s_raw"] = [round(x, 6) for x in self.agent_s[:20000]]
            out["rtt_s_raw"] = [round(x, 6) for x in self.rtt[:20000]]
        return out


# =============================================================================== game-level parsing
def parse_games(steps, final_result: Optional[list]) -> List[dict]:
    """Per-game info from the observation stream of the ACTIVE seat each step.
    Uses current.round / firstPlayer / players[j].win / draw (verified on kaggle-environments 1.33.0)."""
    rounds: Dict[int, dict] = {}
    for t, row in enumerate(steps):
        for seat in (0, 1):
            if getattr(row[seat], "status", None) != "ACTIVE":
                continue
            cur = (row[seat].observation or {}).get("current") if hasattr(row[seat], "observation") else None
            if not cur:
                continue
            r = cur.get("round")
            g = rounds.setdefault(r, {"round": r, "first_step": t, "first_player": None,
                                      "tally_at_start": [cur["players"][0]["win"], cur["players"][1]["win"], cur.get("draw", 0)]})
            fp = cur.get("firstPlayer", -1)
            if fp is not None and fp >= 0 and g["first_player"] is None:
                g["first_player"] = fp
    order = sorted(rounds)
    games = []
    for i, r in enumerate(order):
        g = rounds[r]
        nxt = rounds[order[i + 1]]["tally_at_start"] if i + 1 < len(order) else (list(final_result) if final_result else None)
        winner = None
        if nxt is not None:
            d = [nxt[j] - g["tally_at_start"][j] for j in range(3)]
            winner = {0: 0, 1: 1, 2: "draw"}.get(next((j for j in range(3) if d[j] > 0), None))
        games.append({"round": r, "first_player": g["first_player"], "winner": winner, "first_step": g["first_step"]})
    return games


def overage_stats(steps) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    mins: List[Optional[float]] = [None, None]
    ends: List[Optional[float]] = [None, None]
    for row in steps:
        for seat in (0, 1):
            try:
                v = row[seat].observation["remainingOverageTime"]
            except Exception:  # noqa: BLE001
                continue
            if v is None:
                continue
            ends[seat] = round(float(v), 3)
            mins[seat] = ends[seat] if mins[seat] is None else min(mins[seat], ends[seat])
    return mins, ends


# =============================================================================== one match (runs in a worker process)
def _base_record(job: dict) -> dict:
    return dict(jid=job["jid"], arm=job["arm"], label=job["label"], idx=job["idx"], a_seat=job["a_seat"],
                a_main=job["a_main"], b_main=job["b_main"], a_cfg=job.get("a_cfg"), b_cfg=job.get("b_cfg"),
                engine=job["engine"], harness_version=HARNESS_VERSION)


def _attribute_failures(rec: dict) -> None:
    """fail_seats: {seat: [reasons]} from statuses + per-seat instrumentation.  Strict by design."""
    fs: Dict[str, List[str]] = {}
    for seat in (0, 1):
        why: List[str] = []
        st = (rec.get("statuses") or ["?", "?"])[seat]
        if st != "DONE":
            why.append(f"status:{st}")
        ps = (rec.get("per_seat") or {}).get(str(seat), {})
        if ps.get("exceptions"):
            why.append("exception")
        if ps.get("hang"):
            why.append("hang")
        if ps.get("illegal"):
            why.append("illegal")
        if ps.get("type_coerced"):
            why.append("type_coerced")
        if why:
            fs[str(seat)] = why
    rec["fail_seats"] = fs
    rec["failed"] = bool(fs) or bool(rec.get("error"))
    rec["failure_kind"] = rec.get("failure_kind") or ("seat_failure" if fs else ("error" if rec.get("error") else None))


def _real_match(job: dict) -> dict:
    from kaggle_environments import make
    t0 = time.time()
    rec = _base_record(job)
    cards, _ = r2_card_table()
    ids = {c["cardId"] for c in cards}
    mains: List[Optional[str]] = [None, None]
    mains[job["a_seat"]] = job["a_main"]
    mains[1 - job["a_seat"]] = job["b_main"]
    hosts: List[AgentHost] = []
    recs: List[SeatRecorder] = []
    try:
        for seat in (0, 1):
            h = AgentHost(mains[seat], name=f"{job['jid'].replace(':', '_')}_s{seat}", kaggle_path_mode=job["kpath"], log_dir=job["log_dir"])
            hosts.append(h)
            recs.append(SeatRecorder(f"seat{seat}", h, ids))
        env = make("cabt", configuration={"bo": 3})
        env.run([r.make_agent() for r in recs])
        steps = env.steps
        fin = steps[-1]
        rec["statuses"] = [s.status for s in fin]
        rec["rewards"] = [s.reward for s in fin]
        rec["result"] = list(getattr(env, "result", None) or [])
        rec["n_steps"] = len(steps)
        rec["games"] = parse_games(steps, rec["result"])
        rec["overage_min"], rec["overage_end"] = overage_stats(steps)
        rec["decks"] = [r.deck for r in recs]
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        rec["per_seat"] = {}
        for seat in (0, 1):
            if seat < len(recs):
                ps = recs[seat].summary(hosts[seat].stats(), raw=job.get("keep_lat", False))
                ps["main"] = mains[seat]
                rec["per_seat"][str(seat)] = ps
        import resource
        rec["worker_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        for seat, h in enumerate(hosts):
            h.close()
            scan = scan_stderr(h.err_path)
            rec["per_seat"][str(seat)]["stderr_scan"] = scan
            keep = job.get("keep_logs") or rec.get("error") or scan.get("traceback")
            if keep:
                rec["per_seat"][str(seat)]["stderr_log"] = h.err_path
            else:
                try:
                    os.unlink(h.err_path)
                except OSError:
                    pass
    rec["wall_s"] = round(time.time() - t0, 2)
    _attribute_failures(rec)
    _outcome(rec)
    return rec


def _fake_match(job: dict) -> dict:
    """Plumbing-only simulation.  Never evidence of game strength."""
    t0 = time.time()
    rec = _base_record(job)
    rnd = random.Random(job["seed"])
    if rnd.random() < job["fake_crash_p"]:
        os._exit(17)
    if rnd.random() < job["fake_hang_p"]:
        time.sleep(10 ** 6)
    time.sleep(job.get("fake_sleep", 0.0))
    p = job["fake_p"]
    r = rnd.random()
    outcome = "draw" if r < job["fake_draw_p"] else ("win" if rnd.random() < p else "loss")
    rec_r = {"win": 1, "loss": -1, "draw": 0}[outcome]
    a = job["a_seat"]
    rewards = [0, 0]
    rewards[a], rewards[1 - a] = rec_r, -rec_r
    statuses = ["DONE", "DONE"]
    per_seat = {"0": {"decisions": 100, "illegal": 0, "exceptions": 0, "hang": False, "type_coerced": 0},
                "1": {"decisions": 100, "illegal": 0, "exceptions": 0, "hang": False, "type_coerced": 0}}
    if rnd.random() < job["fake_fail_p"]:
        statuses[a] = "ERROR"
        per_seat[str(a)]["exceptions"] = 1
        rewards[a], rewards[1 - a] = -1, 1
    n_g = 2 if outcome != "draw" else 3
    rec.update(statuses=statuses, rewards=rewards, per_seat=per_seat, n_steps=0, result=[0, 0, 0],
               games=[{"round": i + 1, "first_player": i % 2, "winner": None, "first_step": 0} for i in range(n_g)],
               overage_min=[500.0, 500.0], overage_end=[500.0, 500.0], wall_s=round(time.time() - t0, 3))
    _attribute_failures(rec)
    _outcome(rec)
    return rec


def _outcome(rec: dict) -> None:
    """'win'|'loss'|'draw' from the A agent's seat, or 'failed' if ANY seat failed or the match errored."""
    if rec.get("failed") or not rec.get("rewards"):
        rec["a_outcome"] = "failed"
        return
    r = rec["rewards"]
    a = rec["a_seat"]
    if r[a] is None or r[1 - a] is None:
        rec["a_outcome"] = "failed"
        rec["failed"] = True
        return
    rec["a_outcome"] = "win" if r[a] > r[1 - a] else ("loss" if r[a] < r[1 - a] else "draw")


def _worker_entry(job: dict, q) -> None:
    try:
        os.setsid()  # own process group: the parent can SIGKILL worker + agent hosts + any grandchildren together
    except OSError:
        pass
    try:
        rec = _fake_match(job) if job["engine"] == "fake" else _real_match(job)
    except BaseException as e:  # noqa: BLE001
        rec = _base_record(job)
        rec.update(error=f"{type(e).__name__}: {e}"[:500], failed=True, failure_kind="worker_exception", a_outcome="failed",
                   fail_seats={})
    q.put(rec)


# =============================================================================== scheduler (race-free)
def kill_group(p) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        p.kill()
    except Exception:  # noqa: BLE001
        pass
    try:
        p.join(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def poll_child(p, q, t0: float, deadline_s: float, drain_s: float = 3.0, get_timeout: float = 0.05,
               now=time.time, kill=kill_group):
    """One polling step.  NEVER trusts Queue.empty().  Returns (kind, payload) with kind in
    {'result','died','killed',None}.  If the child is gone, a bounded blocking drain recovers a result that landed
    between the read attempt and the liveness check (the legacy empty()/is_alive() race, defect D1)."""
    try:
        return "result", q.get(timeout=get_timeout)
    except _queue.Empty:
        pass
    if not p.is_alive():
        try:
            return "result", q.get(timeout=drain_s)
        except _queue.Empty:
            return "died", f"child exited rc={p.exitcode} without result"
    if now() - t0 > deadline_s:
        kill(p)
        return "killed", f"wall-clock deadline {deadline_s}s exceeded (process group killed)"
    return None, None


def _fail_record(job: dict, kind: str, msg: str) -> dict:
    rec = _base_record(job)
    rec.update(error=msg, failed=True, failure_kind={"died": "worker_died", "killed": "hang_killed"}.get(kind, kind),
               a_outcome="failed", fail_seats={}, statuses=None, rewards=None)
    return rec


def run(jobs: List[dict], workers: int, deadline_s: float, out_path: str, header: Optional[dict] = None,
        append: bool = False) -> List[dict]:
    ctx = mp.get_context("spawn")
    pending, running, results = list(jobs), [], []
    with open(out_path, "a" if append else "x") as fh:  # 'x': refuse to silently mix runs (defect D8)
        if header:
            fh.write(json.dumps({"_header": header}) + "\n")
        while pending or running:
            while pending and len(running) < workers:
                j = pending.pop(0)
                q = ctx.Queue()
                p = ctx.Process(target=_worker_entry, args=(j, q))
                p.start()
                running.append((p, q, j, time.time()))
            idle = True
            for item in list(running):
                p, q, j, t0 = item
                kind, payload = poll_child(p, q, t0, deadline_s)
                if kind is None:
                    continue
                idle = False
                rec = payload if kind == "result" else _fail_record(j, kind, payload)
                try:
                    p.join(timeout=5)
                finally:
                    if p.is_alive():
                        kill_group(p)
                    try:
                        os.killpg(p.pid, signal.SIGKILL)  # sweep leaked grandchildren (agent subprocesses)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                running.remove(item)
                results.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if idle:
                time.sleep(0.02)
    return results


# =============================================================================== job planning
def build_jobs(cand: str, control: Optional[str], opps: Dict[str, Tuple[str, float]], matches: int, seed0: int,
               engine: str, kpath: str, log_dir: str, h2h: int = 0, keep_logs: bool = False, fake: Optional[dict] = None,
               keep_lat: bool = False) -> List[dict]:
    fake = fake or {}
    defaults = dict(fake_p=0.5, fake_ctrl_p=0.5, fake_h2h_p=0.5, fake_fail_p=0.0, fake_hang_p=0.0, fake_crash_p=0.0, fake_draw_p=0.0, fake_sleep=0.0)
    defaults.update(fake)
    jobs: List[dict] = []

    def mk(arm, label, i, a_dir, b_dir, p_key):
        a_dir, b_dir = os.path.abspath(a_dir), os.path.abspath(b_dir)
        jid = f"{arm}:{label}:{i}"
        j = dict(jid=jid, arm=arm, label=label, idx=i, a_seat=i % 2,
                 a_main=os.path.join(a_dir, "main.py"), b_main=os.path.join(b_dir, "main.py"),
                 a_cfg=dir_fingerprint(a_dir), b_cfg=dir_fingerprint(b_dir), engine=engine, kpath=kpath, log_dir=log_dir,
                 keep_logs=keep_logs, keep_lat=keep_lat, seed=seed0 + zlib.crc32(jid.encode()) % 10 ** 9)
        j.update({k: v for k, v in defaults.items() if k not in ("fake_ctrl_p", "fake_h2h_p")})
        j["fake_p"] = defaults[p_key]
        return j

    arms = [("cand", cand, "fake_p")] + ([("ctrl", control, "fake_ctrl_p")] if control else [])
    for arm, adir, pk in arms:
        for label, (odir, _w) in opps.items():
            for i in range(matches):
                jobs.append(mk(arm, label, i, adir, odir, pk))
    if h2h and control:
        for i in range(h2h):
            jobs.append(mk("h2h", "control", i, cand, control, "fake_h2h_p"))
    random.Random(seed0).shuffle(jobs)  # interleave so a late crash does not bias one cell
    return jobs


# =============================================================================== analysis
def load_results(paths: List[str]) -> Tuple[List[dict], List[dict]]:
    recs, headers = [], []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "_header" in o:
                headers.append(o["_header"])
            else:
                recs.append(o)
    return recs, headers


def classify(rec: dict) -> str:
    """ok | fail_self | fail_opp | fail_unattributed   (self = the A agent: candidate / control / h2h-candidate)."""
    if not rec.get("failed"):
        return "ok"
    fs = rec.get("fail_seats") or {}
    a = str(rec["a_seat"])
    if a in fs:
        return "fail_self"
    if fs:
        return "fail_opp"
    return "fail_unattributed"


def cells_from_records(recs: List[dict], arm: str) -> Dict[Tuple[str, int], dict]:
    cells: Dict[Tuple[str, int], dict] = defaultdict(lambda: dict(W=0, L=0, D=0, fail_self=0, fail_opp=0, fail_unattr=0))
    for r in recs:
        if r["arm"] != arm:
            continue
        c = cells[(r["label"], r["a_seat"])]
        k = classify(r)
        if k == "ok":
            c[{"win": "W", "loss": "L", "draw": "D"}[r["a_outcome"]]] += 1
        elif k == "fail_self":
            c["fail_self"] += 1
        elif k == "fail_opp":
            c["fail_opp"] += 1
        else:
            c["fail_unattr"] += 1
    return cells


def to_stat_cells(cells: Dict[Tuple[str, int], dict], by_seat: bool = False) -> Dict:
    out: Dict = defaultdict(lambda: [0, 0])
    for (label, seat), c in cells.items():
        key = (label, seat) if by_seat else label
        out[key][0] += c["W"]
        out[key][1] += c["W"] + c["L"] + c["D"]
    return {k: (v[0], v[1]) for k, v in out.items()}


def write_comparison_csv(path: str, recs: List[dict], headers: List[dict], weights: Dict[str, float], arms: List[str]) -> None:
    import csv
    eng = (headers[0].get("engine") if headers else None) or {}
    eng_s = f"kaggle-environments {eng.get('kaggle_environments')}; cards={eng.get('card_count')}; libcg={str(eng.get('libcg_sha256'))[:12]}"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_arm", "config_hash", "opponent_archetype", "seat", "N_completed", "wins", "losses", "draws",
                    "win_rate", "ci95_lo", "ci95_hi", "runtime_failures_self", "failures_opponent", "failures_unattributed", "engine_build"])
        for arm in arms:
            cells = cells_from_records(recs, arm)
            cfg = next((r["a_cfg"] for r in recs if r["arm"] == arm), "")
            labels = sorted({k[0] for k in cells} | set(weights))
            for lab in labels:
                tot = dict(W=0, L=0, D=0, fail_self=0, fail_opp=0, fail_unattr=0)
                for seat in (0, 1, "all"):
                    if seat == "all":
                        c = tot
                    else:
                        c = cells.get((lab, seat), dict(W=0, L=0, D=0, fail_self=0, fail_opp=0, fail_unattr=0))
                        for kx in tot:
                            tot[kx] += c[kx]
                    n = c["W"] + c["L"] + c["D"]
                    lo, hi = S.wilson(c["W"], n) if n else (None, None)
                    w.writerow([arm, cfg, lab, seat, n, c["W"], c["L"], c["D"], round(c["W"] / n, 4) if n else "", 
                                round(lo, 4) if lo is not None else "", round(hi, 4) if hi is not None else "",
                                c["fail_self"], c["fail_opp"], c["fail_unattr"], eng_s])
            try:
                fr = S.weighted_rate_ci(to_stat_cells(cells), weights)
                w.writerow([arm, cfg, "__FIELD_WEIGHTED__", "all", fr["n"], "", "", "", round(fr["rate"], 4), round(fr["lo"], 4), round(fr["hi"], 4), "", "", "", eng_s])
            except S.MissingCellError as e:
                w.writerow([arm, cfg, "__FIELD_WEIGHTED__", "all", "", "", "", "", "INCOMPLETE", "", "", "", "", "", str(e)[:200]])
        if "cand" in arms and "ctrl" in arms:
            try:
                d = S.weighted_diff_ci(to_stat_cells(cells_from_records(recs, "cand")), to_stat_cells(cells_from_records(recs, "ctrl")), weights)
                w.writerow(["cand-minus-ctrl", "", "__FIELD_WEIGHTED_DIFF__", "all", f"{d['n_cand']}/{d['n_ctrl']}", "", "", "", round(d["diff"], 4), round(d["lo"], 4), round(d["hi"], 4), "", "", "", eng_s])
            except S.MissingCellError as e:
                w.writerow(["cand-minus-ctrl", "", "__FIELD_WEIGHTED_DIFF__", "all", "", "", "", "", "INCOMPLETE", "", "", "", "", "", str(e)[:200]])


def report(recs: List[dict], weights: Dict[str, float], csv_path: Optional[str] = None, headers: Optional[List[dict]] = None,
           gate: str = "none") -> int:
    arms = [a for a in ("cand", "ctrl", "h2h") if any(r["arm"] == a for r in recs)]
    exit_code = 0
    tot_fail_self = tot_fail_opp = tot_unattr = 0
    for arm in arms:
        if arm == "h2h":
            continue
        cells = cells_from_records(recs, arm)
        print(f"\n=== arm: {arm}   (config {next((r['a_cfg'] for r in recs if r['arm'] == arm), '')[:12]})")
        print(f"{'opponent':<22}{'W-L-D':>12}{'rate':>8}{'Wilson95':>18}{'seat0':>8}{'seat1':>8}{'fail_self':>10}{'fail_opp':>9}{'unattr':>7}")
        for lab in sorted({k[0] for k in cells} | set(weights)):
            c = [cells.get((lab, s), dict(W=0, L=0, D=0, fail_self=0, fail_opp=0, fail_unattr=0)) for s in (0, 1)]
            W, L, D = (sum(x[k] for x in c) for k in ("W", "L", "D"))
            n = W + L + D
            lo, hi = S.wilson(W, n) if n else (float("nan"), float("nan"))
            sr = lambda x: f"{x['W'] / (x['W'] + x['L'] + x['D']):.2f}" if (x['W'] + x['L'] + x['D']) else "n/a"  # noqa: E731
            fs, fo, fu = (sum(x[k] for x in c) for k in ("fail_self", "fail_opp", "fail_unattr"))
            tot_fail_self += fs if arm in ("cand", "ctrl") else 0
            tot_fail_opp += fo
            tot_unattr += fu
            print(f"{lab:<22}{f'{W}-{L}-{D}':>12}{(W / n if n else float('nan')):>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{sr(c[0]):>8}{sr(c[1]):>8}{fs:>10}{fo:>9}{fu:>7}")
        try:
            fr = S.weighted_rate_ci(to_stat_cells(cells), weights)
            print(f"field-weighted win rate: {fr['rate']:.3f}  (stratified MOVER 95% [{fr['lo']:.3f}, {fr['hi']:.3f}], N={fr['n']})")
        except S.MissingCellError as e:
            print(f"field-weighted win rate: INCOMPLETE -> {e}")
            exit_code = 3
    if "cand" in arms and "ctrl" in arms:
        cc, kc = to_stat_cells(cells_from_records(recs, "cand")), to_stat_cells(cells_from_records(recs, "ctrl"))
        try:
            d = S.weighted_diff_ci(cc, kc, weights)
            print(f"\ncandidate-minus-control (field-weighted, independent samples, stratified MOVER): {d['diff']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  N={d['n_cand']}/{d['n_ctrl']}")
            if gate in ("screen", "confirm"):
                g = S.gate_screen(cc, kc, weights) if gate == "screen" else S.gate_confirm(cc, kc, weights)
                print(f"GATE {gate.upper()}: {'PASS' if g['passed'] else 'FAIL/REJECT'}")
                for c in g.get("checks", []):
                    print(f"   [{'ok' if c[1] else 'XX'}] {c[0]}: {c[2]}")
                for r_ in g.get("rejected_reasons", []):
                    print(f"   [XX] {r_}")
                if not g["passed"]:
                    exit_code = exit_code or 5
        except S.MissingCellError as e:
            print(f"\ncandidate-minus-control: INCOMPLETE -> {e}")
            exit_code = 3
    if "h2h" in arms:
        c = cells_from_records(recs, "h2h")
        W = sum(v["W"] for v in c.values()); L = sum(v["L"] for v in c.values()); D = sum(v["D"] for v in c.values())
        n = W + L + D
        if n:
            lo, hi = S.wilson(W, n)
            print(f"\nhead-to-head candidate vs control: {W}-{L}-{D}  rate {W / n:.3f} Wilson95 [{lo:.3f},{hi:.3f}]  "
                  f"SPRT(H0 .50 vs H1 .55): {S.sprt(W, n, .5, .55)[3]}; SPRT(H0 .50 vs H1 .45): {S.sprt(W, n, .5, .45)[3]}")
        for k in ("fail_self", "fail_opp", "fail_unattr"):
            pass
    fh_fail = sum(1 for r in recs if r["arm"] == "h2h" and r.get("failed"))
    print(f"\nRUNTIME FAILURES  self(cand/ctrl)={tot_fail_self}  opponent={tot_fail_opp}  unattributed={tot_unattr}  h2h={fh_fail}   (acceptance requires 0 self/unattributed)")
    if tot_fail_self or tot_unattr or fh_fail:
        exit_code = 3
    if csv_path:
        write_comparison_csv(csv_path, recs, headers or [], weights, [a for a in arms if a != "h2h"])
        print(f"wrote {csv_path}")
    return exit_code


def summarise(results: List[dict], weights: Dict[str, float]) -> int:  # legacy-compatible name
    return report(results, weights)


# =============================================================================== CLI
def _parse_opps(specs: List[str]) -> Dict[str, Tuple[str, float]]:
    opps: Dict[str, Tuple[str, float]] = {}
    for spec in specs:
        lab, rest = spec.split("=", 1)
        path, _, w = rest.partition(":")
        opps[lab] = (os.path.abspath(path), float(w) if w else 1.0)
    return opps


def _cmd_run(a) -> int:
    engine = "fake" if (os.environ.get("PTCG_HARNESS_FAKE") == "1" and a.engine == "real") else a.engine
    opps = _parse_opps(a.opp)
    kpath = a.kaggle_path
    if engine == "real":
        if kpath == "auto":
            kpath = probe_kaggle_path_mode()
        if kpath == "off":
            print("WARNING: /kaggle_simulations/agent bind-mount unavailable; agents that hard-code that path will not find "
                  "their own files (possible SILENT FALLBACK).  Run as root / with CAP_SYS_ADMIN, or verify agents resolve "
                  "paths without it.", file=sys.stderr)
    else:
        kpath = "off"
    log_dir = a.out + ".agentlogs"
    os.makedirs(log_dir, exist_ok=True)
    fake = dict(fake_p=a.fake_p, fake_ctrl_p=a.fake_ctrl_p, fake_h2h_p=a.fake_h2h_p, fake_fail_p=a.fake_fail_p,
                fake_hang_p=a.fake_hang_p, fake_crash_p=a.fake_crash_p, fake_draw_p=a.fake_draw_p, fake_sleep=a.fake_sleep)
    jobs = build_jobs(a.cand, a.control, opps, a.matches, a.seed, engine, kpath, log_dir, a.h2h, a.keep_logs, fake)
    header = dict(harness_version=HARNESS_VERSION, argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  engine_mode=engine, kaggle_path_mode=kpath, n_jobs=len(jobs),
                  engine=engine_fingerprint() if engine == "real" else {"fake": True},
                  python=sys.version.split()[0], note="RNG of the native engine is not seedable: no paired-seed claims.")
    res = run(jobs, a.workers, a.deadline, a.out, header=header, append=a.append)
    weights = {k: v[1] for k, v in opps.items()}
    if len(res) != len(jobs):
        print(f"LOST JOBS: planned {len(jobs)} got {len(res)}", file=sys.stderr)
        return 4
    return report(res, weights, a.csv, [header], a.gate)


def _cmd_compare(a) -> int:
    recs, headers = load_results(a.results)
    weights = {}
    for spec in a.weights:
        k, v = spec.split("=", 1)
        weights[k] = float(v)
    if not weights:
        weights = {lab: 1.0 for lab in sorted({r["label"] for r in recs if r["arm"] in ("cand", "ctrl")})}
    return report(recs, weights, a.csv, headers, a.gate)


def _cmd_selftest(a) -> int:
    """Fake-engine plumbing self-test (NOT evidence of strength)."""
    with tempfile.TemporaryDirectory() as td:
        for n in ("cand", "ctrl", "opp1", "opp2"):
            os.makedirs(os.path.join(td, n))
            open(os.path.join(td, n, "main.py"), "w").write("def agent(o):\n    return []\n")
        out = os.path.join(td, "r.jsonl")
        opps = {"opp1": (os.path.join(td, "opp1"), .6), "opp2": (os.path.join(td, "opp2"), .4)}
        jobs = build_jobs(os.path.join(td, "cand"), os.path.join(td, "ctrl"), opps, 20, 1, "fake", "off", td, 10,
                          fake=dict(fake_p=.6, fake_ctrl_p=.5, fake_h2h_p=.55, fake_draw_p=.02))
        res = run(jobs, 2, 30.0, out, header={"fake": True})
        assert len(res) == len(jobs) and len({r["jid"] for r in res}) == len(jobs), "lost or duplicated jobs"
        print(f"fake selftest: {len(res)} jobs OK")
        return report(res, {k: v[1] for k, v in opps.items()})
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--agent-host":
        return _agent_host_main(argv[1])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    pr = sub.add_parser("run")
    pr.add_argument("--cand", required=True)
    pr.add_argument("--control", help="control agent dir (e.g. Bv1); runs against the SAME panel")
    pr.add_argument("--h2h", type=int, default=0, help="BO3 matches candidate-vs-control (needs --control)")
    pr.add_argument("--opp", action="append", required=True, help="label=dir[:weight]")
    pr.add_argument("--matches", type=int, default=100, help="BO3 matches per (arm, opponent); half per seat")
    pr.add_argument("--workers", type=int, default=2)
    pr.add_argument("--deadline", type=float, default=1500.0, help="wall-clock seconds per BO3 before the process group is killed")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--engine", choices=("real", "fake"), default="real")
    pr.add_argument("--kaggle-path", choices=("auto", "mount", "off"), default="auto")
    pr.add_argument("--keep-logs", action="store_true")
    pr.add_argument("--append", action="store_true")
    pr.add_argument("--gate", choices=("none", "screen", "confirm"), default="none")
    pr.add_argument("--csv")
    pr.add_argument("--out", required=True)
    for k, d in (("fake-p", .5), ("fake-ctrl-p", .5), ("fake-h2h-p", .5), ("fake-fail-p", 0.), ("fake-hang-p", 0.), ("fake-crash-p", 0.), ("fake-draw-p", 0.), ("fake-sleep", 0.)):
        pr.add_argument("--" + k, type=float, default=d)
    pc = sub.add_parser("compare")
    pc.add_argument("--results", nargs="+", required=True)
    pc.add_argument("--weights", nargs="*", default=[], help="label=weight")
    pc.add_argument("--csv")
    pc.add_argument("--gate", choices=("none", "screen", "confirm"), default="none")
    sub.add_parser("selftest")
    if argv and argv[0] not in ("run", "compare", "selftest", "-h", "--help"):
        argv = ["run"] + argv  # legacy flat-flag invocation
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return _cmd_run(a)
    if a.cmd == "compare":
        return _cmd_compare(a)
    if a.cmd == "selftest":
        return _cmd_selftest(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
