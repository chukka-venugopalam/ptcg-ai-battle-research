#!/usr/bin/env python3
"""Run one native-engine self-play smoke game from a submission archive.

Checks entrypoint exec without __file__, deck handshake, legal action indices,
Python-side fallback counters (when available), and optional embedding OOV guards.
This is an operational smoke test, not a win-rate evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import traceback


def _worker(conn, agent_dir: str):
    try:
        os.chdir(agent_dir)
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        namespace = {}
        code = compile(Path(agent_dir, "main.py").read_text(encoding="utf-8"), "main.py", "exec")
        exec(code, namespace)  # Mirrors Kaggle's exec-style runner: no __file__ is provided.
        agent = namespace.get("agent")
        if not callable(agent):
            raise RuntimeError("main.py did not define a callable agent")
        deck = agent({"select": None})
        if not isinstance(deck, (tuple, list)) or len(deck) != 60:
            raise RuntimeError(f"deck handshake must return 60 card IDs; got {type(deck).__name__}/{len(deck) if hasattr(deck, '__len__') else 'n/a'}")
        conn.send({"ready": True, "deck": list(deck), "deck_len": len(deck), "unique_ids": len(set(deck))})
        while True:
            obs = conn.recv()
            if obs is None:
                break
            try:
                action = agent(obs)
                conn.send({"ok": True, "action": action})
            except Exception as exc:
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        try:
            from sa import bcagent
            stats = dict(bcagent.STATS)
            conn.send({"stats": stats, "health": bcagent.health_line()})
        except Exception:
            conn.send({"stats": None})
    except Exception as exc:
        try:
            conn.send({"ready": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        conn.close()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _oov_probe(agent_dir: Path):
    sys.path.insert(0, str(agent_dir))
    import numpy as np
    from sa.policynet import Net
    with np.load(agent_dir / "sa" / "policy_net.npz") as archive:
        net = Net(archive)
    probes = {"card_1300": int(net._emb_row("card_emb", 1300)),
              "card_1431": int(net._emb_row("card_emb", 1431)),
              "attack_1755": int(net._emb_row("atk_emb", 1755))}
    for table, value in (("card_emb", probes["card_1300"]),
                         ("card_emb", probes["card_1431"]),
                         ("atk_emb", probes["attack_1755"])):
        if not 0 <= value < getattr(net, table).shape[0]:
            raise RuntimeError(f"unsafe embedding row for {table}: {value}")
    if net.lut is None:
        if probes["card_1300"] != 0 or probes["card_1431"] != 0 or probes["attack_1755"] != 0:
            raise RuntimeError(f"legacy OOV rows were not guarded to row 0: {probes}")
    return {"status": "PASS", "embedding_rows": {"card_emb": net.card_emb.shape[0], "atk_emb": net.atk_emb.shape[0]}, "probes": probes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--timeout", type=int, default=30, help="seconds to wait per agent action")
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--skip-oov", action="store_true")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        parser.error(f"archive not found: {archive}")
    raw = archive.read_bytes()
    with tempfile.TemporaryDirectory(prefix="ptcg_smoke_") as tmp:
        agent_dir = Path(tmp) / "agent"
        agent_dir.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                # Reuse the validator's path-safety assumptions before extraction.
                for member in tf.getmembers():
                    part = Path(member.name)
                    if part.is_absolute() or ".." in part.parts:
                        raise RuntimeError(f"unsafe archive path: {member.name}")
                tf.extractall(agent_dir, filter="data")
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "stage": "extract", "error": str(exc)}, indent=2))
            return 2

        oov = None
        if not args.skip_oov:
            try:
                oov = _oov_probe(agent_dir)
            except Exception as exc:
                print(json.dumps({"status": "FAIL", "stage": "oov_probe", "error": repr(exc), "traceback": traceback.format_exc()}, indent=2))
                return 1

        ctx = mp.get_context("spawn")
        workers, pipes = [], []
        start = time.time()
        try:
            for _ in range(2):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, str(agent_dir)), daemon=True)
                process.start()
                child.close()
                workers.append(process)
                pipes.append(parent)
            ready = []
            for index, pipe in enumerate(pipes):
                if not pipe.poll(120):
                    raise RuntimeError(f"agent {index} startup timeout")
                ready.append(pipe.recv())
            if not all(item.get("ready") for item in ready):
                raise RuntimeError("agent startup/deck handshake failed: " + json.dumps(ready, default=str))

            os.chdir(agent_dir)
            sys.path.insert(0, str(agent_dir))
            from cg.game import battle_start, battle_select, battle_finish
            obs, status_data = battle_start(ready[0]["deck"], ready[1]["deck"])
            if obs is None:
                raise RuntimeError(f"BattleStart rejected deck(s): player={getattr(status_data, 'errorPlayer', None)} type={getattr(status_data, 'errorType', None)}")
            actions, seat_counts = 0, [0, 0]
            while obs["current"].get("result", -1) == -1:
                selection = obs.get("select")
                if selection is None:
                    raise RuntimeError(f"null select before terminal result at action {actions}")
                seat = obs["current"].get("yourIndex")
                if seat not in (0, 1):
                    raise RuntimeError(f"unexpected seat index: {seat}")
                pipes[seat].send(obs)
                if not pipes[seat].poll(args.timeout):
                    raise RuntimeError(f"agent action timeout at action={actions}, seat={seat}")
                answer = pipes[seat].recv()
                if not answer.get("ok"):
                    raise RuntimeError(f"agent exception at action={actions}: {answer}")
                action = answer.get("action")
                options = selection.get("option") or []
                if not isinstance(action, (list, tuple)) or not all(isinstance(value, int) for value in action):
                    raise RuntimeError(f"non-integer action: {action!r}")
                if (len(action) < selection.get("minCount", 0)
                    or len(action) > selection.get("maxCount", 0)
                    or len(set(action)) != len(action)
                    or any(value < 0 or value >= len(options) for value in action)):
                    raise RuntimeError(f"illegal action at action={actions}: action={action}, options={len(options)}")
                obs = battle_select(list(action))
                actions += 1
                seat_counts[seat] += 1
                if actions >= args.max_actions:
                    raise RuntimeError(f"max actions exceeded ({args.max_actions})")
            result = obs["current"].get("result")
            battle_finish()
            for pipe in pipes:
                pipe.send(None)
            stats = []
            for index, pipe in enumerate(pipes):
                stats.append(pipe.recv() if pipe.poll(20) else {"status": "stats_timeout", "seat": index})
            failures = []
            for index, item in enumerate(stats):
                stat = item.get("stats") if isinstance(item, dict) else None
                if stat is not None and (stat.get("fallbacks", 0) != 0 or stat.get("net_missing", 0) != 0):
                    failures.append({"seat": index, "stats": stat})
            summary = {"status": "PASS" if not failures else "FAIL", "archive": str(archive), "sha256": _sha(archive),
                       "deck_len": ready[0]["deck_len"], "unique_card_ids": ready[0]["unique_ids"],
                       "native_game_result": result, "actions": actions, "actions_by_seat": seat_counts,
                       "elapsed_seconds": round(time.time()-start, 3), "fallback_or_missing_network_failures": failures,
                       "per_agent_stats": stats, "oov_probe": oov,
                       "interpretation": "One smoke game proves operational behavior only; it is not evidence of strategic strength."}
            print(json.dumps(summary, indent=2, default=str))
            return 0 if summary["status"] == "PASS" else 1
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2))
            return 1
        finally:
            for pipe in pipes:
                try: pipe.close()
                except Exception: pass
            for process in workers:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate(); process.join(timeout=2)

if __name__ == "__main__":
    raise SystemExit(main())
