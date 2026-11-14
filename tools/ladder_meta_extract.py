#!/usr/bin/env python3
"""Compact ladder-meta extractor for Kaggle PTCG AI Battle Playground episode replays.

Reads episode JSON files (inside .zip archives, in folders, or loose) and writes ONE small JSON file
(roughly 0.5-1 MB per 1,600 matches) with, per Bo3 match: both teams, both deck hashes, rewards,
per-game winner / first player / turns, and thinking-time pool left. Plus every distinct 60-card deck
list. No third-party dependencies, Python >= 3.8.

Usage (in a Kaggle notebook cell, prefix with !):
  python ladder_meta_extract.py /kaggle/input/<dataset-with-episodes> -o ladder_meta_compact.json
  python ladder_meta_extract.py PATH --team "Your Team Name"   # also prints your matches by opponent deck
  python ladder_meta_extract.py PATH --inspect                 # prints the layout of the first episode, then exits
  python ladder_meta_extract.py --selftest                     # runs on a synthetic episode (no data needed)

The output contains deck lists: keep it private (competition rules restrict publishing deck recipes).

The replay layout is read the way Kelly Mao's scripts/meta_analysis.py (MIT, github.com/kellymao/...) reads
it: decks are the action of step 1 for each seat, rewards/teams come from the top level / info, game
results from the per-step `current.players[].win` counters.
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import zipfile

IS_FIRST_CTX, YES = 41, 1  # select context "go first?" and option type YES (same constants as the reference)


def deck_hash(deck):
    return hashlib.sha1(",".join(map(str, sorted(deck))).encode()).hexdigest()[:8]


def parse_episode(raw):
    ep = json.loads(raw)
    steps = ep["steps"]
    info = ep.get("info") or {}
    teams = info.get("TeamNames") or ["?", "?"]
    decks = [steps[1][s].get("action") if len(steps) > 1 else None for s in (0, 1)]
    games, pools, decisions = {}, {}, [0, 0]
    for i, step in enumerate(steps):
        for seat in (0, 1):
            a = step[seat]
            obs = a.get("observation") or {}
            cur = obs.get("current")
            if not cur:
                continue
            rnd = cur.get("round") or 1
            g = games.setdefault(rnd, {"first": -1, "turns": 0, "wins": None, "chooser": None, "chose_first": None})
            if cur.get("firstPlayer", -1) != -1:
                g["first"] = cur["firstPlayer"]
            g["turns"] = max(g["turns"], cur.get("turn") or 0)
            if g["wins"] is None:
                g["wins"] = [p.get("win") or 0 for p in cur["players"]]
            if a.get("status") == "ACTIVE":
                decisions[seat] += 1
                t = obs.get("remainingOverageTime")
                if t is not None:
                    pr = pools.setdefault(rnd, [[None, None], [None, None]])  # [first seen, last seen] per seat
                    if pr[0][seat] is None:
                        pr[0][seat] = t
                    pr[1][seat] = t
                sel = obs.get("select")
                if sel and sel.get("context") == IS_FIRST_CTX and i + 1 < len(steps):
                    act = steps[i + 1][seat].get("action")
                    if act:
                        g["chooser"], g["chose_first"] = seat, sel["option"][act[0]]["type"] == YES
    rewards = ep.get("rewards") or [None, None]
    rounds = sorted(games)
    out_games = []
    for k, rnd in enumerate(rounds):
        g = games[rnd]
        if k + 1 < len(rounds):
            nxt = games[rounds[k + 1]]["wins"]
            winner = next((s for s in (0, 1) if nxt[s] > g["wins"][s]), 2)
        else:
            winner = 2 if None in rewards or rewards[0] == rewards[1] else int(rewards[1] > rewards[0])
        out_games.append({"winner": winner, "first": g["first"], "turns": g["turns"],
                          "chooser": g["chooser"], "chose_first": g["chose_first"]})
    ok = [isinstance(d, list) and len(d) == 60 for d in decks]
    return {
        "episode": info.get("EpisodeId"),
        "teams": teams,
        "rewards": rewards,
        "statuses": ep.get("statuses"),
        "decks": [deck_hash(d) if o else None for d, o in zip(decks, ok)],
        "lists": {deck_hash(d): sorted(d) for d, o in zip(decks, ok) if o},
        "games": out_games,
        "pool_left_by_round": {str(r): v for r, v in pools.items()},
        "decisions": decisions,
        "module_version": ep.get("module_version"),
    }


def zip_members(zp):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.lower().endswith(".json"):
                yield ("zip", zp, n)


def iter_sources(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    if f.lower().endswith(".zip"):
                        yield from zip_members(fp)
                    elif f.lower().endswith(".json"):
                        yield ("file", fp, None)
        elif p.lower().endswith(".zip"):
            yield from zip_members(p)
        elif p.lower().endswith(".json"):
            yield ("file", p, None)
        else:
            print(f"skipping {p}: not a directory, .zip or .json", file=sys.stderr)


def read_raw(task):
    kind, path, name = task
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            return z.read(name)
    with open(path, "rb") as f:
        return f.read()


def work(task):
    try:
        return parse_episode(read_raw(task))
    except Exception as e:  # one corrupt file must not kill the run
        return {"error": f"{task[2] or task[1]}: {e!r}"}


def tree(x, depth=0, maxdepth=3, key="root"):
    """Layout of a JSON value without printing its content (keys + types + list lengths only)."""
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{key}: dict({len(x)})")
        if depth < maxdepth:
            for k in list(x)[:25]:
                tree(x[k], depth + 1, maxdepth, k)
    elif isinstance(x, list):
        print(f"{pad}{key}: list({len(x)})")
        if depth < maxdepth and x:
            tree(x[0], depth + 1, maxdepth, "[0]")
    else:
        print(f"{pad}{key}: {type(x).__name__}")


def summarize(matches, lists, team=None):
    valid = [m for m in matches if None not in m["decks"] and None not in (m["rewards"] or [None])]

    def score(m):
        r = m["rewards"]
        return 1.0 if r[0] > r[1] else 0.0 if r[0] < r[1] else 0.5

    st = collections.defaultdict(lambda: [0, 0.0])
    for m in valid:
        s = score(m)
        for seat, w in ((0, s), (1, 1 - s)):
            d = m["decks"][seat]
            st[d][0] += 1
            st[d][1] += w
    print(f"\nmatches parsed: {len(matches)}   with both decks + result: {len(valid)}   distinct decks: {len(lists)}")
    print("top decks by usage (hash  matches  share  raw-winrate):")
    tot = sum(v[0] for v in st.values())
    for d, (n, w) in sorted(st.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"  {d}  {n:5d}  {100 * n / max(1, tot):5.1f}%  {w / n:.3f}")
    if team:
        mine = [m for m in valid if team in m["teams"]]
        print(f"\nteam {team!r}: {len(mine)} matches")
        by = collections.defaultdict(lambda: [0, 0.0])
        for m in mine:
            seat = m["teams"].index(team)
            s = score(m) if seat == 0 else 1 - score(m)
            by[m["decks"][1 - seat]][0] += 1
            by[m["decks"][1 - seat]][1] += s
        for d, (n, w) in sorted(by.items(), key=lambda kv: -kv[1][0])[:20]:
            print(f"  vs deck {d}: {n:3d} matches, score {w:.1f}/{n}")
    pools = [v for m in matches for v in m.get("pool_left_by_round", {}).values()]
    if pools:
        firsts = [p[0][s] for p in pools for s in (0, 1) if p[0][s] is not None]
        lasts = [p[1][s] for p in pools for s in (0, 1) if p[1][s] is not None]
        if firsts:
            print(f"\nthinking pool at first decision of a game: min {min(firsts):.1f}s  max {max(firsts):.1f}s; "
                  f"at last decision: min {min(lasts):.1f}s  median {sorted(lasts)[len(lasts) // 2]:.1f}s")


def run(paths, out, team, workers):
    tasks = list(iter_sources(paths))
    print(f"{len(tasks)} episode files found", flush=True)
    if not tasks:
        sys.exit("nothing to read: pass a folder / .zip / .json that contains episode JSON files")
    matches, lists, errors = [], {}, []
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, tasks, chunksize=4)):
            if "error" in r:
                errors.append(r["error"])
                continue
            lists.update(r.pop("lists"))
            matches.append(r)
            if (k + 1) % 100 == 0:
                print(f"  parsed {k + 1}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"parsed {len(matches)} matches, {len(errors)} errors")
    for e in errors[:5]:
        print("  error:", e[:300])
    if not matches:
        sys.exit("no episode parsed: re-run with --inspect and send me its output")
    with open(out, "w") as f:
        json.dump({"matches": matches, "decks": lists, "errors": len(errors)}, f, separators=(",", ":"))
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")
    summarize(matches, lists, team)


def selftest():
    def seat(action=None, obs=None, status="ACTIVE"):
        return {"action": action, "observation": obs, "status": status}

    def cur(rnd, w0, w1, turn=3, first=0):
        return {"round": rnd, "firstPlayer": first, "turn": turn, "players": [{"win": w0}, {"win": w1}]}

    deckA, deckB = list(range(1, 31)) * 2, list(range(100, 130)) * 2
    steps = [[seat(obs=None), seat(obs=None)], [seat(deckA), seat(deckB)]]
    # game 1: seat 0 wins; game 2: seat 1 wins; game 3: seat 0 wins
    for rnd, (w0, w1) in enumerate([(0, 0), (1, 0), (1, 1)], start=1):
        o = {"current": cur(rnd, w0, w1), "remainingOverageTime": 60.0 - rnd,
             "select": {"context": IS_FIRST_CTX if rnd == 2 else 0, "option": [{"type": 0}, {"type": YES}]}}
        steps.append([seat([1], o), seat([0], o)])
        steps.append([seat([0], {"current": cur(rnd, w0, w1, turn=9), "remainingOverageTime": 50.0 - rnd}),
                      seat([0], {"current": cur(rnd, w0, w1, turn=9), "remainingOverageTime": 50.0 - rnd})])
    ep = {"steps": steps, "info": {"TeamNames": ["alpha", "beta"], "EpisodeId": 7}, "rewards": [1, -1],
          "statuses": ["DONE", "DONE"]}
    with tempfile.TemporaryDirectory() as d:
        zp = os.path.join(d, "x.zip")
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("ep7.json", json.dumps(ep))
            z.writestr("ep8.json", json.dumps({"broken": True}))
        out = os.path.join(d, "o.json")
        run([zp], out, "alpha", 1)
        res = json.load(open(out))
    m = res["matches"][0]
    assert len(res["matches"]) == 1 and res["errors"] == 1, "one good episode, one counted error"
    assert m["teams"] == ["alpha", "beta"] and m["rewards"] == [1, -1]
    assert [g["winner"] for g in m["games"]] == [0, 1, 0], m["games"]
    assert m["decks"][0] == deck_hash(deckA) and m["decks"][1] == deck_hash(deckB)
    assert res["decks"][m["decks"][0]] == sorted(deckA)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("-o", "--out", default="ladder_meta_compact.json")
    ap.add_argument("--team", default=None, help="your team name exactly as shown on the ladder")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.paths:
        ap.error("give at least one folder / .zip / .json (or --selftest)")
    if a.inspect:
        first = next(iter_sources(a.paths), None)
        if first is None:
            sys.exit("no .json / .zip found")
        ep = json.loads(read_raw(first))
        print("file:", first[1], first[2] or "")
        tree(ep)
        steps = ep.get("steps") or []
        for i in (0, 1, 2):
            if i < len(steps):
                print(f"-- steps[{i}] has {len(steps[i])} entries; entry 0:")
                tree(steps[i][0], 0, 3, f"steps[{i}][0]")
        return
    run(a.paths, a.out, a.team, a.workers)


if __name__ == "__main__":
    main()
