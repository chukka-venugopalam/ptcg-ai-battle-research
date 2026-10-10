#!/usr/bin/env bash
# Fetch third-party sources at the pinned commits in SOURCE_MANIFEST.csv into ./third_party/.
# Read-only acquisition: nothing is executed. Skips the repo flagged UNSAFE unless --include-unsafe.
set -euo pipefail
cd "$(dirname "$0")"
INCLUDE_UNSAFE=0; [ "${1:-}" = "--include-unsafe" ] && INCLUDE_UNSAFE=1
mkdir -p third_party
python3 - "$INCLUDE_UNSAFE" <<'PY'
import csv, subprocess, sys, os
inc = sys.argv[1] == "1"
UNSAFE = {"Paschal-titre4015/ptcg-population-rl"}
for r in csv.DictReader(open("SOURCE_MANIFEST.csv")):
    repo, sha = r["repo"], r["pinned_commit_sha"]
    if repo in UNSAFE and not inc:
        print("SKIP (unsafe):", repo); continue
    d = os.path.join("third_party", repo.replace("/", "__"))
    if os.path.isdir(os.path.join(d, ".git")):
        print("exists:", d); continue
    os.makedirs(d, exist_ok=True)
    cmds = [["git", "init", "-q", d], ["git", "-C", d, "remote", "add", "origin", r["url"]],
            ["git", "-C", d, "fetch", "-q", "--depth", "1", "origin", sha],
            ["git", "-C", d, "checkout", "-q", "FETCH_HEAD"]]
    ok = all(subprocess.run(c).returncode == 0 for c in cmds)
    print("OK" if ok else "FAILED", repo, sha[:10])
PY
