"""Verify fetched checkouts against SOURCE_MANIFEST.csv and ARTIFACT_HASHES.csv."""
import csv, hashlib, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()
bad = checked = missing = 0
for r in csv.DictReader(open(os.path.join(here, "SOURCE_MANIFEST.csv"))):
    d = os.path.join(here, "third_party", r["repo"].replace("/", "__"))
    if not os.path.isdir(d): missing += 1; continue
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    checked += 1
    if head != r["pinned_commit_sha"]: bad += 1; print("SHA MISMATCH", r["repo"], head[:10])
for a in csv.DictReader(open(os.path.join(here, "ARTIFACT_HASHES.csv"))):
    p = os.path.join(here, "third_party", a["repo"].replace("/", "__"), a["path"])
    if not os.path.isfile(p): continue
    checked += 1
    if sha256(p) != a["sha256"]: bad += 1; print("HASH MISMATCH", a["repo"], a["path"])
print(f"checked={checked} mismatches={bad} repos_not_fetched={missing}")
sys.exit(1 if bad else 0)
