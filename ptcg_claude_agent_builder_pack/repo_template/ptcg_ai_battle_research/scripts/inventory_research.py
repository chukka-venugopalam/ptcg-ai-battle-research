#!/usr/bin/env python3
"""Inventory cloned research repositories without importing or executing them."""
from __future__ import annotations
import argparse, csv, hashlib, pathlib

MODEL_EXT = {".pt", ".pth", ".onnx", ".npz", ".ckpt", ".safetensors", ".bin", ".joblib", ".pkl"}
NOTEBOOK_EXT = {".ipynb"}
DECK_NAMES = {"deck.csv", "decklist.csv"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("external/public_repos"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/manifests/research_artifacts.csv"))
    args = p.parse_args()
    rows = []
    if not args.root.exists():
        print(f"Missing: {args.root}")
        return 1
    for path in args.root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        ext = path.suffix.lower()
        kind = "file"
        if ext in MODEL_EXT:
            kind = "model"
        elif ext in NOTEBOOK_EXT:
            kind = "notebook"
        elif path.name.lower() in DECK_NAMES:
            kind = "deck"
        elif path.name.lower() in {"license", "license.md", "copying"} or path.name.lower().startswith("license."):
            kind = "license"
        elif path.name.lower() in {"readme", "readme.md", "readme.rst"}:
            kind = "documentation"
        rows.append({
            "path": str(path),
            "kind": kind,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path", "kind", "bytes", "sha256"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} artifact rows to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
