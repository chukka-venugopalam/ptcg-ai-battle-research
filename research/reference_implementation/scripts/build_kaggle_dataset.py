#!/usr/bin/env python3
"""Create a Kaggle-friendly dataset folder from already-fetched research repositories.

Default behavior excludes .git metadata. Use --include-binaries to include model files
or other binary artifacts; only do that after checking upstream licensing.
"""
from __future__ import annotations
import argparse, pathlib, shutil

BINARY_EXTS = {".pt", ".pth", ".onnx", ".npz", ".ckpt", ".safetensors", ".bin", ".pkl", ".joblib"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=pathlib.Path, default=pathlib.Path("external/public_repos"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("kaggle_dataset/ptcg_public_research"))
    p.add_argument("--include-binaries", action="store_true")
    args = p.parse_args()
    if args.out.exists(): shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for repo in args.src.iterdir() if args.src.exists() else []:
        if not repo.is_dir(): continue
        dst = args.out / "repos" / repo.name
        for src in repo.rglob("*"):
            if ".git" in src.parts: continue
            if src.is_file() and (not args.include_binaries and src.suffix.lower() in BINARY_EXTS):
                continue
            rel = src.relative_to(repo)
            target = dst / rel
            if src.is_dir(): target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    print(f"Built: {args.out}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
