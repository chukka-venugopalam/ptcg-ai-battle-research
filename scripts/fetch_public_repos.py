#!/usr/bin/env python3
"""Fetch the public research repositories listed in the manifest.

This script intentionally clones upstream repositories as-is. It does not execute
third-party code. Review each repository's license before redistributing any
snapshot or model artifact.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
from typing import Iterable

REPO_TYPES = {"repo"}


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def read_repos(manifest: pathlib.Path) -> Iterable[tuple[str, str, str]]:
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source_type") in REPO_TYPES:
                yield row["name"], row["url"], row.get("notes", "")


def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("data/manifests/public_research_sources.csv"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("external/public_repos"))
    p.add_argument("--fresh", action="store_true", help="delete existing repository directories before cloning")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, url, _notes in read_repos(args.manifest):
        dest = args.out / slugify(name)
        if dest.exists() and args.fresh:
            shutil.rmtree(dest)
        if (dest / ".git").exists():
            run(["git", "-C", str(dest), "pull", "--ff-only"])
        elif not dest.exists():
            run(["git", "clone", "--depth", "1", url, str(dest)])
        else:
            raise RuntimeError(f"Refusing to overwrite non-git directory: {dest}")

    print("\nDone. Review licenses before copying snapshots or model files elsewhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
