#!/usr/bin/env python3
"""Create a consistent research mission folder and register it."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "registries" / "mission_registry.csv"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--status",
        choices=["planned", "in_progress", "blocked", "completed", "not_run"],
        default="planned",
    )
    args = parser.parse_args()

    mission_id = f"M{args.id:02d}"
    folder = ROOT / "research" / "missions" / f"{mission_id}_{slug(args.title)}"

    # Check both registry and directory before creating anything, so a duplicate
    # ID cannot leave behind an orphan scaffold.
    rows: list[list[str]] = []
    if REGISTRY.exists():
        with REGISTRY.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    if not rows:
        rows = [["mission_id", "title", "status", "source_artifact", "summary", "primary_gap"]]
    if any(row and row[0].strip() == mission_id for row in rows[1:]):
        print(f"ERROR: {mission_id} already registered", file=sys.stderr)
        return 2
    if folder.exists():
        print(f"ERROR: folder already exists: {folder}", file=sys.stderr)
        return 2

    today = datetime.date.today().isoformat()
    folder.mkdir(parents=True)
    (folder / "REPORT.md").write_text(
        f"""# {mission_id} — {args.title}

- Status: {args.status}
- Date created: {today}
- Input SHA-256:
- Output SHA-256:

## Objective

## Inputs verified

## Commands/tests actually run

## Results

## Limitations

## Decision

## Reproduction
""",
        encoding="utf-8",
    )
    (folder / "results.csv").write_text(
        "experiment_id,baseline_hash,candidate_hash,opponent,seed,seat,result,actions,fallbacks,errors,elapsed_s\n",
        encoding="utf-8",
    )
    (folder / "run_manifest.json").write_text(
        json.dumps(
            {
                "mission_id": mission_id,
                "title": args.title,
                "status": args.status,
                "created_date": today,
                "inputs": [],
                "commands_run": [],
                "output_files": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows.append(
        [
            mission_id,
            args.title,
            args.status,
            str(folder.relative_to(ROOT)),
            "Scaffold generated; no results claimed",
            "Fill in after execution",
        ]
    )
    # Atomic registry update: avoids a partially-written CSV after interruption.
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=REGISTRY.parent, delete=False
    ) as tmp:
        csv.writer(tmp, lineterminator="\n").writerows(rows)
        tmp_path = pathlib.Path(tmp.name)
    tmp_path.replace(REGISTRY)
    print(folder.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
