#!/usr/bin/env python3
"""Pre-register a PTCG agent experiment before results are observed.

Uses only the Python standard library. Existing historical rows are retained
when the registry gains newer protocol columns.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "research" / "registries" / "experiments.csv"
FIELDS = [
    "experiment_id", "date", "baseline", "candidate", "primary_question",
    "protocol", "opponent_pool", "seat_policy", "planned_games", "stopping_rule",
    "n_games", "result", "status", "decision", "notes",
]


def register_experiment(
    registry: pathlib.Path,
    *,
    experiment_id: str,
    date: str,
    baseline: str,
    candidate: str,
    question: str,
    protocol: str,
    opponent_pool: str,
    seat_policy: str,
    planned_games: int,
    stopping_rule: str,
    notes: str = "",
) -> None:
    if planned_games <= 0:
        raise ValueError("planned_games must be positive")
    if not experiment_id.strip():
        raise ValueError("experiment_id is required")
    registry.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    old_fields: list[str] = []
    if registry.exists() and registry.stat().st_size:
        with registry.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            old_fields = list(reader.fieldnames or [])
            rows = [{field: (row.get(field) or "") for field in old_fields} for row in reader]
        if "experiment_id" not in old_fields:
            raise ValueError(f"Registry has no experiment_id column: {registry}")
        if any(row.get("experiment_id", "").strip().casefold() == experiment_id.strip().casefold() for row in rows):
            raise ValueError(f"Experiment ID already exists: {experiment_id}")

    row = {
        "experiment_id": experiment_id.strip(),
        "date": date,
        "baseline": baseline.strip(),
        "candidate": candidate.strip(),
        "primary_question": question.strip(),
        "protocol": protocol.strip(),
        "opponent_pool": opponent_pool.strip(),
        "seat_policy": seat_policy.strip(),
        "planned_games": str(planned_games),
        "stopping_rule": stopping_rule.strip(),
        "n_games": "",
        "result": "NOT RUN",
        "status": "pre_registered",
        "decision": "NOT RUN",
        "notes": notes.strip(),
    }
    merged = [{field: row_old.get(field, "") for field in FIELDS} for row_old in rows]
    merged.append(row)

    # Atomic replacement prevents a partially written experiment registry.
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=registry.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
        tmp_path = pathlib.Path(tmp.name)
    tmp_path.replace(registry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", required=True, dest="experiment_id", help="Unique ID such as E004")
    ap.add_argument("--baseline", required=True, help="Baseline archive name or SHA-256")
    ap.add_argument("--candidate", required=True, help="Candidate archive name or planned variant")
    ap.add_argument("--question", required=True, help="One primary hypothesis/question")
    ap.add_argument("--protocol", required=True, help="Game-level or match-level protocol")
    ap.add_argument("--opponent-pool", required=True, help="Frozen named opponent list or manifest path")
    ap.add_argument("--seat-policy", required=True, help="How each seat is balanced")
    ap.add_argument("--planned-games", required=True, type=int, help="Fixed planned number of game observations")
    ap.add_argument("--stopping-rule", required=True, help="Predeclared stopping and promotion rule")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--notes", default="")
    ap.add_argument("--registry", type=pathlib.Path, default=DEFAULT_REGISTRY, help=argparse.SUPPRESS)
    args = ap.parse_args()
    try:
        register_experiment(
            args.registry,
            experiment_id=args.experiment_id,
            date=args.date,
            baseline=args.baseline,
            candidate=args.candidate,
            question=args.question,
            protocol=args.protocol,
            opponent_pool=args.opponent_pool,
            seat_policy=args.seat_policy,
            planned_games=args.planned_games,
            stopping_rule=args.stopping_rule,
            notes=args.notes,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Pre-registered {args.experiment_id} in {args.registry}")
    print("status=pre_registered result=NOT RUN; update only after recording complete results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
