import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestRepoTools(unittest.TestCase):
    def test_current_archive_passes_structural_validation(self):
        archive = ROOT / "submissions/current/submission.tar.gz"
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/validate_submission.py"), str(archive)],
            capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        self.assertIn("STRUCTURAL_VALIDATION=PASS", result.stdout)

    def test_registry_contains_current_and_history(self):
        with (ROOT / "submissions/registry.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(any(row["name"] == "R2 compatibility reference" for row in rows))
        self.assertTrue(any(row["name"] == "B2 counter_source" for row in rows))

    def test_mission_7_not_falsely_marked_complete(self):
        p = ROOT / "research/missions/M07_meta_and_deck_upgrade/STATUS.md"
        self.assertIn("not yet executed", p.read_text(encoding="utf-8"))


    def test_new_experiment_pre_registers_and_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "experiments.csv"
            registry.write_text(
                "experiment_id,date,baseline,candidate,primary_question,protocol,n_games,result,status,decision,notes\n"
                "E_OLD,2026-01-01,base,old,old question,old protocol,10,5-5,done,keep,legacy\n",
                encoding="utf-8",
            )
            cmd = [
                sys.executable, str(ROOT / "tools/new_experiment.py"),
                "--id", "E_TEST", "--baseline", "base-hash", "--candidate", "new-hash",
                "--question", "Does candidate improve held-out win rate?",
                "--protocol", "Fixed single-game evaluation", "--opponent-pool", "pool-v1.csv",
                "--seat-policy", "balanced", "--planned-games", "100",
                "--stopping-rule", "Run exactly 100 games", "--registry", str(registry),
            ]
            first = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            with registry.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["experiment_id"] for r in rows], ["E_OLD", "E_TEST"])
            self.assertEqual(rows[0]["n_games"], "10")
            self.assertEqual(rows[1]["planned_games"], "100")
            self.assertEqual(rows[1]["result"], "NOT RUN")
            second = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertIn("already exists", second.stderr)

    def test_duplicate_mission_id_does_not_create_orphan_directory(self):
        folder = ROOT / "research/missions/M07_duplicate_guard_test"
        self.assertFalse(folder.exists())
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/new_mission.py"),
             "--id", "7", "--title", "Duplicate Guard Test", "--status", "planned"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already registered", result.stderr)
        self.assertFalse(folder.exists())

if __name__ == "__main__":
    unittest.main()
