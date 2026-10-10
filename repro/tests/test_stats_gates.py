import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stats_gates import wilson, two_prop_z, required_games, min_detectable, gate_decision

class T(unittest.TestCase):
    def test_wilson_known(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3); self.assertAlmostEqual(hi, 0.5962, places=3)
    def test_wilson_edges(self):
        lo, hi = wilson(0, 10); self.assertAlmostEqual(lo, 0.0, places=9); self.assertLess(hi, 0.35)
        lo, hi = wilson(10, 10); self.assertAlmostEqual(hi, 1.0, places=9); self.assertGreater(lo, 0.65)
    def test_z_symmetry_and_null(self):
        self.assertAlmostEqual(two_prop_z(50, 100, 50, 100), 1.0, places=6)
        self.assertAlmostEqual(two_prop_z(60, 100, 40, 100), two_prop_z(40, 100, 60, 100), places=9)
    def test_required_games_matches_report(self):
        n = required_games(0.05)
        self.assertTrue(600 <= n <= 650, n)
    def test_min_detectable_small_n_is_large(self):
        self.assertGreater(min_detectable(35), 0.15)   # ~ +/-24pp two-sided class at 35 games
        self.assertLess(min_detectable(500), 0.07)
    def test_gate(self):
        ok, _ = gate_decision(300, 500); self.assertTrue(ok)       # 60%
        ok, _ = gate_decision(270, 500); self.assertFalse(ok)      # 54%
        ok, _ = gate_decision(60, 100);  self.assertFalse(ok)      # too few games
        ok, _ = gate_decision(300, 500, control_rates={"x": 0.4}); self.assertFalse(ok)

if __name__ == "__main__": unittest.main()
