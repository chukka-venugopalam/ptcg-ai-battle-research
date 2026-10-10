"""Statistics helpers for game-based acceptance gates (stdlib only)."""
import math
from statistics import NormalDist

def wilson(wins, n, conf=0.95):
    if n <= 0: raise ValueError("n must be > 0")
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def two_prop_z(w1, n1, w2, n2):
    """Two-sided z-test p-value for difference in win rates."""
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0: return 1.0
    z = (w1 / n1 - w2 / n2) / se
    return 2 * (1 - NormalDist().cdf(abs(z)))

def required_games(delta, p0=0.5, alpha=0.05, power=0.8):
    """Games to detect win rate p0+delta vs p0 (one-sided, normal approx.)."""
    za = NormalDist().inv_cdf(1 - alpha); zb = NormalDist().inv_cdf(power)
    p1 = p0 + delta
    n = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / delta) ** 2
    return math.ceil(n)

def min_detectable(n, p0=0.5, alpha=0.05, power=0.8):
    """Smallest win-rate edge detectable with n games (search over delta)."""
    lo, hi = 1e-4, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_games(mid, p0, alpha, power) > n: lo = mid
        else: hi = mid
    return hi

def gate_decision(wins, n, min_rate=0.55, conf=0.95, min_games=500, control_rates=None, control_floor=0.45):
    """Acceptance gate used in IMPLEMENTATION_ROADMAP: pooled rate >= min_rate,
    Wilson lower bound > 0.5, >= min_games, and no control matchup point estimate below control_floor."""
    if n < min_games: return False, f"only {n} games (<{min_games})"
    lo, hi = wilson(wins, n, conf)
    if wins / n < min_rate: return False, f"rate {wins/n:.3f} < {min_rate}"
    if lo <= 0.5: return False, f"Wilson lower bound {lo:.3f} <= 0.5"
    for name, r in (control_rates or {}).items():
        if r < control_floor: return False, f"control {name} {r:.3f} < {control_floor}"
    return True, f"pass: rate {wins/n:.3f}, Wilson [{lo:.3f},{hi:.3f}]"

if __name__ == "__main__":
    print("games for +5pp @80% power:", required_games(0.05))
    print("min detectable edge @500 games:", round(min_detectable(500), 4))
    print("min detectable edge @35 games:", round(min_detectable(35), 4))
