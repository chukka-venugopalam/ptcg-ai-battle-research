"""Tournament-driver tests.  Tests marked REAL-ENGINE never skip: with no engine they FAIL (by design, Mission 6 Phase D).
Tests marked SYNTHETIC use hand-made records and say nothing about any agent."""
import json, os, sys
import pytest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, os.path.join(HERE, "..", "vendor_m3"))
import tournament as T
import ptcg_eval_canonical as C


def rec(jid, arm, label, seat, outcome, games, failed=False, fail_seats=None, kind=None):
    return dict(jid=jid, arm=arm, label=label, idx=0, a_seat=seat, a_outcome=outcome, games=[dict(round=i + 1, first_player=0, winner=w, first_step=i) for i, w in enumerate(games)],
                failed=failed, fail_seats=fail_seats or {}, failure_kind=kind, a_cfg="A", b_cfg="B", wall_s=1.0, per_seat={})


# ---------------- hard engine requirement (the Mission 6 requirement)
def test_engine_missing_is_hard_failure(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "kaggle_environments", None)          # makes `import kaggle_environments` raise ImportError
    with pytest.raises(T.EngineUnavailable):
        T.require_engine()
    out = tmp_path / "run.jsonl"
    rc = T.main(["run", "--cand", str(tmp_path), "--opp", f"x={tmp_path}:1", "--matches", "2", "--out", str(out), "--outdir", str(tmp_path / "o")])
    assert rc == 5 and not out.exists()                                    # no tournament, no partial output, non-zero exit


def test_fake_engine_switch_is_rejected(monkeypatch):
    monkeypatch.setenv("PTCG_HARNESS_FAKE", "1")
    with pytest.raises(T.EngineUnavailable):
        T.require_engine()


def test_REAL_ENGINE_required_no_skip():
    fp = T.require_engine()                                                # raises -> test FAILS when the engine is absent
    assert fp["card_count"] > 0 and len(fp["libcg_sha256"]) == 64


def test_REAL_ENGINE_two_match_tournament(tmp_path):
    import agent_fixtures as F
    deck = F.sdk_deck()
    for n, body in (("rand", F.RAND_BODY), ("first", F.FIRST_BODY)):
        d = tmp_path / n; d.mkdir()
        (d / "main.py").write_text(body.format(deck_loader=F.LOADERS["file_guarded"])); (d / "deck.csv").write_text("\n".join(map(str, deck)) + "\n")
    rc = T.main(["run", "--cand", str(tmp_path / "rand"), "--opp", f"first={tmp_path / 'first'}:1", "--matches", "2", "--workers", "1",
                 "--deadline", "120", "--out", str(tmp_path / "run.jsonl"), "--outdir", str(tmp_path / "o")])
    assert rc == 0
    a = json.load(open(tmp_path / "o" / "tournament_analysis.json"))
    assert a["n_records"] == 2 and a["bo3_rule_mismatches"] == 0
    assert sum(1 for _ in open(tmp_path / "o" / "TOURNAMENT_MATCHES.csv")) == 3   # header + 2 BO3 matches (one row per match)


# ---------------- SYNTHETIC export logic
def test_SYNTHETIC_failure_mapping():
    import bo3_ab_harness as h
    assert T.to_match(rec("a", "cand", "o", 0, "win", [0, 0]), h).status == "ok"
    m = T.to_match(rec("b", "cand", "o", 0, "failed", [0], True, {"0": ["exception"]}, "seat_failure"), h)
    assert (m.status, m.result) == ("cand_fail", "L")                      # candidate fault = loss, never dropped
    assert T.to_match(rec("c", "cand", "o", 0, "failed", [0], True, {"1": ["hang"]}, "seat_failure"), h).status == "opp_fail"
    assert T.to_match(rec("d", "cand", "o", 0, "failed", [], True, {}, "error"), h).status == "infra_fail"


def test_SYNTHETIC_a_view_and_rule_check():
    r = rec("a", "cand", "o", 1, "win", [1, "draw", "draw"])               # seat 1 wins game 1, two draws -> match WIN for seat 1 (cabt rule)
    assert T.a_view_games(r) == ["W", "D", "D"] and C.match_result_from_games(T.a_view_games(r)) == "W"


def test_SYNTHETIC_confirm_without_fresh_is_refused(tmp_path):
    recs = [{"_header": {"argv": ["x", "--seed", "3"]}}]
    for arm in ("cand", "ctrl"):
        for lab in ("o",):
            for seat in (0, 1):
                for i in range(5):
                    recs.append(rec(f"{arm}:{lab}:{seat}:{i}", arm, lab, seat, "win" if i % 2 else "loss", [seat, seat] if i % 2 else [1 - seat, 1 - seat]))
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    out = T.export([str(p)], {"o": 1.0}, str(tmp_path / "o"), stage="confirm", fresh=False)
    assert out["gate"]["passed"] is False and "DesignError" in out["gate"]["reasons"][0]
    row = open(tmp_path / "o" / "TOURNAMENT_MATCHES.csv").read().splitlines()[1].split(",")
    assert row[0].startswith(("cand", "ctrl"))
