"""Mission 6.1 pipeline tests.  SYNTHETIC bundle => HARNESS VALIDATION ONLY (no statement about any real agent).
Engine-backed tests never skip: without the engine they fail."""
import csv, io, json, os, subprocess, sys, tarfile, zipfile
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools")); sys.path.insert(0, str(HERE))
import m61
import synth_bundle


def mk_tar(path, entries):
    with tarfile.open(path, "w:gz") as tf:
        for name, data, mode, typ in entries:
            ti = tarfile.TarInfo(name); ti.mode = mode
            if typ == "sym": ti.type = tarfile.SYMTYPE; ti.linkname = "/etc/passwd"; tf.addfile(ti)
            else: ti.size = len(data); tf.addfile(ti, io.BytesIO(data))


def test_expected_hashes_are_well_formed():
    assert all(m61.HEX64.fullmatch(h) for h in m61.EXPECTED.values()) and m61.HEX64.fullmatch(m61.R2_LIBCG)


def test_extract_rejects_unsafe_members(tmp_path):
    cases = {"traversal": [("../evil.py", b"x", 0o644, "f")], "absolute": [("/etc/x", b"x", 0o644, "f")], "symlink": [("link", b"", 0o777, "sym")],
             "duplicate": [("a.py", b"1", 0o644, "f"), ("a.py", b"2", 0o644, "f")], "casedup": [("A.py", b"1", 0o644, "f"), ("a.py", b"2", 0o644, "f")],
             "elf_outside_cg": [("tool/helper", b"\x7fELF....", 0o755, "f")]}
    for nm, ents in cases.items():
        p = tmp_path / f"{nm}.tar.gz"; mk_tar(p, ents); r = m61.extract_archive(p, tmp_path / f"x_{nm}")
        assert r["problems"] and r["extracted"] is None and not (tmp_path / f"x_{nm}").exists(), nm
    assert not (tmp_path / "evil.py").exists()


def test_extract_allows_native_in_cg_and_flags_exec_bit(tmp_path):
    p = tmp_path / "ok.tar.gz"; mk_tar(p, [("main.py", b"print(1)", 0o755, "f"), ("cg/libcg.so", b"\x7fELF....", 0o755, "f")])
    r = m61.extract_archive(p, tmp_path / "x")
    assert not r["problems"] and r["extracted"] and (tmp_path / "x" / "cg" / "libcg.so").exists()
    assert any("exec-bit" in f and "main.py" in f for f in r["findings"])
    assert oct((tmp_path / "x" / "main.py").stat().st_mode & 0o111) == "0o0"          # modes are not preserved: nothing extracted is executable


def test_malformed_archive_reported_not_raised(tmp_path):
    p = tmp_path / "bad.tar.gz"; p.write_bytes(b"not an archive")
    assert m61.extract_archive(p, tmp_path / "x")["problems"][0].startswith("MALFORMED")


def test_manifest_shapes():
    h = "a" * 64
    assert m61.manifest_pairs({"files": [{"path": "p/a", "sha256": h}]}) == [("p/a", h)]
    assert m61.manifest_pairs([{"name": "b", "hash": h}]) == [("b", h)]
    assert m61.manifest_pairs({"p/c.zip": h}) == [("p/c.zip", h)]
    assert m61.manifest_pairs({"p/d.zip": {"sha256": h}}) == [("p/d.zip", h)]


def test_preflight_reports_exact_missing_paths(tmp_path):
    pf = m61.stage_preflight(tmp_path / "empty_uploads_dir_does_not_exist", tmp_path / "o")
    assert pf["problems"] and "MISSING PATH (exact)" in (tmp_path / "o" / "INPUT_PREFLIGHT.md").read_text()
    d = tmp_path / "uploads"; d.mkdir(); (d / "other.zip").write_bytes(b"x")
    m61.stage_preflight(d, tmp_path / "o2"); t = (tmp_path / "o2" / "INPUT_PREFLIGHT.md").read_text()
    assert "PTCG_MISSION_6_INPUTS_BUNDLE.zip" in t and "MISSING PATH (exact)" in t and "BLOCKED" in t


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("synth"); wrap = synth_bundle.build(base / "bundle"); out = base / "out"
    log = m61.run_all(wrap, out, matches=2, synthetic=True)
    return out, log


def test_REAL_ENGINE_full_pipeline_all_stages_ok(run):
    out, log = run
    assert all(l.endswith(": OK") for l in log), log
    assert (out / "SYNTHETIC_RUN.txt").exists()


def test_synthetic_preflight(run):
    out, _ = run; t = (out / "INPUT_PREFLIGHT.md").read_text()
    assert t.count("| OK |") >= 5 + 0 and "MISMATCH" in t                      # manifest entries OK; the brief's REAL hashes correctly MISMATCH the synthetic files
    pf = json.load(open(out / "preflight.json")); assert set(pf["agents"]) == {"candidate", "Bv1", "B2", "Bv2"} and pf["engine"]["extracted"]
    assert "MEMBER_LISTS" in (out / "MEMBER_LISTS.md").read_text()


def test_synthetic_diffs_show_exact_flag_and_guard_lines(run):
    out, _ = run; t = (out / "ARCHIVE_DIFFS.md").read_text()
    assert "-AGENT_KWARGS = {'counter_source': False}" in t and "+AGENT_KWARGS = {'counter_source': True}" in t          # B2 vs Bv1
    assert "sym_k" in t and "min(i, 1299)" in t and "sa/policynet.py" in t and "cg/game.py" in t
    assert "Deck multiset: identical" in t


def test_synthetic_forensics_has_npz_dims_and_kwargs(run):
    out, _ = run; rows = list(csv.DictReader(open(out / "ARCHIVE_FORENSICS.csv")))
    assert {r["archive"] for r in rows} == {"candidate", "Bv1", "B2", "Bv2"}
    assert all("[1300, 4]:float64" in r["npz_arrays"] and "[1600, 4]:float64" in r["npz_arrays"] for r in rows)
    assert all(r["cold_start_ok"] == "True" for r in rows) and all(r["deck_legal_vs_engine"] == "True" for r in rows)


def test_synthetic_sdk_comparison_and_overlay(run):
    out, _ = run; t = (out / "SDK_FINGERPRINT_COMPARISON.md").read_text(); res = json.load(open(out / "sdk_compare.json"))
    assert res["cg/game.py"]["status"] == "DIFFERS" and res["cg/libcg.so"]["status"] == "IDENTICAL" and "R2 wrapper revision" in t
    r = subprocess.run([sys.executable, "-c", "import os,kaggle_environments.envs.cabt.cg.sim as s; print(os.path.realpath(s.__file__))"], capture_output=True, text=True, env=m61.env_for(out))
    assert str(out / "r2_overlay") in r.stdout


def test_id_probe_detects_unguarded_agent_and_passes_guarded(run):
    out, _ = run; bv1 = {x["variant"]: x for x in json.load(open(out / "idprobe_Bv1.json"))}; cand = {x["variant"]: x for x in json.load(open(out / "idprobe_candidate.json"))}
    assert bv1["raw (unmodified)"]["n_exceptions"] == 0 and bv1["card id=1299"]["n_exceptions"] == 0
    assert bv1["card id=1300"]["n_exceptions"] > 0 and bv1["card id=1431"]["n_exceptions"] > 0                            # unguarded: IndexError at/after 1300
    assert bv1["attack id=1600"]["n_exceptions"] > 0 and bv1["attack id=1755"]["n_exceptions"] > 0 and bv1["attack id=1599"]["n_exceptions"] == 0
    assert all(v["n_exceptions"] == 0 and v["n_illegal"] == 0 for v in cand.values())                                      # guarded agent clean


def test_synthetic_safety_and_pairs_outputs(run):
    out, _ = run; rows = list(csv.DictReader(open(out / "AGENT_SAFETY_AUDITS.csv")))
    assert len(rows) == 4 and all(r["completed"] == "2" and r["illegal"] == "0" for r in rows)
    m = list(csv.DictReader(open(out / "SYNTHETIC_TOURNAMENT_MATCHES.csv")))
    assert len(m) == 6 and {r["pair"] for r in m} == {"candidate_vs_Bv1", "candidate_vs_B2", "candidate_vs_Bv2"} and all(r["bo3_rule_ok"] == "True" for r in m)
    assert {r["agent_seat"] for r in m} == {"0", "1"}
    assert "NOT A STRENGTH RESULT" in (out / "SYNTHETIC_PAIR_RESULTS.md").read_text()
    assert not (out / "TOURNAMENT_MATCHES.csv").exists()                                                                  # synthetic run cannot masquerade as the real deliverable


def test_preflight_accepts_inner_zip_or_preextracted_dir(tmp_path):
    wrap = synth_bundle.build(tmp_path / "b"); inner = tmp_path / "b" / "PTCG_MISSION_6_INPUTS_BUNDLE.zip"
    pf1 = m61.stage_preflight(inner, tmp_path / "o1")                                               # inner bundle given directly
    pre = tmp_path / "pre"; pre.mkdir(); zipfile.ZipFile(inner).extractall(pre)
    pf2 = m61.stage_preflight(pre, tmp_path / "o2")                                                 # already-extracted directory
    for pf in (pf1, pf2): assert set(pf["agents"]) == {"candidate", "Bv1", "B2", "Bv2"} and pf["engine"]["extracted"], pf


def test_r2_layout_accepts_text_only_package(tmp_path):
    cg = tmp_path / "pkg" / "cabt" / "cg"; cg.mkdir(parents=True)
    for n in ("game.py", "sim.py", "__init__.py"): (cg / n).write_text("# x\n")
    (cg.parent / "cabt.py").write_text("# x\n"); (cg.parent / "cabt.json").write_text("{}")
    lay = m61.r2_layout(tmp_path / "pkg")
    assert lay and lay["cg"] == cg and lay["cabt_py"].name == "cabt.py" and lay["cabt_json"].name == "cabt.json"
    assert m61.r2_layout(tmp_path / "empty_dir_that_does_not_exist") is None
