```


====================================================================================================
/kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/build_submission.py
====================================================================================================

--- around line 9 ---
    deck.csv           60 card ids, one per line
    cg/                engine (from the local SDK)
    sa/                agent package (+ value_net.npz / policy_net.npz /
                       deck_library.json if present)

Then smoke-runs the *extracted* bundle from a temp dir: full self-play game,
crash = build failure. Prints latency + time-pool stats.

--- around line 49 ---


def _find_kaggle_python() -> str | None:
    """An interpreter matching Kaggle's episode runner, or None.

    `SA_KAGGLE_PYTHON` wins; otherwise ask uv for a managed 3.11. Returning
    None is not fatal -- the caller falls back to a narrower static check.

--- around line 68 ---


def _scan_pep701(root: Path) -> list[str]:
    """Static fallback: f-strings that only parse on 3.12+ (PEP 701).

    Catches the multi-line replacement field that actually shipped. It is a
    subset of what a real 3.11 parse catches, so it is the fallback, never the

--- around line 105 ---


def _check_kaggle_syntax(build: Path) -> None:
    """Compile every bundled .py the way the grader's interpreter will."""
    py = _find_kaggle_python()
    if py:
        proc = subprocess.run([py, "-m", "compileall", "-q", str(build)],

--- around line 153 ---
# Extra ensemble members, relative to the bundle root. Empty = single net, i.e.
# exactly the behaviour every submission before this one had. When non-empty the
# agent votes across sa/policy_net.npz + these (EVIDENCE: E9).
ENSEMBLE_EXTRA = {ensemble_extra!r}
# Explicit rule flags, pinned at BUILD time. The defaults in sa/bcagent.py are
# tuned for the lw2 net; an optfeat-v3 net wants them OFF (the three rules
# measure 0.427 against it -- report/EVIDENCE.md 8f). Pinning the (net, flags)

--- around line 162 ---


def _read_deck():
    path = os.path.join(_HERE, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as fh:

--- around line 184 ---
        # agent that shipped before this change. Degrade, log, keep playing.
        try:
            _paths = [os.path.join(_HERE, "sa", "policy_net.npz")]
            _paths += [os.path.join(_HERE, p) for p in ENSEMBLE_EXTRA]
            _agent = _A(_deck, "+".join(_paths), **AGENT_KWARGS)
            print("[health] ENSEMBLE OK members=%d" % len(_paths), flush=True)
        except Exception:

--- around line 221 ---


def _log_health(force=False):
    try:
        from sa import bcagent as _bc
        s = _bc.STATS
        if s["fallbacks"] and not _LOG["failed"]:

--- around line 238 ---


def agent(obs):
    out = _agent(obs)
    _log_health(force=obs.get("select") is None)
    return out
'''

--- around line 250 ---
# Load main.py THE WAY KAGGLE DOES: exec the source with no __file__ in globals
# (kaggle_environments/agent.py does exec(code_object, env)). `import main`
# would define __file__ and hide a whole class of crash -- it did once.
with open("main.py", "rb") as _fh:
    _src = _fh.read()
_env = {}
exec(compile(_src, "main.py", "exec"), _env)

--- around line 258 ---


class _M:
    pass


main = _M()

--- around line 279 ---
    _live = _ag.net or _pn.get()
    assert _live is not None, "POLICY NET NOT LOADED -- agent would play random-legal"
    print(f"NET_OK opt_in={_live.opt_in} state_in={_live.state_in}")
    # An ensemble that quietly loaded ONE member is a different agent shipping
    # under the measured one's name, and it would score like the single net
    # while the build log said nothing. Assert the member count explicitly.
    _want = int(os.environ.get("SMOKE_MEMBERS", "1"))

--- around line 293 ---
import cg.game as game

class B:  # opposing agent: trivial legal
    def __call__(self, obs):
        if obs.get("select") is None:
            return deck
        return list(range(obs["select"]["minCount"]))

--- around line 294 ---

class B:  # opposing agent: trivial legal
    def __call__(self, obs):
        if obs.get("select") is None:
            return deck
        return list(range(obs["select"]["minCount"]))


--- around line 329 ---


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--agent", default="search", choices=["search", "bc"])
    ap.add_argument("--no-smoke", action="store_true")

--- around line 333 ---
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--agent", default="search", choices=["search", "bc"])
    ap.add_argument("--no-smoke", action="store_true")
    # Which nets to ship. The SA_NO_* kill-switches are env vars and Kaggle
    # sets none, so anything bundled is LIVE there. Omitting an npz makes the
    # matching net.get() return None and the agent fall back to the
    # handcrafted eval -- the only way to pin the config for the grader.

--- around line 341 ---
                    choices=["both", "policy", "value", "none"],
                    help="nets to include (default both)")
    # Ship a candidate net instead of agents/sa/policy_net.npz. The bundle is
    # what the grader runs, so the net and the rule flags must be pinned
    # TOGETHER -- a v3 net with the lw2 defaults is the 0.427 configuration.
    ap.add_argument("--policy-net", default=None,
                    help="path to an npz to ship as sa/policy_net.npz")

--- around line 345 ---
    # TOGETHER -- a v3 net with the lw2 defaults is the 0.427 configuration.
    ap.add_argument("--policy-net", default=None,
                    help="path to an npz to ship as sa/policy_net.npz")
    # Extra ensemble members (EVIDENCE E9). Each is copied into the bundle as
    # sa/policy_net_<i>.npz and voted with sa/policy_net.npz. Member ORDER is
    # part of the agent's identity -- member 0 supplies the count rule.
    ap.add_argument("--ensemble-net", action="append", default=[],

--- around line 347 ---
                    help="path to an npz to ship as sa/policy_net.npz")
    # Extra ensemble members (EVIDENCE E9). Each is copied into the bundle as
    # sa/policy_net_<i>.npz and voted with sa/policy_net.npz. Member ORDER is
    # part of the agent's identity -- member 0 supplies the count rule.
    ap.add_argument("--ensemble-net", action="append", default=[],
                    help="extra npz to vote with the shipped policy net; "
                         "repeatable")

--- around line 349 ---
    # sa/policy_net_<i>.npz and voted with sa/policy_net.npz. Member ORDER is
    # part of the agent's identity -- member 0 supplies the count rule.
    ap.add_argument("--ensemble-net", action="append", default=[],
                    help="extra npz to vote with the shipped policy net; "
                         "repeatable")
    ap.add_argument("--no-rules", action="store_true",
                    help="disable chip_targeting/energy_spread/counter_source. "

--- around line 352 ---
                    help="extra npz to vote with the shipped policy net; "
                         "repeatable")
    ap.add_argument("--no-rules", action="store_true",
                    help="disable chip_targeting/energy_spread/counter_source. "
                         "Required with an optfeat-v3 net (EVIDENCE 8f).")
    args = ap.parse_args()
    if args.agent == "bc" and args.nets in ("value", "none"):

--- around line 373 ---
    build.mkdir(parents=True)

    ensemble_extra = [f"sa/policy_net_{i + 1}.npz"
                      for i in range(len(args.ensemble_net))]
    (build / "main.py").write_text(
        MAIN_PY.format(agent_kind=args.agent, agent_kwargs=agent_kwargs,
                       ensemble_extra=ensemble_extra),

--- around line 391 ---
    keep_policy = args.nets in ("both", "policy")
    keep_value = args.nets in ("both", "value")
    for npz, keep in (("policy_net.npz", keep_policy),
                      ("value_net.npz", keep_value)):
        if not keep:
            (build / "sa" / npz).unlink(missing_ok=True)
    if args.policy_net:

--- around line 392 ---
    keep_value = args.nets in ("both", "value")
    for npz, keep in (("policy_net.npz", keep_policy),
                      ("value_net.npz", keep_value)):
        if not keep:
            (build / "sa" / npz).unlink(missing_ok=True)
    if args.policy_net:
        src = Path(args.policy_net)

--- around line 395 ---
        if not keep:
            (build / "sa" / npz).unlink(missing_ok=True)
    if args.policy_net:
        src = Path(args.policy_net)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():

--- around line 396 ---
            (build / "sa" / npz).unlink(missing_ok=True)
    if args.policy_net:
        src = Path(args.policy_net)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit(f"--policy-net not found: {src}")

--- around line 403 ---
        if not keep_policy:
            raise SystemExit("--policy-net needs --nets both|policy")
        shutil.copy2(src, build / "sa" / "policy_net.npz")
        print(f"  sa/policy_net.npz <- {src}")
        # The dim guard silently returns None for a net whose feature layout
        # this code cannot feed, and the agent then falls back to
        # list(range(minCount)) -- i.e. a broken agent that still "runs".

--- around line 404 ---
            raise SystemExit("--policy-net needs --nets both|policy")
        shutil.copy2(src, build / "sa" / "policy_net.npz")
        print(f"  sa/policy_net.npz <- {src}")
        # The dim guard silently returns None for a net whose feature layout
        # this code cannot feed, and the agent then falls back to
        # list(range(minCount)) -- i.e. a broken agent that still "runs".
        # Verifying HERE turns that into a build failure. (See policynet.load.)

--- around line 405 ---
        shutil.copy2(src, build / "sa" / "policy_net.npz")
        print(f"  sa/policy_net.npz <- {src}")
        # The dim guard silently returns None for a net whose feature layout
        # this code cannot feed, and the agent then falls back to
        # list(range(minCount)) -- i.e. a broken agent that still "runs".
        # Verifying HERE turns that into a build failure. (See policynet.load.)
        # sa.cards imports cg.sim, so the SDK has to be on the path first. Done

--- around line 414 ---
        _sdk.load()
        from sa import policynet as _pn
        if _pn.load(build / "sa" / "policy_net.npz") is None:
            raise SystemExit(f"{src} FAILS the dim guard -- it would silently "
                             "fall back to random-legal on Kaggle")
        print("  dim guard: net loads OK")
    if args.ensemble_net:

--- around line 430 ---
            if not src.exists():
                raise SystemExit(f"--ensemble-net not found: {src}")
            dst = build / "sa" / f"policy_net_{i + 1}.npz"
            shutil.copy2(src, dst)
            if _pn2.load(dst) is None:
                raise SystemExit(f"{src} FAILS the dim guard")
            print(f"  sa/{dst.name} <- {src} (dim guard OK)")

--- around line 441 ---
        import hashlib
        seen: dict[str, str] = {}
        for f in ["policy_net.npz"] + [f"policy_net_{i + 1}.npz"
                                       for i in range(len(args.ensemble_net))]:
            digest = hashlib.md5((build / "sa" / f).read_bytes()).hexdigest()
            if digest in seen:
                raise SystemExit(

--- around line 450 ---
            seen[digest] = f
        print(f"  ensemble: {len(seen)} distinct members")
    for extra in ("value_net.npz", "policy_net.npz", "deck_library.json"):
        present = (build / "sa" / extra).exists()
        note = "present" if present else "excluded -> handcrafted fallback"
        print(f"  sa/{extra}: {note}")


====================================================================================================
/kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/build_policy_dataset.py
====================================================================================================

--- around line 3 ---
"""Build policy-cloning shards from replay JSONs.

    python scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26

One row per select with >=2 options: state features + per-option features +
multi-hot chosen mask (from the replay's actual action).


--- around line 5 ---
    python scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26

One row per select with >=2 options: state features + per-option features +
multi-hot chosen mask (from the replay's actual action).

⚠ By default this clones BOTH seats of every game -- every archetype and every
skill level in the dump. `--player NAME` keeps only the seats belonging to the

--- around line 6 ---

One row per select with >=2 options: state features + per-option features +
multi-hot chosen mask (from the replay's actual action).

⚠ By default this clones BOTH seats of every game -- every archetype and every
skill level in the dump. `--player NAME` keeps only the seats belonging to the
named team(s), which is how an EXPERT corpus is built from a third-party dump:

--- around line 12 ---
named team(s), which is how an EXPERT corpus is built from a third-party dump:

    python scripts/build_policy_dataset.py --out artifacts/pds_expert \\
        --player "Raja Biswas" --player "Sixth Sense" \\
        replays/sixth_sense_31-07-2026

`--ratings` tags every row with the LB score of the demonstrator who made that

--- around line 19 ---
choice, so a corpus can be reweighted or sliced by demonstrator strength (B7):

    python scripts/build_policy_dataset.py --out artifacts/pds_v3r \\
        --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26
"""
from __future__ import annotations


--- around line 43 ---
sdk.load()

from sa.features import attr_feats, extra_feats, featurize  # noqa: E402
from sa.optfeat import option_features, OPT_DENSE  # noqa: E402

SHARD_ROWS = 60_000
SEL_DENSE = 14

--- around line 44 ---

from sa.features import attr_feats, extra_feats, featurize  # noqa: E402
from sa.optfeat import option_features, OPT_DENSE  # noqa: E402

SHARD_ROWS = 60_000
SEL_DENSE = 14
NO_RATING = np.float32("nan")

--- around line 51 ---


def load_ratings(path: Path) -> tuple[dict[str, float], dict[int, float]]:
    """Kaggle LB export -> (by team name, by teamId). Accepts the .zip that
    `competition_leaderboard_download` writes or the .csv inside it."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:

--- around line 97 ---


def name_id(name: str) -> int:
    """Stable per-team id for dumps with no `episodes_meta.json` sidecar (the
    daily replay dirs). Deterministic across runs and processes, unlike
    `hash()`, so a corpus rebuilt tomorrow keeps the same ids."""
    import hashlib

--- around line 106 ---


def load_episode_meta(d: Path) -> dict[int, dict[int, tuple[int, int]]]:
    """`episodes_meta.json` -> {episode_id: {seat: (submissionId, teamId)}}.
    Only targeted per-team dumps carry it; day dumps return {}."""
    p = d / "episodes_meta.json"
    if not p.exists():

--- around line 122 ---


def sel_features(sel: dict) -> np.ndarray:
    v = np.zeros(SEL_DENSE, dtype=np.float32)
    t = sel.get("type") or 0
    if t < 11:
        v[t] = 1.0

--- around line 133 ---


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0

--- around line 134 ---

class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.reset()

--- around line 140 ---
        self.reset()

    def reset(self):
        self.sd, self.slots, self.seld, self.gid = [], [], [], []
        self.xd, self.xslots, self.attr = [], [], []
        self.bags = {"my_hand": [], "my_discard": [], "opp_discard": []}
        self.od, self.ocard, self.oatk, self.otgt, self.chosen = [], [], [], [], []

--- around line 149 ---
        self.rating, self.opp_rating, self.team_id, self.sub_id = [], [], [], []

    def add(self, dense, bags, seld, opts, chosen_mask, gid, won,
            rating=NO_RATING, opp_rating=NO_RATING, team_id=-1, sub_id=-1,
            extra=None, attr=None):
        self.sd.append(dense)
        # The v4 state block (features.extra_feats). Written unconditionally --

--- around line 153 ---
            extra=None, attr=None):
        self.sd.append(dense)
        # The v4 state block (features.extra_feats). Written unconditionally --
        # a trainer that does not want it simply does not read these arrays,
        # which is what makes the v3 control run on the IDENTICAL rows.
        xd, xids = extra
        self.xd.append(xd)

--- around line 159 ---
        self.xd.append(xd)
        self.xslots.append(xids)
        # The v6 card-attribute block (features.attr_feats), same contract:
        # always written, so a v5 control trains on the IDENTICAL rows.
        self.attr.append(attr)
        self.slots.append(bags["slots"])
        for k in self.bags:

--- around line 182 ---
            self.flush()

    def flush(self):
        if not self.sd:
            return
        arrs = {
            "dense": np.stack(self.sd),

--- around line 215 ---
            arrs[f"bag_{k}_off"] = off
        path = self.out_dir / f"shard_{self.idx:03d}.npz"
        np.savez_compressed(path, **arrs)
        print(f"  wrote {path.name}: {len(self.sd)} rows")
        self.idx += 1
        self.reset()


--- around line 221 ---


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--player", action="append", default=[],

--- around line 225 ---
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--player", action="append", default=[],
                    help="keep only seats owned by this team name (repeatable). "
                         "Default: clone both seats of every game.")
    ap.add_argument("--players-file",
                    help="UTF-8 file of team names, one per line, added to "

--- around line 231 ---
                    help="UTF-8 file of team names, one per line, added to "
                         "--player (for name lists the shell cannot quote)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop these team names even if --player/--players-file "
                         "names them. ⚠ Always exclude OURSELVES from a control "
                         "population: our own agent's selects are what the net "
                         "was fitted to, so leaving `Scio` in inflates "

--- around line 349 ---
                    if not obs or not obs.get("current") or not obs.get("select"):
                        continue
                    state = obs["current"]
                    if state["result"] != -1:
                        continue
                    sel = obs["select"]
                    opts = sel.get("option") or []

--- around line 350 ---
                        continue
                    state = obs["current"]
                    if state["result"] != -1:
                        continue
                    sel = obs["select"]
                    opts = sel.get("option") or []
                    if len(opts) < 2:

--- around line 356 ---
                    if len(opts) < 2:
                        continue
                    action = v.get("selected")
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue

--- around line 357 ---
                        continue
                    action = v.get("selected")
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action

--- around line 358 ---
                    action = v.get("selected")
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]

--- around line 359 ---
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]
                    if len(picked) != len(action):

--- around line 361 ---
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]
                    if len(picked) != len(action):
                        continue
                    me = state["yourIndex"]

--- around line 363 ---
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]
                    if len(picked) != len(action):
                        continue
                    me = state["yourIndex"]
                    if seats is not None and me not in seats:
                        n_skip_seat += 1

--- around line 365 ---
                    if len(picked) != len(action):
                        continue
                    me = state["yourIndex"]
                    if seats is not None and me not in seats:
                        n_skip_seat += 1
                        continue
                    won = 1.0 if rewards[me] > rewards[1 - me] else 0.0

--- around line 370 ---
                        continue
                    won = 1.0 if rewards[me] > rewards[1 - me] else 0.0
                    dense, bags = featurize(state, me)
                    od = np.zeros((len(opts), OPT_DENSE), dtype=np.float32)
                    oc = np.zeros(len(opts), dtype=np.int32)
                    oa = np.zeros(len(opts), dtype=np.int32)
                    ot = np.zeros(len(opts), dtype=np.int32)

--- around line 376 ---
                    ot = np.zeros(len(opts), dtype=np.int32)
                    for i, o in enumerate(opts):
                        od[i], oc[i], oa[i], ot[i] = option_features(obs, o)
                    mask = np.zeros(len(opts), dtype=np.float32)
                    mask[picked] = 1.0
                    writer.add(dense, bags, sel_features(sel),
                               (od, oc, oa, ot), mask, gid, won,

--- around line 379 ---
                    mask = np.zeros(len(opts), dtype=np.float32)
                    mask[picked] = 1.0
                    writer.add(dense, bags, sel_features(sel),
                               (od, oc, oa, ot), mask, gid, won,
                               extra=extra_feats(state, sel, me),
                               attr=attr_feats(state, me),
                               rating=seat_rating.get(me, float("nan")),

--- around line 381 ---
                    writer.add(dense, bags, sel_features(sel),
                               (od, oc, oa, ot), mask, gid, won,
                               extra=extra_feats(state, sel, me),
                               attr=attr_feats(state, me),
                               rating=seat_rating.get(me, float("nan")),
                               opp_rating=seat_rating.get(1 - me,
                                                          float("nan")),

--- around line 382 ---
                               (od, oc, oa, ot), mask, gid, won,
                               extra=extra_feats(state, sel, me),
                               attr=attr_feats(state, me),
                               rating=seat_rating.get(me, float("nan")),
                               opp_rating=seat_rating.get(1 - me,
                                                          float("nan")),
                               team_id=seat_team.get(me, -1),

====================================================================================================
/kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/train_policy.py
====================================================================================================

--- around line 3 ---
"""Train the policy net (behavioral cloning of top players' selects).

    python scripts/train_policy.py --ds artifacts/pds --out agents/sa/policy_net.npz

The state is encoded once per row, then scored against each option.

    state:  dense + slot_emb(12x16) + 3 bag means(16) + seld(14)

--- around line 5 ---
    python scripts/train_policy.py --ds artifacts/pds --out agents/sa/policy_net.npz

The state is encoded once per row, then scored against each option.

    state:  dense + slot_emb(12x16) + 3 bag means(16) + seld(14)
            -> MLP(--state-h) -> state_repr
    option: opt_dense + card_emb(16) + atk_emb(16) + tgt_emb(16)

--- around line 7 ---
The state is encoded once per row, then scored against each option.

    state:  dense + slot_emb(12x16) + 3 bag means(16) + seld(14)
            -> MLP(--state-h) -> state_repr
    option: opt_dense + card_emb(16) + atk_emb(16) + tgt_emb(16)
    score:  MLP([state_repr, option], --head-h) -> 1


--- around line 8 ---

    state:  dense + slot_emb(12x16) + 3 bag means(16) + seld(14)
            -> MLP(--state-h) -> state_repr
    option: opt_dense + card_emb(16) + atk_emb(16) + tgt_emb(16)
    score:  MLP([state_repr, option], --head-h) -> 1

`--loss listwise` optimizes softmax cross-entropy over each select's option

--- around line 10 ---
            -> MLP(--state-h) -> state_repr
    option: opt_dense + card_emb(16) + atk_emb(16) + tgt_emb(16)
    score:  MLP([state_repr, option], --head-h) -> 1

`--loss listwise` optimizes softmax cross-entropy over each select's option
set, which is what the agent actually does at inference (rank the options and
take the top k). `--loss bce` is the original pointwise objective; it treats

--- around line 15 ---
set, which is what the agent actually does at inference (rank the options and
take the top k). `--loss bce` is the original pointwise objective; it treats
every option independently and does not model "which of these is best".

Layer sizes are exported generically (`sfc{i}_w` / `head{i}_w` + counts), so
sa/policynet.py mirrors any depth without a code change.
"""

--- around line 40 ---
sdk.load()

from sa.features import (A_GROUPS, DENSE_DIM, N_ATTR,  # noqa: E402
                         N_CARD_IDS, N_EXTRA, N_XSLOT, X_GROUPS)
from sa.optfeat import (OPT_DENSE, OPT_DENSE_V2, N_ATTACK_IDS,  # noqa: E402
                        pool_width)
from sa.routing import (NAME_TO_ROUTE, ROUTE_NAMES,  # noqa: E402

--- around line 59 ---


def build_remap(vocab_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-table id -> row map with a PAD row and a shared UNK row.

    The shipped tables are allocated over the RAW id space (1300 card ids, 1600
    attack ids) but the corpus only ever touches 104/134/135/57 of those rows.

--- around line 81 ---
    would re-introduce the exact defect this removes, just for fewer rows.
    """
    from sa.features import N_CARD_IDS
    from sa.optfeat import N_ATTACK_IDS
    tabs = json.loads(vocab_path.read_text(encoding="utf-8"))["tables"]
    sizes = {"atk_emb": N_ATTACK_IDS}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

--- around line 102 ---


def apply_remap(data: "Data", remap: dict[str, tuple[np.ndarray, np.ndarray]]
                ) -> None:
    """Rewrite every id column in place. Ids at or past a table's raw size
    cannot appear -- features.py already clamps them to 0 -- but clip anyway so
    a corpus built by an older builder fails to UNK rather than IndexError."""

--- around line 105 ---
                ) -> None:
    """Rewrite every id column in place. Ids at or past a table's raw size
    cannot appear -- features.py already clamps them to 0 -- but clip anyway so
    a corpus built by an older builder fails to UNK rather than IndexError."""
    def m(t: str, a: np.ndarray) -> np.ndarray:
        lut = remap[t][1]
        return lut[np.clip(a, 0, len(lut) - 1)]

--- around line 107 ---
    cannot appear -- features.py already clamps them to 0 -- but clip anyway so
    a corpus built by an older builder fails to UNK rather than IndexError."""
    def m(t: str, a: np.ndarray) -> np.ndarray:
        lut = remap[t][1]
        return lut[np.clip(a, 0, len(lut) - 1)]
    data.slots = m("slot_emb", data.slots)
    data.xslots = m("slot_emb", data.xslots)

--- around line 123 ---


def load_init(model: PolicyNet, path: Path) -> None:
    """Warm-start from an exported .npz (the fine-tuning arm of B7). Refuses on
    any shape mismatch rather than partially loading -- a silently half-loaded
    net trains fine and measures like a fresh one."""
    z = np.load(path)

--- around line 127 ---
    any shape mismatch rather than partially loading -- a silently half-loaded
    net trains fine and measures like a fresh one."""
    z = np.load(path)
    with torch.no_grad():
        for name, emb in (("slot_emb", model.slot_emb), ("bag_emb",
                          model.bag_emb), ("card_emb", model.card_emb),
                         ("atk_emb", model.atk_emb)):

--- around line 129 ---
    z = np.load(path)
    with torch.no_grad():
        for name, emb in (("slot_emb", model.slot_emb), ("bag_emb",
                          model.bag_emb), ("card_emb", model.card_emb),
                         ("atk_emb", model.atk_emb)):
            w = z[name]
            if w.shape != tuple(emb.weight.shape):

--- around line 130 ---
    with torch.no_grad():
        for name, emb in (("slot_emb", model.slot_emb), ("bag_emb",
                          model.bag_emb), ("card_emb", model.card_emb),
                         ("atk_emb", model.atk_emb)):
            w = z[name]
            if w.shape != tuple(emb.weight.shape):
                raise SystemExit(f"--init {path.name}: {name} is {w.shape}, "

--- around line 131 ---
        for name, emb in (("slot_emb", model.slot_emb), ("bag_emb",
                          model.bag_emb), ("card_emb", model.card_emb),
                         ("atk_emb", model.atk_emb)):
            w = z[name]
            if w.shape != tuple(emb.weight.shape):
                raise SystemExit(f"--init {path.name}: {name} is {w.shape}, "
                                 f"model wants {tuple(emb.weight.shape)}")

--- around line 135 ---
            if w.shape != tuple(emb.weight.shape):
                raise SystemExit(f"--init {path.name}: {name} is {w.shape}, "
                                 f"model wants {tuple(emb.weight.shape)}")
            emb.weight.copy_(torch.from_numpy(w))
        for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
            lins = [m for m in seq if isinstance(m, nn.Linear)]
            n = int(z[f"n_{prefix}"][0])

--- around line 137 ---
                                 f"model wants {tuple(emb.weight.shape)}")
            emb.weight.copy_(torch.from_numpy(w))
        for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
            lins = [m for m in seq if isinstance(m, nn.Linear)]
            n = int(z[f"n_{prefix}"][0])
            if n != len(lins):
                raise SystemExit(f"--init {path.name}: {n} {prefix} layers, "

--- around line 142 ---
            if n != len(lins):
                raise SystemExit(f"--init {path.name}: {n} {prefix} layers, "
                                 f"model has {len(lins)}")
            for i, lin in enumerate(lins):
                w, b = z[f"{prefix}{i}_w"], z[f"{prefix}{i}_b"]
                if w.shape != tuple(lin.weight.shape):
                    raise SystemExit(

--- around line 148 ---
                    raise SystemExit(
                        f"--init {path.name}: {prefix}{i}_w is {w.shape}, "
                        f"model wants {tuple(lin.weight.shape)}")
                lin.weight.copy_(torch.from_numpy(w))
                lin.bias.copy_(torch.from_numpy(b))
        # E1 auxiliary heads are append-only. A plain v5 checkpoint has no
        # auxiliary tensors, so warm-starting it deliberately leaves these

--- around line 151 ---
                lin.weight.copy_(torch.from_numpy(w))
                lin.bias.copy_(torch.from_numpy(b))
        # E1 auxiliary heads are append-only. A plain v5 checkpoint has no
        # auxiliary tensors, so warm-starting it deliberately leaves these
        # heads at their seeded initialization while loading the policy
        # byte-for-byte. A later multitask checkpoint restores them as well.
        for prefix, head in (("outcome", model.outcome_head),

--- around line 154 ---
        # auxiliary tensors, so warm-starting it deliberately leaves these
        # heads at their seeded initialization while loading the policy
        # byte-for-byte. A later multitask checkpoint restores them as well.
        for prefix, head in (("outcome", model.outcome_head),
                             ("count", model.count_head)):
            if head is None or f"{prefix}_w" not in z:
                continue

--- around line 155 ---
        # heads at their seeded initialization while loading the policy
        # byte-for-byte. A later multitask checkpoint restores them as well.
        for prefix, head in (("outcome", model.outcome_head),
                             ("count", model.count_head)):
            if head is None or f"{prefix}_w" not in z:
                continue
            w, b = z[f"{prefix}_w"], z[f"{prefix}_b"]

--- around line 156 ---
        # byte-for-byte. A later multitask checkpoint restores them as well.
        for prefix, head in (("outcome", model.outcome_head),
                             ("count", model.count_head)):
            if head is None or f"{prefix}_w" not in z:
                continue
            w, b = z[f"{prefix}_w"], z[f"{prefix}_b"]
            if w.shape != tuple(head.weight.shape):

--- around line 162 ---
            if w.shape != tuple(head.weight.shape):
                raise SystemExit(f"--init {path.name}: {prefix}_w is {w.shape}, "
                                 f"model wants {tuple(head.weight.shape)}")
            head.weight.copy_(torch.from_numpy(w))
            head.bias.copy_(torch.from_numpy(b))
        # E2 adapters are append-only. A plain v5 checkpoint has none, so
        # warm-starting leaves the zero-initialized residuals in place.

--- around line 165 ---
            head.weight.copy_(torch.from_numpy(w))
            head.bias.copy_(torch.from_numpy(b))
        # E2 adapters are append-only. A plain v5 checkpoint has none, so
        # warm-starting leaves the zero-initialized residuals in place.
        if model.adapters is not None and "adapter_names" in z:
            names = [str(x) for x in z["adapter_names"].tolist()]
            for name in names:

--- around line 167 ---
        # E2 adapters are append-only. A plain v5 checkpoint has none, so
        # warm-starting leaves the zero-initialized residuals in place.
        if model.adapters is not None and "adapter_names" in z:
            names = [str(x) for x in z["adapter_names"].tolist()]
            for name in names:
                if name not in model.adapters:
                    raise SystemExit(

--- around line 170 ---
            names = [str(x) for x in z["adapter_names"].tolist()]
            for name in names:
                if name not in model.adapters:
                    raise SystemExit(
                        f"--init {path.name}: unknown adapter {name!r}")
                seq = model.adapters[name]
                lins = [m for m in seq if isinstance(m, nn.Linear)]

--- around line 173 ---
                    raise SystemExit(
                        f"--init {path.name}: unknown adapter {name!r}")
                seq = model.adapters[name]
                lins = [m for m in seq if isinstance(m, nn.Linear)]
                n = int(z[f"adapter_{name}_n"][0])
                if n != len(lins):
                    raise SystemExit(

--- around line 179 ---
                    raise SystemExit(
                        f"--init {path.name}: adapter {name} has {n} layers, "
                        f"model has {len(lins)}")
                for i, lin in enumerate(lins):
                    w, b = (z[f"adapter_{name}{i}_w"],
                            z[f"adapter_{name}{i}_b"])
                    if w.shape != tuple(lin.weight.shape):

--- around line 186 ---
                        raise SystemExit(
                            f"--init {path.name}: adapter_{name}{i}_w is "
                            f"{w.shape}, model wants "
                            f"{tuple(lin.weight.shape)}")
                    lin.weight.copy_(torch.from_numpy(w))
                    lin.bias.copy_(torch.from_numpy(b))
    print(f"warm-started from {path}")

--- around line 193 ---


def _mlp(sizes: list[int], dropout: float, out_dim: int | None) -> nn.Sequential:
    """ReLU MLP over `sizes` hidden widths; `out_dim` appends a linear head."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]

--- around line 203 ---


def _make_adapter(in_dim: int, hidden: int) -> nn.Sequential:
    """Residual logit MLP; final layer is zero-initialized for v5 equivalence."""
    seq = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),

--- around line 215 ---


class PolicyNet(nn.Module):
    def __init__(self, state_h: tuple[int, ...] = (256,),
                 head_h: tuple[int, ...] = (128,), dropout: float = 0.1,
                 opt_cols: int = OPT_DENSE, extra: bool = True,
                 pool: bool = False, outcome: bool = False,

--- around line 216 ---

class PolicyNet(nn.Module):
    def __init__(self, state_h: tuple[int, ...] = (256,),
                 head_h: tuple[int, ...] = (128,), dropout: float = 0.1,
                 opt_cols: int = OPT_DENSE, extra: bool = True,
                 pool: bool = False, outcome: bool = False,
                 count: bool = False, adapter_names: list[str] | None = None,

--- around line 246 ---
        self.atk_emb = nn.Embedding(r.get("atk_emb", N_ATTACK_IDS), EMB,
                                    padding_idx=pi)
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        if extra:                       # the v4 block, appended (features.py)
            in_state += N_EXTRA + N_XSLOT * EMB
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)

--- around line 247 ---
                                    padding_idx=pi)
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        if extra:                       # the v4 block, appended (features.py)
            in_state += N_EXTRA + N_XSLOT * EMB
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)

--- around line 248 ---
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        if extra:                       # the v4 block, appended (features.py)
            in_state += N_EXTRA + N_XSLOT * EMB
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR

--- around line 250 ---
            in_state += N_EXTRA + N_XSLOT * EMB
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB

--- around line 251 ---
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)

--- around line 252 ---
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)
        # Constructed AFTER every policy parameter. Resetting the seed therefore

--- around line 253 ---
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)
        # Constructed AFTER every policy parameter. Resetting the seed therefore
        # gives a control and an auxiliary treatment identical policy weights;

--- around line 254 ---
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)
        # Constructed AFTER every policy parameter. Resetting the seed therefore
        # gives a control and an auxiliary treatment identical policy weights;
        # only the treatment consumes additional RNG after that point.

--- around line 259 ---
        # gives a control and an auxiliary treatment identical policy weights;
        # only the treatment consumes additional RNG after that point.
        self.outcome_head = (nn.Linear(state_h[-1], 1) if outcome else None)
        self.count_head = (nn.Linear(state_h[-1], 1) if count else None)
        # E2 adapters are also append-only and zero-initialized, so an untrained
        # treatment matches the frozen base logits exactly.
        self.adapter_names = list(adapter_names or [])

--- around line 260 ---
        # only the treatment consumes additional RNG after that point.
        self.outcome_head = (nn.Linear(state_h[-1], 1) if outcome else None)
        self.count_head = (nn.Linear(state_h[-1], 1) if count else None)
        # E2 adapters are also append-only and zero-initialized, so an untrained
        # treatment matches the frozen base logits exactly.
        self.adapter_names = list(adapter_names or [])
        self.adapter_route_ids: dict[str, int] = {}

--- around line 280 ---
            self.adapters = None

    def forward(self, dense, slots, bag_flat, bag_off, seld,
                opt_dense, opt_card, opt_atk, opt_tgt, opt_row,
                xdense=None, xslots=None, attrs=None, routes=None,
                return_state: bool = False):
        # The per-option encoding is built FIRST, because the v5 pool feeds it

--- around line 283 ---
                opt_dense, opt_card, opt_atk, opt_tgt, opt_row,
                xdense=None, xslots=None, attrs=None, routes=None,
                return_state: bool = False):
        # The per-option encoding is built FIRST, because the v5 pool feeds it
        # into the state. It is the same tensor the head consumes below, so the
        # pool costs one reduction and no extra embedding lookups.
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,

--- around line 285 ---
                return_state: bool = False):
        # The per-option encoding is built FIRST, because the v5 pool feeds it
        # into the state. It is the same tensor the head consumes below, so the
        # pool costs one reduction and no extra embedding lookups.
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,
        # so `--opt-cols 25` trains the exact v2-feature control on the identical
        # rows -- same games, same selects, same labels, only the features differ.

--- around line 288 ---
        # pool costs one reduction and no extra embedding lookups.
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,
        # so `--opt-cols 25` trains the exact v2-feature control on the identical
        # rows -- same games, same selects, same labels, only the features differ.
        # That is a cleaner control than comparing against the shipped net, which
        # also differs in corpus (2,810 games vs whatever is on disk now).
        oenc = torch.cat([opt_dense[:, :self.opt_cols],

--- around line 289 ---
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,
        # so `--opt-cols 25` trains the exact v2-feature control on the identical
        # rows -- same games, same selects, same labels, only the features differ.
        # That is a cleaner control than comparing against the shipped net, which
        # also differs in corpus (2,810 games vs whatever is on disk now).
        oenc = torch.cat([opt_dense[:, :self.opt_cols],
                          self.card_emb(opt_card),

--- around line 300 ---
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        parts.append(seld)
        # v4 goes LAST so that `--no-extra` reproduces the v3 state vector
        # byte-for-byte on the identical rows -- the same control discipline as
        # `--opt-cols 25` for the option block.
        if self.extra:
            parts.append(xdense)

--- around line 306 ---
            parts.append(xdense)
            parts.append(self.slot_emb(xslots).flatten(1))
        # ...and v5 goes after v4, so `pool=False` reproduces the v4 state
        # vector byte-for-byte. Same discipline, third generation.
        if self.pool:
            parts.append(self._pool(oenc, opt_row, dense.shape[0]))
        # ...and v6 goes after v5, so `attr=False` reproduces the v5 state

--- around line 310 ---
        if self.pool:
            parts.append(self._pool(oenc, opt_row, dense.shape[0]))
        # ...and v6 goes after v5, so `attr=False` reproduces the v5 state
        # vector byte-for-byte. Same discipline, fourth generation.
        if self.attr:
            parts.append(attrs)
        srepr = self.state_fc(torch.cat(parts, dim=1))       # (B, H)

--- around line 314 ---
        if self.attr:
            parts.append(attrs)
        srepr = self.state_fc(torch.cat(parts, dim=1))       # (B, H)
        per_opt = torch.cat([srepr[opt_row], oenc], dim=1)   # (O, ...)
        logits = self.head(per_opt).squeeze(1)               # (O,)
        if (self.adapters is not None and not self.adapters_off
                and routes is not None):

--- around line 327 ---
                        per_opt[mask]).squeeze(1)
            logits = logits + residual
        return (logits, srepr) if return_state else logits

    def _pool(self, oenc: torch.Tensor, opt_row: torch.Tensor,
              n_rows: int) -> torch.Tensor:
        """Segment mean/max of the option encodings + two count scalars.

--- around line 329 ---
        return (logits, srepr) if return_state else logits

    def _pool(self, oenc: torch.Tensor, opt_row: torch.Tensor,
              n_rows: int) -> torch.Tensor:
        """Segment mean/max of the option encodings + two count scalars.

        A permutation-invariant summary of the option SET, which is the one

--- around line 352 ---


def parse_episode_span(spec: str) -> tuple[float, float]:
    """Parse `--episode-span START:END` into inclusive-exclusive fractions."""
    if ":" not in spec:
        raise SystemExit("--episode-span needs START:END (e.g. 0:0.5)")
    a, b = (s.strip() for s in spec.split(":", 1))

--- around line 353 ---

def parse_episode_span(spec: str) -> tuple[float, float]:
    """Parse `--episode-span START:END` into inclusive-exclusive fractions."""
    if ":" not in spec:
        raise SystemExit("--episode-span needs START:END (e.g. 0:0.5)")
    a, b = (s.strip() for s in spec.split(":", 1))
    try:

--- around line 366 ---


def episode_span_mask(gid: np.ndarray, start: float, end: float) -> np.ndarray:
    """True for rows in [floor(n*start), floor(n*end)) within each gid.

    Row order is the order they appear in `gid` (shard-concat chronological
    order from build_policy_dataset). Odd-length games put the middle row in

--- around line 370 ---

    Row order is the order they appear in `gid` (shard-concat chronological
    order from build_policy_dataset). Odd-length games put the middle row in
    the second half when start=0.5 (floor splits)."""
    keep = np.zeros(len(gid), dtype=bool)
    # First pass: counts per gid in appearance order, without sorting the
    # whole array (gids are not contiguous across day dirs).

--- around line 387 ---


def listwise_loss(out: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor, n_rows: int,
                  w: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax cross-entropy within each select's option set, averaged over
    the chosen options of that select. This is the objective that matches

--- around line 415 ---


def count_targets(seld: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target fraction and validity mask for variable-count selects.

    This is the row-level equivalent of `count_fraction_table`: the old table

--- around line 417 ---
def count_targets(seld: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target fraction and validity mask for variable-count selects.

    This is the row-level equivalent of `count_fraction_table`: the old table
    averages these targets within `(selectType, context)` buckets, while E1
    asks the shared state representation to predict each row separately.

--- around line 419 ---
    """Return target fraction and validity mask for variable-count selects.

    This is the row-level equivalent of `count_fraction_table`: the old table
    averages these targets within `(selectType, context)` buckets, while E1
    asks the shared state representation to predict each row separately.
    """
    n_rows = seld.shape[0]

--- around line 421 ---
    This is the row-level equivalent of `count_fraction_table`: the old table
    averages these targets within `(selectType, context)` buckets, while E1
    asks the shared state representation to predict each row separately.
    """
    n_rows = seld.shape[0]
    picked = torch.zeros(n_rows, dtype=chosen.dtype,
                         device=chosen.device).index_add_(0, opt_row, chosen)

--- around line 433 ---


class Data:
    def __init__(self, paths: list[Path], want_attr: bool = True):
        # ⚠ `want_attr=False` skips the v6 attribute block entirely instead of
        # materialising it. It is 276 float32 per row -- 27.6% of this object --
        # and `Model.forward` only reads it when the net was built with

--- around line 434 ---

class Data:
    def __init__(self, paths: list[Path], want_attr: bool = True):
        # ⚠ `want_attr=False` skips the v6 attribute block entirely instead of
        # materialising it. It is 276 float32 per row -- 27.6% of this object --
        # and `Model.forward` only reads it when the net was built with
        # `--attr`, so under the v5 recipe every one of those bytes is loaded,

--- around line 437 ---
        # ⚠ `want_attr=False` skips the v6 attribute block entirely instead of
        # materialising it. It is 276 float32 per row -- 27.6% of this object --
        # and `Model.forward` only reads it when the net was built with
        # `--attr`, so under the v5 recipe every one of those bytes is loaded,
        # copied per batch and moved to the device to be discarded. On the
        # 40.1M-row corpus that is 44.3 GB resident and ~89 GB of load peak for
        # nothing. Verified no-op: train loss identical to 4 dp with and

--- around line 455 ---
        # as the shards store them. The previous version materialised one small
        # numpy array per row per bag -- 249k rows x 3 bags = ~750k objects --
        # and that allocation, not the model, is what OOM'd this 7.3 GB machine
        # on any net above ~1.5M params. Same semantics, ~1 GB less resident.
        bag_flats: dict[str, list] = {n: [] for n in BAGS}
        bag_lens: dict[str, list] = {n: [] for n in BAGS}
        base = 0

--- around line 461 ---
        base = 0
        for p in paths:
            z = np.load(p)
            n = len(z["gid"])
            sd.append(z["dense"])
            slots.append(z["slots"])
            seld.append(z["seld"])

--- around line 521 ---
        self.xslots = np.concatenate(xs).astype(np.int64)
        self.attr = np.concatenate(at)
        self.has_extra = all("xdense" in np.load(p) for p in paths)
        self.has_attr = all("attr" in np.load(p) for p in paths)
        self.gid = np.concatenate(gid)
        self.won = np.concatenate(won)
        self.rating = np.concatenate(rating)

--- around line 522 ---
        self.attr = np.concatenate(at)
        self.has_extra = all("xdense" in np.load(p) for p in paths)
        self.has_attr = all("attr" in np.load(p) for p in paths)
        self.gid = np.concatenate(gid)
        self.won = np.concatenate(won)
        self.rating = np.concatenate(rating)
        self.margin = np.concatenate(margin)

--- around line 551 ---
            self.bag_off["opp_discard"])

    def batches(self, idx: np.ndarray, bs: int,
                rng: np.random.Generator | None):
        order = rng.permutation(idx) if rng is not None else idx
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]

--- around line 593 ---


def td_advantage_weights(data: "Data", is_rl: np.ndarray, beta: float,
                         anchor_w: float) -> np.ndarray:
    """E27: AWR over the PER-DECISION TD residual, not the game result.

    `w = exp(beta * A / sd(A))` on RL rows. **The normalisation by sd(A) is the

--- around line 649 ---


def advantage_weights(data: "Data", is_rl: np.ndarray, beta: float,
                      anchor_w: float, margin_max: float) -> np.ndarray:
    """B8: advantage-weighted regression over our OWN recorded outcomes.

    The weight on an RL row is `exp((won - baseline) / beta)`, so a select from

--- around line 704 ---


def apply_freeze(model: "PolicyNet", spec: str) -> None:
    """Train only the named top-level parameter groups; freeze the rest.

    B8's pre-registered form is "fine-tune a SMALL parameter set", and the
    reason is §8w: 8.2x the parameters bought -43 decisions, so capacity is not

--- around line 713 ---
    """
    keep = {s.strip() for s in spec.split(",") if s.strip()}
    known = {n.split(".", 1)[0] for n, _ in model.named_parameters()}
    unknown = keep - known
    if unknown:
        raise SystemExit(f"--freeze-except names {sorted(unknown)}; "
                         f"parameter groups are {sorted(known)}")

--- around line 719 ---
                         f"parameter groups are {sorted(known)}")
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        if name.split(".", 1)[0] in keep:
            n_train += p.numel()
        else:
            p.requires_grad_(False)

--- around line 729 ---


class StreamData:
    """Shard-at-a-time corpus, for a corpus that does not fit in RAM.

    `Data` concatenates everything: ~4.0 KB/row resident and ~7.8 KB/row at the
    load peak, which is 160 GB / 313 GB on the 40.1M-row host corpus against a

--- around line 749 ---
    """

    def __init__(self, paths: list[Path], buffer: int = 4,
                 want_attr: bool = False):
        self.paths = list(paths)
        self.buffer = max(1, buffer)
        self.want_attr = want_attr

--- around line 760 ---
        t0 = time.time()
        for i, p in enumerate(self.paths):
            with np.load(p) as z:
                n = len(z["gid"])
                gid.append(z["gid"])
                won.append(z["won"])
                nan = np.full(n, np.nan, dtype=np.float32)

--- around line 798 ---
    # `Data` materialises these; nothing in the loop reads them when streaming,
    # so fail loudly rather than return something silently wrong.
    def _refuse(self, what: str):
        raise SystemExit(f"--stream does not support {what}; it needs the whole "
                         f"corpus resident, which is the thing --stream exists "
                         f"to avoid")


--- around line 804 ---

    @property
    def attr(self):
        self._refuse("--drop-a / whole-corpus attr access")

    def _groups(self, idx: np.ndarray):
        """Global row indices -> [(shard, its rows)], in shard order."""

--- around line 807 ---
        self._refuse("--drop-a / whole-corpus attr access")

    def _groups(self, idx: np.ndarray):
        """Global row indices -> [(shard, its rows)], in shard order."""
        sh = np.searchsorted(self._off, idx, side="right") - 1
        order = np.argsort(sh, kind="stable")
        idx_s, sh_s = idx[order], sh[order]

--- around line 816 ---
                for k in range(len(self.paths)) if bounds[k + 1] > bounds[k]]

    def batches(self, idx: np.ndarray, bs: int,
                rng: "np.random.Generator | None"):
        groups = self._groups(np.asarray(idx))
        gorder = (rng.permutation(len(groups)) if rng is not None
                  else np.arange(len(groups)))

--- around line 848 ---


def count_fraction_table_stream(data: "StreamData", idx: np.ndarray
                                ) -> np.ndarray:
    """count_fraction_table over a streamed corpus: same arithmetic, one shard
    of `seld`/`opt_chosen` resident at a time."""
    num = np.zeros((11, 64))

--- around line 850 ---
def count_fraction_table_stream(data: "StreamData", idx: np.ndarray
                                ) -> np.ndarray:
    """count_fraction_table over a streamed corpus: same arithmetic, one shard
    of `seld`/`opt_chosen` resident at a time."""
    num = np.zeros((11, 64))
    den = np.zeros((11, 64))
    for k, rows in data._groups(np.asarray(idx)):

--- around line 856 ---
    for k, rows in data._groups(np.asarray(idx)):
        local = rows - data._off[k]
        with np.load(data.paths[k]) as z:
            seld_all = z["seld"]
            off = z["opt_off"]
            chosen = z["opt_chosen"]
        for j in local:

--- around line 875 ---


def count_fraction_table(data: "Data", idx: np.ndarray) -> np.ndarray:
    """(11, 64) mean of (chosen-min)/(max-min) per (selectType, context) over
    variable-count selects. Unseen cells default to 1.0 (take the max), which
    matches top play for searches/benching."""
    num = np.zeros((11, 64))

--- around line 898 ---


def apply_x_drop(data: "Data", names: list[str]) -> np.ndarray:
    """Zero the named members of the v4 state block and return the mask.

    Zeroing rather than deleting keeps the layer widths, the parameter count and
    the weight init identical to the full-block run, so a drop-one arm differs

--- around line 899 ---

def apply_x_drop(data: "Data", names: list[str]) -> np.ndarray:
    """Zero the named members of the v4 state block and return the mask.

    Zeroing rather than deleting keeps the layer widths, the parameter count and
    the weight init identical to the full-block run, so a drop-one arm differs
    from it in the CONTENT of a few columns and nothing else. An xslot set to 0

--- around line 921 ---


def apply_a_drop(data: "Data", names: list[str]) -> np.ndarray:
    """Zero the named members of the v6 attribute block and return the mask.

    Same discipline as `apply_x_drop`: the block ships whole, so without a
    drop-one arm nothing would say WHICH of energyType / weakness / ability /

--- around line 942 ---


def export_npz(model: PolicyNet, path: Path, count_frac: np.ndarray,
               x_mask: np.ndarray | None = None,
               a_mask: np.ndarray | None = None,
               remap: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
    """Export every Linear generically, so inference mirrors any depth."""

--- around line 947 ---
               remap: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
    """Export every Linear generically, so inference mirrors any depth."""
    def arr(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    out: dict[str, np.ndarray] = {
        "slot_emb": arr(model.slot_emb.weight),

--- around line 951 ---

    out: dict[str, np.ndarray] = {
        "slot_emb": arr(model.slot_emb.weight),
        "bag_emb": arr(model.bag_emb.weight),
        "card_emb": arr(model.card_emb.weight),
        "atk_emb": arr(model.atk_emb.weight),
        "count_frac": count_frac,

--- around line 952 ---
    out: dict[str, np.ndarray] = {
        "slot_emb": arr(model.slot_emb.weight),
        "bag_emb": arr(model.bag_emb.weight),
        "card_emb": arr(model.card_emb.weight),
        "atk_emb": arr(model.atk_emb.weight),
        "count_frac": count_frac,
        # Width of the v5 pooled block, 0 if the net has none. Inference cannot

--- around line 953 ---
        "slot_emb": arr(model.slot_emb.weight),
        "bag_emb": arr(model.bag_emb.weight),
        "card_emb": arr(model.card_emb.weight),
        "atk_emb": arr(model.atk_emb.weight),
        "count_frac": count_frac,
        # Width of the v5 pooled block, 0 if the net has none. Inference cannot
        # derive this from `state_in` alone (the v4 and v5 widths are both

--- around line 954 ---
        "bag_emb": arr(model.bag_emb.weight),
        "card_emb": arr(model.card_emb.weight),
        "atk_emb": arr(model.atk_emb.weight),
        "count_frac": count_frac,
        # Width of the v5 pooled block, 0 if the net has none. Inference cannot
        # derive this from `state_in` alone (the v4 and v5 widths are both
        # legal), so it is recorded explicitly. Nets exported before day 13 have

--- around line 957 ---
        "count_frac": count_frac,
        # Width of the v5 pooled block, 0 if the net has none. Inference cannot
        # derive this from `state_in` alone (the v4 and v5 widths are both
        # legal), so it is recorded explicitly. Nets exported before day 13 have
        # no such key and are read as 0.
        "n_pool": np.array([pool_width(model.opt_cols, EMB) if model.pool
                            else 0], dtype=np.int64),

--- around line 960 ---
        # legal), so it is recorded explicitly. Nets exported before day 13 have
        # no such key and are read as 0.
        "n_pool": np.array([pool_width(model.opt_cols, EMB) if model.pool
                            else 0], dtype=np.int64),
        # Width of the v6 attribute block, 0 if the net has none. Same reason as
        # n_pool: `state_in` alone no longer identifies the layout once three
        # optional blocks exist. Nets exported before day 20 lack this key and

--- around line 963 ---
                            else 0], dtype=np.int64),
        # Width of the v6 attribute block, 0 if the net has none. Same reason as
        # n_pool: `state_in` alone no longer identifies the layout once three
        # optional blocks exist. Nets exported before day 20 lack this key and
        # are read as 0.
        "n_attr": np.array([N_ATTR if model.attr else 0], dtype=np.int64),
    }

--- around line 966 ---
        # optional blocks exist. Nets exported before day 20 lack this key and
        # are read as 0.
        "n_attr": np.array([N_ATTR if model.attr else 0], dtype=np.int64),
    }
    if remap is not None:
        # The raw ids this net's rows stand for, in row order after PAD and UNK.
        # Inference rebuilds the lookup from these, so the map can never drift

--- around line 980 ---
        # applies it, so an ablation arm can never be fed a column it never saw.
        out["x_mask"] = x_mask
    for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
        n = 0
        for mod in seq:
            if isinstance(mod, nn.Linear):
                out[f"{prefix}{n}_w"] = arr(mod.weight)

--- around line 988 ---
                n += 1
        out[f"n_{prefix}"] = np.array([n], dtype=np.int64)
    for prefix, head in (("outcome", model.outcome_head),
                         ("count", model.count_head)):
        if head is not None:
            out[f"{prefix}_w"] = arr(head.weight)
            out[f"{prefix}_b"] = arr(head.bias)

--- around line 989 ---
        out[f"n_{prefix}"] = np.array([n], dtype=np.int64)
    for prefix, head in (("outcome", model.outcome_head),
                         ("count", model.count_head)):
        if head is not None:
            out[f"{prefix}_w"] = arr(head.weight)
            out[f"{prefix}_b"] = arr(head.bias)
    if model.adapters is not None:

--- around line 993 ---
            out[f"{prefix}_w"] = arr(head.weight)
            out[f"{prefix}_b"] = arr(head.bias)
    if model.adapters is not None:
        out["adapter_names"] = np.asarray(model.adapter_names)
        out["adapter_h"] = np.array([model.adapter_h], dtype=np.int64)
        out["adapter_route_ids"] = np.asarray(
            [model.adapter_route_ids[n] for n in model.adapter_names],

--- around line 994 ---
            out[f"{prefix}_b"] = arr(head.bias)
    if model.adapters is not None:
        out["adapter_names"] = np.asarray(model.adapter_names)
        out["adapter_h"] = np.array([model.adapter_h], dtype=np.int64)
        out["adapter_route_ids"] = np.asarray(
            [model.adapter_route_ids[n] for n in model.adapter_names],
            dtype=np.int64)

--- around line 995 ---
    if model.adapters is not None:
        out["adapter_names"] = np.asarray(model.adapter_names)
        out["adapter_h"] = np.array([model.adapter_h], dtype=np.int64)
        out["adapter_route_ids"] = np.asarray(
            [model.adapter_route_ids[n] for n in model.adapter_names],
            dtype=np.int64)
        for name, seq in model.adapters.items():

--- around line 997 ---
        out["adapter_h"] = np.array([model.adapter_h], dtype=np.int64)
        out["adapter_route_ids"] = np.asarray(
            [model.adapter_route_ids[n] for n in model.adapter_names],
            dtype=np.int64)
        for name, seq in model.adapters.items():
            n = 0
            for mod in seq:

--- around line 999 ---
            [model.adapter_route_ids[n] for n in model.adapter_names],
            dtype=np.int64)
        for name, seq in model.adapters.items():
            n = 0
            for mod in seq:
                if isinstance(mod, nn.Linear):
                    out[f"adapter_{name}{n}_w"] = arr(mod.weight)

--- around line 1007 ---
                    n += 1
            out[f"adapter_{name}_n"] = np.array([n], dtype=np.int64)
    np.savez_compressed(path, **out)
    print(f"exported -> {path}")


def main() -> int:

--- around line 1011 ---


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/pds",
                    help="shard dir; comma-separated for several")
    ap.add_argument("--anchor-ds", default="",

--- around line 1030 ---
                    help="weight held by --anchor-ds rows")
    ap.add_argument("--primary-mass", type=float, default=0.0,
                    help="E3: target fraction of total supervised loss assigned "
                         "to --ds rows, with --anchor-ds supplying the remaining "
                         "mass. For example 0.1 gives curated DAgger labels 10%% "
                         "and the frozen BC corpus 90%%. 0 disables.")
    ap.add_argument("--margin-max", type=float, default=0.0,

--- around line 1037 ---
                    help="B8: only re-weight selects whose top1-top2 logit "
                         "margin was <= this. 0 = re-weight every RL row.")
    ap.add_argument("--export-last", action="store_true",
                    help="export the FINAL epoch instead of the best-val one. "
                         "Required on both arms of any A/B where one arm's "
                         "objective is not corpus fit (rule 3) -- otherwise "
                         "the arms export different epochs and the comparison "

--- around line 1053 ---
                         "rating cut or an archetype filter is applied to an "
                         "already-built corpus WITHOUT rebuilding it.")
    ap.add_argument("--stream", action="store_true",
                    help="load shards a buffer at a time instead of "
                         "concatenating the corpus. Required above ~4M rows. "
                         "⚠ shuffling becomes buffer-local; declare it.")
    ap.add_argument("--stream-buffer", type=int, default=4,

--- around line 1066 ---
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--episode-span", default="",
                    help="keep only a chronological fraction of each game's "
                         "rows: START:END in [0,1] (e.g. 0:0.5 = first half, "
                         "0.5:1 = last half). Split is by decision-row count "

--- around line 1068 ---
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--episode-span", default="",
                    help="keep only a chronological fraction of each game's "
                         "rows: START:END in [0,1] (e.g. 0:0.5 = first half, "
                         "0.5:1 = last half). Split is by decision-row count "
                         "per gid, in shard-concat order. Empty = full episode.")
    ap.add_argument("--loss", choices=("bce", "listwise", "both"),

--- around line 1074 ---
    ap.add_argument("--loss", choices=("bce", "listwise", "both"),
                    default="listwise")
    ap.add_argument("--state-h", default="256",
                    help="comma-separated hidden widths for the state MLP")
    ap.add_argument("--head-h", default="128",
                    help="comma-separated hidden widths for the scoring MLP")
    ap.add_argument("--dropout", type=float, default=0.1)

--- around line 1075 ---
                    default="listwise")
    ap.add_argument("--state-h", default="256",
                    help="comma-separated hidden widths for the state MLP")
    ap.add_argument("--head-h", default="128",
                    help="comma-separated hidden widths for the scoring MLP")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",

--- around line 1083 ---
                         "recipes; use cuda for the private E1 GPU sweep.")
    ap.add_argument("--aux-outcome-w", type=float, default=0.0,
                    help="E1: weight of win/loss BCE on the shared state "
                         "representation. 0 disables the outcome head.")
    ap.add_argument("--aux-count-w", type=float, default=0.0,
                    help="E1: weight of soft-label BCE for the selected-count "
                         "fraction on variable-count rows. 0 disables the head.")

--- around line 1087 ---
    ap.add_argument("--aux-count-w", type=float, default=0.0,
                    help="E1: weight of soft-label BCE for the selected-count "
                         "fraction on variable-count rows. 0 disables the head.")
    ap.add_argument("--out", default="agents/sa/policy_net.npz")
    ap.add_argument("--rating-temp", type=float, default=0.0,
                    help="ROADMAP B7: weight each row by "
                         "exp((rating - max) / T), normalised to mean 1. Small "

--- around line 1088 ---
                    help="E1: weight of soft-label BCE for the selected-count "
                         "fraction on variable-count rows. 0 disables the head.")
    ap.add_argument("--out", default="agents/sa/policy_net.npz")
    ap.add_argument("--rating-temp", type=float, default=0.0,
                    help="ROADMAP B7: weight each row by "
                         "exp((rating - max) / T), normalised to mean 1. Small "
                         "T = clone the best demonstrators only; 0 (default) = "

--- around line 1101 ---
                         "Shapes must match the arch flags.")
    ap.add_argument("--opt-cols", type=int, default=OPT_DENSE,
                    help="per-option feature columns to use. Default = all "
                         f"({OPT_DENSE}). Pass {OPT_DENSE_V2} to train the "
                         "v2-feature CONTROL on identical rows (ROADMAP B1).")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed. Vary it to SIZE run-to-run "

--- around line 1103 ---
                    help="per-option feature columns to use. Default = all "
                         f"({OPT_DENSE}). Pass {OPT_DENSE_V2} to train the "
                         "v2-feature CONTROL on identical rows (ROADMAP B1).")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed. Vary it to SIZE run-to-run "
                         "variance, which is the confound behind every "
                         "net-vs-net A/B in this repo (§8z).")

--- around line 1109 ---
                         "net-vs-net A/B in this repo (§8z).")
    ap.add_argument("--drop-x", default="",
                    help="comma-separated members of the v4 state block to "
                         "ABLATE (features.X_GROUPS: "
                         f"{','.join(X_GROUPS)}). The surviving mask is stored "
                         "in the npz and applied at inference.")
    ap.add_argument("--attr", action="store_true",

--- around line 1110 ---
    ap.add_argument("--drop-x", default="",
                    help="comma-separated members of the v4 state block to "
                         "ABLATE (features.X_GROUPS: "
                         f"{','.join(X_GROUPS)}). The surviving mask is stored "
                         "in the npz and applied at inference.")
    ap.add_argument("--attr", action="store_true",
                    help="the v6 block: append per-slot CARD ATTRIBUTES "

--- around line 1113 ---
                         f"{','.join(X_GROUPS)}). The surviving mask is stored "
                         "in the npz and applied at inference.")
    ap.add_argument("--attr", action="store_true",
                    help="the v6 block: append per-slot CARD ATTRIBUTES "
                         "(energyType, weakness, ability, resistance, "
                         "weak-to-facing-type) to the STATE vector. These come "
                         "from the card DB, which covers all 1,267 cards, so "

--- around line 1116 ---
                    help="the v6 block: append per-slot CARD ATTRIBUTES "
                         "(energyType, weakness, ability, resistance, "
                         "weak-to-facing-type) to the STATE vector. These come "
                         "from the card DB, which covers all 1,267 cards, so "
                         "unlike an embedding row they transfer to cards the "
                         "corpus never contained (E6). Default off = the v5 "
                         "state vector, byte-for-byte, on identical rows.")

--- around line 1120 ---
                         "unlike an embedding row they transfer to cards the "
                         "corpus never contained (E6). Default off = the v5 "
                         "state vector, byte-for-byte, on identical rows.")
    ap.add_argument("--drop-a", default="",
                    help="comma-separated members of the v6 attribute block to "
                         "ABLATE (features.A_GROUPS: "
                         f"{','.join(A_GROUPS)}). The surviving mask is stored "

--- around line 1123 ---
    ap.add_argument("--drop-a", default="",
                    help="comma-separated members of the v6 attribute block to "
                         "ABLATE (features.A_GROUPS: "
                         f"{','.join(A_GROUPS)}). The surviving mask is stored "
                         "in the npz and applied at inference.")
    ap.add_argument("--vocab", default="",
                    help="the v7 block: an out/emb/vocab.json census. Collapses "

--- around line 1133 ---
                         "--pad. Default off = the v3-v6 raw id space, i.e. an "
                         "unseen card reads a random untrained row.")
    ap.add_argument("--pad", action="store_true",
                    help="pin embedding row 0 to zero and give it no gradient "
                         "(padding_idx). Alone, this is the v7 block's SECOND "
                         "half only -- the arm that isolates 'id 0 is "
                         "overloaded' from 'unseen cards read noise'.")

--- around line 1138 ---
                         "half only -- the arm that isolates 'id 0 is "
                         "overloaded' from 'unseen cards read noise'.")
    ap.add_argument("--pool", action="store_true",
                    help="the v5 block: append a mean/max pool of the option "
                         "encodings + two count scalars to the STATE vector "
                         "(optfeat.pool_width). Default off = the v4 state "
                         "vector byte-for-byte, i.e. the control.")

--- around line 1140 ---
    ap.add_argument("--pool", action="store_true",
                    help="the v5 block: append a mean/max pool of the option "
                         "encodings + two count scalars to the STATE vector "
                         "(optfeat.pool_width). Default off = the v4 state "
                         "vector byte-for-byte, i.e. the control.")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the v4 state block (features.extra_feats). "

--- around line 1141 ---
                    help="the v5 block: append a mean/max pool of the option "
                         "encodings + two count scalars to the STATE vector "
                         "(optfeat.pool_width). Default off = the v4 state "
                         "vector byte-for-byte, i.e. the control.")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the v4 state block (features.extra_feats). "
                         "This is the day-12 CONTROL: identical rows, "

--- around line 1143 ---
                         "(optfeat.pool_width). Default off = the v4 state "
                         "vector byte-for-byte, i.e. the control.")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the v4 state block (features.extra_feats). "
                         "This is the day-12 CONTROL: identical rows, "
                         "identical recipe, the v3 state vector byte-for-byte.")
    ap.add_argument("--adapters", default="",

--- around line 1144 ---
                         "vector byte-for-byte, i.e. the control.")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the v4 state block (features.extra_feats). "
                         "This is the day-12 CONTROL: identical rows, "
                         "identical recipe, the v3 state vector byte-for-byte.")
    ap.add_argument("--adapters", default="",
                    help="E2: comma-separated residual adapter names "

--- around line 1146 ---
                    help="ignore the v4 state block (features.extra_feats). "
                         "This is the day-12 CONTROL: identical rows, "
                         "identical recipe, the v3 state vector byte-for-byte.")
    ap.add_argument("--adapters", default="",
                    help="E2: comma-separated residual adapter names "
                         "(mirror,alakazam). Append-only; zero-initialized so "
                         "an untrained treatment matches the frozen base.")

--- around line 1153 ---
    ap.add_argument("--adapter-h", type=int, default=64,
                    help="E2: hidden width of each residual adapter MLP")
    ap.add_argument("--adapters-off", action="store_true",
                    help="E2 control: keep adapters in the checkpoint but do "
                         "not add their residual during forward/training")
    args = ap.parse_args()
    adapter_names = [s.strip() for s in args.adapters.split(",") if s.strip()]

--- around line 1154 ---
                    help="E2: hidden width of each residual adapter MLP")
    ap.add_argument("--adapters-off", action="store_true",
                    help="E2 control: keep adapters in the checkpoint but do "
                         "not add their residual during forward/training")
    args = ap.parse_args()
    adapter_names = [s.strip() for s in args.adapters.split(",") if s.strip()]
    if not 1 <= args.opt_cols <= OPT_DENSE:

--- around line 1188 ---
        torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    # Seeded so that a control/treatment pair (e.g. --opt-cols 25 vs 37, ROADMAP
    # B1) differs in its FEATURES and not in dropout masks or batch order. Weight
    # init still differs where the layer widths differ, which cannot be avoided.
    torch.manual_seed(args.seed)
    paths: list[Path] = []
    for d in args.ds.split(","):

--- around line 1271 ---
        keep &= sel
        # ⚠ rule 9: a filter that matches almost nothing must not look like a
        # small corpus. Say what fraction survived, in games as well as rows.
        print(f"--keep-gids {kp.name}: {len(want):,} ids -> "
              f"{int(sel.sum()):,} of {data.n:,} rows kept "
              f"({int(sel.sum())/max(data.n,1):.1%}), "
              f"{len(np.unique(data.gid[sel])):,} episodes matched")

--- around line 1336 ---
        if not n_primary_rows or not n_anchor_rows:
            raise SystemExit("--primary-mass needs non-empty primary and anchor "
                             "datasets")
        primary_w = (args.primary_mass / (1.0 - args.primary_mass)
                     * n_anchor_rows / n_primary_rows)
        data.w = np.ones(data.n, dtype=np.float32)
        data.w[is_rl] = primary_w

--- around line 1350 ---
          f"{len(val_idx)} val ({len(np.unique(data.gid))} games)")

    count_frac = (count_fraction_table_stream(data, train_idx) if args.stream
                  else count_fraction_table(data, train_idx))

    state_h = tuple(int(x) for x in args.state_h.split(","))
    head_h = tuple(int(x) for x in args.head_h.split(","))

--- around line 1351 ---

    count_frac = (count_fraction_table_stream(data, train_idx) if args.stream
                  else count_fraction_table(data, train_idx))

    state_h = tuple(int(x) for x in args.state_h.split(","))
    head_h = tuple(int(x) for x in args.head_h.split(","))
    if not args.no_extra and not data.has_extra:

--- around line 1353 ---
                  else count_fraction_table(data, train_idx))

    state_h = tuple(int(x) for x in args.state_h.split(","))
    head_h = tuple(int(x) for x in args.head_h.split(","))
    if not args.no_extra and not data.has_extra:
        raise SystemExit(f"{args.ds} was built before the v4 state block; "
                         "rebuild it or pass --no-extra")

--- around line 1356 ---
    head_h = tuple(int(x) for x in args.head_h.split(","))
    if not args.no_extra and not data.has_extra:
        raise SystemExit(f"{args.ds} was built before the v4 state block; "
                         "rebuild it or pass --no-extra")
    if args.attr and not data.has_attr:
        raise SystemExit(f"{args.ds} was built before the v6 attribute block; "
                         "rebuild it with scripts/build_policy_dataset.py")

--- around line 1360 ---
    if args.attr and not data.has_attr:
        raise SystemExit(f"{args.ds} was built before the v6 attribute block; "
                         "rebuild it with scripts/build_policy_dataset.py")
    if args.drop_a and not args.attr:
        raise SystemExit("--drop-a ablates the v6 block; it needs --attr")
    if args.pool and args.no_extra:
        raise SystemExit("--pool is the v5 block and is defined as v4 + pool; "

--- around line 1365 ---
    if args.pool and args.no_extra:
        raise SystemExit("--pool is the v5 block and is defined as v4 + pool; "
                         "inference only knows the v4 and v5 state widths")
    model = PolicyNet(state_h=state_h, head_h=head_h, dropout=args.dropout,
                      opt_cols=args.opt_cols, extra=not args.no_extra,
                      pool=args.pool, outcome=args.aux_outcome_w > 0,
                      count=args.aux_count_w > 0,

--- around line 1366 ---
        raise SystemExit("--pool is the v5 block and is defined as v4 + pool; "
                         "inference only knows the v4 and v5 state widths")
    model = PolicyNet(state_h=state_h, head_h=head_h, dropout=args.dropout,
                      opt_cols=args.opt_cols, extra=not args.no_extra,
                      pool=args.pool, outcome=args.aux_outcome_w > 0,
                      count=args.aux_count_w > 0,
                      adapter_names=adapter_names or None,

--- around line 1375 ---
                      attr=args.attr, rows=rows, pad=args.pad).to(device)
    if args.init:
        load_init(model, ROOT / args.init)
    if args.freeze_except:
        if not args.init:
            raise SystemExit("--freeze-except without --init trains a few "
                             "layers on top of RANDOM embeddings; that is not "

--- around line 1381 ---
                             "layers on top of RANDOM embeddings; that is not "
                             "a fine-tune of anything")
        apply_freeze(model, args.freeze_except)
    if args.adapters_off and model.adapters is not None:
        # Control: adapters exist for export shape parity but must not train.
        for p in model.adapters.parameters():
            p.requires_grad_(False)

--- around line 1382 ---
                             "a fine-tune of anything")
        apply_freeze(model, args.freeze_except)
    if args.adapters_off and model.adapters is not None:
        # Control: adapters exist for export shape parity but must not train.
        for p in model.adapters.parameters():
            p.requires_grad_(False)
    if adapter_names:

--- around line 1384 ---
    if args.adapters_off and model.adapters is not None:
        # Control: adapters exist for export shape parity but must not train.
        for p in model.adapters.parameters():
            p.requires_grad_(False)
    if adapter_names:
        route_counts = {
            ROUTE_NAMES[rid]: int((data.routes == rid).sum())

--- around line 1393 ---
        print(f"E2 routes: {route_counts} adapters={adapter_names} "
              f"h={args.adapter_h} off={args.adapters_off}")
    print(f"arch: state{list(state_h)} head{list(head_h)} loss={args.loss} "
          f"opt_cols={args.opt_cols}/{OPT_DENSE} "
          f"extra={not args.no_extra} pool={args.pool} attr={args.attr} "
          f"vocab={bool(args.vocab)} pad={args.pad} "
          f"emb_params={sum(e.weight.numel() for e in (model.slot_emb, model.bag_emb, model.card_emb, model.atk_emb)):,} "

--- around line 1397 ---
          f"extra={not args.no_extra} pool={args.pool} attr={args.attr} "
          f"vocab={bool(args.vocab)} pad={args.pad} "
          f"emb_params={sum(e.weight.numel() for e in (model.slot_emb, model.bag_emb, model.card_emb, model.atk_emb)):,} "
          f"aux_outcome={args.aux_outcome_w:g} "
          f"aux_count={args.aux_count_w:g} "
          f"adapters={adapter_names or []} "
          f"device={device.type} "

--- around line 1402 ---
          f"adapters={adapter_names or []} "
          f"device={device.type} "
          f"params={sum(p.numel() for p in model.parameters())}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
           if trainable else None)
    bcef = nn.BCEWithLogitsLoss()

--- around line 1403 ---
          f"device={device.type} "
          f"params={sum(p.numel() for p in model.parameters())}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
           if trainable else None)
    bcef = nn.BCEWithLogitsLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

--- around line 1413 ---
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for batch in data.batches(train_idx, args.bs, rng):
            (dense, slots, bf, bo, seld, odn, ocd, oat, otg, orow, om,

--- around line 1430 ---
            if opt is not None:
                opt.zero_grad()
            need_state = args.aux_outcome_w > 0 or args.aux_count_w > 0
            result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                           orow, xd, xs, at, routes=routes,
                           return_state=need_state)
            if need_state:

--- around line 1431 ---
                opt.zero_grad()
            need_state = args.aux_outcome_w > 0 or args.aux_count_w > 0
            result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                           orow, xd, xs, at, routes=routes,
                           return_state=need_state)
            if need_state:
                out, srepr = result

--- around line 1433 ---
            result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                           orow, xd, xs, at, routes=routes,
                           return_state=need_state)
            if need_state:
                out, srepr = result
            else:
                out, srepr = result, None

--- around line 1434 ---
                           orow, xd, xs, at, routes=routes,
                           return_state=need_state)
            if need_state:
                out, srepr = result
            else:
                out, srepr = result, None
            loss = torch.zeros((), dtype=out.dtype)

--- around line 1450 ---
                loss = loss + listwise_loss(out, om, orow, len(spans), wrow)
            if args.aux_outcome_w > 0:
                pred = model.outcome_head(srepr).squeeze(1)
                target = torch.from_numpy(data.won[_sel]).to(
                    device=device, dtype=pred.dtype)
                per = bce_none(pred, target)
                aux = (per.mean() if wrow is None else

--- around line 1458 ---
                loss = loss + args.aux_outcome_w * aux
            if args.aux_count_w > 0:
                pred = model.count_head(srepr).squeeze(1)
                target, valid = count_targets(seld, om, orow)
                if valid.any():
                    per = bce_none(pred[valid], target[valid])
                    aux = (per.mean() if wrow is None else

--- around line 1472 ---
            seen += len(om)
        # val: top-1 accuracy on single-choice rows
        model.eval()
        hit = tries = 0
        hit_hi = tries_hi = 0
        route_hit = {name: 0 for name in ROUTE_NAMES.values()}
        route_tries = {name: 0 for name in ROUTE_NAMES.values()}

--- around line 1492 ---
                at = at.to(device)
                routes = routes.to(device)
                need_state = args.aux_outcome_w > 0 or args.aux_count_w > 0
                result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                               orow, xd, xs, at, routes=routes,
                               return_state=need_state)
                if need_state:

--- around line 1493 ---
                routes = routes.to(device)
                need_state = args.aux_outcome_w > 0 or args.aux_count_w > 0
                result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                               orow, xd, xs, at, routes=routes,
                               return_state=need_state)
                if need_state:
                    out, srepr = result

--- around line 1495 ---
                result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                               orow, xd, xs, at, routes=routes,
                               return_state=need_state)
                if need_state:
                    out, srepr = result
                else:
                    out, srepr = result, None

--- around line 1496 ---
                               orow, xd, xs, at, routes=routes,
                               return_state=need_state)
                if need_state:
                    out, srepr = result
                else:
                    out, srepr = result, None
                if args.aux_outcome_w > 0:

--- around line 1501 ---
                    out, srepr = result, None
                if args.aux_outcome_w > 0:
                    pred = model.outcome_head(srepr).squeeze(1)
                    target = torch.from_numpy(data.won[vsel]).to(
                        device=device, dtype=pred.dtype)
                    out_bce += float(bce_none(pred, target).sum())
                    out_ok += float(((pred >= 0) == (target >= 0.5)).sum())

--- around line 1508 ---
                    out_n += len(target)
                if args.aux_count_w > 0:
                    pred = model.count_head(srepr).squeeze(1)
                    target, valid = count_targets(seld, om, orow)
                    if valid.any():
                        count_abs += float(
                            (torch.sigmoid(pred[valid]) - target[valid])

--- around line 1555 ---
              f"{aux_msg} "
              f"({time.time() - t0:.0f}s)")
        # 🔴 rule 3. The default here SELECTS THE CHECKPOINT BY `val_top1`, and
        # that metric is measured not to predict strength in either direction
        # (§8z moved it by 8 decisions for +37 Elo; §8aa moved it by 214 for
        # +14 -- a 70x exchange-rate difference). It is tolerable for a plain
        # clone, whose objective IS corpus fit. It is NOT tolerable for any arm

--- around line 1567 ---
        if args.export_last:
            if epoch == args.epochs - 1:
                export_npz(model, ROOT / args.out, count_frac, x_mask, a_mask,
                           remap)
        elif acc > best:
            best = acc
            export_npz(model, ROOT / args.out, count_frac, x_mask, a_mask,

--- around line 1571 ---
        elif acc > best:
            best = acc
            export_npz(model, ROOT / args.out, count_frac, x_mask, a_mask,
                           remap)
        # 🔴 A HOSTED RUN THAT IS KILLED COMMITS NOTHING. Kaggle terminates a
        # batch kernel at its 12 h cap and discards /kaggle/working with it, so
        # a checkpoint written every epoch still reaches nobody -- the process

--- around line 1575 ---
        # 🔴 A HOSTED RUN THAT IS KILLED COMMITS NOTHING. Kaggle terminates a
        # batch kernel at its 12 h cap and discards /kaggle/working with it, so
        # a checkpoint written every epoch still reaches nobody -- the process
        # has to EXIT CLEANLY for the output to be saved. `--max-hours` turns
        # "trained 11 h, delivered nothing" into "trained N epochs, exported
        # the best one, exit 0". Set it BELOW the platform cap, not at it.
        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:

====================================================================================================
/kaggle/working/ptcg_research_index/cloned_repos/scio/scripts/train_value.py
====================================================================================================

--- around line 10 ---
torch's (out, in) weight orientation, because `valuenet.Net` computes `w @ x`.

**The input is the pure STATE**: `features.featurize` -> dense(242) + slot ids +
three card bags. Deliberately NOT `seld`/`xdense`, which are select-conditional
-- at play time V scores a SUCCESSOR observation returned by `fs.step`, so its
input must be a function of state alone or training and inference diverge and
produce a plausible number (rule 18).

--- around line 13 ---
three card bags. Deliberately NOT `seld`/`xdense`, which are select-conditional
-- at play time V scores a SUCCESSOR observation returned by `fs.step`, so its
input must be a function of state alone or training and inference diverge and
produce a plausible number (rule 18).

**The label is `won`, from the acting seat's point of view** (p26 writes it that
way; draws are 0.5). This is the one column the BC corpus has always carried and

--- around line 43 ---
    sys.path.insert(0, str(ROOT / _sub))

# `sa.features` -> `sa.cards` -> `cg.sim`, so the SDK has to be on the path
# before the first feature import. Same bootstrap as `train_policy.py`.
from ptcg.env import sdk  # noqa: E402

sdk.load()

--- around line 44 ---

# `sa.features` -> `sa.cards` -> `cg.sim`, so the SDK has to be on the path
# before the first feature import. Same bootstrap as `train_policy.py`.
from ptcg.env import sdk  # noqa: E402

sdk.load()


--- around line 49 ---
sdk.load()

from sa.features import DENSE_DIM, N_CARD_IDS  # noqa: E402

EMB = 16
BAGS = ("my_hand", "my_discard", "opp_discard")


--- around line 55 ---


class ValueNet(nn.Module):
    """Mirrors `valuenet.Net` exactly. Any change here is a change there."""

    def __init__(self, h1: int = 256, h2: int = 128, dropout: float = 0.1):
        super().__init__()

--- around line 58 ---
    """Mirrors `valuenet.Net` exactly. Any change here is a change there."""

    def __init__(self, h1: int = 256, h2: int = 128, dropout: float = 0.1):
        super().__init__()
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)

--- around line 69 ---
        self.drop = nn.Dropout(dropout)

    def forward(self, dense, slots, bag_flat, bag_off):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        x = torch.cat(parts, dim=1)

--- around line 79 ---


class Data:
    def __init__(self, paths: list[Path]):
        d, s, w, g = [], [], [], []
        bags: dict[str, list] = {b: [] for b in BAGS}
        offs: dict[str, list] = {b: [] for b in BAGS}

--- around line 80 ---

class Data:
    def __init__(self, paths: list[Path]):
        d, s, w, g = [], [], [], []
        bags: dict[str, list] = {b: [] for b in BAGS}
        offs: dict[str, list] = {b: [] for b in BAGS}
        for p in paths:

--- around line 85 ---
        offs: dict[str, list] = {b: [] for b in BAGS}
        for p in paths:
            z = np.load(p)
            d.append(z["dense"])
            s.append(z["slots"])
            w.append(z["won"])
            g.append(z["gid"])

--- around line 115 ---
            assert len(self.bag_off[b]) == n + 1, (b, len(self.bag_off[b]), n)

    def __len__(self) -> int:
        return len(self.won)

    def batch(self, idx: np.ndarray, dev) -> tuple:
        dense = torch.from_numpy(self.dense[idx]).to(dev)

--- around line 118 ---
        return len(self.won)

    def batch(self, idx: np.ndarray, dev) -> tuple:
        dense = torch.from_numpy(self.dense[idx]).to(dev)
        slots = torch.from_numpy(self.slots[idx].astype(np.int64)).to(dev)
        bf, bo = {}, {}
        for b in BAGS:

--- around line 138 ---


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC over decided rows only (draws carry no order)."""
    m = y != 0.5
    y, p = y[m], p[m]
    if len(np.unique(y)) < 2:

--- around line 151 ---


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)

--- around line 194 ---
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = ValueNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = float("inf"), None, 0

--- around line 195 ---
    torch.manual_seed(args.seed)
    model = ValueNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):

--- around line 198 ---
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(tr_idx)

--- around line 201 ---
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]

--- around line 208 ---
            dense, slots, bf, bo, y = data.batch(idx, dev)
            opt.zero_grad()
            loss = lossf(model(dense, slots, bf, bo), y)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)


--- around line 213 ---
            tot += loss.detach().item() * len(idx)

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(va_idx), args.bs):
                idx = va_idx[i:i + args.bs]

--- around line 219 ---
                idx = va_idx[i:i + args.bs]
                dense, slots, bf, bo, y = data.batch(idx, dev)
                ps.append(torch.sigmoid(model(dense, slots, bf, bo)).cpu().numpy())
                ys.append(y.cpu().numpy())
        p, y = np.concatenate(ps), np.concatenate(ys)
        vl = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
        print(f"ep {ep:2d}  train {tot/len(perm):.4f}  val {vl:.4f}  "

--- around line 228 ---
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:

--- around line 229 ---
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best val {best:.4f})", flush=True)

--- around line 238 ---
    # Export rule pinned in advance (E20 / rule 18's corollary): BEST val
    # logloss on the gid-disjoint split, patience 3. Recorded, not chosen after.
    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(va_idx), args.bs):

--- around line 239 ---
    # logloss on the gid-disjoint split, patience 3. Recorded, not chosen after.
    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(va_idx), args.bs):
            idx = va_idx[i:i + args.bs]

--- around line 245 ---
            idx = va_idx[i:i + args.bs]
            dense, slots, bf, bo, y = data.batch(idx, dev)
            ps.append(torch.sigmoid(model(dense, slots, bf, bo)).cpu().numpy())
            ys.append(y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)

    # ⚠ THE ORIENTATION CONTROL. A sign-flipped V is the single failure that

--- around line 256 ---
          f"AUC {auc(y, p):.4f}")
    if not mw > ml:
        sys.exit("ORIENTATION FAILED: V does not score won states above lost")

    sd = model.state_dict()
    np.savez(
        args.out,

--- around line 258 ---
        sys.exit("ORIENTATION FAILED: V does not score won states above lost")

    sd = model.state_dict()
    np.savez(
        args.out,
        slot_emb=sd["slot_emb.weight"].numpy(),
        bag_emb=sd["bag_emb.weight"].numpy(),

--- around line 259 ---

    sd = model.state_dict()
    np.savez(
        args.out,
        slot_emb=sd["slot_emb.weight"].numpy(),
        bag_emb=sd["bag_emb.weight"].numpy(),
        w1=sd["fc1.weight"].numpy(), b1=sd["fc1.bias"].numpy(),
```