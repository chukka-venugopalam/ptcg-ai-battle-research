# MASTER PROMPT — Build the Next PTCG AI Battle Agent

You are the lead reinforcement-learning / game-AI engineer helping me build the next submission for the Pokémon TCG AI Battle Challenge Playground.

Your task is to **actually design and implement the next competitive agent**, not merely give me a high-level plan. Work from the supplied research pack, inspect the files first, use the public implementations as technical references, and produce runnable code/notebooks/scripts that I can execute in Kaggle.

## 1. Current situation

Our current visible Kaggle baseline is approximately:

- rank 39
- score 781.4
- visible leading score 1135.7

Our current local policy baseline B is the richer public Scio checkpoint:

`out/tmp/sa/policy_net.npz`

In a controlled 40-game Grimmsnarl mirror with the same auxiliary-rule configuration on both sides, B beat the lean Scio checkpoint A by 34–6 (85% win share) with zero fallbacks and zero missing-net events. This is useful evidence, but it is NOT evidence that B will reach 1000 rating across the real ladder.

Our immediate goal is an agent that can plausibly reach **~1000+ rating**, followed by continued iteration toward the leaderboard top. Do not promise that the target will be reached; use experiments to determine what works.

## 2. First action: inspect everything

Before writing substantial code:

1. Read `START_HERE.md`.
2. Read every file in `research/`.
3. Inspect the included CABT SDK under `competition_sdk/cg/`.
4. Inspect `repo_template/`.
5. When network access exists, clone ALL public repositories listed in `research/public_research_sources.csv`.
6. Inventory their agent implementations, feature extraction, legal-action handling, training code, replay/data pipelines, model checkpoints, deck libraries, search/evaluation code, and submission scripts.
7. Do not ignore the public Kaggle writeup directions listed in `public_research_findings.md`.

At the end of this inspection, produce a compact table of:

- agent/repo
- architecture
- training method
- state/action representation
- use of search
- use of value model
- self-play/opponent sampling
- deck specialization
- notable strengths
- likely weaknesses
- which parts are worth reproducing

## 3. Engineering principle

Do NOT make a slightly modified copy of one public agent and call it new.

Use the strongest ideas from several public implementations to make a hybrid system that is still small enough to run reliably under the competition's execution constraints.

The preferred direction is:

**hard legality/rules -> learned policy ranking -> tactical correction/search -> value model -> deck/opponent adaptation**

The learned policy should score the currently legal options; never choose an illegal action and never rely on a huge fixed action space when the engine already provides legal options.

## 4. Architecture to build

Design and implement a v2 agent along these lines, modifying pieces only when experiments justify the change:

### A. Legal-option policy model

Start from the richer Scio representation as the baseline reference:

- richer state representation
- per-option features
- option-set pooled features
- card/attack embeddings
- shared state encoder
- shared option encoder
- ranking head producing one score per legal option

The model should support a variable number of legal options and explicit legal masking.

Prefer a Transformer/set encoder or another permutation-aware option architecture only if it can be trained and benchmarked cleanly within the runtime budget.

### B. Value model

Use a separate value network to estimate the strategic quality of a state. Train it from high-quality replay/self-play outcomes, not just from arbitrary random play.

### C. Tactical checker

Implement a lightweight tactical layer that runs only on selected decisions:

- immediate lethal / provable KO detection
- prize-race checks
- obvious catastrophic blunders
- critical Boss's Orders / gust choices
- energy/retreat legality interactions
- bench and active sequencing when it changes immediate tactical outcome

Do NOT make a slow full search run on every select.

### D. Selective search / lookahead

Use bounded one-ply or shallow search only when:

- the top policy scores are close,
- the decision is strategically important,
- the option count is manageable,
- the remaining time budget is safe.

Use the learned value model to score continuations rather than relying only on handcrafted rollouts.

### E. Deck specialists

Use deck-aware routing. At minimum support:

- a general policy,
- a small number of archetype/deck specialists.

Do not create dozens of specialists until evaluation proves they are useful.

### F. Opponent modelling / BO3 state

The competition is not a single isolated game. Maintain compact per-match / per-opponent state where the simulator exposes enough information to do so safely.

Consider:

- opponent action tendencies
- revealed deck/card patterns
- aggression/control profile
- likely prize targets
- observed sequencing tendencies

Adapt gradually; never overfit to a single early observation.

## 5. Training pipeline

Build a reproducible pipeline in this order:

### Stage 1 — baseline reproduction

Reproduce the current public B baseline in the current CABT SDK and establish a known-good local benchmark.

### Stage 2 — strong teacher

Construct a high-quality teacher using deterministic tactical rules and safe shallow search/value checks.

Do not generate massive low-quality random data.

### Stage 3 — selective replay mining

Generate self-play and teacher-guided games and retain decisions only when they are informative:

- tactical decisions
- close policy/value decisions
- decisions where the teacher/search clearly disagrees with baseline
- strategically important turns
- high-quality completed games

Store state + legal options + chosen option(s) + outcome/value targets.

### Stage 4 — behavioural cloning / ranking

Train a legal-option ranking model.

Prefer listwise/pairwise ranking or cross-entropy over legal options with carefully defined masks.

Use deck-balanced and decision-type-balanced sampling.

### Stage 5 — value fine-tuning

Train the value model from replay/self-play outcomes and calibrate it against the real game result.

### Stage 6 — controlled self-play RL

Use PPO or another stable policy-gradient method only after the initial policy is competent.

Use mixed opponents / population-based opponents rather than self-play against one frozen copy.

Include:

- current best policy
- previous checkpoints
- public-agent baselines when technically compatible
- rule/search teachers
- several deck archetypes

Avoid uncontrolled reward hacking.

### Stage 7 — specialist training

Train only the specialists whose cross-deck evaluation demonstrates a need.

### Stage 8 — tactical integration

Add the tactical/value/search layer only after measuring whether each intervention improves win rate.

## 6. Experimental discipline

Every proposed improvement must have:

1. baseline definition,
2. exact changed component,
3. fixed or clearly randomized evaluation protocol,
4. enough games to reduce noise,
5. seat swapping,
6. cross-deck evaluation,
7. latency measurement,
8. fallback/error counters.

Do not conclude that something is better because it wins one mirror.

For promising variants, use at least:

- 100+ games for screening,
- 300+ games for strong decisions,
- more when the measured gap is small.

Use Wilson confidence intervals or an equivalent uncertainty estimate where appropriate.

## 7. Cross-deck benchmark

Build a reusable benchmark harness using the public decks collected in the research corpus.

At minimum evaluate across a representative set of fast/aggressive, midrange, control, setup-heavy and multi-prize archetypes.

For every matchup:

- A vs B
- B vs A

Record:

- wins/losses/draws
- average turns
- average selects
- mean/p95/max decision latency
- fallbacks
- missing-model events
- tactical intervention frequency
- search frequency
- average search time

Generate a matchup matrix and identify the largest weaknesses.

## 8. Important lessons from the public corpus

Use these as hypotheses, not truths:

- behavioral cloning is a strong initialization but imitation quality does not automatically equal match win rate;
- value models and bounded lookahead can complement a policy;
- specialist policies can help when deck distributions differ;
- self-play opponent diversity matters;
- selective replay is preferable to blindly using every decision;
- legal-action ranking is a natural formulation for this simulator;
- pure search can become too expensive;
- local ladder-free evaluation can disagree with real ladder performance;
- fast inference is valuable, but decision quality is the main objective.

## 9. Current runtime/submission requirements

The final agent must be packaged into the competition submission format used by the supplied builder:

- `main.py` at the archive root
- `deck.csv` at the archive root
- `cg/` engine package
- model files and agent code required by `main.py`

The final archive must pass:

- import/compatibility checks,
- policy dimension guard,
- extracted-bundle smoke test,
- full legal-action game execution,
- time budget test.

Never ship a model if it silently fails to load and falls back to index-order/random-legal behavior.

## 10. Deliverables I want from you

Do the coding and produce these artifacts:

### `notebooks/01_train_next_agent.ipynb`

A complete Kaggle notebook that can:

- install/prepare dependencies,
- locate the competition SDK,
- load card data,
- build features,
- build/load teacher data,
- train the policy,
- train the value model,
- run evaluation,
- save model checkpoints,
- build the submission archive.

### `src/`

Clean reusable implementation for:

- feature extraction,
- legal-option ranking,
- policy model,
- value model,
- tactical checker,
- selective search,
- opponent state,
- deck routing,
- replay/data loader,
- evaluation harness.

### `scripts/`

At minimum:

- data preparation
- replay generation/mining
- training
- evaluation
- benchmark matrix
- submission builder

### `models/`

The trained checkpoint(s), once actually trained and verified.

### `submission/`

A final deployable `submission.tar.gz` built by the same compatibility checks used in this project.

### `reports/`

Include:

- baseline benchmark,
- ablation results,
- cross-deck matrix,
- final model configuration,
- runtime metrics,
- known weaknesses.

## 11. How to handle limited compute

If GPU/CPU/time is limited, prioritize:

1. better data quality,
2. better option/state representation,
3. value-guided selective corrections,
4. deck specialization where justified,
5. small controlled RL fine-tuning.

Do NOT spend most of the budget on giant models.

## 12. What not to do

- Do not claim a 1000 rating before Kaggle evidence exists.
- Do not fabricate training curves.
- Do not invent win rates.
- Do not silently change the evaluation protocol between variants.
- Do not train on illegal choices.
- Do not run expensive search on every select.
- Do not hard-code an answer for one known deck.
- Do not optimize only for Grimmsnarl mirror performance.
- Do not submit a bundle with a model that did not pass an actual extracted-bundle smoke game.
- Do not copy a public submission wholesale and represent it as newly trained research.

## 13. Expected working style

Be decisive and implementation-oriented.

When you have enough information, write the code rather than asking me a long series of questions.

When you encounter a missing competition input, tell me exactly which file is missing and how to obtain it.

When an experiment fails, diagnose it from the logs and fix the code rather than merely describing the error.

At the end of each major step, tell me:

- what changed,
- what files were created,
- what command/notebook cell I should run next,
- what metric decides whether we keep the change.

The objective is a robust, reproducible agent with a realistic path from the current ~781 rating baseline toward ~1000+, and then beyond it through measured iteration.
