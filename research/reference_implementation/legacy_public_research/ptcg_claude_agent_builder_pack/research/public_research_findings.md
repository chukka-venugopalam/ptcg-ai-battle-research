# Public PTCG AI Battle Research Findings

This file records the research directions and concrete artifacts that were inspected in the public-agent corpus used in our Kaggle notebook.

## Public agent repositories

1. **Team Scio — `SiamRahman29/pokemon-tcg-agent`**
   - Policy-only behavioral cloning agent with per-legal-option scoring.
   - Numpy policy/value inference, feature encoders, replay mining, evaluation, self-play, submission packaging.
   - Public policy checkpoints found in the cloned corpus.
   - Model shapes observed in our audit are recorded in `scio_model_audit.md`.

2. **Lawliet — `Lawliet-ai/ptcg-ai-battle-agent`**
   - Search/value/policy stack.
   - PIMC/PUCT-style search and training/evaluation infrastructure.
   - Useful primarily for search, value modelling and uncertainty-aware decision making.

3. **wmh — `wmh/ptcg-abc`**
   - Multiple agents/decks.
   - Replay/meta/prize analysis and evaluation tooling.
   - Useful for deck coverage, matchup analysis and reward signals.

4. **Ducdata — `Ducdata1808/PTCG_Agent`**
   - Policy/value networks, MCTS/IS-MCTS ideas, self-play/training manager and deck collection.
   - Useful as a reference for combining learned policies with tree search.

5. **pckhoa — `pckhoa206/kaggle_challenge_pokemon`**
   - Maskable PPO, heuristics and an ONNX policy path.
   - Flat state representation and fixed legal-option output handling.
   - Useful for legal masking, PPO and deployable ONNX inference.

6. **scha54 — `scha54/Kaggle-Pokemon`**
   - Agent/search/deckbuilder/evaluation/test/submission packaging structure.
   - Useful for engineering discipline and experimental comparisons.

## Public competition writeups/notebooks inspected

- Kiyota: reinforcement learning + MCTS sample notebook.
- 14th-place writeup: behavioral cloning -> archetype expert PPO -> deck-specific specialists -> provable lethal search.
- 9th-place writeup: behavioral cloning -> value fine-tuning -> PPO + bounded engine lookahead.
- 20th-place writeup: action-dynamics contrastive pretraining -> RL.
- 15th-place writeup: recurrent actor-critic specialists + population/self-play ideas.
- 27th-place writeup: multi-deck PPO -> archetype PPO -> deck-specific PPO curriculum.

## Model artifacts observed in the cloned corpus

- `pckhoa/model.onnx`
- `scio/agents/sa/policy_net.npz`
- `scio/agents/sa/value_net.npz`
- `scio/out/tmp/sa/policy_net.npz`

## Scio model observations

The lean policy checkpoint used a 496-wide state input and 25-wide dense option representation. The richer checkpoint used a 708-wide state input, 37-wide dense option representation, and a pooled option-set block (`n_pool=172`). The associated value model used dense state features plus embeddings with a 512 -> 256 -> 1 MLP.

Do not assume these are "better" merely because they are larger. We validated them experimentally on the current SDK.

## Controlled experiment already run

The two Scio policy checkpoints were packaged and smoke-tested on the current CABT engine with zero fallbacks.

Then we ran a controlled 40-game Grimmsnarl mirror with the main auxiliary rules disabled for both models:

- A: `agents/sa/policy_net.npz`
- B: `out/tmp/sa/policy_net.npz`
- `chip_targeting=False`
- `energy_spread=False`
- `counter_source=False`

Results:

- A: 6 wins
- B: 34 wins
- B win share: 85%
- A win share: 15%
- Both seat directions were tested (20 games each).
- Fallbacks: 0
- `net_missing`: 0

This supports using B as the current local policy baseline. It does NOT prove B will achieve a particular competition rating, and it is a single-deck mirror test.

## Current Kaggle baseline

At the time of the captured screenshot, our submitted agent was approximately:

- Rank: 39
- Rating/score: 781.4
- Visible leader: 1135.7

Treat 1000+ as an engineering target, not a guaranteed result.
