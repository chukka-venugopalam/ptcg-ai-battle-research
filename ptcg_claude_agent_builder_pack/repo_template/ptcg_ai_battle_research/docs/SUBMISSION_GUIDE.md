# Submission guide

## Runtime vs research

The research repository can be huge. The submission cannot. `submission/` is therefore a curated runtime package.

## Exact flow

1. Train/evaluate locally or in Kaggle using the competition-authorized simulator.
2. Choose the model checkpoint that is actually needed at inference time.
3. Copy only the runtime Python files, model weights/assets, and the chosen 60-card `deck.csv` into `submission/`.
4. Make sure imports work when the current directory is `/kaggle_simulations/agent/`.
5. Run `bash submission/build_submission.sh`.
6. The generated `submission.tar.gz` must have `main.py` and `deck.csv` at archive root, not inside another directory.
7. Upload the archive through the Playground's My Submissions page.
8. Kaggle first runs a self-play validation episode. A failure marks the submission as Error; a passing submission enters the matchmaking pool.
9. Newer submissions receive more episodes, so early submissions are useful as measurements, but they are not evidence that one short-lived rating is the final strength.

## What `main.py` does

At the initial deck-selection stage, `obs.select` is `None` and the agent returns 60 integer card IDs from `deck.csv`. During normal play, the engine gives an observation containing logs, current state and legal options; the agent returns indices into the legal option list. The engine only presents legal moves.

## Runtime constraints

Treat the official Playground page as authoritative for the current 197.7 MiB submission limit, 5 submissions/day, two active submissions, 2 vCPU and 12.2 GiB RAM limits.
