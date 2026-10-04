# PTCG AI Battle Research Lab

Research repository for the Pokémon TCG AI Battle Challenge Playground.

## Purpose

This project keeps one clean source of truth for:

- the official competition contract and simulator notes;
- public 2026 agent repositories and research writeups;
- a Kaggle bootstrap notebook that can assemble the public research corpus;
- local replay/evaluation experiments;
- trained-model inventories and provenance;
- the final submission build pipeline.

The research corpus is **not** the submission. Do not put every research file or model in the Kaggle submission bundle.

## Public sources tracked

See `data/manifests/public_research_sources.csv`.

The project tracks upstream URLs and retrieval scripts rather than silently copying third-party code into our own namespace. This keeps original attribution and licensing visible.

## Kaggle dataset workflow

1. Create a Kaggle Dataset named something like `ptcg-ai-public-research`.
2. Add the prepared public repository snapshots under `repos/` when licensing permits.
3. Add the competition-provided card data / simulator files as a **separate private or competition-authorized input** when required.
4. Open `notebooks/00_bootstrap_public_research.ipynb` in Kaggle.
5. The notebook indexes the attached corpus, inventories notebooks/models/decks, and writes normalized manifests under `/kaggle/working/ptcg_research_index/`.

When Kaggle Internet is enabled, the notebook can also clone upstream repositories directly using the URLs in the manifest. For reproducibility, the preferred long-term approach is to upload immutable snapshots to a Kaggle Dataset after verifying their licenses.

## Important licensing boundary

Do not publicly redistribute the competition engine/card database or third-party trained weights unless their applicable terms explicitly permit it. The competition package contains its own reuse/license information. Keep competition-provided assets in the authorized environment and use the manifest to track provenance.

## Development stages

`baseline -> replay mining -> behavioral cloning -> specialist models -> self-play/PPO -> opponent model -> tactical search -> BO3 adaptation -> final packaging`

## Final submission contract

The final archive is deliberately small and must have `main.py` and `deck.csv` at the **top level**:

```text
submission.tar.gz
├── main.py
├── deck.csv
└── <only the model/runtime assets actually needed by main.py>
```

Build it with:

```bash
bash submission/build_submission.sh
```

The builder performs a structure/size smoke check. The official competition page is the authority for the submission limit and upload rules.
