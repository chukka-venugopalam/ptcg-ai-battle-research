# Competition Context (as researched through 2026-10-22)

This project targets the Pokémon TCG AI Battle Challenge Playground. Competition-specific rules and limits may change; verify the live Kaggle rules before uploading. Previous research noted: starter skill-rating estimate around 600, matchmaking based on skill rating, a limit on active submissions, and episodes used to update ratings. Historical notes mention up to two active submissions and a daily submission limit; check the current page because live rules can change.

The packaged deck and engine must be compatible with the active R2 SDK. The current R2 package used in our verification exposes card IDs up to 1431 and attack IDs up to 1755. The policy checkpoint in the current compatibility agent has `card_emb` with 1300 rows and `atk_emb` with 1600 rows. The compatibility guard maps out-of-range lookups to a safe row. This prevents index errors, but it does not create learned semantics for unseen cards.
