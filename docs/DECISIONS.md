# Decisions and failed approaches

- Playground engine keeps card ids <= 1267 and attack ids <= 1556 unchanged; new ids 1268-1431 / 1557-1755.
- B (7th-place agent, MIT) has no card-id embeddings; it reads cards via text/atom lookup tables. New ids were zero rows.
- B2 = B + estimated rows for new ids (TF-IDF->ridge for text, kNN for atoms). Old rows bit-identical; 0.22% action change on old-card states; B2 vs B mirror 65-65.
- Opponent-side reading: estimates differ from true-table decisions on 14.3% of decisions vs 16.1% for zeros (small gain).
- Failed / no gain: deck swaps on the Dipam net (A3a/A3b), Slowking exploiter (C), hand-built new-card decks (N3 lost 0-77 to B+F).
- Not enabled: 7th-place endgame energy search (author ships it off; ladder time pool unknown).
- Open: real ladder meta (run tools/ladder_meta_extract.py), thinking-time pool, deck tweaks, training route (no GPU here).

## Rules / licences
Rules 2.4 / 3.6b: this repo must not contain card data, deck lists, model tables or the simulator. Not included here on purpose.
B-line code is by Giovanni (Z3rox-dev), MIT; Dipam's net is MIT; Kelly Mao's format notes are MIT. Keep their LICENSE files in any tarball you submit.
