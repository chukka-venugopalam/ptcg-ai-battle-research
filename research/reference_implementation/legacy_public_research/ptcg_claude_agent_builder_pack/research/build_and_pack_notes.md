# Current CABT / Submission Engineering Notes

The public Scio build script packages a submission as:

- `main.py`
- `deck.csv`
- `cg/`
- `sa/` including policy/value models when selected

It also performs a dimension guard on a candidate policy and runs an extracted-bundle smoke test.

The important discovery during our work was that Scio's `find_sdk_dir()` searches for a directory containing `data/**/cg/api.py`; simply placing `libcg.so` directly under `data/` does not satisfy discovery.

Our fixed working layout used:

```text
data/official_sdk/cg/api.py
data/official_sdk/cg/libcg.so
```

After that, both public Scio policies built and completed smoke games.
