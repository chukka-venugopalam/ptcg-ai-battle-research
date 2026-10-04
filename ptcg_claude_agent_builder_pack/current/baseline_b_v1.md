# Current Baseline B-v1

Policy checkpoint:
`scio/out/tmp/sa/policy_net.npz`

Rules used in the controlled comparison:

- `chip_targeting=False`
- `energy_spread=False`
- `counter_source=False`

Local controlled mirror result:

- 40 total games
- B wins 34
- A wins 6
- B win share 0.85
- 0 fallbacks
- 0 missing-net events

Current visible Kaggle baseline at the time of the screenshot:

- rank 39
- score 781.4
- visible top score 1135.7

The ladder score is the real external metric; local mirror performance is only a diagnostic.
