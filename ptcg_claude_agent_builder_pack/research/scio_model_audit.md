# Scio Public Model Audit

## Policy A

File: `agents/sa/policy_net.npz`

Observed arrays:

- `slot_emb`: (1300, 16)
- `bag_emb`: (1300, 16)
- `card_emb`: (1300, 16)
- `atk_emb`: (1600, 16)
- `count_frac`: (11, 64)
- `sfc0_w`: (512, 496)
- `sfc1_w`: (256, 512)
- `head0_w`: (256, 329)
- `head1_w`: (128, 256)
- `head2_w`: (1, 128)
- `n_sfc` present
- No pooled option-set block in the audited checkpoint

## Policy B

File: `out/tmp/sa/policy_net.npz`

Observed arrays:

- same embedding vocab widths as A
- `sfc0_w`: (512, 708)
- `sfc1_w`: (256, 512)
- `head0_w`: (256, 341)
- `head1_w`: (128, 256)
- `head2_w`: (1, 128)
- `n_pool = 172`
- `n_attr = 0`

## Value model

File: `agents/sa/value_net.npz`

- `w1`: (512, 482)
- `w2`: (256, 512)
- `w3`: (1, 256)

## Compatibility result

Both policy checkpoints and the value model loaded successfully once the official CABT `cg` SDK was placed on the Python path. The build system's own dimension guard then accepted both policy checkpoints.

The richer policy B was subsequently smoke-tested and then won 34/40 controlled Grimmsnarl mirror games against A under the matched rules configuration.
