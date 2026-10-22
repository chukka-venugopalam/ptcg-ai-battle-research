# Pckhoa Public Model Audit

Public artifact: `pckhoa/model.onnx`

Observed graph:

- IR version 10
- Opset 18
- Dense network ending in 50 outputs
- Input shape: `(batch, 164)`
- Output shape: `(batch, 50)`
- Hidden dimensions observed in the ONNX graph: 512 -> 256 -> 256 -> 256, followed by tanh blocks and 50-output head

The corresponding public agent code describes a flat 164-dimensional state vector including board/card/context information plus up to 50 option features.

The agent path combines:

1. Phase-0 tactical/lookahead handling
2. heuristic overrides
3. ONNX inference
4. heuristic fallback

Training infrastructure uses Stable-Baselines3 MaskablePPO and later exports ONNX.

The ONNX file is a reference artifact, not evidence that the architecture is stronger than the Scio policy. Zero-input execution only establishes graph compatibility/executability.
