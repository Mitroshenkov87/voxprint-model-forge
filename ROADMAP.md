# Roadmap

Model Forge prepares models on the operator's own machine: it tracks a pinned
upstream revision, removes refusals when the plan says so, quantizes to a VRAM
budget, records a quality check, and publishes a private model card with
checksums. The GUI is for manual use. The CLI speaks JSON so another program
can drive it.

## Shipped

- CLI with `--json` and stable exit codes.
- Stages: fetch, abliterate, quantize, fit, eval, card, publish, upload.
- Fit chooses a VRAM/RAM split (GPU, KV in RAM, MoE experts on CPU, encoder
  offload, sequential load) and records which split each tier assumes. An
  offload that fits is kept ahead of a smaller quant.
- Preflight, heartbeats, and `forge status`.
- Desktop window for a local run and a remote session.
- Canvas image tiers (Z-Image and Qwen-Image) as a separate entry point.
- Private upload to Hugging Face and Kaggle, with sha256 and size checked
  after the upload.

## Next

- Documentation site generated in CI from the docstrings and from `forge --help`.
- Account setup that keeps Hub, Kaggle, and Drive credentials in the OS keyring.
- Notice when a pinned upstream revision has a newer release.
- A small client that drives the JSON CLI.

## Out of scope

- Bundling the prepared weights in an end-user app.
- Telemetry that is on unless the operator turns it on.
- Abliteration tools whose licence cannot be shipped with this Apache-2.0 tree.
