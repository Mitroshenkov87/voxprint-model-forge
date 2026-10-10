# Manifests

`plans/plan.toml` pins upstream files. It does not invent sha256 values for GGUFs this run has not built.

After an A100 run, bootstrap writes these files under `~/forge-out/manifests/`:

- `<date>-a100.json` — sha256, size, and public HF / private Kaggle / torrent-placeholder sources
- `<date>-a100.MODELS.md` — registry snippet in the same shape

The VM has no GitHub remote. Copy both files back to this repo. Schema: `manifests/schema.json`.
