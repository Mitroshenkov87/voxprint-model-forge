# Manifests

`plans/plan.toml` pins upstream files. It does not invent sha256 values for GGUFs this run has not built.

After an A100 run, bootstrap writes these files under `~/forge-out/manifests/`:

- `<date>-a100.json` — sha256, size, and public HF / private Kaggle / torrent-placeholder sources
- `<date>-a100.MODELS.md` — registry snippet in the same shape

The VM has no GitHub remote. Copy both files back to this repo. Schema: `manifests/schema.json`.

## App models

`model-forge.models.json` (schema `schemas/voxprint-app-models.schema.json`,
`voxprint-app-models/1`) lists the models Model Forge needs (`required`) and can
use (`optional`). Each entry pins a full commit and lists every file with its
size and sha256 from the Hugging Face API. The wheel carries the same bytes as
`voxprint_forge/data/app-models.json`; `voxprint_forge.app_models` reads it.
Regenerate both with `python scripts/gen_app_models.py`. No weights are in git.
