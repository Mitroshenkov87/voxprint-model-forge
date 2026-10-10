# Manifests

`plans/plan.toml` pins upstream files. It does not invent sha256 values for GGUFs this run has not built.

After an A100 run, `scripts/bootstrap.sh` writes:

- `manifests/<date>-a100.json` — sha256, size, and private HF / Kaggle / torrent-placeholder sources
- `manifests/<date>-a100.MODELS.md` — registry snippet in the same shape

Copies also land in `~/forge-out/manifests/`. If `GH_TOKEN` is set, bootstrap commits those two files with `git -c http.extraHeader=...` and pushes. The token is not written into `.git/config`. If `GH_TOKEN` is unset, the run prints the git commands and leaves the commit to you.

Schema: `manifests/schema.json`.
