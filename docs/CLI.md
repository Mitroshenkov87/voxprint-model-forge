# Forge CLI

`forge` prepares GGUF tiers for Voxprint AI Plotweaver. The GUI is later. A bot can drive every command. Add `--json` for one JSON object on stdout. Logs stay on stderr.

```json
{"ok": true, "command": "plan-summary", "exit_code": 0, "result": "..."}
```

On failure the object also has `error.code` and `error.message`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success, including a clean skip of an optional model that did not fit the clock |
| 1 | Unexpected error |
| 2 | Usage, including a missing token when a command needs one |
| 3 | Licence refused |
| 4 | Checksum mismatch |
| 5 | Download failed |
| 6 | Stage failed, including an upload whose remote sha256 or size never matched |
| 7 | Disk short of the next stage. Finished quants stay on disk |
| 8 | Plan failed to load |
| 9 | A required tool is missing |

## Install

Python 3.12 or newer. The A100 bootstrap installs 3.14 and a CUDA torch wheel. Local tests can use a CPU wheel.

```bash
pip install -e ".[dev]"
```

## Commands

`forge plan-summary` prints the plan, the per-model minute estimates, and the required path including the llama.cpp build. It does not import the ablation stack. Bootstrap prints the same text with `PYTHONPATH=src` before torch is installed.

`forge verify-logins` checks `HF_TOKEN` with the Hub API and `KAGGLE_API_TOKEN` with `kaggle` (or `uvx kaggle`) 2.2 or newer. It does not write a token file.

`forge fetch|abliterate|quantize|fit|eval|card|publish-prep|upload --model ID` runs through that stage. Later stages resume earlier markers under `~/forge-out/stages/<id>/`. `fit` accepts repeatable `--budget-gb`.

`forge publish-prep` writes the local card and a registry draft. It does not upload.

`forge upload` and `forge run-all` upload after the model is finished. `run-all` does this per model, then writes the manifest.

Global flags precede the subcommand. `scripts/vm_run.sh` runs:

```bash
forge --plan plans/plan.toml --root <repo> --forge-out ~/forge-out run-all
```

`forge run-all --plan ...` is rejected.

## Private upload

Uploads go to private destinations only.

- Hugging Face: `HF_USER/voxprint-<model-id>`, created with `private=True`. Files are the selected GGUF for each tier, the imatrix, `README.md`, and the `.sha256` sidecars. `HF_HUB_ENABLE_HF_TRANSFER=1` is set when `hf_transfer` imports.
- Kaggle: owner `KAGGLE_USER` (default `mitroshenkov87`). One private model per base (`isPrivate: true`). One variation per tier, framework `gguf`. The CLI is `kaggle` or `uvx kaggle`, version 2.2 or newer. The API token is passed only in the child environment. A temporary config directory is deleted after each call.

Nothing is made public. If a remote repo is public, the stage stops and does not upload. After each upload the remote sha256 and size must match the local file. The client retries with backoff (2s, 4s, 8s, 16s). The `upload` marker is written only after both sides match. Local deliverables stay on disk until that match. F16 and safetensors intermediates are removed once the quants and the imatrix exist.

The run summary lists every uploaded file with its sha256, Hugging Face URL, and Kaggle URL.

## Manifest

`run-all` writes `manifests/<date>-a100.json` and `manifests/<date>-a100.MODELS.md` under `~/forge-out/manifests/`. `torrent` is the placeholder `placeholder`. The A100 box has no GitHub remote. The run prints the two paths so the operator can copy them back. It does not push.

## Environment

| Variable | Role |
| --- | --- |
| `HF_TOKEN` | Hub read and private upload. Not logged. |
| `HF_USER` | Hub account that owns the private model repos |
| `KAGGLE_API_TOKEN` | Kaggle CLI auth. Not logged. |
| `KAGGLE_USER` | Kaggle owner. Default `mitroshenkov87` |
| `FORGE_LLAMA_BIN` | Directory with `llama-quantize`, `llama-imatrix`, `llama-server`, `llama-cli`, `llama-perplexity` |
| `FORGE_LLAMA_SRC` | llama.cpp checkout that contains `convert_hf_to_gguf.py` |
| `FORGE_ROOT` | Repo root, if it is not the working directory |

## Model card

`templates/model_card.md.jinja` renders `README.md`. The file starts with Hugging Face YAML: `license: apache-2.0`, `base_model` (the abliterated upstream for a requant, the T-Bank repo for T-lite and T-pro), `base_model_relation: quantized`, tags `voxprint`, `gguf`, `abliterated`, `llama.cpp`, `imatrix`, `russian`, `language: [ru, en]`, and `pipeline_tag: text-generation`. The body is the base and licence, method, calibration data and licences, quant types, evals, limits, and credits, then `## Made for Voxprint AI Media Suite` with the Plotweaver link. `huggingface_hub.ModelCard` parses that front matter.

## Canvas

`forge canvas run plans/canvas-zimage-tiers-v1.toml` prepares private image tiers for Voxprint AI Canvas. On the A100 that is already running the LLM bootstrap, start it from `scripts/canvas_job.sh` (`python -m voxprint_forge.diffusion`) so the running `scripts/bootstrap.sh` is not replaced. Part D is a separate entry, `scripts/canvas_job_d.sh` (`python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml`). See `docs/CANVAS.md`.

Parts, in order: base (55 min), controlnet (25 min), edit (55 min). Total 135 min. `--deadline <ISO time>` or `FORGE_DEADLINE` skips a part whose estimate exceeds the time remaining and records `deferred to next rental`. `--force` is the only way to start while `scripts/bootstrap.sh` is still running.

Exit codes are the same table as the rest of the CLI. Uploads are private. Base goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo`. ControlNet goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1`. Edit goes to `Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511`. Each part also creates a private Kaggle model. The manifest is `~/forge-out/manifests/<date>-canvas-zimage.json`.

## Fit

`fit --budget-gb 16 --budget-gb 24` keeps tiers of those sizes. A 16 GB tier must fit at the shortest context (8k in the A100 plan). A 24 GB tier must fit every context (8k and 16k). Headroom defaults to 3 GB. KV is q8_0 unless only q4_0 fits. The gate uses computed weights + KV and the llama-server `model buffer size` / `KV buffer size` lines. `nvidia-smi` usage is recorded and is not the gate.
