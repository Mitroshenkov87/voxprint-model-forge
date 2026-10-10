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

`forge status` uses a shorter set, because Hardware Watchdog polls it: 0 when every item is running or finished, 3 when a running item's heartbeat is older than 180 seconds or its pid is gone, 4 when any item is `STAGEFAIL` or `FAILED` or `status.json` is missing. Those two numbers are not the licence and checksum codes above. The other commands keep the table.

## Install

Python 3.12 or newer. The A100 bootstrap installs 3.14 and a CUDA torch wheel. Local tests can use a CPU wheel.

```bash
pip install -e ".[dev]"
```

## Commands

`forge --version` prints `X.Y.Z · build N "Charash"`. `N` is `GITHUB_RUN_NUMBER` plus the `offset` in `BUILD.json`. The first codename is Charash.

`forge plan-summary` prints the plan, the per-model minute estimates, and the required path including the llama.cpp build. It does not import the ablation stack. Bootstrap prints the same text with `PYTHONPATH=src` before torch is installed.

`forge preflight --plan PLAN` checks, in a few seconds, that the converter Python has torch, transformers, and gguf; that each enabled safetensors model answers HEAD for `model.safetensors.index.json` at the pinned revision; that the T-lite pre-tokenizer hash is registered on the pinned converter; that some quant on every quantize tier fits the tier budget (parameters times bits per weight, plus q8_0 KV at the longest plan context); that free disk covers the plan's pinned files; that RAM covers the smallest quant; that `HF_TOKEN` and `KAGGLE_API_TOKEN` are set unless `FORGE_DEFER_UPLOAD=1`; and that the llama.cpp binaries are present or cmake, ninja, and nvcc are on `PATH`. `run-all` and the stage commands run this check before any download or quant. `--json` prints the report, including a `checks` list. Each failed check has a `fix`. A failed preflight still exits 6.

`forge-gui` opens the same checks, the per-model status table, and the upload-target selector. It is a window over this CLI and the library. Tokens typed there are stored in the OS keyring under the service `voxprint-forge`, not in the repo. Install it with `pip install -e ".[gui]"`.

`forge status` reads `~/forge-out/status.json`. Each model or canvas part has `stage`, `pct`, `eta_min`, `last_heartbeat`, `pid`, `state`, and `error`. A `RUNNING` or `READY` item whose heartbeat is older than 180 seconds, or whose pid is dead, is reported as `stalled`.

`forge verify-logins` checks `HF_TOKEN` with the Hub API and `KAGGLE_API_TOKEN` with `kaggle` (or `uvx kaggle`) 2.2 or newer. It does not write a token file.

`forge fetch|abliterate|quantize|fit|eval|card|publish-prep|upload --model ID` runs through that stage. Later stages resume earlier markers under `~/forge-out/stages/<id>/`. `fit` accepts repeatable `--budget-gb`.

`forge publish-prep` writes the local card and a registry draft. It does not upload.

`forge upload` and `forge run-all` upload a model as soon as that model is READY. A failure is `STAGEFAIL <stage>` for that model and the others keep going. `run-all --jobs N` (or `FORGE_JOBS`, default 1; `scripts/vm_run.sh` exports 2) overlaps CPU quantize of one model with GPU work of another. The run prints one line per model: `READY`, `UPLOADED`, `STAGEFAIL <stage>`, or `DEFERRED`. `ALL-DONE` is printed only when no model failed. A failed model makes the process exit 6.

`FORGE_DEFER_UPLOAD=1` leaves the model `READY`, writes `<id>-upload-deferred`, and does not write the upload marker, so a later run can upload. When Hugging Face and Kaggle both fail, the same files are copied with rclone to `Voxprint/model-forge/<date>/<model>`. Configure the remote with `RCLONE_CONFIG_*`, or set `FORGE_DRIVE_TOKEN` or `FORGE_DRIVE_SERVICE_ACCOUNT` (never a token file). The remote name defaults to `forge` (`FORGE_DRIVE_REMOTE`). After the copy, `rclone sha256sum` must match the local file.

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
| `FORGE_PYTHON` | Interpreter for `convert_hf_to_gguf.py`. Defaults to the forge interpreter |
| `FORGE_JOBS` | Parallel models for `run-all` when `--jobs` is omitted. Default 1 |
| `FORGE_DEFER_UPLOAD` | `1` / `true` / `yes` / `on` skips upload and leaves the model READY |
| `FORGE_UPLOAD_TARGETS` | Comma-separated `hf`, `kaggle`, and/or `drive`. Default `hf,kaggle,drive`. Drive is used when it is selected and the selected hubs fail, or when it is the only target |
| `FORGE_HEARTBEAT_S` | Seconds between progress lines during quantize, imatrix, fit, eval, and upload. Default 60 |
| `FORGE_DRIVE_TOKEN` | OAuth token for the Drive fallback. Passed only in the rclone environment |
| `FORGE_DRIVE_SERVICE_ACCOUNT` | Service-account JSON for the Drive fallback, as a string |
| `FORGE_DRIVE_REMOTE` | rclone remote name. Default `forge` |
| `GITHUB_RUN_NUMBER` | Added to `BUILD.json` `offset` for `forge --version` |

## Model card

`templates/model_card.md.jinja` renders `README.md`. The file starts with Hugging Face YAML: `license: apache-2.0`, `base_model` (the abliterated upstream for a requant, the T-Bank repo for T-lite and T-pro), `base_model_relation: quantized`, tags `voxprint`, `gguf`, `abliterated`, `llama.cpp`, `imatrix`, `russian`, `language: [ru, en]`, and `pipeline_tag: text-generation`. The body is the base and licence, method, calibration data and licences, quant types, evals, limits, and credits, then `## Made for Voxprint AI Media Suite` with the Plotweaver link. `huggingface_hub.ModelCard` parses that front matter.

## Canvas

`forge canvas run plans/canvas-zimage-tiers-v1.toml` prepares private image tiers for Voxprint AI Canvas. On the A100 that is already running the LLM bootstrap, start it from `scripts/canvas_job.sh` (`python -m voxprint_forge.diffusion`) so the running `scripts/bootstrap.sh` is not replaced. Part D is a separate entry, `scripts/canvas_job_d.sh` (`python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml`). See `docs/CANVAS.md`.

Parts, in order: base (55 min), controlnet (25 min), edit (55 min). Total 135 min. `--deadline <ISO time>` or `FORGE_DEADLINE` skips a part whose estimate exceeds the time remaining and records `deferred to next rental`. `--force` is the only way to start while `scripts/bootstrap.sh` is still running. `FORGE_DEFER_UPLOAD=1` or `--defer-upload` prepares tier files and writes `<part>-upload-deferred` without tokens. `forge canvas upload PLAN --parts A,B,C` (and `bash scripts/canvas_upload.sh --parts A,B,C,D`) uploads those parts later and is idempotent.

Exit codes are the same table as the rest of the CLI. Uploads are private. Base goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo`. ControlNet goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1`. Edit goes to `Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511`. Each part also creates a private Kaggle model. The manifest is `~/forge-out/manifests/<date>-canvas-zimage.json`.

## Fit

`fit --budget-gb 16 --budget-gb 24` keeps tiers of those sizes. A 16 GB tier must fit at the shortest context (8k in the A100 plan). A 24 GB tier must fit every context (8k and 16k). Headroom defaults to 3 GB. KV is q8_0 unless only q4_0 fits. The gate uses computed weights + KV and the llama-server `model buffer size` / `KV buffer size` lines. `nvidia-smi` usage is recorded and is not the gate.
