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

`forge verify-logins` checks `HF_TOKEN` with the Hub API and `KAGGLE_API_TOKEN` with `kaggle` (or the venv `kaggle`, or `uvx kaggle`) 2.2 or newer. It does not write a token file.

`forge accounts add|list|test|remove --json` stores Hugging Face, Kaggle (KGAT), and Google Drive credentials in the OS keyring (`keyring`: Windows Credential Manager, Linux Secret Service). Nothing is written to a token file. `add --target hf --login browser` uses the Hugging Face device-code flow. `add --target drive --login browser` runs `rclone authorize drive` and keeps the token JSON in the keyring. `list` and `test` return the account name, not the token. `test` checks the connection. A VM upload passes those tokens on ssh stdin for that command only, then the remote shell unsets them.

`forge fetch|abliterate|quantize|fit|eval|card|publish-prep|upload --model ID` runs through that stage. Later stages resume earlier markers under `~/forge-out/stages/<id>/`. `fit` accepts repeatable `--budget-gb`.

`forge publish-prep` writes the local card and a registry draft. It does not upload.

`forge upload` and `forge run-all` upload a model as soon as that model is READY. A failure is `STAGEFAIL <stage>` for that model and the others keep going. `run-all --jobs N` (or `FORGE_JOBS`, default 1; `scripts/vm_run.sh` exports 2) overlaps CPU quantize of one model with GPU work of another. The run prints one line per model: `READY`, `UPLOADED`, `STAGEFAIL <stage>`, or `DEFERRED`. `ALL-DONE` is printed only when no model failed. A failed model makes the process exit 6.

`FORGE_DEFER_UPLOAD=1` leaves the model `READY`, writes `<id>-upload-deferred`, and does not write the upload marker, so a later run can upload. When Hugging Face and Kaggle both fail, the same files are copied with rclone to `Voxprint/model-forge/<date>/<model>`. Configure the remote with `RCLONE_CONFIG_*`, or set `FORGE_DRIVE_TOKEN` or `FORGE_DRIVE_SERVICE_ACCOUNT` (never a token file). The remote name defaults to `forge` (`FORGE_DRIVE_REMOTE`). After the copy, `rclone sha256sum` must match the local file.

Global flags precede the subcommand. `scripts/vm_run.sh` runs:

```bash
forge --plan plans/plan.toml --root <repo> --forge-out ~/forge-out run-all
```

`forge run-all --plan ...` is rejected.

## Upload

Hugging Face repos are public. Kaggle models stay private.

- Hugging Face: `HF_USER/voxprint-<model-id>`, created with `private=False`. The stage does not call `update_repo_visibility`. Before that create, the licence must be Apache-2.0, MIT, or another permissive tag. A non-commercial tag is refused. The model card must contain `## License`, `## Credits`, and `## Modification notice`, and the licence section is one line plus a link, not a fenced copy of the licence. Files are the selected GGUF for each tier, the imatrix, `README.md`, and the `.sha256` sidecars. `HF_HUB_ENABLE_HF_TRANSFER=1` is set when `hf_transfer` imports.
- Kaggle: owner `KAGGLE_USER` (default `mitroshenkov87`). One private model per base (`isPrivate: true`). One variation per tier, framework `gguf`. Slugs use `[a-z0-9-]` (`t-lite-it-2.1` becomes `voxprint-t-lite-it-2-1`). Hugging Face keeps the dotted name. Bootstrap installs the `kaggle` CLI into the forge venv; otherwise the CLI is `kaggle` on `PATH` or `uvx kaggle`, version 2.2 or newer. On a local machine the API token is passed only in the child environment. A temporary config directory is deleted after each call.

A refused licence or an incomplete card does not fall through to Drive. After each upload the remote sha256 and size must match the local file. The client retries with backoff (2s, 4s, 8s, 16s). The `upload` marker is written only after both sides match. Local deliverables stay on disk until that match. F16 and safetensors intermediates are removed once the quants and the imatrix exist.

The run summary lists every uploaded file with its sha256, Hugging Face URL, and Kaggle URL.

## Manifest

`run-all` writes `manifests/<date>-a100.json` and `manifests/<date>-a100.MODELS.md` under `~/forge-out/manifests/`. `torrent` is the placeholder `placeholder`. The A100 box has no GitHub remote. The run prints the two paths so the operator can copy them back. It does not push.

## Environment

| Variable | Role |
| --- | --- |
| `HF_TOKEN` | Hub read and public upload. Not logged. |
| `HF_USER` | Hub account that owns the public model repos |
| `FORGE_HOST_KIND` | `local` (default) or `rented-vm`. See `docs/VM.md` |
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

`templates/model_card.md.jinja` renders `README.md`. The file starts with Hugging Face YAML: `license: apache-2.0`, `base_model` (the abliterated upstream for a requant, the T-Bank repo for T-lite and T-pro), `base_model_relation: quantized`, tags `voxprint`, `gguf`, `abliterated`, `llama.cpp`, `imatrix`, `russian`, `language: [ru, en]`, and `pipeline_tag: text-generation`. The body is the base and licence, method, calibration data and licences, quant types, evals, limits, and credits, then a one-line `## License` section with a link to the Apache-2.0 text, then `## Made for Voxprint AI Media Suite` with the Plotweaver link. `huggingface_hub.ModelCard` parses that front matter.

## Canvas

`forge canvas run plans/canvas-zimage-tiers-v1.toml` prepares image tiers for Voxprint AI Canvas. On the A100 that is already running the LLM bootstrap, start it from `scripts/canvas_job.sh` (`python -m voxprint_forge.diffusion`) so the running `scripts/bootstrap.sh` is not replaced. Parts D and E (Qwen-Image-2512) are retired. `scripts/canvas_job_d.sh` (`python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml`) exits 2 unless `--parts` omits D and E or you pass `--force`. See `docs/CANVAS.md`.

Parts, in order: base (55 min), controlnet (25 min), edit (55 min). Total 135 min. Part C converts the shared Qwen2.5-VL-7B encoder. `--deadline <ISO time>` or `FORGE_DEADLINE` skips a part whose estimate exceeds the time remaining and records `deferred to next rental`. `--force` is the only way to start while `scripts/bootstrap.sh` is still running, and the only way to run retired parts D and E. `FORGE_DEFER_UPLOAD=1` or `--defer-upload` prepares tier files and writes `<part>-upload-deferred` without tokens. `forge canvas upload PLAN --parts A,B,C` (and `bash scripts/canvas_upload.sh --parts A,B,C`) uploads those parts later and is idempotent.

Exit codes are the same table as the rest of the CLI. Hugging Face repos are public. Base goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo`. ControlNet goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1`. Edit goes to `Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511`. Each part also creates a private Kaggle model. The manifest is `~/forge-out/manifests/<date>-canvas-zimage.json`.

## Nunchaku edit matrix

`scripts/nunchaku_edit_test.sh` and `scripts/nunchaku_edit_test.ps1` run `python -m voxprint_forge.diffusion.nunchaku_edit`. This is not a `forge` subcommand. The tag is `nunchaku-test-1`. `--json` prints one result object on stdout. Logs stay on stderr. The scripts do not prompt.

```bash
bash scripts/nunchaku_edit_test.sh --json --smoke
pwsh scripts/nunchaku_edit_test.ps1 --arms qwen-r128,qwen-r32,zimage-r128 --caps unconstrained,cap16
```

| Flag | Meaning |
| --- | --- |
| `--arms` | `qwen-r128`, `qwen-r32`, `zimage-r128`, `zimage-r256`. Default omits r256 |
| `--cases` | `c1,c2,c3` for Qwen and `z1,z2,z3` for Z-Image |
| `--out` | Default `results/nunchaku-edit/nunchaku-test-1/` |
| `--mode` | `auto`, `model-offload`, or `block-offload` |
| `--caps` | `unconstrained` and/or `cap16` (16/24 of a 24 GB card). Default is both |
| `--smoke` | One case per arm. Qwen uses 20 steps instead of 40 |
| `--cleanup` | Delete the INT4 safetensors after the run. Bases stay |
| `--budget-min` | Default 90. r256 is dropped first when the budget is exceeded |

`cap16` calls `torch.cuda.set_per_process_memory_fraction(16/24)`. `auto` tries model CPU offload and then block offload (`set_offload` with one pinned block, then sequential offload). Unconstrained Z-Image tries a resident GPU load first. A warm-up image is not included in `seconds_per_image`. `fits_16gb` is true when the nvidia-smi peak, sampled every 0.5 s, is at most 15.0 GiB.

Exit codes for this script are separate from the `forge` table above: 0 pass, 1 a case failed or cap16 did not fit, 2 setup (wheel, sha256, disk, download), 3 no CUDA GPU, under 16 GB, or compute capability below 8.9. The wheel is the pinned vitoom-nunchaku build, checked by sha256. Upstream `nunchaku` is refused because it has no `zero_cond_t`. Resume markers are `~/.done/nunchaku-edit/<step>`. `result.json` uses schema `voxprint-forge-nunchaku-edit/1` and records the GPU, driver, torch, diffusers, nunchaku, wheel hash, and weight hash. See `docs/VM.md` for the 4090 steps.

## Fit

`fit --budget-gb 16 --budget-gb 24` keeps tiers of those sizes. A 16 GB tier must fit at the shortest context (8k in the A100 plan). A 24 GB tier must fit every context (8k and 16k). Headroom defaults to 3 GB. KV is q8_0 unless only q4_0 fits. For each quant, best first, fit tries placements in order: everything on the GPU, KV in RAM (`--no-kv-offload`), MoE experts on CPU (`--n-cpu-moe`, when the plan sets `moe_layers` and `expert_fraction`), then a diffusion text encoder in RAM or sequential loading (when the plan sets `encoder_gib`). An offload that fits is kept instead of a smaller quant. GPU and flagged offloads are measured with llama-server on the target. Encoder offload and sequential loading have no llama-server flag, so those two are the byte formula. The fit marker, the registry draft, and the model card record the split that tier assumes. The gate uses on-card weights + KV and the llama-server `model buffer size` / `KV buffer size` lines. `nvidia-smi` usage is recorded and is not the gate.
