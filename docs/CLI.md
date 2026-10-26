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

`forge compare --job ID` uses another short set: 0 when every candidate passes, 1 when the quality bar fails, 2 for setup, 3 when there is no CUDA GPU. `forge compare --make-fp8 --job D1|D4 --out DIR` writes local FP8 files and does not need a GPU. See `docs/COMPARE.md`.

`forge kl` uses the table above. It writes a perplexity and KL report for one L-job and exits 6 when conversion, the log parse, or the Qwen3.6 MTP check fails. See `docs/KL.md`.

## Install

Python 3.14 or newer. The shared library `voxprint-common` requires 3.14. The A100 bootstrap installs 3.14 and a CUDA torch wheel. Local tests can use a CPU wheel.

`voxprint-common` is not on PyPI yet. The wheel v0.5.2 is the official release asset (https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.5.2/voxprint_common-0.5.2-py3-none-any.whl, merge commit c3a74eaa5e834391b499af08b110efa37441209d) vendored at `vendor/voxprint_common-0.5.2-py3-none-any.whl` (sha256 `dd2ac2352a63313ff9fc23c2dfddfb2ade8d7ac49cf95802fcae1000e1f7f9c1`). Install that wheel, then the project. The core pin is `voxprint-common==0.5.2`; the `gui` extra needs `voxprint-common[ui]==0.5.2`.

```bash
pip install vendor/voxprint_common-0.5.2-py3-none-any.whl
pip install -e ".[dev]"
```

The Windows installer ships the `te-quant` extra from `installer/windows/requirements-te-quant.txt`: bitsandbytes==0.50.2, the PyPI win_amd64 wheel, sha256 `c697963c8fda3dcd0d7ebd9b5211ae4067feef7cd06e0350d4e816a434fe683d`.

Model downloads through MDL honour `VOXPRINT_MDL_ROUTE=direct|system|auto` (default `auto`).

## Commands

`forge --version` prints `X.Y.Z build N "Charash"`. `N` is `GITHUB_RUN_NUMBER` plus the `offset` in `BUILD.json`. `VOXPRINT_BUILD` replaces that sum. With neither value, and no recorded build, the line says `dev` instead of a number. The raw run number is not the build when `offset` is not zero. The first codename is Charash. The Windows pack writes that number and `rc` into the wheel as `build_info.json` and into the installed `BUILD.json`. An installed `voxprint-forge-cli --version` prints the recorded build when those variables are unset.

`forge plan-summary` prints the plan, the per-model minute estimates, and the required path including the llama.cpp build. It does not import the ablation stack. Bootstrap prints the same text with `PYTHONPATH=src` before torch is installed.

`forge preflight --plan PLAN` checks, in a few seconds, that the converter Python has torch, transformers, and gguf; that each enabled safetensors model answers HEAD for `model.safetensors.index.json` at the pinned revision; that the T-lite pre-tokenizer hash is registered on the pinned converter; that some quant on every quantize tier fits the tier budget (parameters times bits per weight, plus q8_0 KV at the longest plan context); that free disk covers the plan's pinned files; that RAM covers the smallest quant; that `HF_TOKEN` and `KAGGLE_API_TOKEN` are set unless `FORGE_DEFER_UPLOAD=1`; and that the llama.cpp binaries are present or cmake, ninja, and nvcc are on `PATH`. `run-all` and the stage commands run this check before any download or quant. `--json` prints the report, including a `checks` list. Each failed check has a `fix`. A failed preflight still exits 6.

`forge-gui` is the internal operator window (pre-Yitro, not shipped): the same checks, the per-model status table, and the upload-target selector, over this CLI and the library. The user app `voxprint-model-forge` is the LoRA trainer on the shared Yitro shell; see [ui/FEATURE-MAP.md](ui/FEATURE-MAP.md). Tokens typed there are stored in the OS keyring under the service `voxprint-forge`, not in the repo. Install it with `pip install -e ".[gui]"`. The installed binaries are `voxprint-model-forge.exe` (GUI) and `voxprint-forge-cli.exe` (CLI). See the install names in the repository README.

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

- Hugging Face: `HF_USER/voxprint-<model-id>`, created with `private=False`. The stage does not call `update_repo_visibility`. Before that create, the licence must be Apache-2.0, MIT, or another permissive tag. A non-commercial tag is refused. The model card must contain `## License`, `## Credits`, and `## Modification notice`, and the licence section is one line plus a link, not a fenced copy of the licence. Files are the selected GGUF for each tier, the imatrix, `README.md`, the `.sha256` sidecars, and `voxprint-model.json` at the repo root next to the card. `HF_HUB_ENABLE_HF_TRANSFER=1` is set when `hf_transfer` imports.
- Kaggle: owner `KAGGLE_USER` (default `mitroshenkov87`). One private model per base (`isPrivate: true`). One variation per tier, framework `gguf`. Slugs use `[a-z0-9-]` (`t-lite-it-2.1` becomes `voxprint-t-lite-it-2-1`). Hugging Face keeps the dotted name. Bootstrap installs the `kaggle` CLI into the forge venv; otherwise the CLI is `kaggle` on `PATH` or `uvx kaggle`, version 2.2 or newer. On a local machine the API token is passed only in the child environment. A temporary config directory is deleted after each call.

A refused licence or an incomplete card does not fall through to Drive. After each upload the remote sha256 and size must match the local file. The client retries with backoff (2s, 4s, 8s, 16s). The `upload` marker is written only after both sides match. Local deliverables stay on disk until that match. F16 and safetensors intermediates are removed once the quants and the imatrix exist.

The run summary lists every uploaded file with its sha256, Hugging Face URL, and Kaggle URL.

## Manifest

`run-all` writes `manifests/<date>-a100.json` and `manifests/<date>-a100.MODELS.md` under `~/forge-out/manifests/`. `torrent` is the placeholder `placeholder`. The A100 box has no GitHub remote. The run prints the two paths so the operator can copy them back. It does not push.

Each published model also gets `voxprint-model.json` (schema `voxprint-model/1`) at the root of its Hugging Face repo, next to the card, and on the Kaggle and Drive mirrors. That file is separate from the Plotweaver run export above. See `docs/MODEL-MANIFEST.md`. `voxprint-forge-cli index build` writes a local `index.json` and does not upload. `index build` writes the signed-index object (`head`, `serial`, `generated_at`, `models`); `serial` is the published serial plus one unless `--serial` is given. `index push` uploads only when `--push` is set: it signs the exact bytes of `index.json` with the Ed25519 key in `VOXPRINT_MDL_INDEX_KEY_FILE` and uploads `index.json` and `index.json.sig` in one commit. A bare list or a serial that is not above the published one is refused. See `docs/KEY-ROTATION.md`. `index backfill` writes manifests for already published `Mitroshenkov87/voxprint-*` repos and does not upload.

## Environment

| Variable | Role |
| --- | --- |
| `HF_TOKEN` | Hub read and public upload. Not logged. |
| `HF_USER` | Hub account that owns the public model repos |
| `FORGE_HOST_KIND` | `local` (default) or `rented-vm`. See `docs/VM.md` |
| `FORGE_NONINTERACTIVE` | When set, bootstrap does not open `/dev/tty` for a missing token |
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
| `VOXPRINT_BUILD` | Build number. Wins over `GITHUB_RUN_NUMBER` |
| `VOXPRINT_MODELS_DIR` | Absolute path of the shared model-component directory. Default is the suite models folder |
| `VOXPRINT_MODEL_CACHE` | Alternate absolute directory for model components |
| `VOXPRINT_FORGE_MOSS_LORA` | `0` hides MOSS-Image Base as a LoRA base (app list and `lora train`). Default `1` (`MOSS_LORA_DEFAULT`) |
| `GITHUB_RUN_NUMBER` | Added to `BUILD.json` `offset` for `forge --version`. Not used on its own |

## Model card

`templates/model_card.md.jinja` renders `README.md`. The file starts with Hugging Face YAML: `license: apache-2.0`, `base_model` (the abliterated upstream for a requant, the T-Bank repo for T-lite and T-pro), `base_model_relation: quantized`, tags `voxprint`, `gguf`, `abliterated`, `llama.cpp`, `imatrix`, `russian`, `language: [ru, en]`, and `pipeline_tag: text-generation`. The body is the base and licence, method, calibration data and licences, quant types, evals, limits, and credits, then a one-line `## License` section with a link to the Apache-2.0 text, then `## Made for Voxprint AI Media Suite` with the Plotweaver link. `huggingface_hub.ModelCard` parses that front matter.

## Canvas

`forge canvas run plans/canvas-zimage-tiers-v1.toml` prepares image tiers for Voxprint AI Canvas. On the A100 that is already running the LLM bootstrap, start it from `scripts/canvas_job.sh` (`python -m voxprint_forge.diffusion`) so the running `scripts/bootstrap.sh` is not replaced. Qwen-Image-Edit and Qwen-Image-2512 are removed. `scripts/canvas_job_d.sh` exits 2. See `docs/CANVAS.md`.

Parts, in order: base (55 min), controlnet (25 min). Total 80 min. `--deadline <ISO time>` or `FORGE_DEADLINE` skips a part whose estimate exceeds the time remaining and records `deferred to next rental`. `--force` is the only way to start while `scripts/bootstrap.sh` is still running. `FORGE_DEFER_UPLOAD=1` or `--defer-upload` prepares tier files and writes `<part>-upload-deferred` without tokens. `forge canvas upload PLAN --parts A,B` (and `bash scripts/canvas_upload.sh --parts A,B`) uploads those parts later and is idempotent.

Exit codes are the same table as the rest of the CLI. Hugging Face repos are public. Base goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo`. ControlNet goes to `Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1`. Each part also creates a private Kaggle model. The manifest is `~/forge-out/manifests/<date>-canvas-zimage.json`.

## Diffusion INT4

`forge diffusion-int4 run` converts Wan 2.2 T2V-A14B to GGUF (Q4_K_M and Q5_K_M for both experts) and FP8, runs a short smoke, then cards and uploads. `--svd-rank` defaults to 256 and is stored on the run. `--dry-run` is the CPU smoke: tiny files, no CUDA, no downloads. `FORGE_SKIP_WAN=1` omits the Wan steps.

`forge diffusion-int4 estimates` prints the A100 and H100 planning minutes. Wan minutes are the same on both (190). `forge diffusion-int4 disk-floor` exits 2 when free space stays under 15 percent after verified sources are deleted. JSON `error.code` is `disk_floor`. Exit 7 stays the general "next stage does not fit its byte budget" code. `forge diffusion-int4 fetch --dest DIR` downloads the Wan experts. A refused licence exits 3 and does not fall through to Drive.

Markers are `~/.done/diffusion-int4/<step>` (`FORGE_DONE_DIR`). `~/watch.json` (`FORGE_WATCH`) lists the jobs, which are done, and the expected minutes. Heartbeats stay in `~/forge-out/status.json`.

`bash scripts/vm_run_diffusion_int4.sh` is the A100 (compute 8.0) and H100 rental script. It clones Wan2.2, ComfyUI-GGUF, and llama.cpp, applies `lcpp.patch`, builds `llama-quantize` for `80;90`, creates `venv-wan`, and exports `FORGE_WAN_REPO`, `FORGE_WAN_PYTHON`, `FORGE_COMFY_GGUF`, `FORGE_LLAMA_SRC`, and `FORGE_LLAMA_BIN`. `--dry-run` skips CUDA and those clones. Qwen-Image-Edit is not part of this script.

## Z-Image INT4

The Windows RTX 4090 laptop check is still the 16 GB fit. Nunchaku has sm_89 kernels. INT4 runs from compute capability 8.0. The Linux driver floor is 525.60.13. Ampere FP8 timings are not representative of RTX 40/50.

`scripts/smoke_laptop.ps1` runs only the `zimage` step. `scripts/nunchaku_zimage_test.ps1` (Windows) and `scripts/nunchaku_zimage_test.sh` (Linux) run tag `nunchaku-test-2`: `zimage-r128` and optional `zimage-r256`, with `cap16` as the 16 GB cap. The module is `python -m voxprint_forge.diffusion.nunchaku_zimage`.

## LoRA review rounds

`voxprint-forge-cli lora` trains a LoRA. With review rounds on, the first stage is short so the first pictures arrive before the rest of the preset.

```bash
voxprint-forge-cli lora train --project DIR --review-rounds 3 \
  --first-round-steps 120 --first-round-samples 6 --samples-per-round 8 --sample-steps 28
```

- `--review-rounds` is how many review pauses (1–6, default 3). `--review-rounds 0` trains straight through, including a resume of a run that never paused for review.
- `--first-round-steps` is the first stage (100–500, default 120). Later rounds share the preset steps that remain; the last of them takes the remainder. The preset total does not change. One planned round is only the first-round length. The steps still left are what `lora continue --unattended` runs.
- `--first-round-samples` (default 6) and `--samples-per-round` (default 8) are the pictures at the first review and at later reviews. Both are drawn at the training resolution. `--sample-steps` (default 28) is passed to the preview pipeline as `num_inference_steps`. Test pictures during a stage keep the pipeline default.
- `--sample-attire {auto,as_photos,clothed,unclothed}` chooses the attire word on those pictures (default `auto`). It is stored as `sample_attire` on `project.json`. Each prompt is the trigger, one framing from a fixed rotation (close-up portrait, waist-up, full body), and one attire word. The rotation start is `(seed + round number) % 3`. `auto` uses unclothed when any prepared caption sets `unclothed`, and otherwise `as_photos`. `as_photos` uses the photos' attire word and does not add clothing. A `--preview-prompts` file still replaces this list. The review screen's list applies from the next round.
- `lora continue PROJECT` trains the next planned round on the original photos and the liked samples, then pauses for another review.
- `lora continue PROJECT --unattended` keeps the current liked set, uses disliked samples as negatives where the base supports them, and trains every remaining preset step with no more pauses. It is allowed on the last review; with nothing left it saves the LoRA and draws a final preview.
- `lora finish PROJECT` saves the current LoRA, draws a final preview, and records the steps it got. `train.json` and the export store `steps` and `steps_planned`.
- `lora finish PROJECT --discard` stops and removes temporary checkpoints and samples. Nothing is saved. Photos stay. Without a terminal, `--discard` also needs `--yes`.
- `lora export PROJECT [-o PATH]` writes `<stage name or name>.voxlora`, an uncompressed zip with `model.safetensors`, `voxprint-model.json` and previews. The description schema is `docs/schemas/voxprint-model-1.json`. `lora export --project DIR` still writes the `.safetensors` file. `--preset quick` is 150 steps and follows the same path as the other presets.
- `lora status PROJECT` prints `train.json`. During round 1 the result includes `first_pictures` (steps left times measured seconds per step, plus samples times measured seconds per sample; until a sample is timed, one denoising step counts as half a training step).

The rest of the trainer is in `docs/LORA.md`.

## Fit

`fit --budget-gb 16 --budget-gb 24` keeps tiers of those sizes. A 16 GB tier must fit at the shortest context (8k in the A100 plan). A 24 GB tier must fit every context (8k and 16k). Headroom defaults to 3 GB. KV is q8_0 unless only q4_0 fits. For each quant, best first, fit tries placements in order: everything on the GPU, KV in RAM (`--no-kv-offload`), MoE experts on CPU (`--n-cpu-moe`, when the plan sets `moe_layers` and `expert_fraction`), then a diffusion text encoder in RAM or sequential loading (when the plan sets `encoder_gib`). An offload that fits is kept instead of a smaller quant. GPU and flagged offloads are measured with llama-server on the target. Encoder offload and sequential loading have no llama-server flag, so those two are the byte formula. The fit marker, the registry draft, and the model card record the split that tier assumes. The gate uses on-card weights + KV and the llama-server `model buffer size` / `KV buffer size` lines. `nvidia-smi` usage is recorded and is not the gate.
