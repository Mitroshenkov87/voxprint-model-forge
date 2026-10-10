# Canvas add-on

Three private image-model parts, after the LLM bootstrap on the same A100. Unpack only this add-on tarball onto the tree that is already there. Do not unpack a newer full `voxprint-model-forge-<sha>.tar.gz` while `scripts/bootstrap.sh` is running.

| Part | Work | Estimate |
| --- | --- | --- |
| base | Z-Image-Turbo and Z-Image dtype tiers | 55 min |
| controlnet | Union 2.1 lite FP8, full BF16 | 25 min |
| edit | Qwen-Image-Edit-2511 FP8 | 55 min |

Total when every part runs: 135 min. Each part uploads to its own private Hugging Face repo and a private Kaggle model, then deletes that part's FP32/BF16 download intermediates. Finished tier files stay in `~/forge-out`.

`FORGE_DEADLINE` (or `--deadline`) is an ISO time. Before a part starts, the job skips it when the estimate is longer than the time remaining and records `deferred to next rental`. Equal time still runs.

The job refuses to start while `scripts/bootstrap.sh` is alive (`~/forge-out/bootstrap.lock` with a live pid, or that process on the machine) unless you pass `--force`. A tmux session named `forge` can outlive the script, so the session name is not the signal.

## Operator steps

On a machine that can reach GitHub, from a checkout that already contains this add-on:

```bash
bash scripts/make_vm_bundle.sh --canvas-only
scp dist/voxprint-model-forge-canvas-<sha>.tar.gz <user>@<vm>:
```

On the VM, in a second tmux window. Do not replace `scripts/bootstrap.sh`.

```bash
cd ~/voxprint-model-forge
tar -xzf ~/voxprint-model-forge-canvas-<sha>.tar.gz
export HF_TOKEN=...
export KAGGLE_API_TOKEN=...
export KAGGLE_USER=mitroshenkov87
# HF_USER defaults to Mitroshenkov87 inside scripts/canvas_job.sh.
export FORGE_DEADLINE=2026-10-10T18:00:00Z
tmux new -s canvas 'bash scripts/canvas_job.sh'
```

Re-attach with `tmux attach -t canvas`. Progress lines append to `~/forge-out/progress.log`.

If bootstrap is still running, `canvas_job.sh` exits 2. After it has exited:

```bash
tmux new -s canvas 'bash scripts/canvas_job.sh'
```

`--force` starts anyway:

```bash
bash scripts/canvas_job.sh --force
```

## What the script checks

`nvcc` must be on `PATH`. If it is not, the script prepends `/usr/local/cuda/bin` and sets `LD_LIBRARY_PATH` from `/usr/local/cuda/lib64` and `/usr/local/cuda/lib`. It then requires `torch.cuda.is_available()` in `~/forge-out/venv`. It does not install cmake.

Canvas packages are installed with `--no-deps` so the shared venv keeps its current `huggingface_hub`:

```text
diffusers==0.41.0
torchao==0.18.0
lpips==0.1.4
pillow-avif-plugin==1.6.0
torchvision==0.29.1
```

`torchvision` is the CUDA 13.0 build that matches `torch==2.14.1`. It is installed with `--no-deps` so it does not replace torch. `lpips` imports it.

The entry point is `python -m voxprint_forge.diffusion run plans/canvas-zimage-tiers-v1.toml --json --forge-out ~/forge-out --root <repo>`. Those flags follow `run PLAN` because this entry is one parser, not a `forge` subcommand. `HF_USER` defaults to `Mitroshenkov87`. If `HF_TOKEN` or `KAGGLE_API_TOKEN` is unset and stdin is not a terminal, the script exits 2 and does not prompt. Tokens stay in the environment. On EXIT, INT, and TERM the script removes Hugging Face and Kaggle token files and unsets the two token variables. It does not call `hf auth login` or `huggingface-cli login`.

Preflight is `nvcc` on `PATH` (or `/usr/local/cuda/bin`), `torch.cuda.is_available()`, and the diffusers imports. This job does not compile llama.cpp, so cmake and ninja are not required, and it does not read an imatrix calibration file.

## Private destinations

- `Mitroshenkov87/voxprint-mirror-z-image-turbo`
- `Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1`
- `Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511`

Kaggle owner `KAGGLE_USER` (default `mitroshenkov87`), one private model per part. A variation is marked done only after the remote sha256 and size match. Nothing is made public.

Copy the manifest back when you want a local registry. The VM does not push it:

```bash
scp <user>@<vm>:~/forge-out/manifests/<date>-canvas-zimage.json .
scp <user>@<vm>:~/forge-out/manifests/<date>-canvas-zimage.MODELS.md .
```

Exit codes match the forge CLI: 0 success (including a part deferred for the deadline), 2 usage or a live LLM job, 3 licence, 4 checksum, 5 download, 6 stage or upload mismatch, 7 disk, 8 plan, 9 missing tool.

## Part D

Qwen-Image-2512 FP8, in its own entry so a running `scripts/canvas_job.sh` is not replaced. INT4 stays a marker: there is no official licence-clean nunchaku-tech SVDQuant pin, and this rental does not calibrate. The Qwen2.5-VL-7B text encoder is byte-checked against Edit-2511. When the pins match and part C has already written `qwen2.5-vl-7b-text-encoder-fp8.safetensors`, part D links that file. It does not copy the weights into a second encoder. A pin match with no part C file yet is `reuse-pending`. A hash mismatch converts the encoder. Stable files land in `~/forge-out/canvas/qwen-image-2512`.

| Selection | Work | Estimate |
| --- | --- | --- |
| D (default) | Qwen-Image-2512 transformer FP8, INT4 marker, encoder dedupe | 120 min |
| D, encoder reused | Planning figure when part C's FP8 encoder is linked | 280 min |
| D, encoder recomputed | Planning figure when the encoder hashes do not match | 340 min |
| E | Encoder dedupe only | 15 min |
| C | Qwen-Image-Edit-2511, the existing part | 55 min |

`--parts` accepts `A,B,C,D,E` or `base,controlnet,qwen,edit,encoder`. The run order is A, B, D, C, E, so `--parts C,D` runs D first. Default is D. Part C's FP8 encoder exists only after C has finished, so a same-rental `--parts C,D,E` links it when E runs.

Private destination: `Mitroshenkov87/voxprint-mirror-qwen-image-2512`. Kaggle slug `voxprint-mirror-qwen-image-2512`. Nothing is made public.

On a machine that can reach the checkout:

```bash
bash scripts/make_vm_bundle.sh --canvas-only
```

On the VM, in a tmux window that is not the LLM job and not the A–C canvas job:

```bash
cd ~/voxprint-model-forge
tar -xzf ~/voxprint-model-forge-canvas-<sha>.tar.gz
export HF_TOKEN=...
export KAGGLE_API_TOKEN=...
export KAGGLE_USER=mitroshenkov87
# HF_USER defaults to Mitroshenkov87 inside scripts/canvas_job_d.sh.
tmux new -s canvas-d 'bash scripts/canvas_job_d.sh'
```

The script runs:

```bash
python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml --json --forge-out ~/forge-out --root <repo>
```

Selectors:

```bash
bash scripts/canvas_job_d.sh --parts C,E
bash scripts/canvas_job_d.sh --parts C,D,E
```

`scripts/canvas_job_d.sh` exits 2 while `scripts/canvas_job.sh` is still running, unless you pass `--force`. A missing token on a non-interactive stdin exits 2 and does not prompt. The INT4 directory contains `DEFERRED.txt` and no weights.

## Token-free prepare, upload later

The A100 can run with no `HF_TOKEN` and no `KAGGLE_API_TOKEN`. Downloads of the public inputs use `HF_TOKEN` only when it is set. Set `FORGE_DEFER_UPLOAD=1` (or pass `--defer-upload`). The scripts skip the token prompt. Each part still runs fetch, convert, quality, card, and sha256 sidecars, writes `~/forge-out/stages/<job>/<part>-upload-deferred.json`, keeps the tier files, and deletes the FP32/BF16 download. The part status is `upload-deferred`.

```bash
export FORGE_DEFER_UPLOAD=1
bash scripts/canvas_job.sh
bash scripts/canvas_job_d.sh
bash scripts/canvas_job_d.sh --parts C,D,E
```

The python entry is the same command as the script, plus `--defer-upload`:

```bash
python -m voxprint_forge.diffusion run plans/canvas-zimage-tiers-v1.toml --json --forge-out ~/forge-out --root <repo> --defer-upload
python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml --json --forge-out ~/forge-out --root <repo> --defer-upload
```

When the tokens are back, one batch uploads every deferred part, checks sha256 and size, and flips the marker to `<part>-upload`. A second run is a no-op.

```bash
export HF_TOKEN=...
export KAGGLE_API_TOKEN=...
bash scripts/canvas_upload.sh --parts A,B,C,D
forge --forge-out ~/forge-out --root <repo> canvas upload plans/canvas-zimage-tiers-v1.toml --parts A,B,C
forge --forge-out ~/forge-out --root <repo> canvas upload plans/canvas-qwen-image-2512-v1.toml --parts D,E
```

`bash scripts/canvas_upload.sh` with no `--parts` uploads every deferred part, including E. `--parts C,D,E` uploads edit, Qwen-Image-2512, and the encoder. A reused encoder is still not copied.
