# Canvas add-on

Two image-model parts, after the LLM bootstrap on the same A100. Unpack only this add-on tarball onto the tree that is already there. Do not unpack a newer full `voxprint-model-forge-<sha>.tar.gz` while `scripts/bootstrap.sh` is running.

| Part | Work | Estimate |
| --- | --- | --- |
| base | Z-Image-Turbo and Z-Image dtype tiers | 55 min |
| controlnet | Union 2.1 lite FP8, full BF16 | 25 min |

Total when every part runs: 80 min. Each part uploads to its own public Hugging Face repo and a private Kaggle model, then deletes that part's scratch directory. Shared downloads live in `~/forge-out/work/<job>/sources/<id>` until every name in that source's `consumers` list has released. Base and turbo stay until bake-off releases. Qwen is not a consumer. The controlnet file has no later consumer and is removed when that part finishes. `forge --plan plans/canvas-zimage-tiers-v1.toml sources release --source turbo --consumer bake-off` records the external bake-off and deletes the tree when it was the last name. Finished tier files stay in `~/forge-out`.

`FORGE_DEADLINE` (or `--deadline`) is an ISO time. Before a part starts, the job skips it when the estimate is longer than the time remaining and records `deferred to next rental`. Equal time still runs.

The job refuses to start while `scripts/bootstrap.sh` is alive (`~/forge-out/bootstrap.lock` with a live pid, or a real `bash scripts/bootstrap.sh` / `python -m voxprint_forge` process) unless you pass `--force`. A tmux server whose command line still contains the pane command is not a job.

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
export FORGE_HOST_KIND=rented-vm
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

The entry point is `python -m voxprint_forge.diffusion run plans/canvas-zimage-tiers-v1.toml --json --forge-out ~/forge-out --root <repo>`. Those flags follow `run PLAN` because this entry is one parser, not a `forge` subcommand. `HF_USER` defaults to `Mitroshenkov87`. If `HF_TOKEN` or `KAGGLE_API_TOKEN` is unset and stdin is not a terminal, the script exits 2 and does not prompt. Tokens stay in the environment. `FORGE_HOST_KIND=rented-vm` logs in once (`hf auth login --token`, Kaggle token at `~/.kaggle/access_token`) and leaves those logins for the rental. The default `local` does not call `hf auth login` or `huggingface-cli login` and does not write token files. The script does not clear the token variables on exit.

Preflight is `nvcc` on `PATH` (or `/usr/local/cuda/bin`), `torch.cuda.is_available()`, and the diffusers imports. This job does not compile llama.cpp, so cmake and ninja are not required, and it does not read an imatrix calibration file.

## Destinations

Hugging Face repos are public. The licence gate (Apache-2.0, MIT, or another permissive tag; non-commercial refused) and a complete card run before the create.

- `Mitroshenkov87/voxprint-mirror-z-image-turbo`
- `Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1`

Kaggle owner `KAGGLE_USER` (default `mitroshenkov87`), one private model per part. A variation is marked done only after the remote sha256 and size match.

Copy the manifest back when you want a local registry. The VM does not push it:

```bash
scp <user>@<vm>:~/forge-out/manifests/<date>-canvas-zimage.json .
scp <user>@<vm>:~/forge-out/manifests/<date>-canvas-zimage.MODELS.md .
```

Exit codes match the forge CLI: 0 success (including a part deferred for the deadline), 2 usage or a live LLM job, 3 licence, 4 checksum, 5 download, 6 stage or upload mismatch, 7 disk, 8 plan, 9 missing tool.

## Qwen-Image-Edit and Qwen-Image-2512 are removed

Qwen-Image-Edit (the old part C) and Qwen-Image-2512 (parts D and E) are removed from this forge. `scripts/canvas_job_d.sh` exits 2 and does not start a job. Canvas is base and controlnet.

## Token-free prepare, upload later

The A100 can run with no `HF_TOKEN` and no `KAGGLE_API_TOKEN`. Downloads of the public inputs use `HF_TOKEN` only when it is set. Set `FORGE_DEFER_UPLOAD=1` (or pass `--defer-upload`). The scripts skip the token prompt. Each part still runs fetch, convert, quality, card, and sha256 sidecars, writes `~/forge-out/stages/<job>/<part>-upload-deferred.json`, keeps the tier files, and deletes the FP32/BF16 download. The part status is `upload-deferred`.

```bash
export FORGE_DEFER_UPLOAD=1
bash scripts/canvas_job.sh
```

The python entry is the same command as the script, plus `--defer-upload`:

```bash
python -m voxprint_forge.diffusion run plans/canvas-zimage-tiers-v1.toml --json --forge-out ~/forge-out --root <repo> --defer-upload
```

When the tokens are back, one batch uploads every deferred part, checks sha256 and size, and flips the marker to `<part>-upload`. A second run is a no-op.

```bash
export HF_TOKEN=...
export KAGGLE_API_TOKEN=...
export FORGE_HOST_KIND=rented-vm
bash scripts/canvas_upload.sh --parts A,B
forge --forge-out ~/forge-out --root <repo> canvas upload plans/canvas-zimage-tiers-v1.toml --parts A,B
```

`bash scripts/canvas_upload.sh` with no `--parts` uploads every deferred base and controlnet part.
