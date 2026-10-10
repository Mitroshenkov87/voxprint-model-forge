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
