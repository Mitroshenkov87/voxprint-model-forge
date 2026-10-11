# A100 80GB VM

Ubuntu 24.04 on immers.cloud: 1x A100 80GB, 16 vCPU, 128 GB RAM, 480 GB NVMe. The rental is about 3 hours. Model work is budgeted at 150 minutes inside that window. The separate RTX 4090 is for app tests later, so fit on this box is computed weights + KV plus a llama-server measurement on the A100. `min_vram_gb` in the registry is 16 or 24 from that measurement.

RAM: the resident ablation path wants the bf16 weights plus about 6 GB. T-pro bf16 is about 61 GiB, so 128 GB is the minimum this plan is written for. Disk: keep at least a few tens of GiB free after the F16 source, the quants, and the reused GGUF. As soon as one mirror has verified a model's finals (sha256, or for a small Hugging Face file the size plus the downloaded hash), Forge deletes that model's source weights, f16/bf16 GGUF, safetensors downloads, and upload staging. A shared canvas source stays until every name in its `consumers` list has released. If the next stage does not fit, Forge first deletes verified leftovers, finished staging, stale partials with no `.sha256`, and orphaned Hugging Face cache blobs (the blob, not only the snapshot symlink), logs each path and size, and checks again. It does not sit waiting for disk while those files are still there. Bootstrap stops with exit 7 when the next stage would still not fit, and it leaves finished quants in place.

The VM has no GitHub access. The repo arrives as a `git archive` bundle, so the tree has no `.git` directory.

## Operator steps

On a machine that can reach GitHub, from a checkout of this repo:

```bash
bash scripts/make_vm_bundle.sh
scp dist/voxprint-model-forge-<sha>.tar.gz <user>@<vm>:
```

`make_vm_bundle.sh` runs `git archive`, writes the commit sha to `VERSION`, and packs llama.cpp `b11476` (`scripts/llama.rev`) under `third_party/`. Bootstrap prints that sha and does not call git or GitHub.

On the VM:

```bash
ssh <user>@<vm>
tar -xzf voxprint-model-forge-<sha>.tar.gz
cd voxprint-model-forge
export HF_TOKEN=...
export KAGGLE_API_TOKEN=...
export KAGGLE_USER=mitroshenkov87
export FORGE_HOST_KIND=rented-vm
tmux new -s forge 'bash scripts/bootstrap.sh'
```

`HF_USER` defaults to `Mitroshenkov87`. `FORGE_HOST_KIND` is `local` unless you set `rented-vm`.

On a disposable rented VM, export `FORGE_HOST_KIND=rented-vm` before bootstrap or a canvas script. After the venv exists the script logs in once: `hf auth login --token` from `HF_TOKEN`, and the Kaggle token written to `~/.kaggle/access_token` (mode 600). Those logins stay for the whole rental. The VM scripts do not remove token files on exit and do not clear the two token variables. `set +x` stays in force around the prompt and the login. Logs go through `scripts/logscrub.py`, so a token value that reaches a log line is replaced.

On the operator's own machine (`FORGE_HOST_KIND=local`, the default) the GUI and `accounts` keep credentials in the OS keyring. A GUI or `ssh` upload sends the keyring tokens on stdin for that one command, and that remote shell clears them before it exits. The local CLI still passes the Kaggle token only in a child environment and deletes that temporary config directory after each call. Local mode does not call `hf auth login` or `huggingface-cli login`.

Bootstrap installs the `kaggle` CLI into `~/forge-out/venv`. Edit quality uses 8 pairs and seed 0 when `~/forge-out/EDIT_TRIM` exists. If `HF_TOKEN` or `KAGGLE_API_TOKEN` is unset and stdin is not a terminal, or `CI` or `FORGE_NONINTERACTIVE` is set, bootstrap exits 2 and does not open `/dev/tty`. A real interactive stdin, with neither of those variables set, still prompts once with `read -s` on `/dev/tty`.

Re-attach: `tmux attach -t forge`.

If the session is gone, export the tokens again and re-run `tmux new -s forge 'bash scripts/bootstrap.sh'`. Stage markers in `~/forge-out/stages` skip finished work, including uploads whose remote sha256 and size already matched.

Progress lines (timestamp, model, stage, percent, ETA) append to `~/forge-out/progress.log` at least every 60 seconds during quantize, imatrix, fit, eval, renders, and uploads. `tail -f ~/forge-out/progress.log` while the session runs. The same pulse updates `~/forge-out/status.json` (stage, percent, ETA, last heartbeat, pid, state, error). Hardware Watchdog can poll `forge status` over ssh. Exit 0 means the items are running or finished, 3 means a heartbeat is older than 180 seconds or the pid is gone, 4 means a model failed or the file is missing.

When the run finishes, copy the manifest back:

```bash
scp <user>@<vm>:~/forge-out/manifests/<date>-a100.json .
scp <user>@<vm>:~/forge-out/manifests/<date>-a100.MODELS.md .
```

Bootstrap prints those two paths. There is no push from the VM.

## What bootstrap does

1. Print the forge commit from `VERSION`. Refuse anything other than Ubuntu 24.04.
2. Require `HF_TOKEN` and `KAGGLE_API_TOKEN` (prompt only when stdin is a terminal), default `HF_USER` to `Mitroshenkov87`, then check `nvidia-smi`. Preflight then requires `cmake`, `ninja`, and `nvcc` on `PATH`, and that the calibration file tokenises to at least `max(2×ctx, ctx×chunks)` tokens. The file is fetched from the pinned gist when the tree does not already contain a copy with the pinned sha256.
3. Print the plan estimates with system Python (no torch import).
4. Install uv, Python 3.14, and `torch==2.14.1` from the cu130 index. If that wheel is missing it retries Python 3.13, then 3.12.
5. When `FORGE_HOST_KIND=rented-vm`, log in once with `hf auth login --token` and write `~/.kaggle/access_token`, then `forge verify-logins`. A failed Hub or Kaggle login stops the script before the llama.cpp build and before any download. Local mode skips that login and only runs `forge verify-logins`.
6. Register the T-lite-it-2.1 Qwen2 pre-tokenizer on the bundled llama.cpp tree, then build with `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=80;89;120`. The registration is `python -m voxprint_forge.llama_patch "$LLAMA_SRC/conversion/base.py"`. It inserts sha256 `e9b7dbd66e0308c6e89983d5b6e1ca047106d862879a0fd33a12c8491b91ec5c` as `qwen2` after the Kimi-K2 entry in `conversion/base.py` at commit `98819068`. The probe text is `src/voxprint_forge/data/llama_b11476_chktxt.txt`. Applying the patch twice is a no-op. `FORGE_PYTHON` is the forge venv interpreter, and `convert_hf_to_gguf.py` uses that interpreter.
7. `scripts/vm_run.sh` runs `forge preflight` and then `forge --plan plans/plan.toml --root <repo> --forge-out ~/forge-out run-all`. Global flags precede the subcommand. `FORGE_JOBS` defaults to 2. Each model uploads when it is READY. If Hugging Face and Kaggle fail, rclone copies the files to `Voxprint/model-forge/<date>/<model>` and checks sha256. A failed model is `STAGEFAIL <stage>`; the others continue. The scripts print `ALL-DONE` only after a zero exit.
8. Tar `~/forge-out/forge-results-<date>.tar` plus a sha256 file, and print the manifest paths to copy back. `ALL-DONE` is printed only when that run succeeded.

Copy results off the box with `scp` or `rsync` as a second copy. The public Hub upload and the private Kaggle upload are the primary copy.

## Order and time

| Step | Minutes |
| --- | --- |
| llama.cpp build (`bootstrap_build`) | 12 |
| Huihui Qwen3.6 27B requant | 45 |
| Gemma 4 31B requant | 48 |
| T-lite-it-2.1 ablation and quants | 35 |
| T-pro-it-2.1, only if time remains | 80 |

Required path including the build: 140 minutes. The narrative target is about 2.5 hours once network variance is included. Mistral-Small-3.2-24B-Instruct-2506 is pinned and disabled. Mistral Small 4 119B is out of scope. YandexGPT-5-Lite and anything under `smkrv/` are refused.

Rough split of the model minutes: Huihui downloads the reused Q5 (and Q4 only if Q5 misses the cap) plus the F16 source, a 128-chunk imatrix, three 16 GB quants, then a light measure and eval. Gemma is the same shape with a larger F16 and only the Q4 reuse. T-lite downloads about 15 GiB, ablates on the GPU, converts to F16, builds the imatrix, and quantizes Q5, Q4, IQ4_XS, and IQ3_M. T-pro is the same work at about 61 GiB of bf16.

## Calibration file

The imatrix corpus is bartowski's `calibration_datav3` (public gist, no explicit licence). It is not stored in this git repository. Bootstrap downloads it on first use into `~/forge-out/calibration/imatrix.txt` and checks the sha256. A copy already at `src/voxprint_forge/data/imatrix.txt` is used only when that sha256 matches, so a hot-patched tree does not download again.

- Source: https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8
- Pinned raw file: https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/41ae165614f4041c262fa8fe668bbe3efd9eb695/calibration_datav3.txt
- sha256: `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`
- GPT-2 content tokens: 74048. llama.cpp at `98819068` errors below `2×ctx` (1024 at ctx 512) and will not produce 128 chunks unless the file has at least `ctx×chunks` tokens (65536). 74048 meets both.
- Licence status: licence unverified, calibration-only, not redistributed in model files. Flagged for GitHub Bot review. Do not copy this text into a published model repo.

`src/voxprint_forge/data/imatrix-ru-pd.txt` is an optional follow-up slice: Pushkin's 1829 lyric, original Russian, public domain (author died 1837). It is not part of the default corpus. Set `FORGE_IMATRIX_RU=1` to append it for that run only.

Eval is a short RU and EN refusal set, one perplexity chunk, and one Russian prose sample. Each tier records llama-server peak weights and KV at 8k and 16k, plus the computed figure, perplexity, refusal rates, and the sample.

## Two tiers

Same source, imatrix, and session.

- 24 GB: best of Q5_K_M / Q4_K_M that leaves 3 GB of headroom at both 8k and 16k (weights + KV at most 21 GiB).
- 16 GB: best of IQ4_XS / IQ3_M (and Q3_K_M for the two requants) that leaves 3 GB at 8k (weights + KV at most 13 GiB). 16k is still measured.

Huihui 24 GB reuses `Huihui-Qwen3.6-27B-abliterated-ggml-model-Q5_K.gguf` (sha256 `a4fd0da6…`, 19535700288 bytes) from `huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF` at `2da8c0432f0a42c25c14cbc99f90028a90d3dcd3`. The filename omits `_M`; the size matches K_M, and the card labels it Q5_K_M. Q4_K from the same repo is the fallback. Both are the same abliteration author and Apache-2.0. The 16 GB tier is quantized here from the pinned F16. Upstream Q3_K is not reused.

Gemma 24 GB reuses `gemma-4-31b-abliterated-Q4_K_M.gguf` (sha256 `380f2f50…`, 18687057536 bytes) from `paperscarecrow/Gemma-4-31B-it-abliterated` at `2491ee58472b82a2372d288f982943b9d5e6a82c`. That is the ablation author. There is no same-ablation Q5. `mradermacher/gemma-4-31B-it-abliterated-GGUF` quantizes `wangzhang/gemma-4-31B-it-abliterated`, a different ablation, and the Q5 file is about 20.35 GiB plus about 1 GiB of KV at 16k, which misses the 21 GiB cap. It is not pinned. If the reused Q4 misses the measurement, Forge quantizes its own from the pinned F16.

T-lite and T-pro are ablated here (difference of means, then `W := W - r r^T W` on the writing matrices). Both tiers are our quants. Heretic is not imported or invoked. The layer offload path remains; on this GPU a 24–33B bf16 model can stay resident when the weights plus 6 GB fit in free VRAM.

F16 and safetensors intermediates are deleted per model after the quants and the imatrix exist. Selected GGUFs, the imatrix, the card, and the sha256 files stay until the public Hub upload and the private Kaggle upload verify.

## Upload

`artifact_host` in the plan is empty. Hugging Face repos are created public (`private=False`, with no forced re-privatize) because the private storage limit was hit. Kaggle stays private. Before the Hub create, the licence must be Apache-2.0, MIT, or another permissive tag. A non-commercial tag is refused, and the card must be complete.

Right after each model, the `upload` stage pushes the finished GGUFs, the imatrix, the model card, and the sha256 sidecars to:

- a public Hugging Face model repo under `HF_USER`
- a private Kaggle model owned by `KAGGLE_USER`, one variation per tier, framework `gguf`

Logins are verified before the build. A remote sha256 or size mismatch retries with backoff and does not mark the stage done. The summary lists each file with its sha256 and both URLs. `~/forge-out/manifests/<date>-a100.json` and the MODELS snippet are the failover list: Hugging Face, Kaggle, and a torrent placeholder. Copy them back; the VM does not push.

The card template is `templates/model_card.md.jinja`. It starts with Hub YAML (`license`, `base_model`, `base_model_relation: quantized`, tags, `language`, `pipeline_tag`). `base_model` is the abliterated upstream for Huihui and Gemma, and the T-Bank repo for T-lite and T-pro. The body is the base and licence, method, calibration data and licences, quant types, evals, limits, and credits. A modification notice stays. `## License` is one line and a link to the Apache-2.0 text, not a fenced copy of the licence. A short "Made for Voxprint AI Media Suite" section links Plotweaver and records the 16 GB and 24 GB tiers with `min_vram_gb`.

## Nunchaku matrix on the A100

Tag `nunchaku-test-2`. Phase 0 runs on the Linux A100 80GB (compute capability 8.0). `scripts/nunchaku_edit_test.sh` is the entry. The Windows script remains for a later 8.9 box. Neither script prompts, and neither installs upstream `nunchaku`. INT4 is the only precision: it covers sm_80, sm_86, and sm_89. NVFP4 stays out.

The run creates its own uv venv. `uv python install 3.11` downloads CPython, then `uv venv --python 3.11 --managed-python` uses that build, so the image does not need a distro Python 3.11 package. The process re-execs into that venv when `sys.prefix` is not the venv directory. Resolved `python` paths are not compared, because uv points every venv at one interpreter. `NUNCHAKU_PYTHON` remains an override. The pinned Linux wheel is `nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311-cp311-linux_x86_64.whl` (sha256 `bcdfe6fcc36cd37da34c1d0f1e0fa7c9b4201b2a70a89a3274d22420221afd2f`). Its `_C` extension contains sm_80 cubins. CUDA 12.x minor-version compatibility needs a Linux driver of at least 525.60.13, which is the usual A100 floor for this wheel. Torch 2.11 comes from the cu128 index. `diffusers==0.37.1` and `transformers==4.57.6` are installed beside the wheel. A hash mismatch deletes the file and exits 2. The fork implements `zero_cond_t` and also ships the Z-Image transformer, so both families share this one environment.

Default arms are QuantFunc Qwen-Image-Edit-2511 INT4 r128 and r32, plus nunchaku-ai Z-Image-Turbo INT4 r128. r256 is off by default so the pass stays inside about 90 minutes. Add it with `--arms qwen-r128,qwen-r32,zimage-r128,zimage-r256`. If the estimate or the elapsed time exceeds `--budget-min` (default 90), r256 is the first arm dropped. Each INT4 arm runs unconstrained and again under `torch.cuda.set_per_process_memory_fraction(16/total)` (`--caps unconstrained,cap16`). On an 80 GB card that fraction is 16/80. Qwen uses `QwenImageEditPlusPipeline`, 40 steps, `true_cfg_scale` 4.0, seed 0, and the committed inputs in `tests/assets/nunchaku_edit/`. Z-Image uses `ZImagePipeline`, 8 steps, guidance 0, and three text-to-image prompts (a clothed adult, a hand, and a VOXPRINT / печать sign). `--smoke` keeps one case per arm and shortens Qwen to 20 steps. One warm-up image per cap is not included in seconds per image.

FP8 baselines are `Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511` and `Mitroshenkov87/voxprint-mirror-z-image-turbo`, run once unconstrained when the dequantized weights fit. Below compute capability 8.9 the baseline is weight-only dequant, because the A100 has no FP8 tensor cores. `result.json` and `MESSAGE.md` mark those timings `not representative of RTX 40/50`. Layerwise FP8 casting is reserved for sm_89 and newer, and only when the bf16 transformer is already on disk. The script does not download the bf16 transformer. Base snapshots keep `transformer/config.json` and skip `transformer` weight files. The download runs even when `HF_HUB_OFFLINE` is set, then the pipeline loads with `local_files_only`. A setup or load error on one arm is recorded on that arm and the matrix continues. A missing `nvidia-smi` or no CUDA GPU still stops the run.

Exit 0 means every requested cap finished and every cap16 nvidia-smi peak is at most 15.0 GiB. Exit 1 means a case failed or cap16 did not fit. Exit 2 is setup: wheel, hash, disk, or download. Exit 3 means no CUDA GPU, under 16 GB of VRAM, compute capability below 8.0, or a Linux driver below 525.60.13. Markers live in `~/.done/nunchaku-edit/`. `--cleanup` deletes only the INT4 files. Results land in `results/nunchaku-edit/nunchaku-test-2/` (`result.json` schema `voxprint-forge-nunchaku-edit/1`, WebP q95, `MESSAGE.md`) and are copied to `/workspace/vm-results/model-forge/nunchaku-test-2/` when that directory's parent exists.

## Diffusion INT4 rental

This is a separate session from the LLM bootstrap. The box is a fresh Ubuntu A100 or H100 80GB (`immers.cloud` `teslaa100-1.16.128.320` or `h100-1.16.128.320`). Export `FORGE_HOST_KIND=rented-vm` so the script logs in once and leaves `HF_TOKEN` in place. The script does not unset tokens and does not enable command tracing. It reads `nvidia-smi` (`sm_80` A100, `sm_90` H100), exports `FORGE_GPU` and `FORGE_COMPUTE_CAP`, and prints the A100 and H100 minute tables. Torch 2.11 comes from the cu128 index. `llama-quantize` is built for `80;90`.

```bash
export HF_TOKEN=...
export HF_USER=Mitroshenkov87
export FORGE_HOST_KIND=rented-vm
export FORGE_DRIVE_TOKEN=...
tmux new -s int4 'bash scripts/vm_run_diffusion_int4.sh'
```

`--dry-run` skips CUDA, clones, venvs, and the token prompt. `--smoke-tiny` is the one-block Qwen calib, INT4 rank 8, convert, and one transformer latent. It exits before the token prompt, `nvidia-smi`, and the clone, and prints `ALL-DONE`. On CUDA, with the vitoom wheel already importable in `FORGE_PYTHON`, it also loads that tiny checkpoint and runs one forward. `--qwen-image` adds the optional Qwen-Image INT4 steps. `--svd-rank` defaults to 256. `--own-edit` defaults to `on` (the own Qwen-Edit-2511 INT4 r256 quant). `--phase0-result` defaults to `/workspace/vm-results/model-forge/nunchaku-test-1/result.json`. When that file is present it is an eval and card baseline. It does not skip the quant unless `--own-edit auto` and QuantFunc r128 passed every hard case with `fits_16gb`. `FORGE_SKIP_WAN=1` omits Wan. `FORGE_WAIT_FOR` holds the live GPU stages until that file exists. `FORGE_PRIVATE_CALIB` may be a directory or a private Hugging Face dataset id; when it is unset or cannot be loaded, that calibration set is skipped.

Preflight requires `nvidia-smi`, `nvcc`, `git`, `cmake`, and `ninja`, then clones the pinned deepcompressor (`lantudou/deepcompressor2` `019cbdeae7e2add52eaf4c27a361cfc1e57993b8`, branch `syh`), ComfyUI-GGUF, llama.cpp `b3962`, and Wan2.2 trees. The deepcompressor `LICENSE` must say Apache License and Version 2.0 or the script exits 3. `69f3473` is not the quant pin: it has no Qwen struct factory. nunchaku-ai/deepcompressor issue 104 is still open, so this fork is the Apache-2.0 Qwen tree. Two Python 3.11 venvs keep the vitoom-nunchaku wheel (cp311, torch 2.11, cu128) off the forge 3.14 stack. `fetch` then downloads QuantFunc (sha256 checked), Qwen-Image-Edit-2511, Wan2.2-T2V-A14B, and the FP8 reference. Set `FORGE_FP8_EDIT` to a local canvas part C tree to skip the public mirror.

Free space under 15 percent exits 2 (`error.code` `disk_floor`) after verified sources are deleted. Uploads run at the end of each model's publish step. Sources are deleted only after a mirror sha256 matches. `~/watch.json` lists the jobs. `~/forge-out/status.json` is the heartbeat file. `ALL-DONE` is printed only after a zero exit.

| Step | A100 minutes | H100 minutes |
| --- | --- | --- |
| Qwen-Edit-2511 calibration | 30 | 30 |
| Qwen-Edit-2511 INT4 r256 | 900 (planning 15 h; the A100 range is 12–18 h) | 500 |
| Nunchaku export | 20 | 20 |
| Eval against QuantFunc r128, r256, and FP8 | 90 | 90 |
| Edit card, sha256, upload | 25 | 25 |
| Edit source delete | 5 | 5 |
| Wan GGUF Q4_K_M and Q5_K_M, both experts | 100 | 100 |
| Wan FP8 | 40 | 40 |
| Wan smoke, 5 frames at 832×480, 4 steps | 20 | 20 |
| Wan card, sha256, upload | 25 | 25 |
| Wan source delete | 5 | 5 |

Required total: 1260 minutes on an A100 (21 h at the 15 h quant midpoint; about 18 h if quant finishes in 12 h and about 24 h if it takes 18 h) and 860 minutes on an H100. The H100 quant step is planned at 1.8×. Optional Qwen-Image adds 1010 minutes on an A100 and 610 on an H100 (quant 900 or 500, export 20, eval 60, publish 25, reclaim 5), for 2270 and 1470 minutes. The Wan block is 190 minutes on either GPU. On sm_89 or newer the FP8 baseline uses native FP8 tensor cores. On an A100, FP8 is emulated and weight-only, and the comparison records that speed comparisons vs FP8 are not representative. The pinned nunchaku wheel lists sm_75, sm_80, sm_86, sm_89, and sm_120a, so INT4 eval on an H100 may fail to launch; quant files can still be published, and sources stay until a mirror is verified. On the live GPU, identity, hands, and text stay unset, so the winner is the fastest arm that fits 16 GB (smi peak at most 15.0 GiB), otherwise 24 GB (23.0 GiB). The dry-run scores are synthetic and are not that measurement.

Proven here: the CPU one-block chain (calib, fast INT4 rank 8, convert, Nunchaku key shapes), the issue 116 scale broadcast, and a QuantFunc r128 header check that plans the missing `img_mod` `smooth_factor` keys. Unproven here: loading QuantFunc r128 or r32 through the shim on the Linux wheel, nunchaku kernel numerics on that wheel, and the 12–18 h full quant. The planned full quant remains 900 minutes on an A100 and 500 minutes on an H100.

## Windows laptop smoke

`scripts/smoke_laptop.ps1` is the unattended check for a wiped Windows 11 machine with an RTX 4090 Laptop (16 GB, sm_89). It does not prompt and it does not read a token. It installs uv into the user profile, builds a Python 3.12 venv under `%LOCALAPPDATA%\VoxprintForgeSmoke`, and installs torch 2.11 from the cu130 index plus the pinned Windows vitoom wheel `nunchaku-1.3.0.dev20260629` (cp312, cu130, sha256 `ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955`) and `hf_xet` / `hf_transfer`. That wheel is not the Linux wheel `dev20260622` (cp311, cu128) used on the H100. A pass proves the QuantFunc smooth shim and the nunchaku kernels on Windows sm_89. It does not prove the Linux wheel. Weights are prefetched with resume and retries before the 90 minute compute cap starts. A download failure exits 2 and is labelled `setup: download failed`. The text encoder and VAE stay off the GPU until the transformer file is complete. `-Force` reruns steps that already passed. Without it, a rerun reuses passed steps and keeps the weight cache after a failure so the download can resume. A successful run deletes the cache. `calib-quant-timing` times 2 calibration samples and a syh INT4 quant at SVD rank 256 on the first N blocks (default 2). If deepcompressor cannot use CUDA, that step records a CPU fallback. It writes `extrapolation` for 60 blocks on this GPU and a scaled H100 estimate.

Before the long quant, run a Linux preflight on the H100 itself. It is about five minutes once the weights are on disk. A non-zero exit means do not start the 12–18 h quant:

```bash
export HF_TOKEN=...
bash scripts/vm_run_diffusion_int4.sh --preflight
```

`--preflight` runs smoke-tiny and a QuantFunc r128 `from_pretrained` through the shim, prints `PREFLIGHT-OK`, and exits. The long quant does not start in that process.

Fetch the laptop script from the tag and launch it with no console. A dropped remote session leaves the log and `result.json` in place:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-model-forge/diffusion-int4-v4-smoke2/scripts/smoke_laptop.ps1 -OutFile "$env:TEMP\smoke_laptop.ps1"
Start-Process pwsh -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$env:TEMP\smoke_laptop.ps1" -WindowStyle Hidden
```

Results are `%LOCALAPPDATA%\VoxprintForgeSmoke\results\<timestamp>\` (`result.json` schema `voxprint-forge-smoke-laptop/1`, WebP images, `smoke.log`). The path is printed and written to `%LOCALAPPDATA%\VoxprintForgeSmoke\latest.txt`. `result.json` records the nunchaku wheel version and whether the Qwen, `zero_cond_t`, and `smooth_factor` paths imported. `-Steps` selects `smoke-tiny`, `quantfunc`, `zimage`, and `calib-quant-timing`. `-Blocks` defaults to 2. The compute cap is 90 minutes and does not include the download. `-KeepVenv` keeps the venv. Exit 0 is every selected step passed, 1 is a failed or skipped step, 2 is setup, and 3 is no suitable GPU.
