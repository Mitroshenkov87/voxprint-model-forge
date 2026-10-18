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

## synth4 setup (`scripts/vm_setup_synth4.sh`)

Needs `cmake`, `ninja` (`ninja-build`) and the CUDA toolkit. The script stops with exit 9 and names what is missing before it downloads anything. Set `CUDACXX` to the `nvcc` path when it is not `/usr/local/cuda/bin/nvcc` or on `PATH`. Install is two steps: torch and torchvision from the PyTorch CUDA index, then the `compare` extra from PyPI with torch pinned by a constraint file. Do not add `--extra-index-url` to the second step; uv then resolves index-first and installs torchmetrics 1.0.3. llama.cpp is configured with `-G Ninja`. `scripts/bootstrap.sh` does the same: apt installs `cmake`, `ninja-build` and `build-essential` when missing, puts `/usr/local/cuda/bin` on `PATH`, exports `CUDACXX`, installs torch from cu130 and the rest from PyPI. An older Makefile build tree is reset so it can switch to Ninja.

### Check the image first: `forge preflight --vm`

Run it on a fresh image before any paid GPU job, after `scripts/vm_setup_synth4.sh`:

```
FORGE_OUT=~/forge-out ~/forge-out/venv/bin/python -m voxprint_forge.cli --root ~/voxprint-model-forge preflight --vm
```

For a LoRA-only VM add `--for lora`: it skips `cmake`, `ninja`, `nvcc`, the C++ compiler, `third_party/llama.cpp` and the converter imports, which only the GGUF build needs.

It prints one `PASS` or `FAIL` line per check, with a fix hint on each FAIL, and exits 6 when anything fails (`--json` for one object). Checks: Python 3.12 or newer; torch imports, is a CUDA build, sees the GPU, and the driver is new enough for that CUDA (580 for cu130, 570 for cu128); every base and `[compare]` requirement installed at a matching version (for example `torchmetrics>=1.6`); `cmake`, `ninja`, `nvcc` (`CUDACXX`, else `/usr/local/cuda/bin/nvcc`, else `PATH`) and a C++ compiler; `third_party/llama.cpp` present at `scripts/llama.rev`; every module `convert_hf_to_gguf.py` imports at that pin (`numpy`, `torch`, `safetensors`, `transformers`, `sentencepiece` and its protobuf model (protobuf pinned to 5.29.6; 4.25.9 crashes on Python 3.14), `yaml`, `tqdm`, `requests`) imports in the forge venv, because some load only inside `set_vocab` after the BF16 download; the vendored voxprint_common wheel for the version pyproject.toml pins (0.3.1) present and installed; `HF_TOKEN` set (the value is never printed); free disk at `FORGE_OUT` (`--min-free-gb`, default 150). Nothing is installed or downloaded.

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

## Wan 2.2 GGUF and FP8

`bash scripts/vm_run_diffusion_int4.sh` is the rental entry. It clones Wan2.2 at `1ea34ff48f87168174e12956e200b1d908b1c5ff`, city96/ComfyUI-GGUF at `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`, and llama.cpp at `c8c07d658a6cefc5a50cfdf6be7d726503612303`, applies `ComfyUI-GGUF/tools/lcpp.patch`, and builds `llama-quantize` for architectures `80;90`. It creates `venv-wan` and exports `FORGE_WAN_REPO`, `FORGE_WAN_PYTHON`, `FORGE_COMFY_GGUF`, `FORGE_LLAMA_SRC`, and `FORGE_LLAMA_BIN`. `forge diffusion-int4` then converts Wan2.2-T2V-A14B to GGUF and FP8. Qwen-Image-Edit INT4 is removed.

`FORGE_PYTHON` is the forge interpreter. When it is unset, the live run creates `$FORGE_OUT/venv-forge` with Python 3.14 before the first disk-floor check and exports that interpreter as `FORGE_PYTHON`. The venv installs torch from the cu130 index when `nvidia-smi` reports driver 580 or newer, and from the cu128 index otherwise, then `safetensors`, `huggingface_hub>=0.26`, `gguf`, `jinja2>=3.1`, and an editable install of this checkout. A preset `FORGE_PYTHON` is used as-is and the venv is not created. `--dry-run` does not create it.

`--dry-run` is a CPU smoke with tiny files. It does not download weights, clone those trees, or start CUDA. `FORGE_SKIP_WAN=1` omits the Wan clones and steps. `forge diffusion-int4 fetch --dest DIR` downloads the Wan experts after the Apache-2.0 licence check. `forge diffusion-int4 estimates` prints 190 minutes on both the A100 and the H100.

## Z-Image INT4

The Windows RTX 4090 laptop (sm_89, 16 GB) is the fit target. Nunchaku ships sm_89 kernels. INT4 covers compute capability 8.0 and newer. The Linux cu128 wheel needs driver 525.60.13 or newer. FP8 timings below compute capability 8.9 are not representative of RTX 40/50.

`scripts/smoke_laptop.ps1` runs the `zimage` step (Z-Image-Turbo INT4 r128, 16 GB peak). `scripts/nunchaku_zimage_test.sh` and `scripts/nunchaku_zimage_test.ps1` run the matrix tagged `nunchaku-test-2`. Default arm is `zimage-r128`. `zimage-r256` is optional. `cap16` is the 16 GB budget.

| Step | Minutes |
| --- | --- |
| Wan GGUF Q4_K_M and Q5_K_M, both experts | 100 |
| Wan FP8 | 40 |
| Wan smoke, 5 frames at 832×480, 4 steps | 20 |
| Wan card, sha256, upload | 25 |
| Wan source delete | 5 |

Free space under 15 percent exits 2 (`error.code` `disk_floor`) after verified sources are deleted. Sources are deleted only after a mirror sha256 matches. `~/watch.json` lists the jobs. `~/forge-out/status.json` is the heartbeat file.
