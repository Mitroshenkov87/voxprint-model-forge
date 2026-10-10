# A100 80GB VM

Ubuntu 24.04 on immers.cloud: 1x A100 80GB, 16 vCPU, 128 GB RAM, 480 GB NVMe. The rental is about 3 hours. Model work is budgeted at 150 minutes inside that window. The separate RTX 4090 is for app tests later, so fit on this box is computed weights + KV plus a llama-server measurement on the A100. `min_vram_gb` in the registry is 16 or 24 from that measurement.

RAM: the resident ablation path wants the bf16 weights plus about 6 GB. T-pro bf16 is about 61 GiB, so 128 GB is the minimum this plan is written for. Disk: keep at least a few tens of GiB free after the F16 source, the quants, and the reused GGUF. Bootstrap stops with exit 7 when the next stage would not fit, and it leaves finished quants in place.

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
tmux new -s forge 'bash scripts/bootstrap.sh'
```

`HF_USER` defaults to `Mitroshenkov87`. Pass the tokens in the environment over SSH. Do not write them to disk. If `HF_TOKEN` or `KAGGLE_API_TOKEN` is unset and stdin is not a terminal, bootstrap exits 2 and does not prompt. A real interactive stdin still prompts once with `read -s` on `/dev/tty`. That value stays in the process only. `set +x` stays in force around the prompt. Logs go through `scripts/logscrub.py`. On EXIT, INT, and TERM the script deletes `~/.cache/huggingface/token`, `~/.cache/huggingface/stored_tokens`, `~/.kaggle/`, and any `kaggle.json` or `access_token` file it finds, then unsets the two token variables. It does not call `hf auth login` or `huggingface-cli login`.

Re-attach: `tmux attach -t forge`.

If the session is gone, export the tokens again and re-run `tmux new -s forge 'bash scripts/bootstrap.sh'`. Stage markers in `~/forge-out/stages` skip finished work, including uploads whose remote sha256 and size already matched.

Progress lines (timestamp, model, stage, percent, ETA) append to `~/forge-out/progress.log`. `tail -f ~/forge-out/progress.log` while the session runs.

When the run finishes, copy the manifest back:

```bash
scp <user>@<vm>:~/forge-out/manifests/<date>-a100.json .
scp <user>@<vm>:~/forge-out/manifests/<date>-a100.MODELS.md .
```

Bootstrap prints those two paths. There is no push from the VM.

## What bootstrap does

1. Print the forge commit from `VERSION`. Refuse anything other than Ubuntu 24.04.
2. Require `HF_TOKEN` and `KAGGLE_API_TOKEN` (prompt only when stdin is a terminal), default `HF_USER` to `Mitroshenkov87`, then check `nvidia-smi`. Preflight then requires `cmake`, `ninja`, and `nvcc` on `PATH`, and that `src/voxprint_forge/data/imatrix.txt` tokenises to at least `max(2×ctx, ctx×chunks)` tokens.
3. Print the plan estimates with system Python (no torch import).
4. Install uv, Python 3.14, and `torch==2.14.1` from the cu130 index. If that wheel is missing it retries Python 3.13, then 3.12.
5. `forge verify-logins`. A failed Hub or Kaggle login stops the script before the llama.cpp build and before any download.
6. Build the bundled llama.cpp with `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=80;89;120`.
7. `scripts/vm_run.sh` runs `forge --plan plans/plan.toml --root <repo> --forge-out ~/forge-out run-all`. Global flags precede the subcommand.
8. Tar `~/forge-out/forge-results-<date>.tar` plus a sha256 file, and print the manifest paths to copy back.

Copy results off the box with `scp` or `rsync` as a second copy. The private Hub and Kaggle uploads are the primary copy.

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

`src/voxprint_forge/data/imatrix.txt` is bartowski's `calibration_datav3` (public gist, no explicit licence).

- Source: https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8
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

F16 and safetensors intermediates are deleted per model after the quants and the imatrix exist. Selected GGUFs, the imatrix, the card, and the sha256 files stay until both private uploads verify.

## Private upload

`artifact_host` in the plan is empty. There is no public publish.

Right after each model, the `upload` stage pushes the finished GGUFs, the imatrix, the model card, and the sha256 sidecars to:

- a private Hugging Face model repo under `HF_USER`
- a private Kaggle model owned by `KAGGLE_USER`, one variation per tier, framework `gguf`

Logins are verified before the build. A remote sha256 or size mismatch retries with backoff and does not mark the stage done. The summary lists each file with its sha256 and both URLs. `~/forge-out/manifests/<date>-a100.json` and the MODELS snippet are the failover list: Hugging Face, Kaggle, and a torrent placeholder. Copy them back; the VM does not push.

The card template is `templates/model_card.md.jinja`. It starts with Hub YAML (`license`, `base_model`, `base_model_relation: quantized`, tags, `language`, `pipeline_tag`). `base_model` is the abliterated upstream for Huihui and Gemma, and the T-Bank repo for T-lite and T-pro. The body is the base and licence, method, calibration data and licences, quant types, evals, limits, and credits, then a short "Made for Voxprint AI Media Suite" section that links Plotweaver and records the 16 GB and 24 GB tiers with `min_vram_gb`. The Apache-2.0 text and a modification notice sit above that section.
