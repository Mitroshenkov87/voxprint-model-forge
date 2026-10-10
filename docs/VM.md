# A100 80GB VM

Fresh Ubuntu on immers.cloud: 1x A100 80GB, 16 vCPU, 128 GB RAM, 480 GB NVMe. The rental is about 3 hours. Model work is budgeted at 150 minutes inside that window. The separate RTX 4090 is for app tests later, so fit on this box is computed weights + KV plus a llama-server measurement on the A100. `min_vram_gb` in the registry is 16 or 24 from that measurement.

RAM: the resident ablation path wants the bf16 weights plus about 6 GB. T-pro bf16 is about 61 GiB, so 128 GB is the minimum this plan is written for. Disk: keep at least a few tens of GiB free after the F16 source, the quants, and the reused GGUF. Bootstrap stops with exit 7 when the next stage would not fit, and it leaves finished quants in place.

## Command

Set the tokens in the environment. The script prompts once on `/dev/tty` with a silent read if `HF_TOKEN` or `KAGGLE_API_TOKEN` is missing. It never echoes them, never writes them to disk, and scrubs them from `~/forge-out/logs`.

```bash
export GH_TOKEN=...
export HF_TOKEN=...
export HF_USER=...
export KAGGLE_API_TOKEN=...
export KAGGLE_USER=mitroshenkov87

curl -fsSL -H "Authorization: Bearer $GH_TOKEN" \
  https://raw.githubusercontent.com/Mitroshenkov87/voxprint-model-forge/main/scripts/bootstrap.sh | bash
```

From a checkout:

```bash
git -c "http.extraHeader=Authorization: Bearer $GH_TOKEN" \
  clone https://github.com/Mitroshenkov87/voxprint-model-forge.git
cd voxprint-model-forge
bash scripts/bootstrap.sh
```

`GH_TOKEN` is only required for the private clone and, if you want it, the manifest push. Do not put the token in the remote URL or in `git config`.

## What bootstrap does

1. Prompt for missing tokens, then check `nvidia-smi`.
2. Print the plan estimates with system Python (no torch import).
3. Install uv, Python 3.14, and `torch==2.14.1` from the cu130 index. If that wheel is missing it retries Python 3.13, then 3.12.
4. `forge verify-logins`. A failed Hub or Kaggle login stops the script before the llama.cpp build and before any download.
5. Build llama.cpp `b11476` (`988190680d5a89fce97de3c20df2c2813731fd61`) with `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=80;89;120`.
6. `scripts/vm_run.sh` runs `forge run-all`.
7. Tar `~/forge-out/forge-results-<date>.tar` plus a sha256 file.

Stage markers live in `~/forge-out/stages`. Re-running the script skips finished stages.

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

Rough split of the model minutes: Huihui downloads the reused Q5 (and Q4 only if Q5 misses the cap) plus the F16 source, an 8-chunk imatrix, three 16 GB quants, then a light measure and eval. Gemma is the same shape with a larger F16 and only the Q4 reuse. T-lite downloads about 15 GiB, ablates on the GPU, converts to F16, builds the imatrix, and quantizes Q5, Q4, IQ4_XS, and IQ3_M. T-pro is the same work at about 61 GiB of bf16.

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

Logins are verified before the build. A remote sha256 or size mismatch retries with backoff and does not mark the stage done. The summary lists each file with its sha256 and both URLs. The manifest committed back to this repo (`manifests/<date>-a100.json` and the MODELS snippet) is the failover list: Hugging Face, Kaggle, and a torrent placeholder.

The card template is `templates/model_card.md.jinja`. It carries the base revision and licence, the ablation or requant note, calibration licences, quant and imatrix data, Plotweaver purpose, eval numbers, limits (reduced refusals, not for safety-critical use, Russian quality is a sample), credits (base authors, T-Bank and Qwen for T-lite and T-pro, abliteration authors for Huihui and Gemma, llama.cpp), and the Apache-2.0 text plus a modification notice.
