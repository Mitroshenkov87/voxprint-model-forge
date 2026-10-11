# Diffusion INT4

`forge diffusion-int4` is the Phase 1 stage for an A100 or H100 80GB rental. It wraps pinned tools. It does not reimplement SVDQuant.

## Pins

Checked 2026-10-11.

| Tool | Commit | Licence |
| --- | --- | --- |
| [lantudou/deepcompressor2](https://github.com/lantudou/deepcompressor2) branch `syh` | `019cbdeae7e2add52eaf4c27a361cfc1e57993b8` | Apache-2.0. Fork of nunchux-ai/deepcompressor (the nunchaku-ai/deepcompressor redirect). Copyright 2024 Yujun Lin, Muyang Li, Zhekai Zhang, Haotian Tang, Shang Yang, Song Han. |
| [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | Apache-2.0 |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tag `b3962` | `c8c07d658a6cefc5a50cfdf6be7d726503612303` | MIT |
| [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | `1ea34ff48f87168174e12956e200b1d908b1c5ff` | Apache-2.0 |
| Wan-AI/Wan2.2-T2V-A14B | `c8c270b13ee05bfa474194ac9fb07a5868a97cea` | Apache-2.0 |
| Qwen/Qwen-Image-Edit-2511 | `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` | Apache-2.0 |
| Qwen/Qwen-Image | `75e0b4be04f60ec59a75f475837eced720f823b6` | Apache-2.0 |
| QuantFunc/Nunchaku-Qwen-Image-EDIT-2511 | `4ca543a1e91db4183ee16fd5aa5950e0b02803e5` | Apache-2.0 |
| tonera/vitoom-nunchaku | `85bad5510d409570527b018387491fd0c81e4727` | Apache-2.0 wheel `nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311-cp311-linux_x86_64.whl` |

nunchaku-ai/deepcompressor issue 104 is still open. The nunchaku team has not published a Qwen-Image quant tree. `69f3473f5e1c1504bae35cc50c7858ef900a9b17` has no `DiffusionModelStruct` factory for `QwenImageTransformer2DModel` and raises `KeyError`. `syh` is six commits ahead of that base and registers the Qwen block. The stage applies an idempotent patch: image-edit prompts, `DiffusionPipeline.from_pretrained` for names that start with `qwen-image`, `register_to_config(zero_cond_t=True)` for names that start with `qwen-image-edit`, the calibration image, and a broadcast of a `[1, 1, 1, 1]` W4 scale before the `scale.shape[0] == oc` assert (nunchaku-ai/deepcompressor issue 116). A checkout whose text does not match the anchors exits 6 (`patch_anchor`). `zero_cond_t` is true for Qwen-Image-Edit-2511 and false for Qwen-Image. Edit inference uses the vitoom wheel, in a Python 3.11 venv, because upstream Nunchaku does not apply that flag.

QuantFunc INT4 r128 and r32 store `img_mod` / `txt_mod` as TinyChat W4A16 (int32 `qweight`, `wscales`, `wzeros`) and do not store `smooth_factor`. The pinned Linux wheel `dev20260622` and the Windows wheel `dev20260629` are the same Qwen source and both use `SVDQW8A16Linear` for those mods, so `patch_scale_key` raises on the missing smooth keys. There is no Linux cp312 cu128 wheel and no Linux wheel dated 20260629. Rebuilding the same vitoom source would hit the same check. Z-Image INT4 r128 loads on this wheel because it does not use that module. The forge wraps `patch_scale_key` for the Qwen family: it inserts identity `smooth_factor` and unpacks the W4 matrix to a float `.weight` so the wheel's existing CUDA requant path can run. That requant has not been executed on an A100 in this change.

`forge diffusion-int4 smoke-tiny` and `scripts/vm_run_diffusion_int4.sh --smoke-tiny` run the one-block chain (calibration from `data/diffusion_calib`, fast INT4 rank 8, convert, one transformer latent) and exit before the token prompt and the clone. On CUDA, with the vitoom wheel importable, that command also loads the tiny checkpoint through the shim and runs one forward. The latent is the image. VAE decode is not part of this command. The packed low-rank tile is 16, so rank 8 is stored as 16 with the extra singular values zero. The A100 quant of the full 60-block model stays 900 minutes (12–18 h). The H100 quant stays 500 minutes.

## Calibration

`data/diffusion_calib/` holds original Russian and English edit prompts (people, text, hands) and 16×16 synthetic swatches. Licences are in `data/diffusion_calib/LICENCES.md`. `FORGE_CALIB_IMAGE_DIR` can replace the swatch root. `FORGE_PRIVATE_CALIB` may be a directory or a private Hugging Face dataset id. When it is unset or cannot be loaded, that calibration set is skipped. The English identity, hands, and text lines are the canvas hard cases. The winner gate uses those three. The Russian lines are calibration.

## Winner

When every arm has identity, hands, and text: identity within 0.05 of FP8, text exact, hands no worse than FP8. A 16 GB fit is an nvidia-smi peak of at most 15.0 GiB. A 24 GB fit is at most 23.0 GiB. A passing 16 GB arm wins over a passing 24 GB arm.

On the live GPU those three scores stay unset, because this stage times the pipeline and does not run an identity or OCR model. The winner is then the fastest arm that fits 16 GB, otherwise 24 GB. No arm inside either cap exits 6 (`no_winner`) and leaves the sources in place. The published file is the local export for an `own-*` winner, the QuantFunc safetensors for `quantfunc-r128` or `quantfunc-r256`, or the FP8 tree for `fp8`. Dry-run scores are synthetic. They pick `own-r256` so the export path is what CI uploads to a local mirror.

## Own quant

`--own-edit` defaults to `on`. That is the diffusion-int4-v1 plan: calibrate, quantize Qwen-Image-Edit-2511 INT4 r256, export, evaluate, publish, then Wan 2.2. Wan runs unless `FORGE_SKIP_WAN=1`, which omits those steps (190 minutes).

`--phase0-result` points at a Phase 0 `result.json` (the rental default is `/workspace/vm-results/model-forge/nunchaku-test-1/result.json`). A missing file is ignored. A present file is copied into `comparison.json` as `phase0_baseline` and onto the card. It records QuantFunc `qwen-r128` and `qwen-r32` cap16 (`fits_16gb`, c1, c2, c3). It does not choose the winner. Invalid JSON exits 6 (`phase0`).

`--own-edit auto` is opt-in. It skips the own quant only when `qwen-r128` cap16 finished c1, c2, and c3 and `fits_16gb` is true. The stage then mirrors `nunchaku_qwen_image_edit_2511_balance_int4.safetensors` and `nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors` to `HF_USER/voxprint-qwen-image-edit-2511-quantfunc-int4` and goes to Wan. The card says these files are mirrored QuantFunc weights. `--own-edit off` mirrors without reading the gate. `status.json` stores `own_edit` and `phase0_baseline`. A missing Phase 0 file under `auto` keeps the own quant.

## Publish

Public Hugging Face repos are `HF_USER/voxprint-<target>-svdquant-int4` and `HF_USER/voxprint-wan2.2-t2v-a14b` (`HF_USER` defaults to `Mitroshenkov87`). The card is `templates/canvas_model_card.md.jinja` with autoescape on. The licence gate is the same one as the rest of the forge: Apache-2.0, MIT, or another permissive tag, and a card with `## License`, `## Credits`, and `## Modification notice`. A refusal exits 3 and is not sent to Drive. Drive is the fallback and a backup copy after a Hub success. The folder id is `13qzgTnnkqvZrf_DK9fmymJ8ICdOuwDJT` (`FORGE_DRIVE_ROOT_FOLDER_ID` for that call). One verified mirror is enough. Sources are deleted only after that match.

Wan publishes Q4_K_M and Q5_K_M for the high-noise and low-noise experts, plus a voxprint-fp8-v1 file per expert. The smoke test is the pinned `generate.py`: 5 frames, `832*480`, 4 steps, offload, on the bf16 checkpoint, before those sources are deleted.

## Clock

The H100 quant step is planned at 1.8× the A100 step (900 minutes becomes 500). Other steps stay the same. `forge diffusion-int4 estimates` prints both columns. The run follows the H100 column when `FORGE_GPU=h100` or `FORGE_COMPUTE_CAP=9.0`. With neither set, the A100 column is the plan.

| Step | A100 minutes | H100 minutes |
| --- | --- | --- |
| `qwen-edit-2511-calib` | 30 | 30 |
| `qwen-edit-2511-quant` | 900 | 500 |
| `qwen-edit-2511-export` | 20 | 20 |
| `qwen-edit-2511-eval` | 90 | 90 |
| `qwen-edit-2511-publish` | 25 | 25 |
| `qwen-edit-2511-reclaim` | 5 | 5 |
| `wan-2.2-gguf` | 100 | 100 |
| `wan-2.2-fp8` | 40 | 40 |
| `wan-2.2-smoke` | 20 | 20 |
| `wan-2.2-publish` | 25 | 25 |
| `wan-2.2-reclaim` | 5 | 5 |

Required total: 1260 minutes on an A100, 860 minutes on an H100. Optional `qwen-image-*` adds quant 900 (A100) or 500 (H100), export 20, eval 60, publish 25, reclaim 5. Optional total: 2270 minutes on an A100, 1470 minutes on an H100.

`--own-edit auto` or `off`, when the own quant is skipped, replaces the six edit steps with `qwen-edit-2511-mirror` (25) and `qwen-edit-2511-reclaim` (5). That required total is 220 minutes on either GPU. Optional Qwen-Image still adds 1010 minutes on an A100 and 610 on an H100.

## GPU

`sm_80` is an A100. `sm_90` is an H100. The script reads `nvidia-smi` and exports `FORGE_GPU` (`a100`, `h100`, or `other`) and `FORGE_COMPUTE_CAP`. A capability other than 9.0 keeps the A100 minute table.

FP8 tensor cores start at sm_89. On sm_89 or newer the FP8 baseline keeps float8 weights and runs them with native FP8 tensor cores. On an A100 the same baseline is emulated and weight-only, and `comparison.json` records that speed comparisons vs FP8 are not representative. A run that cannot read a compute capability records that same caveat and times the baseline as weight-only BF16.

Torch 2.11 and torchvision 0.26 come from the cu128 wheel index, which includes sm_80 and sm_90. `llama-quantize` is built with `CMAKE_CUDA_ARCHITECTURES=80;90`. The pinned vitoom-nunchaku wheel lists sm_75, sm_80, sm_86, sm_89, and sm_120a. On an H100 the INT4 eval may fail to launch. Quantization uses deepcompressor. If eval fails, sources stay until a mirror is verified.

## Proven on this change

- CPU, one random-init block, `zero_cond_t` true: two calibration swatches, fast INT4 group 64 rank 8, convert, and the Nunchaku key/shape contract. Modulation is W4A16 (`qweight`, `wscales`, `wzeros`, `bias`) with no `smooth_factor`. The other linears are W4A4 with `smooth_factor`, `proj_down`, and `proj_up`. Packed rank is 16.
- Issue 116: a scale of shape `[1, 1, 1, 1]` converts for an output width divisible by 6 and does not trip the `oc` assert.
- QuantFunc r128 header, HTTP range only: `transformer_blocks.0.img_mod.1` has no `smooth_factor`, attention `smooth_factor` is `[3072]`, `qweight` is int32 `[4608, 1536]`, and `wscales` is `[48, 18432]`. The shim plans `smooth_factor` `[3072]`.
- A synthetic TinyChat W4 roundtrip unpacks to the post-adanorm float. The preflight replica does not raise after the shim adds `.weight` and identity smooth.

## Not proven on this change

- `from_pretrained` of QuantFunc r128 or r32 on the Linux wheel after the shim, including the CUDA requant inside `SVDQW8A16Linear`.
- Nunchaku CUDA kernel output matching the float block.
- Wall clock of `--smoke-tiny` on an RTX 4090, RTX 5090, or A10. The CPU run is one block and is the proof that exists here.
- The 12–18 h A100 quant (900 minutes planned; 500 minutes on an H100) and the quality of that export.
- VAE decode of the smoke-tiny latent.

## Script

`scripts/vm_run_diffusion_int4.sh --preflight` is the short H100 gate: smoke-tiny, then QuantFunc r128 `from_pretrained` through the shim. It prints `PREFLIGHT-OK` and exits. A failure aborts before the long quant. The Windows vitoom wheel is a different build, so `scripts/smoke_laptop.ps1` on an RTX 4090 does not prove this Linux wheel. `scripts/vm_run_diffusion_int4.sh` is the non-interactive entry for a fresh Ubuntu A100 or H100 80GB. `FORGE_HOST_KIND=local` (the default) does not write token files. `FORGE_HOST_KIND=rented-vm` logs in once and leaves the logins. `--dry-run` is what CI runs. Exit codes match `docs/CLI.md`, with the 15 percent floor as exit 2. `FORGE_SKIP_WAN=1` skips the Wan clones, venv, and weight download. `FORGE_WAIT_FOR` is a file path; the live script sleeps until that file exists before the GPU stages. `--dry-run` does not wait. The cp311 venv installs deepcompressor with `--no-deps` plus its diffusion imports (`datasets>=3.0,<4`, `transformers==4.57.6`). voxprint-forge stays on `PYTHONPATH`. `FORGE_COMMON_SRC`, when set, is the directory that contains `voxprint_common`. A checkout whose `PIN` file and `HEAD` already match the pin skips `git fetch`. Generated deepcompressor configs go under `examples/diffusion/configs/voxprint/` when `FORGE_DEEPCOMPRESSOR` is set.
