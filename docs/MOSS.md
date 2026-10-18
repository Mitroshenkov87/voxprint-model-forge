# MOSS-Image

MOSS-Image (OpenMOSS, Apache-2.0) is a 5B single-stream DiT in the FLUX.2 latent space. It uses a Qwen3.5-9B text encoder and the FLUX.2 small VAE decoder.

## Sources

| Row | Author (primary) | Our backup mirror |
|---|---|---|
| `moss-image-base` | `OpenMOSS-Team/MOSS-Image-Base` @ `1ceede80` | `Mitroshenkov87/voxprint-mirror-moss-image-base` @ `85cc5ca1` |
| `moss-image-sft` | `OpenMOSS-Team/MOSS-Image-SFT` @ `fc5de0d5` | `Mitroshenkov87/voxprint-mirror-moss-image-sft` @ `2d0da610` |

Each repo holds 37 files, 27.23 GB. The mirrors are byte-identical to the author's files at those commits. The only difference is `README.md`, which is the mirror card. `text_encoder/`, `tokenizer/` and `vae/` are identical in Base and SFT.

**Licence check (2026-10-17).** Both `LICENSE` files are byte-identical to the canonical Apache-2.0 text. The bundled third-party parts are also Apache-2.0: Qwen3.5-9B (`Qwen/Qwen3.5-9B`) and the FLUX.2 small decoder (`black-forest-labs/FLUX.2-small-decoder`).

**Third-party repacks.** `Comfy-Org/MOSS-Image` (INT8 ConvRot, W4A8 GPTQ) and `lodestones/MOSS-Image-SFT-8-steps-dmd` (a work-in-progress DMD distill) are derivatives. They are never primary sources. The Comfy INT8 files are used only as the fit job's speed and VRAM reference.

## Loading

The pipeline file is vendored verbatim at `src/voxprint_forge/vendored/moss_image/`; see `VENDOR.md` there for the review. `voxprint_forge.diffusion.moss.load_pipeline(snapshot)` builds each component from a local snapshot:

- offline (`local_files_only=True`, `HF_HUB_OFFLINE=1`);
- safetensors only;
- `trust_remote_code=False`;
- no `custom_pipeline`.

It needs diffusers>=0.38 and transformers>=5.3. The app venv has diffusers 0.41 and transformers 5.3 or newer; the fit job keeps its own venv (the `moss` extra).

## LoRA

`moss-image-base` trains with the LoRA trainer (`voxprint-forge-cli lora train --base moss-image-base`). The DiT stays bf16 and the LoRA stays fp32. 4-bit and FP8 storage break the MOSS forward (it casts activations to `weight.dtype`), so both memory presets keep the bf16 weights. The text encoder stays on the CPU; the transformer moves onto the card only after the VAE latents and prompt embeddings are cached.

A `lora export` safetensors file loads at generation time. The adapter is merged into the transformer linears (`W + scale * (alpha / rank) * B @ A`):

```bash
python -m voxprint_forge.diffusion.moss --snapshot SNAPSHOT --prompt "a red cube" --out out.png --lora export.safetensors --lora-scale 1
```

`load_pipeline(..., lora_path=..., lora_scale=1.0)`, `generate(...)` and `diffusion.generate.generate_moss(...)` take the same arguments.

## Fit job (synth4)

`plans/moss-synth4.toml` and `scripts/vm_run_moss_fit.sh` run the SFT model at 1024², 50 steps, CFG 3, with memory caps of 16 and 24 GB:

| Arm | Role | Notes |
|---|---|---|
| `bf16-te-offload` | candidate | 16 GB: text encoder on CPU. 24 GB: diffusers model CPU offload. |
| `c-dq` | candidate | FP8 storage, BF16 compute. Skipped until `FORGE_MOSS_FP8_DIR` points at built weights. |
| `comfy-int8` | reference | ComfyUI `a639f187` (#17095 + #17099), comfy-kitchen 0.2.38, files sha256-pinned. |

The caps are emulated: `set_per_process_memory_fraction` for diffusers, `--reserve-vram` for ComfyUI. A real 16/24 GB card check follows separately.

Dry run, with no downloads and no GPU:

```bash
python -m voxprint_forge.diffusion.moss_preflight --plan plans/moss-synth4.toml
python -m voxprint_forge.diffusion.moss_fit --plan plans/moss-synth4.toml --dry-run
scripts/vm_run_moss_fit.sh --preflight-only   # on the VM: adds the CUDA, runtime and disk checks
```

### 16 GB fit

The #82 rerun put `bf16-te-offload` at 16 GB at 14.30 GB allocated and 16.36 GB reserved (30.5 s/img), with a
16 GiB cap. A 16 GB card also holds the CUDA context outside torch's pool, so the 16 GB cells now target
`reserved_target_gb = 15.2` (decimal GB of peak reserved).

- Budget: transformer bf16 10.17 GB resident, VAE fp32 0.25 GB, text encoder on CPU (it never reaches the GPU).
  The untiled fp32 FLUX.2 decode at 1024x1024 needs about 3.2 GB more (measured on CPU, `decode_latents`), so the
  decode is the peak phase (denoise activations at 4096+T tokens are estimated under 1 GB; the memprobe measures both).
- The 2.06 GB gap between allocated and reserved is allocator fragmentation: denoise-sized blocks stay cached
  and the decode's larger blocks need new segments. `alloc_conf = "expandable_segments:True"` (plan default,
  set before CUDA starts) and the split decode (`output_type="latent"`, `empty_cache`, then decode) remove it.
- Tiled VAE decode is not used: 512 px tiles cut the decode to 0.94 GB, but give 23.3 dB PSNR against the untiled
  decode (a global colour shift; the VAE mid-block attention and group norms see one tile). Sequential offload
  is not needed. `decode_evict = { "16" = true }` (transformer to CPU for the decode) is the fallback.
- `python -m voxprint_forge.diffusion.moss_fit --memprobe --arm bf16-te-offload --cap 16 ...` runs one prompt for
  `memprobe_steps` steps with the target as a hard cap and writes per-phase peaks, the CUDA context size and the
  evicted-decode figure; `--baseline` repeats the #82 settings. `scripts/vm_run_moss_fit.sh` runs both first and
  stops if the fit probe fails; `--te-offload-only` then runs only the `bf16-te-offload` arm.
