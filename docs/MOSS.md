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

It needs diffusers>=0.38 and transformers>=5.3, which is the `moss` extra. The forge venv still pins diffusers<0.38, so the fit job uses its own venv.

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
