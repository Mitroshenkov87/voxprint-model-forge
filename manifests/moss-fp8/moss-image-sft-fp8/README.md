---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-Image-SFT
tags:
- voxprint
- quantized
- fp8
---

# MOSS-Image SFT DiT FP8

This is a **modified** Apache-2.0 derivative of `OpenMOSS-Team/MOSS-Image-SFT` at revision `fc5de0d56fc910cbca6f6625a70f5f582b26deb1`.
The bf16 backup mirror is `Mitroshenkov87/voxprint-mirror-moss-image-sft` at `2d0da610484827b96bdb81c44d702b5cf79a0689`.
LICENSE and NOTICE are included. The NOTICE states the modification.

The DiT weights are stored as FP8 e4m3 (`float8_e4m3fn`) with per-row scales.
torchao `0.18.0`, `Float8WeightOnlyConfig(weight_dtype=float8_e4m3fn, granularity=PerRow(), version=2, set_inductor_config=False)`.
Quant id: `float8_e4m3fn,per_row,v2`. Format: `voxprint-moss-fp8-v1`.
Activations and the matmul stay in bf16. `Float8Tensor.dtype` reports bf16, which is what the MOSS block reads before it casts activations.

## Source files

| Path | Size | sha256 |
| --- | --- | --- |
| `transformer/config.json` | 240 | `3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666` |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | 40298 | `b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03` |
| `transformer/diffusion_pytorch_model-00001-of-00003.safetensors` | 4991978016 | `e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01` |
| `transformer/diffusion_pytorch_model-00002-of-00003.safetensors` | 4983137016 | `d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033` |
| `transformer/diffusion_pytorch_model-00003-of-00003.safetensors` | 189549184 | `11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d` |

## Quantized file

`dit-fp8.safetensors` sha256: pending (mirror revision `unpublished`; placeholder `0000000000000000000000000000000000000000000000000000000000000000` until the file is built and the mirror is pinned).

Primary download: `Mitroshenkov87/voxprint-mirror-moss-image-sft-fp8` revision `unpublished`.
Until that revision is a published commit, and if a download of the pinned file fails, the loader quantizes the bf16 snapshot above and saves the result here.

Card date: 2026-10-25.
