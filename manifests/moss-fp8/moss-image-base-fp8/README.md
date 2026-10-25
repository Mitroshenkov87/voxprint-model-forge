---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-Image-Base
tags:
- voxprint
- quantized
- fp8
---

# MOSS-Image Base DiT FP8

This is a **modified** Apache-2.0 derivative of `OpenMOSS-Team/MOSS-Image-Base` at revision `1ceede80badddd6a7812069619edaa5633587192`.
The bf16 backup mirror is `Mitroshenkov87/voxprint-mirror-moss-image-base` at `85cc5ca1dc0b6b74555715dec6164bb3e81f3c55`.
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
| `transformer/diffusion_pytorch_model-00001-of-00003.safetensors` | 4991978016 | `ae085fba123b6df67daf90aa9698fadee60c73e45bcaf3f3043d15f34e7a1804` |
| `transformer/diffusion_pytorch_model-00002-of-00003.safetensors` | 4983137016 | `806e876d80c7fba560663a3a95580e6c7033da3f9ea3c52765245ef0e1818f75` |
| `transformer/diffusion_pytorch_model-00003-of-00003.safetensors` | 189549184 | `5f319b72c40324294007c270d78112425627b537fcd8769b3c00f9f070e506a9` |

## Quantized file

`dit-fp8.safetensors` sha256: pending (mirror revision `unpublished`; placeholder `0000000000000000000000000000000000000000000000000000000000000000` until the file is built and the mirror is pinned).

Primary download: `Mitroshenkov87/voxprint-mirror-moss-image-base-fp8` revision `unpublished`.
Until that revision is a published commit, and if a download of the pinned file fails, the loader quantizes the bf16 snapshot above and saves the result here.

Card date: 2026-10-25.
