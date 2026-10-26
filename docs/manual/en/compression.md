# Compression tests

Some model parts can run in a compressed (quantized) form that needs less video memory. Model Forge uses a compressed part only after it has been tested against the original, and the results are published here.

## MOSS-Image Base text encoder

The text encoder turns your prompt and descriptions into numbers for the picture model. On the 16 GB tier it runs on the graphics card in 8-bit form (int8); on the 24 GB tier it runs in full precision (bf16). On the 16 GB tier the picture model (the DiT) is stored as FP8 e4m3 weights and computed in bf16. On the 24 GB tier the picture model stays in full precision. The table below is the text encoder only.

Test: 8 prompts, each drawn with the full-precision encoder and with each compressed one, same seed; then the pictures were scored for how well they match the prompt (CLIP score, higher is better) and how much they differ from the full-precision picture (LPIPS, lower is closer).

| Text encoder | Video memory while encoding | Prompt match (CLIP) | Difference from full precision (LPIPS, mean) |
| --- | --- | --- | --- |
| Full precision (bf16) | 15.7 GB | 28.38 | — |
| 8-bit (int8), the 16 GB default | 8.9 GB | 28.43 | 0.134 |
| 4-bit (nf4) | 5.9 GB | 28.50 | 0.167 |

The 8-bit encoder matches the prompts as well as the full-precision one while using about half the memory, so it is the default on 16 GB cards. The 4-bit encoder also matches the prompts, but its pictures move further from the original, so it is offered only as an option.

The choice can be changed on the command line with `--te-quant auto|none|int8|nf4` on `lora train` (`auto` is the tier default). See [MOSS-Image](../../MOSS.md).

## Training on each tier

Training was measured with the default encoder of each tier; see [Presets → Measured figures](presets.md#measured-figures).

## Tests for advanced users

The command-line tool also scores compressed language and picture models against their originals. These tests need an NVIDIA graphics card and are meant for advanced users:

- `voxprint-forge-cli kl` compares compressed language models with the original: [KL](../../KL.md).
- `voxprint-forge-cli compare` compares compressed picture models with the original: [Compare](../../COMPARE.md).
