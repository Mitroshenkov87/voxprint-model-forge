# KL stage

`forge kl` scores GGUF quants against an F16 base for the synth4 L-jobs. The corpus is `src/voxprint_forge/data/ppl.txt` (licence in `data/LICENCES.md`): 100 chunks of 512 tokens.

## Command

```bash
forge kl --bf16 t-tech/T-lite-it-2.1@d125c970c553de58fcee3c937d5e4867d4a448d8 \
  --quant Mitroshenkov87/voxprint-t-lite-it-2.1@d85d9393:*IQ4_XS*.gguf
```

`scripts/vm_run_kl.sh --job L1` reads `plans/kl-synth4.toml`. `--json` is a global flag and goes before the subcommand.

Safetensors jobs download the pinned BF16 repo and run `convert_hf_to_gguf.py --outtype f16` from llama.cpp b11476 (`scripts/llama.rev`, commit `988190680d5a89fce97de3c20df2c2813731fd61`). Set `FORGE_LLAMA_SRC` to that tree and `FORGE_LLAMA_BIN` to `llama-perplexity` and `llama-server`.

`scripts/bootstrap.sh` and `scripts/vm_setup_synth4.sh` build llama.cpp with `FORGE_CUDA_ARCHS` (default `80;89;90;120`). The `llama-cpp.done` stamp includes that list, so an older sm_80-only build is rebuilt. On synth4, `FORGE_CUDA_ARCHS=90-real` is a single-arch Hopper build and is about three times faster to compile. Before a live `forge kl` run, the harness checks `llama-perplexity --version` and, when `cuobjdump` is installed, that the binary contains SASS for the current GPU. A missing arch is exit 9. Base and quant runs both pass `-fa on`.

The base run passes `--kl-divergence-base`. On b11476 that flag is an alias of `--save-all-logits` and writes the base logits. Each quant run adds `--kl-divergence`, which reads the same file.

## Metrics

For the base: perplexity, file size of the F16 GGUF, and tokens/s. For each quant: perplexity, mean KL, KL p99 (`99.0% KLD`), top-1 agreement (`Same top p`), delta-p RMS, file size, and tokens/s.

Tokens/s is the A100 figure. It is not a consumer-GPU number.

## 16 GB and 24 GB fit

`fitmath.fits_budget` uses the quant file size plus q8_0 KV at context 8192 and 3 GB headroom. The KV shape is read from the F16 GGUF (`block_count`, `attention.head_count_kv`, `attention.key_length` or `embedding_length / head_count`). A sliding-window pattern (true means SWA) or `full_attention_interval` splits full and sliding layers.

The same row is also filled from `llama-server` at that context. CUDA model and KV buffers count. `CPU_Mapped` and `*_Host` do not. Both the computed fit and the server fit are in `summary.json`.

## L3 and L4

L3 converts `huihui-ai/Huihui-Qwen3.6-27B-abliterated`. The job fails if the checkpoint declares an MTP head and the F16 GGUF `nextn_predict_layers` does not match. That is the b11476 Qwen3.6 check.

L2's pinned paperscarecrow revision is already an F16 GGUF (`gemma-4-31b-abliterated-f16.gguf`), so that job uses the file as the base and does not reconvert it. Recheck the `google/gemma-4-31b-it` Apache-2.0 listing before publishing.

L4 scores the upstream t-tech Q4_K_M mirror. We did not quantise it. `voxprint-mirror-t-pro-it-2.1-gguf` is missing from `Mitroshenkov87/voxprint-model-index`. The summary records that gap. This command does not upload.

## Disk

When the job finishes, the BF16 tree, the converted F16 file, and the base logits are deleted so the next L-job can use the disk. Downloaded quant files stay under `results/kl/<job>/quants/` until you remove them.

Exit codes are the normal forge table: 0 when the report is written, 6 when conversion, KL parsing, or the MTP check fails, 9 when a llama.cpp tool is missing.
