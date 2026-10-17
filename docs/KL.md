# KL stage

`forge kl` scores GGUF quants against an F16 base for the synth4 L-jobs. The corpus is `src/voxprint_forge/data/ppl.txt` (licence in `data/LICENCES.md`): 100 chunks of 512 tokens.

## Command

```bash
forge kl --bf16 t-tech/T-lite-it-2.1@d125c970c553de58fcee3c937d5e4867d4a448d8 \
  --quant Mitroshenkov87/voxprint-t-lite-it-2.1@d85d9393:*IQ4_XS*.gguf
```

`scripts/vm_run_kl.sh --job L1` reads `plans/kl-synth4.toml`. `--json` is a global flag and goes before the subcommand.

Safetensors jobs download the pinned BF16 repo and run `convert_hf_to_gguf.py --outtype f16` from llama.cpp b11476 (`scripts/llama.rev`, commit `988190680d5a89fce97de3c20df2c2813731fd61`). A job whose source is already an F16 GGUF downloads that file only and does not reconvert it. Set `FORGE_LLAMA_SRC` to that tree and `FORGE_LLAMA_BIN` to `llama-perplexity` and `llama-server`.

`scripts/bootstrap.sh` and `scripts/vm_setup_synth4.sh` build llama.cpp with `FORGE_CUDA_ARCHS` (default `80;89;90;120`). The `llama-cpp.done` stamp includes that list, so an older sm_80-only build is rebuilt. On synth4, `FORGE_CUDA_ARCHS=90-real` is a single-arch Hopper build and is about three times faster to compile. Before a live `forge kl` run, the harness checks `llama-perplexity --version` and, when `cuobjdump` is installed, that the binary contains SASS for the current GPU. A missing arch is exit 9. Base and quant runs both pass `-fa on`.

The base run passes `--kl-divergence-base`. On b11476 that flag is an alias of `--save-all-logits` and writes the base logits. Each quant run adds `--kl-divergence`, which reads the same file.

## Metrics

For the base: perplexity, file size of the F16 GGUF, and tokens/s. For each quant: perplexity, mean KL, KL p99 (`99.0% KLD`), top-1 agreement (`Same top p`), delta-p RMS, file size, and tokens/s.

Tokens/s is the figure from the GPU that ran the job. `summary.json` names that GPU in `tokens_per_second_gpu`. `tokens_per_second_is_a100` is true only on compute capability 8.0. The number is not a consumer-GPU figure. The llama.cpp build includes sm_90 (`FORGE_CUDA_ARCHS`, default `80;89;90;120`, or `90-real` for a single-arch synth4 build).

## 16 GB and 24 GB fit

`fitmath.fits_budget` uses the quant file size plus q8_0 KV at context 8192 and 3 GB headroom. The KV shape is read from the F16 GGUF (`block_count`, `attention.head_count_kv`, `attention.key_length` or `embedding_length / head_count`). A sliding-window pattern (true means SWA) or `full_attention_interval` splits full and sliding layers.

The same row is also filled from `llama-server` at that context. CUDA model and KV buffers count. `CPU_Mapped` and `*_Host` do not. Both the computed fit and the server fit are in `summary.json`.

## L3 and L4

L3 uses the published F16 GGUF `Huihui-Qwen3.6-27B-abliterated-ggml-model-f16.gguf` from `huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF` at `2da8c0432f0a42c25c14cbc99f90028a90d3dcd3`. That is the file our Q3_K_M was quantised from, so the job does not reconvert the BF16 safetensors repo. The job fails if that GGUF has no MTP head (`nextn_predict_layers` is missing or zero). A safetensors job still fails when the checkpoint declares an MTP head and the converted F16 does not keep it. That remains the b11476 Qwen3.6 check.

L2's pinned paperscarecrow revision is already an F16 GGUF (`gemma-4-31b-abliterated-f16.gguf`), so that job uses the file as the base and does not reconvert it. Recheck the `google/gemma-4-31b-it` Apache-2.0 listing before publishing.

A job or a quant may set `sha256`. After the download, that digest is checked against the F16 GGUF (the job) or the quant GGUF. A mismatch fails the job before llama.cpp runs. L2's Q4_K_M, L3's Q5_K, and the L3 F16 reference carry the published digests. Leave the field unset to skip the check. A safetensors job has no single source file, so a digest on that job is a plan error.

L4 scores the upstream t-tech Q4_K_M mirror. We did not quantise it. `voxprint-mirror-t-pro-it-2.1-gguf` is missing from `Mitroshenkov87/voxprint-model-index`. The summary records that gap. This command does not upload.

## Disk

When the job finishes, the downloaded source, the F16 file, and the base logits are deleted so the next L-job can use the disk. Downloaded quant files stay under `results/kl/<job>/quants/` until you remove them.

Exit codes are the normal forge table: 0 when the report is written, 6 when conversion, KL parsing, or the MTP check fails, 9 when a llama.cpp tool is missing.

## KV-cache arms and the spec/prefix bench (Plotweaver)

`plans/kv-synth4.toml` and `scripts/vm_run_kv.sh` use llama.cpp b11476.

- **KV arms.** Each GGUF (L1, L3, L4; L2 once the L1/L2 eval fix is merged) is scored against the same file with an f16 KV cache. Arms are K/V = q8_0/q8_0, q8_0/q4_0 and q4_0/q4_0, run at ctx 512 (100 chunks) and ctx 8192 (12 chunks) on wikitext-2-raw test. The corpus is downloaded at runtime from `ggml-org/ci` @ `927b3642` and sha256-checked. `perplexity_argv` and `kl_argv` take `cache_k`/`cache_v`. A quantized cache always runs with `-fa on`. Proposed bar: KL ≤ 0.01 and PPL within 1% of f16 KV. Where PPL is unusable, KL alone decides.
- **Bench.** `llama-server` runs with no draft and with `--spec-type ngram-simple | ngram-map-k | ngram-mod`, on a fixed rewrite set (wikitext passages returned with light edits) and a free-prose set. Settings: temperature 0, seed 7, 256 tokens. Prefix: one 12k-character shared system prompt and four turns, with the default prompt cache and with `--cache-reuse 256`. Reported: tokens/s, accept rate (`draft_n_accepted / draft_n`), `prompt_ms` and `cache_n`. Hybrid (L3) and SWA (L2) models may refuse speculation or cache reuse. The bench records `server_refused`.
- **Preflight.** `python -m voxprint_forge.kl.kv --preflight` checks that `llama-server --help` lists `--spec-type` with every ngram type and `--cache-reuse`, that `llama-perplexity` has `-ctk`/`-ctv`/`--kl-divergence`, `protobuf==5.29.6`, and, for L2, the gemma4 GGUF BOS flag, BOS id and chat template. `--dry-run` prints every step.
