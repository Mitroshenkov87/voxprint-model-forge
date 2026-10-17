# KL stage

`forge kl` scores GGUF quants against an F16 base for the synth4 L-jobs. The corpus is `src/voxprint_forge/data/ppl.txt` (licence in `data/LICENCES.md`), at most 100 chunks of 512 tokens. The file is about 1,700 words, so a run scores only 3 to 5 chunks; treat small KL gaps between quants as noise.

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

## Ablated quants (L1)

Our `mode = "ablate"` quants were made from the BF16 after Forge's refusal-direction ablation, not from the stock BF16. Scoring them against the stock weights measures the ablation (L1 on 2026-10-17: KL 0.18 for both IQ4_XS and Q5_K_M). `reference = "recover-ablation"` with `ablation_quant` fixes the reference: `kl.recover` reads the removed direction back from that quant (the smallest eigenvector of the summed `W W^T` over `attn_output` and `ffn_down`; the job fails when its eigenvalue ratio is above 0.1, which is what an un-ablated file gives), projects it out of the pinned BF16 exactly as the ablation step does, and converts that tree. Re-running the ablation is not used because it is not bit-reproducible. Each quant row records its GGUF `general.name` and whether it matches the reference's name.

## PPL validity (L2)

llama-perplexity stores the base log-probs floored at the top logit minus 16. When the base PPL read back in a KL run differs from the base run by more than 2%, many tokens sit below that floor and the corpus is far outside what the model predicts. The summary then sets `ppl_valid = false`, prints PPL as n/a and says to compare by KL, top-1 and delta-p. Gemma 4 31B instruct does this on raw prose at context 512 (L2: base 6827, read back 3867). `forge kl --render <results folder>` rewrites an existing summary with these fields without a GPU.

The pinned L2 GGUF matches google/gemma-4-31B-it byte for byte in the tensors checked, including layer 59 `attn_output` and `ffn_down`, which the author's ablation script changes. The ablated weights are only in the repo's safetensors folder.

## Local reference

A `gguf` job whose `f16_file` is an absolute path uses that file in place: it is not downloaded and not deleted
afterwards. `scripts/vm_gemma4_requant.sh` (GEMMA4-REQUANT) uses this to score its re-quantised tiers against the F16
it converted from the author's safetensors. Its other checks (tensor diff against stock, private refusal probe,
verdict) are in `python -m voxprint_forge.requant_check`.
