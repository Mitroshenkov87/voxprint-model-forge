# Compare harness

`forge compare --job ID` scores an original arm against one or more compressed
arms for the synth4 A100 queue. The plan is `plans/compare-synth4.toml`.
`scripts/vm_run_compare.sh --job D1 --json` calls the same command.

Exit codes for this command are separate from the rest of `forge`:

| Code | Meaning |
| --- | --- |
| 0 | Every enabled candidate passed the quality bar |
| 1 | The run finished and at least one candidate failed the bar |
| 2 | Setup failed: plan, weights, manifest, or a missing tool |
| 3 | No CUDA GPU |

Results land in `$FORGE_OUT/results/<job>/`:

- `pairs.jsonl`, one row per original-versus-candidate image, resumed on the next run
- `summary.json` and `summary.md`
- `images/<arm>/`, review pictures that stay on the machine

Job weights are downloaded under `results/<job>/weights` and deleted when the
run ends. The review images are not deleted.

## Prompts

Public prompts are `BASE_PROMPTS` (32 texts, seeds 0 and 1) or, for D3,
`CONTROL_PROMPTS` times canny, depth, and pose times seeds 0 and 1.

Private prompts load only from the directory in `FORGE_PRIVATE_PROMPTS`.
That variable follows the same idea as `FORGE_PRIVATE_CALIB`: unset skips the
private set, and a value that is not a directory is a setup error. The harness
reads two manifest layouts and keeps prompt, negative prompt, seed, and size:

- `<set>/manifest.json`, a dict with a `_run` key plus one key per image
- `<set>/dataset/manifest.json`, a dict keyed by image id

Steps and CFG in those files are ignored. Turbo uses 8 steps and CFG 0.
Z-Image Creative uses 50 steps and CFG 4. D3 uses the control scale 0.75.

D1 and D2 add up to 50 prompts from each of `vxp_yuna` and `vxp_mei` at the
manifest seed. A set that is not on disk yet is skipped and named in the
summary. D4 stays on the 64 public images.

Every arm of a case uses `torch.Generator("cuda").manual_seed(seed)`.

Logs, `pairs.jsonl`, and both summaries store `sha256(prompt)[:12]` and never
the prompt text. A CI test fails if private prompt text appears in those
outputs. Review images use the same hash in the file name.

## Metrics

Each non-warmup pair records:

| Metric | Definition |
| --- | --- |
| LPIPS | Alex net, perceptual distance, lower is closer |
| SSIM | torchmetrics structural similarity on RGB at the native size, data range 1 |
| PSNR | torchmetrics peak signal to noise ratio on the same RGB tensors. Identical images have unbounded PSNR and are stored as null |
| DINOv2 | cosine of `facebook/dinov2-base` class tokens |
| CLIP-I | cosine of CLIP ViT-B/32 image embeddings |
| CLIP-T | CLIP ViT-B/32 text-image cosine for each arm. The drop is `(reference - candidate) / abs(reference)` |
| s/img | candidate generation time. The first case is a warm-up and is left out of the mean |
| VRAM | peak `torch.cuda.max_memory_allocated`, and the peak of a 1 Hz `nvidia-smi` memory sample |
| file bytes | size of each weight file used by the arm |

The pass bar is the existing Forge gate: mean LPIPS <= 0.10 and CLIP-T drop
<= 1%. SSIM, PSNR, DINOv2, and CLIP-I are reported until a bar is set for them.

## A100 FP8 caveat

sm_80 has no FP8 tensor cores. An FP8 arm is dequantised to BF16 before the
forward pass (`voxprint-fp8-v1` scales). Seconds per image and VRAM measured
on the A100 are not RTX 40/50 numbers. Quality metrics and file sizes still
apply. INT4 W4A4 is native on sm_80. The upstream nunchaku INT4 arm is a
measurement-only baseline and stays disabled until the loader proves the INT4
kernels actually loaded.
