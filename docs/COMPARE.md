# Compare harness

`forge compare --job ID` scores an original arm against one or more compressed
arms for the synth4 queue. The plan is `plans/compare-synth4.toml`.
`scripts/vm_run_compare.sh --job D1 --json` calls the same command.
`scripts/vm_setup_synth4.sh` installs the `compare` extra and builds llama.cpp.
It does not run a job. `--dry-run` prints the torch index and CUDA arch list.
Before any download it checks for `cmake`, `ninja` and `nvcc` and exits 9 with
the missing names. `CUDACXX` defaults to `/usr/local/cuda/bin/nvcc` when that
exists, else `nvcc` on `PATH`; llama.cpp builds with `-G Ninja`. Torch and
torchvision come from the CUDA index (cu130, or cu128 for drivers below 580)
in a first step. Everything else then comes from PyPI only, with torch and
torchvision held by a constraint file: mixing the CUDA index into that step
made uv resolve index-first and pick torchmetrics 1.0.3 (`[compare]` needs
1.6 or newer). See docs/VM.md.

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
reads two manifest layouts:

- `<set>/manifest.json`, a dict with a `_run` key plus one key per image
- `<set>/dataset/manifest.json`, a dict keyed by image id

Each image row uses these fields:

| Field | Required | Use |
| --- | --- | --- |
| `prompt` | yes | Text sent to the model. A row without it is a setup error that names the image key. The text is not written to the results. |
| `seed` | yes | Integer seed for every arm of that row. |
| `negative_prompt` or `negative` | no | Empty when omitted. |
| `width` and `height`, or `size` | no | `size` may be an int, `[width, height]`, or `"WxH"`. A row without a size uses the job size, 1024 in this plan. |
| `model`, `model_id`, `revision` | no | Ignored. The job names the model under test. |
| `steps`, `guidance`, `guidance_scale`, `cfg` | no | Ignored. Turbo is 8 steps and CFG 0. Z-Image Creative is 50 steps and CFG 4. D3 uses control scale 0.75. |
| `keep` | no | `keep: false` drops the row. |

`vxp_yuna` and `vxp_mei` use that per-model layout: prompt, seed, model and
revision, steps, and CFG. Size may be omitted. `vxp_lena` is the nested
`dataset/manifest.json` layout and may omit size. A row that also omits
`prompt` fails that row and names its key.

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
| s/img | generation time for that arm, including the reference. The warm-up case and the pipeline load are left out of the mean |
| VRAM | peak `torch.cuda.max_memory_allocated`, and the peak of a 1 Hz `nvidia-smi` memory sample |
| file bytes | size of each weight file used by the arm |

The pass bar is the existing Forge gate: mean LPIPS <= 0.10 and CLIP-T drop
<= 1%. SSIM, PSNR, DINOv2, and CLIP-I are reported until a bar is set for them.
The summary also reports reference seconds per image over the same non-warmup
cases.

## Generation

Arms run one at a time. The reference arm renders every case and the PNGs stay
in `images/<reference>/`. Each candidate runs after that pipeline has been
deleted, collected, and the CUDA cache emptied. LPIPS, CLIP, and DINOv2 load
after the last pipeline is released, so the peak VRAM of an arm is that arm
alone.

The first case is the warm-up for every arm. It is stored and left out of
seconds per image. `load_arm` runs before the image timer. A resumed run that
still has images to render loads the pipeline before it times the first of
those images, so load time is not part of that sample.

A mirror split across safetensors shards is merged per module before the
coverage check. A `*.safetensors.index.json` weight map selects the shard
files when it is present. One shard of the text encoder or the transformer is
not required to hold half of that module on its own.

## FP8 modes

`compute = "fp8-native"` keeps the `float8_e4m3fn` file bytes and runs them
with dynamic per-tensor FP8 activations. It needs compute capability 8.9 or
newer. On an older GPU those arms are skipped with `fp8-native needs sm_89+`
and the rest of the job still runs. `compute = "bf16-dequant"` expands the
same file to BF16 (`load_dequant_bf16`). D1 arm `c-dq` is that weight-only
path. Native arms are valid only with `quant = "fp8"`.

Every arm of a job uses one attention setting from the plan (`sdpa` by
default: flash, cuDNN, and efficient kernels, never MATH). `sdpa-math` is
MATH only. Any other value is passed to `pipe.transformer.set_attention_backend`
and is opt-in, including Hopper FlashAttention-3. cuDNN benchmark and TF32
stay off. `summary.json` records the GPU. `fp8_note` is one sentence per enabled FP8 arm:
dequant says the seconds and VRAM are BF16 numbers, and native names the
detected GPU and its compute capability.

## Shared downloads

Each `repo@revision` is downloaded once per run into `$FORGE_OUT/compare-cache/`.
An arm may set `sha256`. After the download, that digest is checked against the
one named file in `files`. A mismatch, or more than one file, stops the job.
Leave it unset when the arm downloads a whole repo.
Arm folders hardlink the files they load. A local FP8 arm does not link
`transformer/*.safetensors` from the base snapshot; the config stays, and the
FP8 file replaces the weights. That transformer is built from `config.json`,
so it starts random: the overlay must fill every parameter and persistent
buffer, with no shape mismatches, or the job stops with `compare_weights`.
An overlay onto real base weights still accepts a match of half the weight
keys. `component_bytes` counts only the files in the arm folder.

`[defaults] queue` is `D1`, `D3`, `D2`, `D4`. A cache entry is deleted when
the last job in that queue which would download it finishes. A job the GPU
gate skips holds nothing, so on sm_90 the Turbo base is released at the end
of D3. `forge compare --reclaim-cache` deletes the whole cache. Job weights
under `results/<job>/weights` are still deleted at the end of that job.

```bash
forge compare --make-fp8 --job D1 --out "$FORGE_FP8_DIR"
forge compare --make-fp8 --job D4 --out "$FORGE_FP8_DIR"
```

`--make-fp8` writes `zimage-turbo-transformer-fp8.safetensors` and
`qwen3-4b-text-encoder-fp8.safetensors` for D1 (encoder profile
`qwen3_encoder`), and `zimage-transformer-fp8.safetensors` for D4. An existing
`voxprint-fp8-v1` file is left in place. The files stay local.

Arms `c`, `c-dq`, and `d` on D1, and the D4 FP8 arm, read those files from
`FORGE_FP8_DIR`. When the variable is unset those arms are skipped and the
other arms still run.

D2 lists `requires_cc` for the INT4 arms. On sm_90 they are skipped with
`nunchaku INT4 has no sm_90 kernels (Hopper lacks 4-bit tensor cores)` and the
job exits 2 before nunchaku is imported. Do not build nunchaku for sm_90.
Where the kernels exist, D2 loads `NunchakuZImageTransformer2DModel` and
records `nunchaku.__version__`. If the package, the class, or the load fails,
the job exits 2. The upstream INT4 weights are a measurement-only baseline.
