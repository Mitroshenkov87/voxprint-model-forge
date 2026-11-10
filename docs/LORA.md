# LoRA trainer (engine and CLI)

`forge lora ...` (also `voxprint-forge-cli lora ...`) trains a LoRA from 10 to 50 photos of a person, an object or a style. The GUI drives the same engine. Plan: project-notes `model-forge/LORA-TRAINER-GUI-PLAN.md`.

## Commands

| Command | What it does |
| --- | --- |
| `lora prepare --project DIR --photos DIR --kind KIND ...` | Checks, crops and describes the photos. Photos that are unreadable, too small, a collage, blurry or fail the safety check are listed in `drops.jsonl` with a reason key, one line per photo. A near-duplicate (a photo within `--dup-threshold` bits of an earlier one by difference hash) is kept, and a warning (`lora.warn.near_duplicate`) names both files. `--drop-near-duplicates` sets near-duplicates aside as `lora.drop.duplicate` instead. A byte-identical copy of a photo already taken is always skipped. `--keep FILE` (repeatable) keeps a photo that would be set aside as a collage (`lora.drop.collage`) or, with `--drop-near-duplicates`, a near-duplicate (`lora.drop.duplicate`); the file is added to the photos if needed. A `--keep` photo still goes through every other check, the safety check included, and the same rules apply to it as to any other photo. `--exclude` wins over `--keep`. Running it again on the same project adds the new photos as the next round (for `extend`); a photo set aside again keeps its single line. |
| `lora train --project DIR` | Trains on the prepared photos of the project. `--dry-run` prints the plan and the time estimate. `--review-rounds N` (1–6, default 3; `0` trains straight through) pauses after each planned stage. The first stage is `--first-round-steps` (100–500, default 120). Later stages share the preset steps that remain; the last of them takes the remainder. The preset total does not change. `--first-round-samples` (default 6) and `--samples-per-round` (default 8) are the pictures at the first review and at later reviews. `--sample-steps` (default 28) is the inference step count for those pictures. `--sample-attire {auto,as_photos,clothed,unclothed}` (default `auto`, stored on the project) builds those prompts from the trigger, a framing, and one attire word. `--sample-framing {mixed,portrait}` (default `mixed`, stored on the project) picks the framing: `mixed` rotates close-up portrait, waist-up and full body; `portrait` makes every sample a face close-up, for judging likeness. A paused run waits for `lora review` / `continue` / `finish`; see [Headless runs](#headless-runs). |
| `lora extend --project DIR --from FILE` | Continues an existing LoRA with the newest photos of the project plus `--replay` share (default 0.25) of the older ones. A LoRA made on another base is refused (`lora.err.base_mismatch`). |
| `lora export PROJECT` | Writes `<stage name or name>.voxlora` (`-o FILE` to choose the place): an uncompressed zip with `model.safetensors`, `voxprint-model.json` (schema `voxprint-model/1`, `docs/schemas/voxprint-model-1.json`) and previews from samples that passed the output check. Source photos are not included. `steps` and `steps_planned` are in the description. |
| `lora export --project DIR` | Saves the last version (or `--checkpoint N`, any saved version; a failed export changes nothing, so run it again without retraining) as `<id>.safetensors` with `voxprint-model.json`, LICENSE and NOTICE (`--out FILE` to choose the place). `--install-to-mdl` also adds it to your models for Canvas. |
| `lora refset round --project DIR --ref FILE ... --kind KIND ...` | Build a set from 1-10 references. The first round gates every reference (any that does not pass blocks the loop) and has the judge VLM write one prompt per reference (`refset/prompts.json`, local only). Then Z-Image-Turbo img2img (`ZImageImg2ImgPipeline`, BF16, model CPU offload) draws `--candidates` (8-64, default 24) pictures at `--strength` (0.25-0.90, default 0.5) and `--steps` (4-16, default 9); every candidate is gated before it is listed; ones that do not pass are deleted and a minor signal blocks the loop. DINOv2-base marks near-duplicates. Later rounds start from the kept candidates plus the references. Stages run one at a time with a full unload between them. |
| `lora refset finish --project DIR` | "This is it": copies the references and kept candidates into `refset/approved/` (10-50, or 5-60 with `--mode expert`) and records rounds, seeds, settings, the base and the teacher with its licence in `refset/metadata.json` and `project.json`. Then run `lora prepare` on `refset/approved/`. |
| `lora review PROJECT` | Lists the samples of a paused review, with the original photos. `--like ID` and `--dislike ID` mark samples (repeat either flag). `--from-file FILE` reads `like ID` / `dislike ID` lines, or a JSON object with `like`, `dislike` and `unmarked` lists. |
| `lora continue PROJECT` | Trains the next round from the review checkpoint, on the original photos plus the liked samples, then pauses for another review. `--unattended` keeps that liked set and trains every remaining preset step with no more pauses. Disliked samples are negatives only when the base supports them. `--sample-attire` and `--sample-framing` change the samples from this round on and are stored on the project. |
| `lora finish PROJECT` | Saves the current LoRA, draws a final preview, records the steps it got (`steps` and `steps_planned` in `train.json` and in the export), and writes `<project>/export/<stage name or name>.voxlora` (`voxlora` in the result). `--no-preview` skips the final preview. If the preview fails, a warning is printed (`lora.warn.final_preview`), the result has `preview: "failed"`, and the LoRA is still saved, finished and exported. `--discard` removes temporary checkpoints and samples and saves nothing; photos stay. `--discard` needs `--yes` when the terminal is not interactive. |
| `lora eval PROJECT` | Likeness score (see [Likeness score](#likeness-score)). Scores the previews of the saved LoRA's step by default; `--step N` picks another preview step, `--step all` scores every step and names the best one, `--images DIR` scores a folder of pictures. Fetches the face models with MDL when stdin is a terminal or with `--fetch-missing`. |
| `lora status PROJECT` | Prints the latest `train.json` progress. Adds the review phase and counts when a review is saved. During round 1 it also prints `first_pictures`. |

Training refuses to start unless the project was prepared and passed the safety check (`lora.err.gate_missing`). It needs at least 10 prepared photos, or 5 in Expert mode.

Presets: `--preset quick|balanced|best`, `--likeness flexible|balanced|close`, `--vram auto|16|24`. `quick` is 150 steps and uses the same pipeline as the other presets. **Train a person LoRA with `--preset balanced` (the default).** `quick` is a smoke test of the setup, not a usable likeness: on an RTX 4090 person run, 120 quick steps scored 1.68/5 for likeness and 1000 balanced steps scored 3.95/5. Z-Image on the 16 GB tier uses the measured 9.7 s/step at 1024 px. The 24 GB Z-Image time, 1.6 s/step at 1024 px, has not been measured. Other preset values are not final until the GPU calibration run.

### Z-Image 16 GB recipe (release path)

This is the default training path on full Z-Image (`Tongyi-MAI/Z-Image@04cc4abb`). Z-Image-Turbo is the generation default and is not trained. Use `--base z-image` or omit `--base` on a new project. Pass `--vram 16` (or `--vram auto` on a card under 20 GiB). The 16 GB tier stores the Z-Image DiT as FP8; gradient checkpointing is on. CLI preset names stay `quick`, `balanced` and `best` (`fast` / `quality` in prose map to `quick` / `best`).

| Preset | Steps | Rank / alpha | lr | Resolution | batch × accum | Planned peak | Need (plan + 1 GiB) | Seconds / optimizer step | Reserved peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `quick` (fast; smoke test) | 150 | 16 / 16 | 1e-4 | 768 | 1 × 1 | 9.03 GiB | 10.0 GiB | 5.46 s, scaled from the 9.7 s/step at 1024 px | unmeasured |
| `balanced` (recommended) | 1500 | 32 / 32 | 1e-4 | 1024 | 1 × 2 | 10.08 GiB | 11.1 GiB | 19.4 s, the measured 9.7 s/step at 1024 px × accum 2 | unmeasured |
| `best` (quality) | 3000 | 32 / 32 | 8e-5 | 1024 | 1 × 2 | 10.08 GiB | 11.1 GiB | 19.4 s, same scaling as `balanced` | unmeasured |

Figures: planned peaks are from `ZImageAdapter.memory_plan` (no GPU, pinned shapes) at rank and resolution above. The only measured Z-Image time is 9.7 s/step on a 16 GB RTX 4090 laptop at 1024 px, batch 1, accum 1 (`presets.MEASURED`, peak not recorded). `seconds_per_step` scales that by `(resolution/1024)² × batch × grad_accum`. The reserved peak on the 16 GB tier is **unmeasured**; a short VM train with `--dry-run` then a few steps is needed to fill it in. The 24 GB Z-Image plan is about 14.87 GiB (`quick`) and 16.27 GiB (`balanced`/`best`); its 1.6 s/step time and reserved peak are unmeasured.

 Expert flags (`--rank`, `--alpha`, `--steps`, `--lr`, `--scheduler`, `--warmup`, `--batch`, `--grad-accum`, `--resolution`, `--precision`, `--grad-checkpointing`, `--caption-mode`, `--caption-dropout`, `--seed`, `--save-every`, `--preview-every`) override single values. There are no safety settings.

Memory tiers: both tiers train with gradient checkpointing on (`--no-grad-checkpointing` turns it off in Expert). Without it, a Balanced 1024 px batch of 2 kept every block's activations and reached 79.3 GB on the synth4 calibration run, so the 24 GB tier could not fit. The 24 GB tier keeps the base in BF16; the 16 GB tier also stores it in FP8. The transformer moves to the card only after the text encoder and VAE have left it, and the test prompts are encoded once up front, so previews never load the text encoder next to the transformer.

Memory plan: before any weight reaches the card, `lora train` projects the peak of each phase from the base's shapes and stored sizes (encode: text encoder and VAE; train: transformer, LoRA weights, gradients and AdamW moments, one micro-batch of activations, cached samples; plus 10 % or 0.5 GiB slack) and stops with `lora.err.vram` when the peak exceeds the card, or `FORGE_VRAM_CAP_GB`, minus 1 GiB. `--dry-run` and every result report it as `memory_plan`. For Z-Image Balanced at 1024 px the plan is about 16.3 GiB on the 24 GB tier (BF16 base) and about 10.1 GiB on the 16 GB tier (FP8 base storage). The text encoder and VAE are frozen and every embedding is computed without autograd, so nothing can pin them on the card; when they leave, whatever is still allocated must be the cached latents and embeddings, or the run stops with `lora.err.encoders_stuck` before the transformer moves. CALIB-R2 and R3 OOMed exactly there: the test-prompt embeddings were computed with autograd on, and their graphs kept the 7.5 GiB text encoder and about 8.8 GiB of its activations on the card. `PYTORCH_CUDA_ALLOC_CONF` defaults to `expandable_segments:True`. Previews run on the cached embeddings with the text encoder hidden from the pipeline, so the execution device is the card the transformer is on, and the preview noise comes from a CPU generator (same picture on every card for a seed); a device mismatch stops with `lora.err.preview_device`. CALIB-R4 crashed there: diffusers picked the parked text encoder's CPU as the execution device while the generator was on CUDA.

`FORGE_VRAM_CAP_GB` (GiB) is a test knob for calibration: it caps the allocator with `torch.cuda.set_per_process_memory_fraction`, so a large card behaves like a smaller one. With it set, `lora train` also reports `vram_cap_gb`, and every run reports `peak_reserved_gb` next to `peak_vram_gb`.

Pause with Ctrl+C, or by creating an empty file named `PAUSE` in the project folder: the run saves a version and stops. Continue with `--resume`; the saved settings are reused.

### Pause file (GUI contract, Windows and Linux)

- File: `<project>/PAUSE`, where `<project>` is the `--project` folder. Content is ignored; an empty file is enough.
- To pause, the GUI creates the file. After each training step the engine checks for it. If it exists, the engine saves a version at that step, sends an event with `state: "paused"` (`message_key: "lora.progress.paused"`), prints the result with `"state": "paused"` and exits with code 0. If test pictures are being drawn at that step, the pause comes after they finish.
- The engine never creates or deletes `PAUSE`.
- To continue, the GUI deletes `PAUSE`, then runs `lora train --project <project> --resume` (or `extend ... --resume`).
- If `PAUSE` exists when `train` or `extend` starts, the engine refuses with `lora.err.pause_file` (exit code 2) and trains nothing.
- To stop for good, pause and then export the last version, or end the process. Ctrl+C on Linux does the same as `PAUSE`.

### Headless runs

`lora train` pauses for review by default (`--review-rounds 3`). The process then exits with the result `"state": "review"`, and nothing trains until someone runs `lora continue` or `lora finish`. When stdin is not a terminal (a script, a CI job, a scheduled task), the pause also prints a hint to stderr with the next commands: `lora review PROJECT` to list and mark samples, `lora continue PROJECT` (or `--unattended`), and `lora finish PROJECT`. stdout and `--json` output are unchanged. To train straight through with nobody watching, pass `--review-rounds 0`:

```text
voxprint-forge-cli lora train --project my-lora --preset balanced --review-rounds 0
```

`lora finish` on that project uses the base training stored in `project.json`. It does not need `review/state.json`. If those base files are missing, finish fetches that model with `voxprint-forge-cli models fetch --model <id>` before the final preview.

## Export and the Canvas hand-off

The deliverable is the exported `.voxlora` package from `lora finish PROJECT` or `lora export PROJECT`. Keep, share and open that file in Canvas. `checkpoints/step-NNNNNN.safetensors` are intermediate training files: they have no description, previews or subject record, and Canvas does not accept them. To ship an earlier version, export it with `lora export PROJECT --checkpoint N`.

Header metadata (strings): `voxprint.subject.name`, `stage_name`, `trigger`, `consent_obtained`, `adult_confirmed`, `provenance_note`, `release_attached`, `release_sha256`, `consent_method` (`verbal` or `file`) for a person, and `rights_confirmed` for objects and styles; `voxprint.lora.base` (legacy MDL id, unchanged), `voxprint.lora.trained_base` (`z-image@04cc4abb`, `z-image-turbo@f332072a` or `moss-image-base@` plus the first eight characters of the pin), `voxprint.lora.compatible_bases` (compact JSON: `["z-image","z-image-turbo"]` for Z-Image, `["z-image-turbo"]` only for a Turbo-trained LoRA, `["moss-image-base"]` for MOSS), `base_revision`, `rank`, `alpha`, `steps`, `kind`, `trained_on`, `adult_content` (`true` or `false`, always present). Nothing else: no paths, usernames, machine ids or EXIF, and no key that reads like a gate or threshold setting. Short export tags stay `zimg6b`, `zimgturbo` and `moss5b`. A LoRA trained on full Z-Image is accepted on Z-Image-Turbo. A LoRA trained on Turbo (with the de-distillation adapter) is accepted on Turbo only, not on full Z-Image.

`voxprint-model.json` (`schema: voxprint-model/1`): `min_schema` 1, `id`, `kind`, `type` lora, `producer`, `file`, `sha256`, `size`, `base.mdl_id`, `base.revision`, `licence` (from the base), `default_strength` 0.8, `strength_range` [0, 1.5], `tags`, `voxprint.lora.adult_content` (JSON `true` or `false`, always present), `voxprint.subject.consent_method` (`"verbal"` or `"file"`, only for a person), and `preview` with `preview_sha256` when a test picture exists (`preview.webp`).

`voxprint.lora.adult_content` is a label the user sets: the training photos contain adult content. Apps such as Canvas can use it to label, sort or filter LoRAs (for example, hide them unless the user opts in). It can be `true` only for a person confirmed as 18 or older. It is not a safety setting and changes no checks. `voxprint.subject.consent_method` says how a person's consent was given: `verbal` (no file, `release_sha256` empty) or `file` (a release is attached; only its sha256 is recorded). LoRAs exported before these fields have no `adult_content` key; treat that as `false`.

Both `voxprint-model.json` files (the sidecar and the one in the `.voxlora`) also carry `likeness: {mean, median, ref_baseline, threshold, status, n, no_face}` when the likeness score was measured on the exported step. It is informational only.

`--install-to-mdl` writes `<id>.safetensors`, `voxprint-model.json`, LICENSE, NOTICE and the preview to `<models dir>/user-models/lora/<id>/`, where the models dir is always `voxprint_common.paths.models_dir()` (`VOXPRINT_MODELS_DIR`, then `models_dir` in `suite.json`, then `<suite home>/models`). The id is the trigger word plus 8 hex digits of the LoRA tensors. Tensors keep the dtype they were trained in (BF16 by default; FP32 for an FP16 base); the id hashes their raw bytes.

## Photos and consent

Simple mode (default) takes 10 to 50 photos. Expert mode (`--mode expert` or `--allow-count-outside-range`) warns outside 10-50, needs at least 5 and takes at most 60.

| Kind | Required |
| --- | --- |
| `person` | `--name` or `--stage-name`, `--consent-obtained`, `--subject-adult`, and `--consent-method verbal` or `--consent-method file --release FILE` |
| `object` | `--name` or `--stage-name`, `--rights-confirmed` (I own this object or have the right to use these photos) |
| `style` | `--name` or `--stage-name`, `--rights-confirmed` (I have the right to use these images; the style is not presented as a named living artist's work) |

Example for a person, from any folder once the app is installed:

```text
voxprint-forge-cli lora prepare --project my-lora --photos my-photos --kind person --name Garnis_AI --consent-obtained --subject-adult --consent-method verbal --fetch-missing
```

- `--name NAME`: the subject's name or stage name, for example `Garnis_AI`. Required for every kind (`--stage-name` alone also works). Without it, `lora prepare` stops with exit 2 and prints `--name is required: give the subject's name or stage name, for example --name Garnis_AI --consent-method verbal`. The name does not become the trigger word.
- `--consent-method verbal|file`: how a person's consent was given. `verbal`: given in person, no file. `file`: a signed release, passed with `--release FILE`.
- `--fetch-missing`: download any missing required models before the photos are checked. `--no-fetch` never downloads. With neither flag, models are fetched only when the command runs in an interactive terminal.

`--release FILE` stays in the project folder; only `release_attached` and its sha256 are recorded. Without `--consent-method`, a person's consent is `file` when `--release` is given, else `verbal`. `--consent-method verbal` with `--release` is refused (`lora.err.consent_method`), and `--consent-method file` without it too (`lora.err.no_release`).

`--adult-content` (person only, needs `--subject-adult`, else `lora.err.adult_content`) marks that the photos contain adult content. It is written as `voxprint.lora.adult_content`.

## Safety check (always on)

Every photo of every kind goes through the safety check (`voxprint_forge.lora.gate`, settings in [Gate config](spec/gate-config.md)). A person set needs `--consent-obtained` and `--subject-adult`. A photo that does not pass is set aside as `lora.drop.safety`; some cases block the project (`lora.err.blocked`). If a check model is not installed, nothing runs (`lora.err.gate_missing`). MDL ids: `qwen3-vl-4b-instruct`, `mivolo-v2`, `yunet-2023mar`.

## Captions

Qwen3-VL-4B-Instruct (`qwen3-vl-4b-instruct`, Apache-2.0) by default, Florence-2-large (`florence2-large`, MIT) as the fallback. `prepare` runs in phases: the safety check runs over every photo with only the judge and the age estimator loaded, both are unloaded, and only then is the captioner loaded to describe the kept photos and unloaded again before training. Photos are decoded one at a time in each phase, so a large set does not sit in memory as decoded pictures. The captions state the attire as it appears, in neutral words: clothing that is worn, or that the subject is unclothed, nude or topless. The caption text is stored as the captioner wrote it. `unclothed` in `captions.jsonl` is true when the captioner answered yes or its caption says unclothed, nude, naked or undressed. Edit captions with `--captions FILE` (JSONL `id`, `long`, `short`) on `prepare` or `train`.

## Bases

The base is chosen by MDL id (`--base`, default `z-image`, registry id `z-image`). MDL installs it in `<models dir>/model-forge/<id>`. The default weights are Z-Image. A project or LoRA with no base recorded stays Z-Image. The stored short tag is `zimg6b` (`zimgturbo` for Turbo, `moss5b` for MOSS).

| MDL id | Source | Licence | State |
| --- | --- | --- | --- |
| `z-image` | Tongyi-MAI/Z-Image | Apache-2.0 | Default (`--base z-image`, Tongyi-MAI/Z-Image@04cc4abb). A Z-Image LoRA is accepted on Z-Image-Turbo; strength and likeness on Turbo are unmeasured until a VM check. A file with no base recorded is still this base. The 16 GB preset stores the DiT as FP8; the 24 GB preset keeps bf16. |
| `z-image-turbo` | Tongyi-MAI/Z-Image-Turbo | Apache-2.0 | Optional (`--base z-image-turbo`, Tongyi-MAI/Z-Image-Turbo@f332072a). Trains only with a frozen de-distillation training adapter (`--training-adapter`, default `ostris/zimage_turbo_training_adapter` @ `654cd1bf8b3589d9442cb1c5e35221879e1af4f3`, file `zimage_turbo_training_adapter_v2.safetensors`). The adapter is loaded at scale 1.0 during training and is left out of checkpoints, export, previews and inference, so the LoRA runs on plain Turbo. `trained_base` is `z-image-turbo@f332072a`; `compatible_bases` is `["z-image-turbo"]` only. The 16 GB preset stores the DiT as FP8; the 24 GB preset keeps bf16. Official FP32 Turbo shards are not left in VRAM. |
| `moss-image-base` | OpenMOSS-Team/MOSS-Image-SFT | Apache-2.0 | Optional (`--base moss`, variant `sft`). fp32 LoRA. The 24 GB preset keeps the bf16 DiT. The 16 GB preset stores the DiT as FP8 e4m3 weights (torchao weight-only) and computes in bf16. |

A new base is a `BaseAdapter` subclass registered with `voxprint_forge.lora.bases.register`. The CLI does not change.

## Progress

Each step sends an event `{step, message_key, params, reason_key, percent, eta_seconds, state}` with `state` in `running|paused|done|failed|blocked`. The CLI prints it as an English sentence. With `--json` every event is one line `{"event": {...}}`, and the usual result object follows at the end. The time left appears after two warm-up steps.

During a stage the engine also writes `<project>/train.json` (`step`, `total`, `loss`, `elapsed_seconds`, `eta_seconds`, `stage`), one line in `train.log`, and one row in `vram.csv` (`elapsed_seconds`, `allocated_gb`, `reserved_gb`). A write happens at least every 60 seconds and whenever the stage name changes. Each write is flushed and fsynced, so a crash keeps the last report. `lora status` and the train step read `train.json`.

During round 1, `train.json` also carries `round`, `seconds_per_step`, `first_pictures_seconds`, `seconds_per_sample` and `sample_seconds_estimated`. The first-pictures time is the steps still to run times the measured seconds per step, plus the pictures still to draw times the seconds per picture. Until one picture has been timed, each denoising step is estimated as half a training step (`sample_seconds_estimated` is true and `seconds_per_sample` is omitted). After that, the measured seconds per picture are stored. The sentence is localized in the GUI and in `lora status`; `train.json` keeps the numbers. Later rounds and a train-fully stage do not write these fields. The final `stage` of `review` omits them.

All strings are in `src/voxprint_forge/lora/i18n/en.json`. The UI is English only; Russian stays a content language (captions, prompts, the safety check).

## Likeness score

`lora eval` and `lora finish` score how close generated pictures are to the training photos. It is a quality score only: it never refuses, never blocks export, never filters a picture, and is not part of the safety check.

- Models: OpenCV YuNet (`yunet-2023mar`) finds the largest face and SFace (`sface-2021dec`, Apache-2.0) embeds it. Both come from MDL (`voxprint-forge-cli models fetch --model sface-2021dec --model yunet-2023mar`). The embedding runs in a `voxprint_common.worker` process when that module and OpenCV are available, otherwise in a subprocess under a Python that has OpenCV (this env, else the MiVOLO env).
- Reference: the mean of the L2-normalised embeddings of the kept training photos (`prepared/`). `ref_baseline` is each photo against the mean of the others, averaged.
- Per picture: cosine against the reference. A picture with no face counts in `no_face` and is never a pass. `mean`, `median` and `n` cover the pictures with a face.
- Status: `fail` when the mean is below 0.363 (SFace same-person line); `warn` below 0.50, below 0.6 x `ref_baseline`, or when no picture has a face; else `pass`. `skipped` when the score could not run (for example, the models are not installed). A `warn` or `fail` prints one line to stderr; nothing else changes.
- Files: `<project>/eval/likeness.jsonl` (one row per picture: `group`, `image`, `cos`, `face`, `face_px`, `pass`), `eval/report.json` and `eval/report.md` (the summary per step), and `likeness` in `project.json`.

`lora finish` scores its final previews automatically, before writing the `.voxlora`. Without the face models it records `skipped` and goes on.

## Previews

Without `--preview-prompts`, training previews use one prompt that shows the subject clearly: for a person, `<trigger> woman, head-and-shoulders portrait, face clearly visible, looking at camera` (the noun comes from the captions, else `person`); for an object, a centred view on a plain background; for a style, a simple scene in that style. `lora finish` draws that prompt first, then the review prompts, unless the run used its own `--preview-prompts`. Review samples keep `--sample-framing` and `--sample-attire`.

Preview PNGs in `previews/` carry text chunks: `voxprint.seed`, `voxprint.lora_scale` (1 for previews), `voxprint.base_model` (MDL id), `voxprint.lora_sha256` (sha256 of the step's checkpoint file) and `voxprint.step`. Pictures copied into a `.voxlora` are re-encoded without metadata.

## Project folder

A project folder has a fixed layout. Every path is built through the `Project` properties in `src/voxprint_forge/lora/project.py` (`project.prepared`, `project.checkpoints`, `project.previews`, `project.review`, `project.eval`, `project.export`, `project.release`, `project.refset`, `project.train_state`, `project.train_log`, `project.vram_log`, ...), so the on-disk names below are stable and a project created by an older version opens unchanged. Paths stored inside files (an item's `file` in `captions.jsonl`, a review sample's `file`, the review `checkpoint`) are relative to the project root; `Project.resolve()` turns them back into absolute paths and passes absolute paths through.

| Entry | Content |
| --- | --- |
| `project.json` | Subject, settings, gate record, `train` and `likeness` records (including `sample_attire` when a review run has stored it) |
| `prepared/` | The cropped training photos, one PNG per photo id |
| `captions.jsonl` | One line per prepared photo: `id`, `file`, `long`, `short`, `round`, and `unclothed` when the captioner set it |
| `drops.jsonl` | Photos set aside, one line per photo id, with a `lora.drop.*` reason key |
| `gate_trace.jsonl` | One line per checked photo: the safety gate's codes and numbers, no image data |
| `checkpoints/` | `step-NNNNNN.safetensors` intermediate versions plus `state.pt` training state |
| `previews/` | Test pictures drawn during training (`step-NNNNNN-I.png`) |
| `review/` | `state.json` and `round-NNNN/*.png` samples while a review is in progress |
| `eval/` | Likeness score rows and `report.md` |
| `export/` | Finished `.voxlora` packages and `.safetensors` exports |
| `release/` | An attached consent release file |
| `refset/` | Reference-set rounds: `state.json`, `prompts.json`, `round-N/`, `approved/`, `metadata.json` |
| `train.json` | Latest training progress: step, total, loss, elapsed, ETA |
| `train.log` | One progress line per report |
| `vram.csv` | VRAM samples on the same cadence as `train.json` |
| `PAUSE` | While this file exists, training saves a version and stops |

Folders are created on demand. Sample pictures stay in the project folder and carry no extra metadata. A train with `--review-rounds 0` does not write `review/`. Omitting the flag uses the default of 3. Original source photos are not copied into the project; `lora prepare` reads them from where they are and keeps only the cropped copies in `prepared/`.

### Safety-check log

`drops.jsonl` stays neutral: a safety drop is only `lora.drop.safety`. A failing check stage prints one generic line to stderr (stage and exception type).

Each checked photo gets one tier (constants in `lora/gate.py`). **strong**: the judge says `child_or_adolescent` and MiVOLO's youngest face is under 18, or MiVOLO is under 16 and the judge under 18, or any other under-18 reading the rules below do not cover. **ambiguous**: one model says minor while the other says 20 or older, a minor reading averaging in [18, 20), one model under 16 with the other 20 or older, or judge confidence under 0.5. **margin**: no minor reading, but not both models at 20 or older (the 18-20 adult margin). **unchecked**: a check could not run on the photo. **clear**: both models 20 or older, no markers. Only clear photos are trained on. The job, over the n checked photos: a child keyword in a filename, photo metadata, a caption, the name, stage name, trigger or provenance note refuses it at any ratio. Otherwise, when at least 90 % are clear, every other photo is a false flag and is dropped. Otherwise any strong photo refuses, and more ambiguous photos than the allowance refuse (1 up to 30 photos, 10 from 60, `1 + round((n - 30) * 9 / 30)` between, half rounding up). Margin and unchecked photos are dropped and never refuse by themselves. A judge or estimator that is missing, or errors on any photo, still refuses the whole job (`lora.err.check_unavailable`, fail closed), and the photo-count minimums apply to what is left.

`gate_trace.jsonl`, beside it, has one line per photo the safety check ran on: `file` (name only), `id`, `judge` (`ok` or the error type), `reason_code`, the `markers` that were `present`, `judge_min_age`, `judge_confidence`, `mivolo_ages`, the v4 `estimate`, the `tier`, `decision` (`keep`, `drop` or `minor` for a strong photo) and the gate `reasons`. Face fields: `face_rule`, `face_why`, `faces` (`i`, `box`, `area`, `score`, `age`), `primary`, `stray`. It holds no image data, paths or metadata. Each row is appended and synced as soon as that photo is decided, so a crash keeps the rows already written. It is written before a refusal, so a refused run can still be measured. In Expert mode, each dropped or minor-flagged photo also prints one `safety gate:` line to stderr with the tier and the same numbers, and one `safety gate job:` line gives the job verdict and the tier counts.

A safety drop is appended to `drops.jsonl` the same way. When prepare finishes, that file is compacted so each set-aside photo still has one line. Rows from a previous run are not duplicated. `gate_run.json` is written before the first photo (`run_id`, `rule_version`, `gate_sha` as the sha256 of `gate.json` or empty until the pin exists, `started`, `state` `running`) and set to `complete` at the end. Training does not start while that state is `running` or `interrupted`. Prepare has to be run again. A photo with no trace row was not checked and is not treated as kept. That block does not set the photo aside.
