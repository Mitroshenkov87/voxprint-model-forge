# LoRA trainer (engine and CLI)

`forge lora ...` (also `voxprint-forge-cli lora ...`) trains a LoRA from 10 to 30 photos of a person, an object or a style. The GUI comes later and drives the same engine. Plan: project-notes `model-forge/LORA-TRAINER-GUI-PLAN.md`.

## Commands

| Command | What it does |
| --- | --- |
| `lora prepare --project DIR --photos DIR --kind KIND ...` | Checks, crops and describes the photos. Photos that are unreadable, too small, blurry, near-duplicate or fail the safety check are listed in `drops.jsonl` with a reason key. Running it again on the same project adds the new photos as the next round (for `extend`). |
| `lora train --project DIR` | Trains on the prepared photos of the project. `--dry-run` prints the plan and the time estimate. `--review-rounds N` (1–6, default 3; `0` trains straight through) pauses after each planned stage. The first stage is `--first-round-steps` (100–500, default 200). Later stages share the preset steps that remain; the last of them takes the remainder. The preset total does not change. `--first-round-samples` (default 6) and `--samples-per-round` (default 8) are the pictures at the first review and at later reviews. `--sample-steps` (default 28) is the inference step count for those pictures. |
| `lora extend --project DIR --from FILE` | Continues an existing LoRA with the newest photos of the project plus `--replay` share (default 0.25) of the older ones. A LoRA made on another base is refused (`lora.err.base_mismatch`). |
| `lora export --project DIR` | Saves the last version (or `--checkpoint N`, any saved version; a failed export changes nothing, so run it again without retraining) as `<id>.safetensors` with `voxprint-model.json`, LICENSE and NOTICE (`--out FILE` to choose the place). `--install-to-mdl` also adds it to your models for Canvas. |
| `lora refset round --project DIR --ref FILE ... --kind KIND ...` | Build a set from 1-10 references. The first round gates every reference (any that does not pass blocks the loop) and has the judge VLM write one prompt per reference (`refset/prompts.json`, local only). Then Z-Image-Turbo img2img (`ZImageImg2ImgPipeline`, BF16, model CPU offload) draws `--candidates` (8-64, default 24) pictures at `--strength` (0.25-0.90, default 0.5) and `--steps` (4-16, default 9); every candidate is gated before it is listed; ones that do not pass are deleted and a minor signal blocks the loop. DINOv2-base marks near-duplicates. Later rounds start from the kept candidates plus the references. Stages run one at a time with a full unload between them. |
| `lora refset finish --project DIR` | "This is it": copies the references and kept candidates into `refset/approved/` (10-30, or 5-60 with `--mode expert`) and records rounds, seeds, settings, the base and the teacher with its licence in `refset/metadata.json` and `project.json`. Then run `lora prepare` on `refset/approved/`. |
| `lora review PROJECT` | Lists the samples of a paused review, with the original photos. `--like ID` and `--dislike ID` mark samples (repeat either flag). `--from-file FILE` reads `like ID` / `dislike ID` lines, or a JSON object with `like`, `dislike` and `unmarked` lists. |
| `lora continue PROJECT` | Trains the next stage from the review checkpoint, on the original photos plus the liked samples. `--no-more-reviews` trains every remaining preset step in one stage and then finishes, with no further reviews. Disliked samples are negatives only when the base supports them. |
| `lora finish PROJECT` | Keeps the latest checkpoint and ends the review rounds. |
| `lora status PROJECT` | Prints the latest `train.json` progress. Adds the review phase and counts when a review is saved. During round 1 it also prints `first_pictures`. |

Training refuses to start unless the project was prepared and passed the safety check (`lora.err.gate_missing`). It needs at least 10 prepared photos, or 5 in Expert mode.

Presets: `--preset quick|balanced|best`, `--likeness flexible|balanced|close`, `--vram auto|16|24`. The values are not final until the GPU calibration run. Expert flags (`--rank`, `--alpha`, `--steps`, `--lr`, `--scheduler`, `--warmup`, `--batch`, `--grad-accum`, `--resolution`, `--precision`, `--grad-checkpointing`, `--caption-mode`, `--caption-dropout`, `--seed`, `--save-every`, `--preview-every`) override single values. There are no safety settings.

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

## Export and the Canvas hand-off

Header metadata (strings): `voxprint.subject.name`, `stage_name`, `trigger`, `consent_obtained`, `adult_confirmed`, `provenance_note`, `release_attached`, `release_sha256`, `consent_method` (`verbal` or `file`) for a person, and `rights_confirmed` for objects and styles; `voxprint.lora.base`, `base_revision`, `rank`, `alpha`, `steps`, `kind`, `trained_on`, `adult_content` (`true` or `false`, always present). Nothing else: no paths, usernames, machine ids or EXIF, and no key that reads like a gate or threshold setting.

`voxprint-model.json` (`schema: voxprint-model/1`): `min_schema` 1, `id`, `kind`, `type` lora, `producer`, `file`, `sha256`, `size`, `base.mdl_id`, `base.revision`, `licence` (from the base), `default_strength` 0.8, `strength_range` [0, 1.5], `tags`, `voxprint.lora.adult_content` (JSON `true` or `false`, always present), `voxprint.subject.consent_method` (`"verbal"` or `"file"`, only for a person), and `preview` with `preview_sha256` when a test picture exists (`preview.webp`).

`voxprint.lora.adult_content` is a label the user sets: the training photos contain adult content. Apps such as Canvas can use it to label, sort or filter LoRAs (for example, hide them unless the user opts in). It can be `true` only for a person confirmed as 18 or older. It is not a safety setting and changes no checks. `voxprint.subject.consent_method` says how a person's consent was given: `verbal` (no file, `release_sha256` empty) or `file` (a release is attached; only its sha256 is recorded). LoRAs exported before these fields have no `adult_content` key; treat that as `false`.

`--install-to-mdl` writes `<id>.safetensors`, `voxprint-model.json`, LICENSE, NOTICE and the preview to `<models dir>/user-models/lora/<id>/`, where the models dir is always `voxprint_common.paths.models_dir()` (`VOXPRINT_MODELS_DIR`, then `models_dir` in `suite.json`, then `<suite home>/models`). The id is the trigger word plus 8 hex digits of the LoRA tensors. Tensors keep the dtype they were trained in (BF16 by default; FP32 for an FP16 base); the id hashes their raw bytes.

## Photos and consent

Simple mode (default) takes 10 to 30 photos. Expert mode (`--mode expert` or `--allow-count-outside-range`) warns outside 10-30, needs at least 5 and takes at most 60.

| Kind | Required |
| --- | --- |
| `person` | `--name` or `--stage-name`, `--consent-obtained`, `--subject-adult`, and `--consent-method verbal` or `--consent-method file --release FILE` |
| `object` | `--name` or `--stage-name`, `--rights-confirmed` (I own this object or have the right to use these photos) |
| `style` | `--name` or `--stage-name`, `--rights-confirmed` (I have the right to use these images; the style is not presented as a named living artist's work) |

`--release FILE` stays in the project folder; only `release_attached` and its sha256 are recorded. Without `--consent-method`, a person's consent is `file` when `--release` is given, else `verbal`. `--consent-method verbal` with `--release` is refused (`lora.err.consent_method`), and `--consent-method file` without it too (`lora.err.no_release`).

`--adult-content` (person only, needs `--subject-adult`, else `lora.err.adult_content`) marks that the photos contain adult content. It is written as `voxprint.lora.adult_content`.

## Safety check (always on)

Every photo of every kind goes through the safety check (`voxprint_forge.lora.gate`, settings in [Gate config](spec/gate-config.md)). A person set needs `--consent-obtained` and `--subject-adult`. A photo that does not pass is set aside as `lora.drop.safety`; some cases block the project (`lora.err.blocked`). If a check model is not installed, nothing runs (`lora.err.gate_missing`). MDL ids: `qwen3-vl-4b-instruct`, `mivolo-v2`, `yunet-2023mar`.

## Captions

Qwen3-VL-4B-Instruct (`qwen3-vl-4b-instruct`, Apache-2.0) by default, Florence-2-large (`florence2-large`, MIT) as the fallback. The captioner is loaded, describes every photo, and is unloaded before training. Edit captions with `--captions FILE` (JSONL `id`, `long`, `short`) on `prepare` or `train`.

## Bases

The base is chosen by MDL id (`--base`, default `z-image`). MDL installs it in `<models dir>/model-forge/<id>`.

| MDL id | Source | Licence | State |
| --- | --- | --- | --- |
| `z-image` | Tongyi-MAI/Z-Image | Apache-2.0 | Default. LoRAs are for Z-Image; loading on Z-Image-Turbo is untested until a Canvas test. |
| `moss-image-base` | OpenMOSS-Team/MOSS-Image-Base | Apache-2.0 | Trains in bf16 with an fp32 LoRA. The 16 GB and 24 GB presets both keep the bf16 DiT. |

A new base is a `BaseAdapter` subclass registered with `voxprint_forge.lora.bases.register`. The CLI does not change.

## Progress

Each step sends an event `{step, message_key, params, reason_key, percent, eta_seconds, state}` with `state` in `running|paused|done|failed|blocked`. The CLI prints it as a sentence in English or Russian (`VOXPRINT_LANG=ru` or a Russian locale). With `--json` every event is one line `{"event": {...}}`, and the usual result object follows at the end. The time left appears after two warm-up steps.

During a stage the engine also writes `<project>/train.json` (`step`, `total`, `loss`, `elapsed_seconds`, `eta_seconds`, `stage`), one line in `train.log`, and one row in `vram.csv` (`elapsed_seconds`, `allocated_gb`, `reserved_gb`). A write happens at least every 60 seconds and whenever the stage name changes. Each write is flushed and fsynced, so a crash keeps the last report. `lora status` and the train step read `train.json`.

During round 1, `train.json` also carries `round`, `seconds_per_step`, `first_pictures_seconds`, `seconds_per_sample` and `sample_seconds_estimated`. The first-pictures time is the steps still to run times the measured seconds per step, plus the pictures still to draw times the seconds per picture. Until one picture has been timed, each denoising step is estimated as half a training step (`sample_seconds_estimated` is true and `seconds_per_sample` is omitted). After that, the measured seconds per picture are stored. The sentence is localized in the GUI and in `lora status`; `train.json` keeps the numbers. Later rounds and a train-fully stage do not write these fields. The final `stage` of `review` omits them.

All strings are in `src/voxprint_forge/lora/i18n/en.json` and `ru.json`; a test keeps their keys and placeholders identical.

## Project folder

`project.json`, `prepared/`, `captions.jsonl` (`id`, `file`, `long`, `short`, `round`), `drops.jsonl`, `checkpoints/step-NNNNNN.safetensors` plus `state.pt`, `previews/`, `review/state.json` and `review/round-NNNN/*.png` while a review is in progress. Sample pictures stay in the project folder and carry no extra metadata. A train with `--review-rounds 0` does not write `review/`. Omitting the flag uses the default of 3.

### Safety-check log

`drops.jsonl` stays neutral: a safety drop is only `lora.drop.safety`. A failing check stage prints one generic line to stderr (stage and exception type).
