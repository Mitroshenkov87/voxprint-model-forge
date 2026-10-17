# LoRA trainer (engine and CLI)

`forge lora ...` (also `voxprint-forge-cli lora ...`) trains a LoRA from 10 to 30 photos of a person, an object or a style. The GUI comes later and drives the same engine. Plan: project-notes `model-forge/LORA-TRAINER-GUI-PLAN.md`.

## Commands

| Command | What it does |
| --- | --- |
| `lora prepare --project DIR --photos DIR --kind KIND ...` | Checks, crops and describes the photos. Photos that are unreadable, too small, blurry, near-duplicate or fail the safety check are listed in `drops.jsonl` with a reason key. Running it again on the same project adds the new photos as the next round (for `extend`). |
| `lora train --project DIR` | Trains on the prepared photos of the project. `--dry-run` prints the plan and the time estimate. |
| `lora extend --project DIR --from FILE` | Continues an existing LoRA with the newest photos of the project plus `--replay` share (default 0.25) of the older ones. A LoRA made on another base is refused (`lora.err.base_mismatch`). |
| `lora export --project DIR` | Saves the last version (or `--checkpoint N`) as `<id>.safetensors` with `voxprint-model.json`, LICENSE and NOTICE (`--out FILE` to choose the place). `--install-to-mdl` also adds it to your models for Canvas. |

Training refuses to start unless the project was prepared and passed the safety check (`lora.err.gate_missing`). It needs at least 10 prepared photos, or 5 in Expert mode.

Presets: `--preset quick|balanced|best`, `--likeness flexible|balanced|close`, `--vram auto|16|24`. The values are not final until the GPU calibration run. Expert flags (`--rank`, `--alpha`, `--steps`, `--lr`, `--scheduler`, `--warmup`, `--batch`, `--grad-accum`, `--resolution`, `--precision`, `--grad-checkpointing`, `--caption-mode`, `--caption-dropout`, `--seed`, `--save-every`, `--preview-every`) override single values. There are no safety settings.

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

`--install-to-mdl` writes `<id>.safetensors`, `voxprint-model.json`, LICENSE, NOTICE and the preview to `$VOXPRINT_USER_MODELS_DIR/lora/<id>/`, by default `<models dir>/user-models/lora/<id>/`. The id is the trigger word plus 8 hex digits of the LoRA tensors.

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

The minors ban and the age check run on every photo of every kind, whatever the checkboxes say. There is no setting and no override. A visible face (or an unsure answer) needs both judges to read every person as 20 or older: the VLM judge (Qwen3-VL-4B-Instruct, Apache-2.0) and the age estimator (MiVOLO v2, Apache-2.0, on YuNet face crops). Any age cue answered present or unsure, a low confidence, a malformed answer or a judge error sets the photo aside as `lora.drop.safety`. A clear minor signal blocks the whole project (`lora.err.blocked`). If a judge is not installed, nothing runs (`lora.err.gate_missing`). MDL ids: `qwen3-vl-4b-instruct`, `mivolo-v2`, `yunet-2023mar`.

## Captions

Qwen3-VL-4B-Instruct (`qwen3-vl-4b-instruct`, Apache-2.0) by default, Florence-2-large (`florence2-large`, MIT) as the fallback. The captioner is loaded, describes every photo, and is unloaded before training. Edit captions with `--captions FILE` (JSONL `id`, `long`, `short`) on `prepare` or `train`.

## Bases

The base is chosen by MDL id (`--base`, default `z-image`). MDL installs it in `<models dir>/model-forge/<id>`.

| MDL id | Source | Licence | State |
| --- | --- | --- | --- |
| `z-image` | Tongyi-MAI/Z-Image | Apache-2.0 | Default. LoRAs are for Z-Image; loading on Z-Image-Turbo is untested until a Canvas test. |
| `moss-image-base` | OpenMOSS-Team/MOSS-Image-Base | Apache-2.0 | Registered; the training loader lands after GPU calibration. |

A new base is a `BaseAdapter` subclass registered with `voxprint_forge.lora.bases.register`. The CLI does not change.

## Progress

Each step sends an event `{step, message_key, params, reason_key, percent, eta_seconds, state}` with `state` in `running|paused|done|failed|blocked`. The CLI prints it as a sentence in English or Russian (`VOXPRINT_LANG=ru` or a Russian locale). With `--json` every event is one line `{"event": {...}}`, and the usual result object follows at the end. The time left appears after two warm-up steps.

All strings are in `src/voxprint_forge/lora/i18n/en.json` and `ru.json`; a test keeps their keys and placeholders identical.

## Project folder

`project.json`, `prepared/`, `captions.jsonl` (`id`, `file`, `long`, `short`, `round`), `drops.jsonl`, `checkpoints/step-NNNNNN.safetensors` plus `state.pt`, `previews/`.
