# LoRA trainer (engine and CLI)

`forge lora ...` (also `voxprint-forge-cli lora ...`) trains a LoRA from 10 to 30 photos of a person, an object or a style. The GUI comes later and drives the same engine. Plan: project-notes `model-forge/LORA-TRAINER-GUI-PLAN.md`.

## Commands

| Command | What it does |
| --- | --- |
| `lora train --project DIR` | Trains on the prepared photos of the project. `--dry-run` prints the plan and the time estimate. |
| `lora extend --project DIR --from FILE` | Continues an existing LoRA with the newest photos of the project plus `--replay` share (default 0.25) of the older ones. A LoRA made on another base is refused (`lora.err.base_mismatch`). |

Training refuses to start unless the project was prepared and passed the safety check (`lora.err.gate_missing`). It needs at least 10 prepared photos, or 5 in Expert mode.

Presets: `--preset quick|balanced|best`, `--likeness flexible|balanced|close`, `--vram auto|16|24`. The values are not final until the GPU calibration run. Expert flags (`--rank`, `--alpha`, `--steps`, `--lr`, `--scheduler`, `--warmup`, `--batch`, `--grad-accum`, `--resolution`, `--precision`, `--grad-checkpointing`, `--caption-mode`, `--caption-dropout`, `--seed`, `--save-every`, `--preview-every`) override single values. There are no safety settings.

Pause with Ctrl+C: the run saves a version and stops. Continue with `--resume`; the saved settings are reused.

## Bases

The base is chosen by MDL id (`--base`, default `z-image`). MDL installs it in `<models dir>/model-forge/<id>`.

| MDL id | Source | Licence | State |
| --- | --- | --- | --- |
| `z-image` | Tongyi-MAI/Z-Image | Apache-2.0 | Default. LoRAs load on Z-Image and Z-Image-Turbo in Canvas. |
| `moss-image-base` | OpenMOSS-Team/MOSS-Image-Base | Apache-2.0 | Registered; the training loader lands after GPU calibration. |

A new base is a `BaseAdapter` subclass registered with `voxprint_forge.lora.bases.register`. The CLI does not change.

## Progress

Each step sends an event `{step, message_key, params, reason_key, percent, eta_seconds, state}` with `state` in `running|paused|done|failed|blocked`. The CLI prints it as a sentence in English or Russian (`VOXPRINT_LANG=ru` or a Russian locale). With `--json` every event is one line `{"event": {...}}`, and the usual result object follows at the end. The time left appears after two warm-up steps.

All strings are in `src/voxprint_forge/lora/i18n/en.json` and `ru.json`; a test keeps their keys and placeholders identical.

## Project folder

`project.json`, `prepared/`, `captions.jsonl` (`id`, `file`, `long`, `short`, `round`), `drops.jsonl`, `checkpoints/step-NNNNNN.safetensors` plus `state.pt`, `previews/`.
