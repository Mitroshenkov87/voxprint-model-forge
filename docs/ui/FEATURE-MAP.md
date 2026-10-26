# Model Forge GUI feature map

The user app `voxprint-model-forge` (`voxprint_forge.app`) runs on the shared Yitro shell, voxprint-common v0.5.2 (merge commit c3a74eaa), surface `model-forge`. It is the LoRA trainer from the plan (project-notes `model-forge/LORA-TRAINER-GUI-PLAN.md`, 7b824cf with the decisions in f2104ea and the v1 ship list a0e8c0a). Training steps run `voxprint-forge-cli lora ... --json`. The Photos step checks images itself, on a background worker, and does not load a model until Check images is pressed.

The pre-Yitro operator window (`forge-gui`, `voxprint_forge.gui`) is retired from the user app. It stays in the tree for internal VM work and is not shipped. Its features are CLI-only for users.

Status is `mapped`, `CLI-only`, `operator-only` (internal mirror work, not a user feature), `waits-engine` (the control calls a `lora` command this build does not have yet and says so in plain words), or `dropped`.

## LoRA trainer (CLI `lora`)

| Feature | CLI | Where it lives now | Status |
| --- | --- | --- | --- |
| Photos add/remove, folders, drag and drop | `--photo`, `--photos`, `--exclude` | Train a model → Photos | mapped |
| Photo limits | Simple 10-50; `--allow-count-outside-range` (floor 5) | Photos: count gate plus Check images; Expert mode warns outside 10-50 | mapped |
| What it is | `--kind person\|object\|style` | Photos: chips | mapped |
| Name, Stage name, Trigger | `--name`, `--stage-name`, `--trigger` | Subject and consent | mapped |
| Consent and 18+ (person) | `--consent-obtained`, `--consent-method verbal\|file`, `--subject-adult` | Subject and consent: Consent obtained with Verbal on its right (or attach a release), 18+ | mapped |
| Adult content (person, 18+) | `--adult-content` | Subject and consent: right of 18+, shown only while 18+ is ticked | mapped |
| Rights (object, style) | `--rights-confirmed` | Subject and consent: one required box per kind | mapped |
| Model release, provenance note | `--release`, `--provenance-note` | Subject and consent | mapped |
| Crop | `--crop subject\|whole` | Check and prepare: chips | mapped |
| Prepare (filter, safety gate, crop, captions) | `lora prepare` | Check and prepare: Check photos | mapped |
| Safety drops | `drops.jsonl` | Check and prepare: Set aside list; no override anywhere | mapped |
| Caption edit | `--captions FILE` | Check and prepare: review table (edits go to `--captions`) | mapped |
| Caption model | `--caption-model` | Expert panel (Check and prepare) | mapped |
| Blur / duplicate thresholds | `--min-side`, `--blur-threshold`, `--dup-threshold` | — (engine defaults; GUI control is a follow-up) | CLI-only |
| Preset, likeness, memory | `--preset`, `--likeness`, `--vram` | Train: chips; values are live on the Expert panel | mapped |
| Base model | `--base` (MDL id) | Train: combo from the engine base registry | mapped |
| Test pictures | `--previews/--no-previews`, `--preview-every` | Train: checkbox; Expert panel | mapped |
| Preview prompts | `--preview-prompts FILE` | — | CLI-only |
| Rank, alpha, steps, lr, scheduler, warm-up, batch, grad accum, resolution, precision, checkpointing, text encoder, caption mode, caption dropout, seed, save every | Expert flags | Expert panel (Train) | mapped |
| Replay share | `--replay` | Expert panel (Train), sent for extend | mapped |
| Estimate time | `--dry-run` | Train: Estimate time | mapped |
| Train / Pause / Continue / Stop | `lora train`, `<project>/PAUSE` file, `--resume` | Train: buttons, plain-words progress and time left | mapped |
| Live progress | `lora status`, `<project>/train.json` | Train: the step line and bar read `train.json` while a job runs | mapped |
| Review rounds | `lora train --review-rounds --first-round-steps --first-round-samples --samples-per-round --sample-steps --sample-attire`, `lora review`, `lora continue`, `lora continue --unattended`, `lora finish`, `lora finish --discard` | Train: review rounds (1–6, default 3) and first-round steps (default 120) under Details; review screen with Continue with reinforcement (▾ switches to Just continue; the choice is `continue_mode` on the project), Sample pictures (`sample_attire`, applied from the next round), and Finish (Save, or Force, without saving) when **Extend the set from samples** is on | mapped |
| Extend | `lora extend --from` | Result: Add more photos; overflow: Add more photos to a model | mapped |
| Checkpoint pick | `--checkpoint N` | Result: Version | mapped |
| Export LoRA | `lora export PROJECT -o` | Result: Export LoRA | mapped |
| Save | `lora export --project --out` | Result: Save; overflow: Export | mapped |
| Open in Canvas | `lora export --install-to-mdl` | Result: Open in Canvas | mapped |
| Preview strength slider | — (no engine call for previews at a strength yet) | — | waits-engine |
| Open project | project folder | overflow: Open project | mapped |
| User guide | docs/manual (rendered by `scripts/build_guide.py` into `app/guide/`) | overflow: Open guide, offline, in the app language | mapped |
| Manage models | `voxprint-mdl --gui --app model-forge --manifest {app}\\mdl.json` | overflow: Manage models; Check this computer | mapped |
| Check and repair models | `voxprint-mdl-gui --app model-forge` | Check this computer | mapped |
| Graphics card, memory, disk, models | `preflight`, `disk`, MDL `ready` | Check this computer | mapped |

## Operator window (`forge-gui`, retired from the user app)

| Feature | Where it was | Where it lives now | Status |
| --- | --- | --- | --- |
| Check before run | operator window button | `forge preflight`; for users, Check this computer | mapped |
| Per-model status table | operator window table | `forge status` | CLI-only |
| Retry from a stage | operator window table | `forge-gui` (internal operator window, library `pipeline.retry_stage`) | operator-only |
| Upload targets (HF, Kaggle, Drive) and tokens | operator window | `forge accounts`, `forge upload`, `forge verify-logins` | CLI-only |
| Attach to VM, upload on VM | operator window | VM scripts in `scripts/` | CLI-only |
| VRAM, RAM and disk gauges | operator window | Check this computer | mapped |
| About / version | operator window dialog | `forge --version` | CLI-only |

## Shell notes (v0.2.5 through v0.4.1)

The v0.2.5 wizard draws only text and choice fields for a step and its Start button records a placeholder run. Model Forge therefore fills each step body itself through `ForgeWidget._parts()` (one place, covered by `tests/test_app_window.py::test_shell_parts_this_app_extends_exist`). The Expert panel shows a float with one decimal, so learning rate, caption dropout and replay share are lists of values.
