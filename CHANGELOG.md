# Changelog

## Unreleased

- `lora finish`: fixed "Expected all tensors to be on the same device, cuda:0 vs cpu" in the final preview. The preview loaded the base and the LoRA but never moved the transformer to the card (the adapters keep it on the CPU until the encoders are released), so the VAE and the prompt embeddings were on `cuda:0` and the transformer with its LoRA on the CPU. The final preview now follows the training order (encoders, prompts, encoders off, transformer and LoRA onto the card) and checks that the model is on one device first. `lora finish` also writes the `.voxlora` to `<project>/export/`. `--no-preview` skips the preview. A preview that fails prints `lora.warn.final_preview` and still finishes and exports the LoRA.
- `lora train` / `lora continue`: when a run pauses for review and stdin is not a terminal, a hint on stderr lists `lora review`, `lora continue` and `lora finish`, and `--review-rounds 0` for unattended runs. The GUI is unchanged.
- `--sample-framing {mixed,portrait}` on `lora train` and `lora continue` (stored as `sample_framing`). `portrait` draws every review sample as a face close-up for judging likeness. The default `mixed` is the old rotation. `lora continue` also takes `--sample-attire`.
- Docs and help: use `--preset balanced` for a person LoRA; `quick` is a smoke test (RTX 4090 person run: quick 120 steps 1.68/5 likeness, balanced 1000 steps 3.95/5). Only the exported `.voxlora` is the deliverable; `checkpoints/step-*.safetensors` are intermediate and Canvas does not accept them.
- `lora prepare`: `drops.jsonl` has one line per set-aside photo. A re-run on the same project no longer appends a second copy of earlier lines.
- Windows setup: adds `{app}\venv\Scripts` (`voxprint-forge-cli.exe`) to the current user's `Path` once, and the uninstaller removes only that entry. The old RU guide shortcut is now removed by an ASCII pattern.
- `preflight` and the other plan commands work on an installed copy without a checkout: the default plan ships in the package (`voxprint_forge/data/plan.toml`, kept equal to `plans/plan.toml` by a test). A checkout's `plans/plan.toml` and an explicit `--plan` still take precedence.
- `lora prepare --help` explains `--name` and `--consent-method` and shows an example. A missing `--name` prints one line with an example, `--name Garnis_AI --consent-method verbal` (still exit 2). docs/LORA.md documents `--name`, `--consent-method` and `--fetch-missing`.

## 0.1.0

- Windows setup: one Start menu launcher (`VOXPrint AI Suite\VOXPrint AI Model Forge`), AppUserModelID `VOXPrint.AISuite.ModelForge`, and the shared app icon.
