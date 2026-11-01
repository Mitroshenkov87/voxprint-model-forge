# Changelog

## Unreleased

- `lora prepare`: `drops.jsonl` has one line per set-aside photo. A re-run on the same project no longer appends a second copy of earlier lines.
- Windows setup: adds `{app}\venv\Scripts` (`voxprint-forge-cli.exe`) to the current user's `Path` once, and the uninstaller removes only that entry. The old RU guide shortcut is now removed by an ASCII pattern.
- `preflight` and the other plan commands work on an installed copy without a checkout: the default plan ships in the package (`voxprint_forge/data/plan.toml`, kept equal to `plans/plan.toml` by a test). A checkout's `plans/plan.toml` and an explicit `--plan` still take precedence.
- `lora prepare --help` explains `--name` and `--consent-method` and shows an example. A missing `--name` prints one line with an example, `--name Garnis_AI --consent-method verbal` (still exit 2). docs/LORA.md documents `--name`, `--consent-method` and `--fetch-missing`.

## 0.1.0

- Windows setup: one Start menu launcher (`VOXPrint AI Suite\VOXPrint AI Model Forge`), AppUserModelID `VOXPrint.AISuite.ModelForge`, and the shared app icon.
