# voxprint-model-forge

Internal tooling for preparing models used by the Voxprint AI Media Suite: abliteration, quantization (GGUF/imatrix), evaluation, model cards and checksums. Not part of the end-user apps.

## Product vision

VOXPrint AI Model Forge (Aleksandr, 2026-10-10) is a standalone desktop application, not part of the end-user suite and not embedded in any app. It runs on the user's own RTX 40+ machine and prepares models: tracks new upstream releases, abliterates (removes refusals), quantizes (GGUF/imatrix) to fit VRAM targets such as 16 GB with headroom, evaluates (quality, refusal rate, speed, peak VRAM), and publishes pinned model cards and checksums that suite apps download. Interfaces: a GUI for manual use, and a full CLI with --json and documented exit codes from day one, because a bot connects to it as a client; a plugin for Grok Bot (and a shareable bot template) is planned. Repo: Mitroshenkov87/voxprint-model-forge (private for now, Apache-2.0). Owner: VOXPrint AI Model Forge (took over from Voxprint AI Plotweaver on 2026-10-10). Suite rules apply (Apache-2.0, local-first, opt-in telemetry, forward-only hardware). Model Forge uses its own Apache-2.0 refusal-direction ablation; Heretic (AGPL-3.0) is never bundled in or invoked by Model Forge.

## Install names

The display name is VOXPrint AI Model Forge. Shortcuts use that name inside the Start menu folder VOXPrint AI Suite. Windows installs under `C:\Program Files\VOXPrint AI Suite\Model Forge\`. The GUI binary is `voxprint-model-forge.exe` and the CLI binary is `voxprint-forge-cli.exe`. Linux installs under `/opt/voxprint-ai-suite/model-forge/`. Per-user data lives in `%LOCALAPPDATA%\VOXPrint AI Suite` and `~/.local/share/voxprint-ai-suite` (voxprint-common 0.2.6); models, and the TensorRT cache (`<home>/trt-cache`), are always under `voxprint_common.paths.models_dir()`. The Windows setup moves a legacy `%LOCALAPPDATA%\Voxprint` there once, before any Python runs. An upgrade moves an older program folder into this install location, updates shortcuts and PATH, and leaves the user data in place. The package name stays `voxprint-forge`. Repo slugs and URLs stay lower-case. The in-tree commands remain `forge` and `forge-gui` (the internal operator window, not shipped). The GUI binary runs `voxprint_forge.app`, the LoRA trainer on the shared Yitro shell (voxprint-common v0.2.6).

## Roadmap

The public roadmap is [ROADMAP.md](ROADMAP.md). The API docs, the CLI reference, and that roadmap are published at <https://mitroshenkov87.github.io/voxprint-model-forge/>.

Wan 2.2 GGUF and FP8 is `forge diffusion-int4`. The minute tables are `forge diffusion-int4 estimates`.

## Testing

Checks follow [docs/TESTING.md](docs/TESTING.md). CI is the cheap tier: CLI, dry runs, and JSON contracts on plain Linux and Windows, with remotes mocked. GPU evals use a few hard cases and stop when the confidence interval decides the gate. A person reviews a few full-resolution labelled frames.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party models keep their own licences; record both code and weights licences in each model card.

The models we publish (compressed and fine-tuned) and the official builds are also subject to the [Acceptable Use Policy](ACCEPTABLE_USE.md), in addition to each model's upstream licence. The code remains Apache-2.0.

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
