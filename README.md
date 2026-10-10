# voxprint-model-forge

Internal tooling for preparing models used by the Voxprint AI Media Suite: abliteration, quantization (GGUF/imatrix), evaluation, model cards and checksums. Not part of the end-user apps.

## Product vision

Voxprint AI Model Forge (Aleksandr, 2026-10-10) is a standalone desktop application, not part of the end-user suite and not embedded in any app. It runs on the user's own RTX 40+ machine and prepares models: tracks new upstream releases, abliterates (removes refusals), quantizes (GGUF/imatrix) to fit VRAM targets such as 16 GB with headroom, evaluates (quality, refusal rate, speed, peak VRAM), and publishes pinned model cards and checksums that suite apps download. Interfaces: a GUI for manual use, and a full CLI with --json and documented exit codes from day one, because a bot connects to it as a client; a plugin for Grok Bot (and a shareable bot template) is planned. Repo: Mitroshenkov87/voxprint-model-forge (private for now, Apache-2.0). Owner: Voxprint AI Model Forge (took over from Voxprint AI Plotweaver on 2026-10-10). Its own GitHub Pages page comes later with the rest. Suite rules apply (Apache-2.0, local-first, opt-in telemetry, forward-only hardware). Model Forge uses its own Apache-2.0 refusal-direction ablation; Heretic (AGPL-3.0) is never bundled in or invoked by Model Forge.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party models keep their own licences; record both code and weights licences in each model card.

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
