# Lean testing

Cheap checks come first. A GPU run is a few hard cases, and it stops when the
answer is already clear. A person reviews a handful of full-resolution labelled
images.

## Order

1. **Plain Linux, no GPU.** CLI help, `--dry-run`, and `--json` contracts. Pytest
   on Ubuntu and Windows is this tier. It mocks remotes and does not load a
   model onto a device.
2. **A few hard cases.** Image checks use the lean lists in
   `voxprint_forge.diffusion.prompts` (fine detail, a clothed adult, thin
   structure, one of each control type). One seed. The full prompt catalogues
   stay in the file and are not the job.
3. **Stop when the interval decides.** After a minimum count, a 95% interval
   that lies entirely inside a gate, or entirely outside it, ends that
   component. Refusal rate uses a Wilson interval around one half. LPIPS and
   CLIP use a Student-t interval on the per-sample scores. A clear failure of
   one metric stops the component. The recorded mean is the samples that ran.
4. **A review pack, not a thumbnail wall.** The job writes up to four
   full-resolution reference and candidate PNGs plus `labels.md` under
   `checks/review-<name>/`.

## Caps

| Check | Hard cases | Minimum before a stop |
| --- | --- | --- |
| Base image | 8 prompts, seed 0 | 4 |
| Control | 6 typed cases, seed 0, one of each type first | 3 |
| Inpaint | 4 prompts | 2 |
| Edit | 6 pairs, seed 0 | 4 |
| LLM refusal | plan cap, default 4 prompts per language | 4 |
| Perplexity and prose | 1 chunk, 1 sample | already one shot |

`~/forge-out/EDIT_TRIM` can only shrink the edit list. It does not turn the
lean list back into the 24-pair catalogue.

## Eager disk freeing

These checks stay on the CPU. Remotes are mocked. Nothing in this tier needs a GPU.

`tests/test_diffusion_int4.py` is the same tier for `forge diffusion-int4`. It runs `--dry-run` with tiny Wan files, mocks Hub and Drive, and does not download weights or start CUDA.

After a model's or a part's final files match on one mirror, delete that item's
source weights, f16/bf16 GGUF, safetensors downloads, and upload staging. A
Hugging Face match is the remote sha256, or for a small file the size plus the
hash of the downloaded bytes. A shared source stays while any later name in
its `consumers` list has not released.

A disk check does not wait while those leftovers exist. It lists
verified-uploaded sources, finished staging, stale partials that have no
`.sha256`, and orphaned Hugging Face cache blobs, deletes each one, logs the
path and the size, then checks free space again. Deleting a snapshot directory
only removes symlinks. The bytes go when
`scan_cache_dir().delete_revisions(...).execute()` runs, or when a blob that
no snapshot still references is removed.

## Windows installer smoke

`.github/workflows/installer.yml` builds `installer/windows/voxprint-model-forge.iss`
with Inno Setup 6 on `windows-latest`, with the version from `BUILD.json`. It fails
if the staging folder or a bundled wheel holds model weights. Then it runs
`scripts/windows_smoke.ps1` against the new setup and uploads
`voxprint-model-forge-setup-<version>` (the exe) and `windows-smoke-results`.

The `pdf-guide` job (Ubuntu) builds `VOXPrint-Model-Forge-Guide-en.pdf` and `-ru.pdf` from
`docs/manual` with `scripts/build_pdf_guide.py` (WeasyPrint, embedded DejaVu fonts, pins in
`installer/requirements-pdf.txt`), checks them with `tests/test_pdf_guide.py` (more than 10 pages,
screenshots, extractable Cyrillic) and uploads them as `voxprint-model-forge-guide-pdf`. The Windows
job puts them into the setup (`{app}\docs`). The app opens the guide from its menu.

`scripts/windows_smoke.ps1` is the same checklist for a real PC. Run it from an
elevated PowerShell 7 (`pwsh`) next to the downloaded setup. Windows PowerShell 5.1
also works, but PowerShell 7 is recommended:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\windows_smoke.ps1 -Setup .\VoxprintModelForgeSetup.exe -RequireGpu
```

The setup runs its PowerShell steps hidden with PowerShell 7 when it is installed.
If it is missing, an interactive setup offers to download and install the pinned
official MSI (sha256 checked); if you decline or it fails, the setup continues with
Windows PowerShell 5.1. A silent setup (`/VERYSILENT`) never prompts. If the app is still
running, a silent setup exits 20 before it deletes or copies any file, and the previous
install keeps working. The installer
workflow downloads the pinned MSI and checks its sha256 and Microsoft signature.

| Step | Passes when |
|---|---|
| home-plant | CI only (`-MigrationTest`, clean profile): a fake `%LOCALAPPDATA%\Voxprint\models\smoke-dummy\smoke-dummy-model.bin` |
| install | the silent setup exits 0; both venvs, the CLI, the compiled launcher and its icon, the PDF guide (en, ru) in `docs\`, one Start menu launcher (target, icon, AppUserModelID; no Uninstall or guide shortcut; no desktop shortcut; old shortcuts removed) and the uninstall entry (name, icon, publisher) exist; no weights |
| upgrade-plant | `voxprint_common-0.4.0-py3-none-any.whl` and `voxprint_forge-0.0.9-py3-none-any.whl` are planted in `{app}\wheels` before setup |
| upgrade-wheels | those planted wheels are gone, and the app venv's voxprint-common version is the pin recorded in the app wheel |
| upgrade-busy | a process holds `voxprint-model-forge.exe`; the silent upgrade exits 20 and leaves the previous install runnable (`--help` exits 0, the same wheels are present) |
| te-quant | `import bitsandbytes` in the app venv reports version 0.50.2 |
| version | the installed `voxprint-forge-cli --version`, with the checkout off `PATH` and without `GITHUB_RUN_NUMBER`, prints the build number recorded in the installed wheel |
| preflight | `preflight --refset` reports diffusers, img2img and the MiVOLO env OK (and CUDA with `-RequireGpu`); missing models are listed only |
| lora-dry-run | `lora train --dry-run --base z-image --vram 24` on a 12-picture stub project returns `dry-run` |
| home-migration | the setup moved `%LOCALAPPDATA%\Voxprint` to `%LOCALAPPDATA%\VOXPrint AI Suite` before any Python ran: the old folder is gone and the dummy is under `paths.models_dir()` with the same sha256 |
| reinstall-noop | with `-MigrationTest`: a second silent install exits 0, logs `no-op` and leaves the data home unchanged |
| uninstall | the program folder, shortcut and uninstall entry are gone; `%LOCALAPPDATA%\VOXPrint AI Suite` stays |

Results go to `%LOCALAPPDATA%\VoxprintForgeSmoke\windows\<timestamp>\result.json`.
It downloads no models and needs no token. Exit 0 means every step passed.
