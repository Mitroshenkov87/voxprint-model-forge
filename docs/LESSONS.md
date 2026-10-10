# Lessons

## 2026-10-10 A100 field fixes

The rental lost about 40 minutes to three startup bugs.

- `scripts/vm_run.sh` put `--plan`, `--root`, and `--forge-out` after `run-all`. argparse on this CLI only accepts those flags before the subcommand, so the operator command never reached the job.
- `bootstrap.sh` prompted for `HF_USER` and for missing tokens whenever `/dev/tty` was readable. A non-interactive stdin (and a tmux pty with nobody attached) blocked instead of exiting. `HF_USER` now defaults to `Mitroshenkov87`. Missing tokens exit 2 when stdin is not a terminal.
- `imatrix.txt` was about 184 tokens. llama-imatrix at the pinned commit errors when the file tokenises to fewer than `2×ctx` tokens, and it silently runs fewer chunks than requested when the file is shorter than `ctx×chunks`. The corpus is bartowski's calibration_datav3 (74048 GPT-2 tokens, sha256 `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`). The gist has no explicit licence: licence unverified, calibration-only, not redistributed in model files. Flagged for GitHub Bot review. Source: https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8. The bytes are fetched at runtime from the pinned raw URL and are not committed.
- `os.kill(pid, 0)` is not a liveness probe on Windows. `CTRL_C_EVENT` is 0, so the call raises KeyboardInterrupt in the pytest process. The lock check uses `OpenProcess` there and still uses signal 0 on Linux.
- Plan `imatrix_chunks` is 128. Bootstrap preflight checks cmake, ninja, and nvcc on `PATH`, and that the calibration token count covers `max(2×ctx, ctx×chunks)`, before the llama.cpp build.
- An optional Russian slice lives in `src/voxprint_forge/data/imatrix-ru-pd.txt` (Pushkin, 1829, public domain). It is off unless `FORGE_IMATRIX_RU=1`.

## Dry run before a bundle

Every future bundle or overlay, including canvas part D, ships only after an end-to-end CPU dry run is green. The dry run uses the exact operator command from the docs, a tiny model, and a mocked remote, and it walks the real stages: fetch, quant or convert, fit, eval, card, and upload. Canvas uses `python -m voxprint_forge.diffusion run plans/canvas-zimage-tiers-v1.toml --json --forge-out <dir> --root <repo>`. The LLM job uses `forge --plan <plan> --root <repo> --forge-out <dir> run-all`.
