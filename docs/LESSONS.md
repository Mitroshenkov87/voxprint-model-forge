# Lessons

## 2026-10-10 A100 field fixes

The rental lost about 40 minutes to three startup bugs.

- `scripts/vm_run.sh` put `--plan`, `--root`, and `--forge-out` after `run-all`. argparse on this CLI only accepts those flags before the subcommand, so the operator command never reached the job.
- `bootstrap.sh` prompted for `HF_USER` and for missing tokens whenever `/dev/tty` was readable. A non-interactive stdin (and a tmux pty with nobody attached) blocked instead of exiting. `HF_USER` now defaults to `Mitroshenkov87`. Missing tokens exit 2 when stdin is not a terminal.
- `imatrix.txt` was about 184 tokens. llama-imatrix at the pinned commit errors when the file tokenises to fewer than `2×ctx` tokens, and it silently runs fewer chunks than requested when the file is shorter than `ctx×chunks`. The corpus is bartowski's calibration_datav3 (74048 GPT-2 tokens, sha256 `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`). The gist has no explicit licence: licence unverified, calibration-only, not redistributed in model files. Flagged for GitHub Bot review. Source: https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8. The bytes are fetched at runtime from the pinned raw URL and are not committed.
- `os.kill(pid, 0)` is not a liveness probe on Windows. `CTRL_C_EVENT` is 0, so the call raises KeyboardInterrupt in the pytest process. The lock check uses `OpenProcess` there and still uses signal 0 on Linux.
- The pipeline dry-run plan declares a 50 GiB reuse so the fit stage skips it. `ensure_free` still asks for that declared size plus the 5 GiB margin, and windows-latest only has about 28 GiB free. Those tests report a large disk. The production margin and the A100 reservation are unchanged.
- Plan `imatrix_chunks` is 128. Bootstrap preflight checks cmake, ninja, and nvcc on `PATH`, and that the calibration token count covers `max(2×ctx, ctx×chunks)`, before the llama.cpp build.
- An optional Russian slice lives in `src/voxprint_forge/data/imatrix-ru-pd.txt` (Pushkin, 1829, public domain). It is off unless `FORGE_IMATRIX_RU=1`.

## 2026-10-10 quant ladder, shard index, and the T-lite pre-tokenizer

- Gemma 4 31B on a 16 GB card: the tier budget is 13 GB. IQ3_M at 3.716 bits per weight was 14.4 GB and does not fit. The ladder continues through IQ3_XXS (3.0) and IQ2_M (2.7).
- `t-lite-it-2.1` and `t-pro-it-2.1` fetch `model.safetensors.index.json`. Without that sidecar the sharded safetensors download has no map.
- Qwen text-encoder checkpoints use either `model.layers...` or `layers...`. `diffusion/generate.py` aliases those prefixes. The bakeoff that compares encoders is `bakeoff/run_bakeoff.py` in voxprint-canvas, not in this repo.
- Canvas base and controlnet `min_free_gb` is 40 when the sources are already on disk. Edit stays at 100. The run-level floor stays 120.
- `FORGE_PYTHON` is the forge interpreter. `convert_hf_to_gguf.py` uses it, and falls back to `sys.executable` when the variable is unset.
- llama.cpp b11476 does not recognise the T-lite Qwen2 pre-tokenizer. Bootstrap patches `conversion/base.py` at build time. The digest of the pinned probe is `e9b7dbd66e0308c6e89983d5b6e1ca047106d862879a0fd33a12c8491b91ec5c` and the converter name is `qwen2`. A check that Hugging Face `encode` ids equal `llama-tokenize` ids belongs next to that patch; the ids are the input to the digest.
- A failed model used to abort the rest and could still be followed by `ALL-DONE`. The summary is now one line per model (`READY`, `UPLOADED`, `STAGEFAIL <stage>`, `DEFERRED`) and `ALL-DONE` is printed only when every model finished. `forge status` is what Hardware Watchdog polls. Long stages write `~/forge-out/status.json` at least every 60 seconds.
- Private Hub or Kaggle failure falls back to a Drive folder `Voxprint/model-forge/<date>/<model>` via rclone. The token stays in the environment.

## Dry run before a bundle

Every future bundle or overlay, including canvas part D, ships only after an end-to-end CPU dry run is green. The dry run uses the exact operator command from the docs, a tiny model, and a mocked remote, and it walks the real stages: fetch, quant or convert, fit, eval, card, and upload. Canvas A–C uses `python -m voxprint_forge.diffusion run plans/canvas-zimage-tiers-v1.toml --json --forge-out <dir> --root <repo>`. Part D uses `python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml --json --forge-out <dir> --root <repo>`. The LLM job uses `forge --plan <plan> --root <repo> --forge-out <dir> run-all`.
