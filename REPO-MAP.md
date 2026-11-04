# REPO-MAP

Map of the main files, generated 2026-11-04 from a `grep -n "^def \|^class "` scan of each file (the quoted purpose is the module docstring). Files were not read end to end. Tests never download models: `tests/conftest.py` fails any test that reaches for real weights unless `VOXPRINT_REAL_MODELS=1` is set. `src/voxprint_forge/lora/gate.py` is listed without detail, by request.

## src/voxprint_forge — core forge library (builds quantized LLMs)
- `cli.py`, `__main__.py` — the `forge` command line; `--json` on stdout — `build_parser`, `main`.
- `plan.py` — loads `plans/plan.toml` (`Plan`, `ModelPlan`, `Tier`, `PinnedFile`, `load_plan`); `pipeline.py`, `pipeline_state.py`, `schedule.py` — per-model stages and the GPU slot gate — `Host`, `retry_stage`, `clear_from_stage`, `Gate`.
- `abliterate.py` — refusal-direction ablation (difference of means) — `select_direction`, `apply_orthogonalization`, `AblationResult`.
- `evaluate.py`, `requant_check.py`, `g4r_rerun.py` — refusal/ppl probes, re-quant and G4R verification — `refusal_rate`, `perplexity`, `RemoteTensor`, `Stock`.
- `llama.py`, `llama_patch.py` — llama.cpp b11476 command lines (runner injected so tests need no binary); T-lite pre-tokenizer patch — `Runner`, `tool`, `apply_t_lite`.
- `fitmath.py`, `measure.py`, `placement.py` — pick a quant that fits the card budget; `fetch.py`, `calibration.py` — pinned downloads, checksum-enforced — `fits_budget`, `fitting_candidates`, `ensure_pinned`, `ensure_calibration`.
- `upload.py`, `drive.py` — Hugging Face / Kaggle uploads, Drive fallback — `Artifact`, `Remote`, `upload_drive`; `accounts.py`, `keystore.py`, `scrub.py` — OS-keyring tokens and log secret-scrubbing — `SecretStore`, `KeyringStore`, `scrub`.
- `mdl.py`, `mdl_adapter.py`, `mdl_app.py`, `model_views.py`, `app_models.py`, `components.py` — `voxprint-model/1` manifests, the shared blob store, the app's required-model list — `build_document`, `MdlAdapterError`, `ensure_view`, `AppModel`, `resolve_model_component`.
- `cards.py`, `cardenv.py`, `manifest.py`, `hashing.py`, `licence.py` — model-card rendering, run manifest + MODELS.md, sha256 sidecars, licence gate — `hub_title`, `what_changed`, `write_run_manifest`, `sha256_file`, `assert_repo_allowed`.
- `errors.py`, `outcome.py`, `jsonio.py`, `progress.py`, `statusfile.py`, `layout.py`, `disk.py`, `reclaim.py`, `version.py` — stable exit codes, `~/forge-out` layout and markers, progress/status files, free space and cleanup — `ForgeError`, `Layout`, `Progress`, `ensure_free`, `reclaim_model_work`.
- `preflight.py`, `vmcheck.py`, `vram_probe.py`, `model_park.py`, `plural.py`, `cli_reference.py` — seconds-long pre-run checks, fresh-VM probe, VRAM readings, weight residency, i18n plurals, CLI docs generator — `missing_imports`, `Probe`, `tier_gb`, `park`, `ru_plural`.

## src/voxprint_forge/lora — LoRA trainer (photos in, trained LoRA out)
- `cli.py` — `forge lora prepare|train|extend|export|review|continue|finish|status|eval` — `add_command`.
- `prepare.py` — the `lora prepare` step — `prepare`, `open_photo`.
- `photos.py`, `image_check.py`, `image_job.py` — photo import/filters and the dataset quality check — `collect`, `ImageChecker`, `measure_check`.
- `train.py`, `presets.py`, `layers.py`, `memory.py`, `vram_stage.py`, `progress_log.py` — training loop, presets, LoRA layers, VRAM plan, live progress on disk — `estimate`, `TrainConfig`, `LoRALinear`, `MemoryPlan`, `ProgressWriter`.
- `project.py`, `refset.py`, `review.py`, `export.py` — project folder, 1–10-photo reference sets, review rounds, safetensors export with metadata — `Project`, `Item`, `check_settings`, `can_continue`, `header_metadata`.
- `captioner.py`, `backends.py`, `bases.py` — Qwen3-VL / Florence captioners, judge workers, base-model adapters — `Captioner`, `parse_verdict`, `BaseAdapter`, `register`.
- `consent.py`, `attr.py`, `job_gate.py`, `refusal.py` — subject/consent checks and why simple mode set a photo aside — `Subject`, `check_subject`, `Clearance`, `Report`.
- `likeness.py`, `likeness_worker.py`, `mivolo_worker.py` — SFace likeness score vs training photos, MiVOLO age worker in its own env; `sample_prompts.py`, `trigger.py`, `te_quant.py`, `events.py`, `errors.py`, `gate_run.py`, `limits.py`, `i18n/` — review prompts, rare triggers, text-encoder plan, events, photo-count rules, crash-safe gate marker — `cosine`, `summarize`, `build_sample_prompts`, `rare_trigger`, `TePlan`, `open_gate_run`, `min_photos`.
- `gate.py` — photo gate (checks before training).

## src/voxprint_forge/diffusion — Canvas add-on (Z-Image tiers, ControlNet, FP8/INT4)
- `cli.py`, `job.py`, `planload.py`, `common.py` — `run PLAN`, parts in file order, stage markers — `execute`, `CanvasPlan`, `Ctx`, `run_stage`.
- `part_base.py`, `part_control.py` — part A (BF16/FP8 tiers), part B (ControlNet Union 2.1) — `_fetch`, `_convert`, `_quality`.
- `weights.py`, `fp8.py`, `fp8_native.py`, `moss_fp8.py` — BF16 casts, per-tensor FP8 e4m3 files; `moss.py`, `moss_fit.py`, `moss_preflight.py` — MOSS-Image loader (vendored pipeline, no `trust_remote_code`) and its VRAM/speed fit — `index_tensors`, `quantize_weight`, `Fp8Linear`, `MossRepo`, `verify_vendored`.
- `policy.py`, `generate.py`, `scenes.py`, `prompts.py` — publish gates (LPIPS/CLIP), quality renders, synthetic CC0-style scenes; `cardrender.py`, `registry.py`, `names.py` — canvas model cards and MODELS.md rows — `decide_base`, `render_scene`, `render_canvas_card`, `render_models_md`.
- `publish.py`, `release.py`, `defer.py`, `retain.py`, `deadline.py`, `sources.py` — uploads, deferred-upload release, source retention, pinned downloads; `controlnet_link.py`, `controlnet_union.py`, `nunchaku_zimage.py`, `int4/` — ControlNet swap, nunchaku INT4 stage — `release_canvas`, `defer_part`, `download_source`, `swap_controlnet`, `union_config`.
- `alias.py`, `spec.py`, `guard.py`, `stage_free.py`, `smoke_laptop.py`, `smoke_prefetch.py` — small helpers: key aliasing, LLM-bootstrap guard, eval-stage memory release, laptop smoke and prefetch — `cmdline_is_job`, `parse_steps`, `call_with_retry`.

## src/voxprint_forge/kl and src/voxprint_forge/compare — measurement harnesses (synth4 queue)
- `kl/` — perplexity/KL stage vs a reference: `cli.py` (`forge kl`), `plan.py` (`KlPlan`), `run.py` (`run_kl`), `parse.py` (`parse_kl`), `arch.py` (MTP checks), `kv.py` (KV-cache arms), `preflight.py` (SASS check), `recover.py` (rebuild F16 reference).
- `compare/` — original-vs-compressed image compare: `cli.py`, `plan.py` (`ComparePlan`), `run.py` (`run_compare`), `engine.py` (generation), `score.py`/`metrics.py` (LPIPS/CLIP/DINOv2, SSIM/PSNR), `cases.py`, `manifests.py`, `results.py`, `vram.py`, `attention.py`, `cache.py`, `fp8_make.py`, `int4.py`, `privacy.py` (prompt text stays private), `codes.py`, `reclaim.py`.

## src/voxprint_forge/app, gui, ui — desktop front ends
- `app/` — Model Forge window on the shared Yitro shell (Qt); buttons call the CLI, stages are not reimplemented — `window.py` (window), `engine.py` (`EngineEvent`, `Outcome`), `runner.py` (`ProcessRunner`), `draft.py` (`Draft`, headless state), `settings.py`, `image_checks.py`, `thumbs.py` (`ThumbService`), `flow.py`, `details.py`, `basepicker.py`, `branding.py`, `text.py` (`tr`), `__main__.py`.
- `gui/` — thinner PySide6 window over `forge --json`, local or over SSH; `ui/` — `check_gate.py`, a red/green dataset gate widget mirroring `voxprint_common.ui.check_gate` (`GateState`, `CheckItem`) — `window.py` (`MainWindow`), `client.py` (`SubprocessForge`, `forge_args`), `gauges.py`, `icons.py`, `backdrop.py` (Windows 11 Acrylic), `secrets.py`.

## scripts/ — repo maintenance and VM job runners
- `bench_prepare.py`, `bench_prepare_io.py` — peak RSS / wall time of `lora prepare` on synthetic photos, CPU stand-ins, no downloads — `make_photos`, `peak_rss_mb`.
- `build_guide.py`, `build_pdf_guide.py` — render `docs/manual/` into the offline guide and the PDF (WeasyPrint) — `render`, `build`.
- `forge_app_shots.py`, `render_ui_screens.py`, `make_icon.py`, `logscrub.py`, `publish_moss_fp8.py` — screenshots of every app step offscreen; draw the app icon; strip tokens from a log stream; MOSS FP8 model card + upload steps.
- `gen_app_models.py` — regenerate `manifests/model-forge.models.json` from the Hugging Face API — `app_document`.
- `release_assets.py` — collect installer + guide PDFs from a tag's CI run and publish the GitHub release — `choose_run`, `write_sha256sums`.
- `*.sh` / `*.ps1` (`vm_run_*.sh`, `canvas_*.sh`, `bootstrap.sh`, `vm_setup_synth4.sh`, `windows_smoke.ps1`, …) — run the pipelines on rented GPU VMs and the Windows smoke.

## tests/ — shared helpers (the rest are `test_*.py` suites)
- `conftest.py` — offscreen Qt, per-suite temp home, blocks real model downloads unless `VOXPRINT_REAL_MODELS=1` — `REAL_MODELS`, `pytest_collection_modifyitems`.
- `forge_app_stub.py` — procedural photos plus a stub engine runner for app tests — `make_photos`, `write_prepared`, `fake_image_checks`, `StubRunner`.
- `image_worker_fake.py` — stand-in image-check handlers for the worker process — `load_model`, `check`.
- `lora_fakes.py` — tiny CPU stand-ins for LoRA tests (random 2-layer model, procedural images) — `TinyNet`, `TinyAdapter`, `gradient`, `make_project`.

## installer/ — Windows packaging
- `stamp_build.py` — record build number / rc into the wheel and installer payload — `resolve_build`, `rc_from_tag`, `frozen_document`.
- `windows/` — Inno Setup script (`voxprint-model-forge.iss`), `install.ps1`, `launcher.cs`, `migrate-home.ps1`, extra requirements; `requirements-mivolo.txt`, `requirements-pdf.txt` at the root.

## .github/workflows/
- `ci.yml` — on push/PR: vendor pin check, test matrix across OSes, model-index check.
- `pin-check.yml` — reusable `voxprint-common` pin check (`workflow_call`).
- `installer.yml` — builds the PDF guide and the Windows installer.
- `docs.yml` + `release.yml` — deploy the MkDocs site; publish a GitHub release from a tag, with a release-identity check.
