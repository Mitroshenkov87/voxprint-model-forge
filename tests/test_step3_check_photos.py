"""Step 3 (Check photos): explicit photo files, readable failures, a random seed per run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from voxprint_forge.lora import cli as lora_cli
from voxprint_forge.lora.errors import LoraUsageError


def test_explicit_photo_files_skip_the_photos_root_rule(monkeypatch, tmp_path: Path) -> None:
    # An empty photos root would make the single-folder rule fail; named files must not care.
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "Документы"))
    args = argparse.Namespace(photos=None, photo=[tmp_path / "Папка с фото" / "a.jpg"])
    assert lora_cli._photos_dir(args) is None


def test_no_photos_at_all_still_uses_the_photos_root_rule(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "docs"))
    with pytest.raises(LoraUsageError):
        lora_cli._photos_dir(argparse.Namespace(photos=None, photo=[]))


def test_a_failure_with_a_parameterised_key_shows_the_engine_sentence() -> None:
    from voxprint_forge.app import engine
    from voxprint_forge.app.window import _failure, register_catalogs

    register_catalogs()

    outcome = engine.Outcome(False, None, "lora.err.photos_resolution", "No person folders were found.", 2)
    key, params, _ = _failure(outcome)
    assert key == "forge.app.err.unexpected"
    assert params == {"message": "No person folders were found."}
    plain = engine.Outcome(False, None, "lora.err.not_prepared", "Prepare first.", 2)
    assert _failure(plain)[0] == "lora.err.not_prepared"


def _train_args(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**{"seed": None, "dry_run": False, "resume": False, "json": True, **values})


def test_a_new_run_without_seed_gets_a_random_seed_and_reports_it(capsys) -> None:
    from voxprint_forge.lora.presets import build_config

    config = build_config("balanced", "balanced", 16)
    seeds = {lora_cli._run_seed(config, _train_args()).seed for _ in range(5)}
    assert len(seeds) > 1
    assert "lora.train.seed" in capsys.readouterr().out


def test_an_explicit_seed_a_resume_and_a_dry_run_keep_the_seed() -> None:
    from voxprint_forge.lora.presets import build_config

    config = build_config("balanced", "balanced", 16)
    for args in (_train_args(seed=7), _train_args(resume=True), _train_args(dry_run=True)):
        assert lora_cli._run_seed(config, args) is config
