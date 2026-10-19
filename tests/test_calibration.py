"""The imatrix corpus must be the pinned gist and long enough for llama-imatrix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from transformers import AutoTokenizer
from voxprint_common.download import DownloadError as CommonDownloadError
from voxprint_common.download import download as shared_download

from voxprint_forge.calibration import (
    CALIBRATION_RAW_URL,
    CALIBRATION_SHA256,
    CALIBRATION_TOKENS,
    CALIBRATION_URL,
    LICENCE_STATUS,
    assert_calibration,
    ensure_calibration,
    tokens_required,
)
from voxprint_forge.errors import ChecksumError
from voxprint_forge.evaluate import data_dir
from voxprint_forge.hashing import sha256_file
from voxprint_forge.plan import load_plan

ROOT = Path(__file__).resolve().parents[1]


def _bash(script: Path) -> list[str]:
    """Git bash on Windows. System32\\bash.exe is a stub and is not used."""
    exe = "bash"
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        exe = str(git_bash) if git_bash.is_file() else "bash"
    return [exe, script.as_posix()]


def test_pinned_calibration_meets_the_llama_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "src/voxprint_forge/data/imatrix.txt"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert tracked.stdout.strip() == ""
    empty = tmp_path / "nodata"
    empty.mkdir()
    monkeypatch.setattr("voxprint_forge.calibration.data_dir", lambda: empty)
    monkeypatch.setenv("FORGE_OUT", str(tmp_path))
    calls = {"n": 0}

    def _counting(destination: Path, sources: list[str] | tuple[str, ...], **kwargs: object) -> Path:
        calls["n"] += 1
        assert list(sources) == [CALIBRATION_RAW_URL]
        assert kwargs["sha256"] == CALIBRATION_SHA256
        return shared_download(destination, sources, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("voxprint_forge.calibration.download", _counting)
    path = ensure_calibration()
    assert calls["n"] == 1
    assert path == tmp_path / "calibration" / "imatrix.txt"
    assert sha256_file(path) == CALIBRATION_SHA256
    assert ensure_calibration() == path
    assert calls["n"] == 1
    assert tokens_required(512, 1) == 1024
    assert tokens_required(512, 128) == 65536
    plan = load_plan(ROOT / "plans" / "plan.toml")
    assert plan.imatrix_chunks == 128
    assert plan.imatrix_ctx == 512
    assert assert_calibration(plan) == CALIBRATION_TOKENS
    tokens = AutoTokenizer.from_pretrained("gpt2").encode(path.read_text(encoding="utf-8"))
    assert len(tokens) == CALIBRATION_TOKENS
    assert CALIBRATION_TOKENS >= tokens_required(plan.imatrix_ctx, plan.imatrix_chunks)
    assert "gist.github.com/bartowski1182" in CALIBRATION_URL
    assert LICENCE_STATUS == "licence unverified, calibration-only, not redistributed in model files"
    docs = (ROOT / "docs" / "VM.md").read_text(encoding="utf-8")
    lessons = (ROOT / "docs" / "LESSONS.md").read_text(encoding="utf-8")
    for text in (docs, lessons):
        assert CALIBRATION_SHA256 in text
        assert CALIBRATION_URL in text
        assert LICENCE_STATUS in text
        assert "GitHub Bot" in text
    assert CALIBRATION_RAW_URL in docs
    assert "not stored in this git repository" in docs
    assert (data_dir() / "imatrix-ru-pd.txt").is_file()
    assert "FORGE_IMATRIX_RU" in docs
    assert "public domain" in (data_dir() / "imatrix-ru-pd.txt").read_text(encoding="utf-8").lower()
    local = tmp_path / "pkgdata"
    local.mkdir()
    pinned = local / "imatrix.txt"
    pinned.write_bytes(path.read_bytes())
    monkeypatch.setattr("voxprint_forge.calibration.data_dir", lambda: local)
    monkeypatch.setenv("FORGE_OUT", str(tmp_path / "other"))
    assert ensure_calibration() == pinned
    assert calls["n"] == 1


def test_calibration_download_rejects_a_mismatched_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "nodata"
    empty.mkdir()
    monkeypatch.setattr("voxprint_forge.calibration.data_dir", lambda: empty)
    monkeypatch.setenv("FORGE_OUT", str(tmp_path))
    def _bad(destination: Path, sources: list[str] | tuple[str, ...], **kwargs: object) -> Path:
        del kwargs
        raise CommonDownloadError(
            f"all sources failed for {destination.name}: {sources[0]}: abc does not match {CALIBRATION_SHA256}"
        )

    monkeypatch.setattr("voxprint_forge.calibration.download", _bad)
    with pytest.raises(ChecksumError):
        ensure_calibration()
    assert not (tmp_path / "calibration" / "imatrix.txt").exists()
    assert not (tmp_path / "calibration" / "imatrix.txt.partial").exists()


def _bootstrap_missing_token(env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        _bash(ROOT / "scripts/bootstrap.sh"),
        input=b"",
        capture_output=True,
        env=env,
        # Git bash on a busy Windows runner can take longer than a few seconds to start.
        timeout=30,
        check=False,
    )


def test_bootstrap_refuses_a_missing_token_when_stdin_is_not_a_tty(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_OUT"] = tmp_path.as_posix()
    proc = _bootstrap_missing_token(env)
    assert proc.returncode == 2
    assert b"non-interactive" in proc.stderr
    forced = dict(env)
    forced.pop("CI", None)
    forced["FORGE_NONINTERACTIVE"] = "1"
    forced_proc = _bootstrap_missing_token(forced)
    assert forced_proc.returncode == 2
    assert b"non-interactive" in forced_proc.stderr
