"""The imatrix corpus must be the pinned gist and long enough for llama-imatrix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from voxprint_forge.calibration import (
    CALIBRATION_SHA256,
    CALIBRATION_TOKENS,
    CALIBRATION_URL,
    LICENCE_STATUS,
    assert_calibration,
    tokens_required,
)
from voxprint_forge.evaluate import data_dir
from voxprint_forge.hashing import sha256_file
from voxprint_forge.plan import load_plan

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_calibration_meets_the_llama_requirement() -> None:
    path = data_dir() / "imatrix.txt"
    assert sha256_file(path) == CALIBRATION_SHA256
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
    assert (data_dir() / "imatrix-ru-pd.txt").is_file()
    assert "FORGE_IMATRIX_RU" in docs
    assert "public domain" in (data_dir() / "imatrix-ru-pd.txt").read_text(encoding="utf-8").lower()


def test_bootstrap_refuses_a_missing_token_when_stdin_is_not_a_tty(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_OUT"] = str(tmp_path)
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/bootstrap.sh")],
        input=b"",
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )
    assert proc.returncode == 2
    assert b"non-interactive" in proc.stderr
