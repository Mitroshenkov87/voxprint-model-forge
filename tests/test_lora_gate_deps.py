"""The LoRA safety-gate deps are core deps: a fresh install must not fail closed on them."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIN_OPENCV = "opencv-python-headless==5.0.0.93"
PIN_TIMM = "timm==0.8.13.dev0"


def test_lora_safety_gate_deps_are_core_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert PIN_OPENCV in project["dependencies"]
    assert PIN_TIMM in project["dependencies"]
    for group in project["optional-dependencies"].values():
        assert PIN_OPENCV not in group
        assert PIN_TIMM not in group
