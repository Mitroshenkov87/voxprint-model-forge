"""UI modules must not import ML runtimes.

A fresh interpreter imports each UI module and fails when torch, transformers,
diffusers, onnxruntime, peft, accelerate, nunchaku, bitsandbytes, or llama_cpp
is in ``sys.modules``. The app window still imports ImageChecker, which imports
torch while the module loads; that case is expected to fail until PR4. Gauges
and ``preflight.memory_status`` do not import those libraries at import time
(the CUDA read stays inside ``preflight._probe_cuda`` until PR2), so the
operator window is held to the boundary now.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from voxprint_forge.errors import UsageError
from voxprint_forge.layout import Layout, write_marker
from voxprint_forge.pipeline_state import ORDER, clear_from_stage, retry_stage

ROOT = Path(__file__).resolve().parents[1]
_UI = ROOT / "src" / "voxprint_forge" / "ui"
BANNED = (
    "torch",
    "transformers",
    "diffusers",
    "onnxruntime",
    "peft",
    "accelerate",
    "nunchaku",
    "bitsandbytes",
    "llama_cpp",
)
_HELPER = (
    "import importlib\n"
    "import sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "sys.stdout.write('MODULES\\n')\n"
    "sys.stdout.write('\\n'.join(sorted(sys.modules)))\n"
    "sys.stdout.write('\\n')\n"
)


def _ui_module_names() -> list[str]:
    """``voxprint_forge.ui`` and every submodule on disk."""
    names = ["voxprint_forge.ui"]
    for path in sorted(_UI.rglob("*.py")):
        relative = path.relative_to(_UI)
        if path.name == "__init__.py":
            if relative.parent == Path("."):
                continue
            parts = relative.parent.parts
        else:
            parts = relative.with_suffix("").parts
        names.append("voxprint_forge.ui." + ".".join(parts))
    return names


def _boundary_targets() -> list[object]:
    targets: list[object] = [
        pytest.param(
            "voxprint_forge.app.window",
            marks=pytest.mark.xfail(
                reason=(
                    "PR4: the app window still imports ImageChecker, and "
                    "lora.image_check imports torch while that module loads"
                ),
            ),
        ),
        "voxprint_forge.gui.window",
        "voxprint_forge.pipeline_state",
    ]
    targets.extend(_ui_module_names())
    return targets


def _modules_after_import(module: str) -> set[str]:
    """Module names present after importing ``module`` in a subprocess."""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", _HELPER, module],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import\n{proc.stderr}"
    marker = proc.stdout.rfind("MODULES\n")
    assert marker >= 0, proc.stdout
    body = proc.stdout[marker + len("MODULES\n"):]
    return {line.strip() for line in body.splitlines() if line.strip()}


def _present(name: str, modules: set[str]) -> bool:
    prefix = name + "."
    return name in modules or any(item.startswith(prefix) for item in modules)


def _banned(modules: set[str]) -> list[str]:
    return [name for name in BANNED if _present(name, modules)]


def test_import_boundary_lists_the_ui_package() -> None:
    """The boundary covers the real ``ui`` package, not a hard-coded guess."""
    names = _ui_module_names()
    assert names[0] == "voxprint_forge.ui"
    assert "voxprint_forge.ui.check_gate" in names


@pytest.mark.parametrize("module", _boundary_targets())
def test_ui_import_boundary(module: str) -> None:
    """Importing one UI module leaves every ML runtime out of ``sys.modules``."""
    modules = _modules_after_import(module)
    leaked = _banned(modules)
    assert not leaked, f"{module} imported {leaked}"
    if module == "voxprint_forge.gui.window":
        assert "voxprint_forge.abliterate" not in modules
        assert "voxprint_forge.pipeline" not in modules


def test_pipeline_reexports_stage_helpers() -> None:
    """Callers that still import the helpers from ``pipeline`` get the same objects."""
    from voxprint_forge import pipeline

    assert pipeline.ORDER is ORDER
    assert pipeline.retry_stage is retry_stage
    assert pipeline.clear_from_stage is clear_from_stage


def test_retry_and_clear_keep_their_stage_rules(tmp_path: Path) -> None:
    """Restart commands and marker deletion match the old pipeline helpers."""
    assert retry_stage("") == "fetch"
    assert retry_stage("start") == "fetch"
    assert retry_stage("imatrix") == "quantize"
    assert retry_stage("eval") == "eval"
    with pytest.raises(UsageError):
        retry_stage("no-such-stage")

    layout = Layout(tmp_path)
    model = "gemma4-31b"
    extras = ("imatrix", "upload-deferred", "skipped")
    for name in (*ORDER, *extras):
        write_marker(layout.stage(model, name), {"ok": True})
    removed = clear_from_stage(layout, model, "eval")
    assert layout.stage(model, "fetch").is_file()
    assert layout.stage(model, "quantize").is_file()
    assert layout.stage(model, "imatrix").is_file()
    assert not layout.stage(model, "eval").is_file()
    assert not layout.stage(model, "card").is_file()
    assert not layout.stage(model, "upload").is_file()
    assert not layout.stage(model, "upload-deferred").is_file()
    assert not layout.stage(model, "skipped").is_file()
    assert removed[0] == "eval"
    assert "imatrix" not in removed

    for name in (*ORDER, *extras):
        write_marker(layout.stage(model, name), {"ok": True})
    clear_from_stage(layout, model, "imatrix")
    assert layout.stage(model, "fetch").is_file()
    assert layout.stage(model, "abliterate").is_file()
    assert not layout.stage(model, "quantize").is_file()
    assert not layout.stage(model, "imatrix").is_file()
    assert not layout.stage(model, "upload").is_file()
