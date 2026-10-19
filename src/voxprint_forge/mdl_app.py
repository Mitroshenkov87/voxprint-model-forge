"""Model Forge's MDL manifest and the ``models`` commands.

Every MDL invocation passes this app's schema 1 file with ``--manifest``.
Discovery of ``model-forge`` is not required.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from voxprint_common.mdl.catalog import VramTier, load_manifest, plan_files
from voxprint_common.mdl.job import detect_tier
from voxprint_common.mdl.names import CLI_NAME
from voxprint_common.paths import models_dir

from voxprint_forge.app_models import load as load_app_models
from voxprint_forge.errors import StageError, UsageError
from voxprint_forge.lora.backends import LEGACY_IDS
from voxprint_forge.model_views import APP_DIR, MARKER, ensure_view
from voxprint_forge.model_views import manifest_path as packaged_manifest

APP_ID = "model-forge"
FETCH_COMMAND = "voxprint-forge-cli models fetch --required"


def manifest_path() -> Path:
    """``{app}\\mdl.json`` in an installed build, otherwise the packaged ``data/mdl.json``."""
    installed = _installed_manifest()
    if installed is not None:
        return installed
    packaged = packaged_manifest()
    if packaged.is_file():
        return packaged
    return packaged


def _installed_manifest() -> Path | None:
    """The copy next to an installed app, when this process is that install."""
    here = Path(__file__).resolve()
    if "site-packages" not in here.parts:
        return None
    cursor = here
    venv: Path | None = None
    for _ in range(6):
        if cursor.name.lower() == "venv":
            venv = cursor
            break
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if venv is None:
        return None
    candidate = venv.parent / "mdl.json"
    return candidate if candidate.is_file() else None


def _tier(tier: VramTier | None) -> VramTier:
    if tier is not None:
        return tier
    try:
        return detect_tier(None)
    except (OSError, ValueError):
        return VramTier.UNKNOWN


def required_ids(tier: VramTier | None = None) -> list[str]:
    """Required model ids that fit ``tier`` (the machine tier when omitted)."""
    chosen = _tier(tier)
    path = manifest_path()
    manifest = load_manifest(path)
    optional = {model.id for model in manifest.models if model.optional}
    planned = plan_files([manifest], chosen)
    found: set[str] = set()
    for item in planned:
        found.update(item.model_ids)
    order = [model.id for model in manifest.models if model.id not in optional]
    return [mdl_id for mdl_id in order if mdl_id in found]


def present(mdl_id: str) -> bool:
    """True when ``mdl_id`` has its files, or a legacy folder of the same id."""
    folder = ensure_view(mdl_id)
    if folder is not None and folder.is_dir() and _has_payload(folder):
        return True
    root = models_dir() / APP_DIR
    for name in (mdl_id, *LEGACY_IDS.get(mdl_id, ())):
        legacy = root / name
        if legacy.is_dir() and any(legacy.iterdir()):
            return True
    return False


def missing_ids(tier: VramTier | None = None) -> list[str]:
    """Required ids for ``tier`` that are not installed, in manifest order."""
    return [mdl_id for mdl_id in required_ids(tier) if not present(mdl_id)]


def missing_id_text(tier: VramTier | None = None) -> str:
    """Comma-separated :func:`missing_ids`."""
    return ", ".join(missing_ids(tier))


def mdl_executable() -> list[str]:
    """``voxprint-mdl`` beside this interpreter, else on ``PATH``, else ``python -m``."""
    suffix = ".exe" if os.name == "nt" else ""
    bundled = Path(sys.executable).with_name(CLI_NAME + suffix)
    if bundled.is_file():
        return [str(bundled)]
    found = shutil.which(CLI_NAME)
    if found:
        return [found]
    return [sys.executable, "-m", "voxprint_common.mdl"]


def manage_argv() -> list[str]:
    """``voxprint-mdl --gui --app model-forge --manifest <path>``."""
    return [*mdl_executable(), "--gui", "--app", APP_ID, "--manifest", str(manifest_path())]


def download_argv(
    ids: Sequence[str] = (),
    *,
    gui: bool = False,
    json_mode: bool = False,
) -> list[str]:
    """``voxprint-mdl download --app model-forge --manifest <path>`` plus the given ids."""
    command = [*mdl_executable()]
    if json_mode:
        command.append("--json")
    command += ["download", "--app", APP_ID, "--manifest", str(manifest_path())]
    for mdl_id in ids:
        command += ["--model", mdl_id]
    if gui:
        command.append("--gui")
    return command


def fetch(ids: Sequence[str], gui: bool = False) -> dict[str, object]:
    """Download ``ids`` through MDL and build views when the result is ready.

    The command is ``voxprint-mdl --json download --app model-forge --manifest <path>``.
    Stderr is the MDL progress text.
    """
    chosen = [mdl_id for mdl_id in ids if mdl_id]
    if not chosen:
        return {"ok": True, "ready": True, "ids": []}
    command = download_argv(chosen, gui=gui, json_mode=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    payload = _last_object(completed.stdout)
    ready = _ready(payload, completed.returncode)
    if ready:
        for mdl_id in chosen:
            ensure_view(mdl_id)
    return {
        "ok": completed.returncode == 0,
        "ready": ready,
        "exit_code": completed.returncode,
        "ids": chosen,
        "result": payload,
    }


def status_report(ids: Sequence[str] = (), *, required: bool = False) -> dict[str, object]:
    """One row per app-model id: required flag and whether the files are present."""
    chosen = set(ids)
    rows = []
    for model in load_app_models():
        if required and not model.required:
            continue
        if chosen and model.id not in chosen:
            continue
        rows.append({"id": model.id, "required": model.required, "present": present(model.id)})
    return {"models": rows}


def execute(args: object) -> dict[str, object]:
    """Run ``models status`` or ``models fetch``."""
    command = str(getattr(args, "models_command", "") or "")
    ids = tuple(getattr(args, "model", None) or ())
    required = bool(getattr(args, "required", False))
    gui = bool(getattr(args, "gui", False))
    if command == "status":
        return status_report(ids, required=required)
    if command == "fetch":
        selected = list(required_ids()) if required or not ids else []
        for mdl_id in ids:
            if mdl_id not in selected:
                selected.append(mdl_id)
        if not selected:
            selected = missing_ids()
        result = fetch(selected, gui=gui)
        if not result["ok"]:
            raise StageError(FETCH_COMMAND, code="mdl")
        return result
    raise UsageError(f"unknown models command {command}", code="usage")


def _has_payload(folder: Path) -> bool:
    return any(path.name != MARKER for path in folder.iterdir())


def _last_object(text: str) -> dict[str, object]:
    found: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            found = payload
    return found


def _ready(payload: dict[str, object], code: int) -> bool:
    if code != 0:
        return False
    if payload.get("type") == "result":
        return bool(payload.get("ok")) and bool(payload.get("ready", True))
    result = payload.get("result")
    if isinstance(result, dict):
        return bool(payload.get("ok", True)) and bool(result.get("ready", True))
    return True
