"""Calls into voxprint-common 0.5.0 for check, repair, and RAM staging.

The vendored pin is still 0.4.1. That wheel has neither
``voxprint_common.mdl.run_verify_repair`` nor ``model_park.stage``, and its
``voxprint-mdl verify --repair`` only neutralises runtime-manifest paths.
This module is the only forge code that names the 0.5.0 surface:

* ``run_verify_repair(model_ids, repair=...)`` opens the shared progress
  window, hashes each model, reports OK / missing / corrupt, repairs only
  the bad ones when asked, then shows a summary.
* ``voxprint-mdl verify [--repair]`` is the table for a terminal.
* ``model_park.stage`` is the shared RAM hot tier.

Nothing here wraps ``progress_ui``. A missing name is an error, not a
fallback onto the 0.4.1 download window.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import voxprint_common.mdl as mdl
from voxprint_common.mdl.descriptor import DescriptorError, parse_descriptor_text
from voxprint_common.mdl.names import CLI_NAME
from voxprint_common.paths import models_dir

from voxprint_forge.errors import EXIT_CHECKSUM, EXIT_TOOL, ForgeError
from voxprint_forge.model_views import APP_DIR, BLOBS_DIR

APP_ID = "model-forge"
MANIFEST_NAME = "voxprint-model.json"
USER_DATA_DIRS = frozenset({"user-models", "projects", "lora"})
WALK_SKIP = USER_DATA_DIRS | frozenset({BLOBS_DIR, APP_DIR})

# 0.4.1 does not ship model_park. Importing it at startup would stop the app.
# 0.5.0 adds the module; until that wheel is pinned this stays None.
try:
    from voxprint_common import model_park as _model_park
except ImportError:
    _model_park = None


class MdlAdapterError(ForgeError):
    """The installed voxprint-common does not provide a 0.5.0 name this command needs."""

    exit_code = EXIT_TOOL


@dataclass(frozen=True)
class VerifyRepairResult:
    """Exit code, per-model rows, and the summary text from ``run_verify_repair``."""

    exit_code: int
    models: tuple[Mapping[str, object], ...]
    text: str


def installed_model_ids(manifest: Path, *, root: Path | None = None) -> list[str]:
    """Installed app models, then plug-in ids, in a stable order.

    A catalog row with nothing on disk is left out. ``user-models``,
    projects, and other user data are not walked and are not returned.
    """
    base = root if root is not None else models_dir()
    found = _app_ids(manifest, base)
    seen = set(found)
    for model_id in _plugin_ids(base):
        if model_id not in seen:
            found.append(model_id)
            seen.add(model_id)
    return found


def run_verify_repair(model_ids: Sequence[str], *, repair: bool = False) -> VerifyRepairResult:
    """Call ``voxprint_common.mdl.run_verify_repair(model_ids, repair=repair)``.

    ``repair`` re-downloads only missing or corrupt files. The call still
    happens when ``model_ids`` is empty, so the progress window opens.
    """
    upstream = getattr(mdl, "run_verify_repair", None)
    if not callable(upstream):
        raise MdlAdapterError(
            "voxprint-common 0.5.0 is required for models verify "
            "(voxprint_common.mdl.run_verify_repair). The installed package does not provide it."
        )
    return _normalize(upstream(list(model_ids), repair=repair))


def stage(*args: object, **kwargs: object) -> object:
    """Stage weights through ``voxprint_common.model_park.stage``.

    That function is the shared RAM hot tier. Arguments are forwarded as
    given; 0.5.0 owns the signature.
    """
    fn = getattr(_model_park, "stage", None) if _model_park is not None else None
    if not callable(fn):
        raise MdlAdapterError(
            "voxprint-common 0.5.0 is required to stage a model "
            "(voxprint_common.model_park.stage). The installed package does not provide it."
        )
    return fn(*args, **kwargs)


def verify_argv(
    *,
    repair: bool,
    model_ids: Sequence[str],
    manifest: Path,
    executable: Sequence[str] | None = None,
) -> list[str]:
    """``voxprint-mdl verify --app model-forge --manifest <path> [--model id] [--repair]``."""
    command = list(executable) if executable is not None else _mdl_executable()
    command.append("verify")
    command += ["--app", APP_ID, "--manifest", str(manifest)]
    for model_id in model_ids:
        command += ["--model", model_id]
    if repair:
        command.append("--repair")
    return command


def execute_verify(
    *,
    manifest: Path,
    repair: bool,
    gui: bool,
    root: Path | None = None,
    executable: Sequence[str] | None = None,
) -> dict[str, object]:
    """Check installed models and return ``ok``, ``exit_code``, ``models``, and ``text``.

    ``gui`` calls :func:`run_verify_repair`, including when every installed
    model is already OK. Without ``gui``, ``voxprint-mdl verify`` prints the
    table and its exit code is returned. An empty install is success and does
    not ask MDL to plan the whole catalog.
    """
    ids = installed_model_ids(manifest, root=root)
    if gui:
        outcome = run_verify_repair(ids, repair=repair)
        return {
            "ok": outcome.exit_code == 0,
            "exit_code": outcome.exit_code,
            "models": [dict(row) for row in outcome.models],
            "text": outcome.text,
        }
    if not ids:
        return {"ok": True, "exit_code": 0, "models": [], "text": _empty_table()}
    command = verify_argv(repair=repair, model_ids=ids, manifest=manifest, executable=executable)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    text = completed.stdout
    if text and not text.endswith("\n"):
        text += "\n"
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "models": [],
        "text": text,
    }


def _normalize(value: object) -> VerifyRepairResult:
    if isinstance(value, bool) or value is None:
        raise MdlAdapterError("run_verify_repair returned an unusable result")
    if isinstance(value, int):
        return VerifyRepairResult(value, (), "")
    if isinstance(value, Mapping):
        return _from_mapping(value)
    code = _code(value, "exit_code")
    if code is None:
        code = _code(value, "code")
    models = _model_rows(getattr(value, "models", ()))
    text = getattr(value, "text", "")
    if code is None:
        raise MdlAdapterError("run_verify_repair returned an unusable result")
    return VerifyRepairResult(code, models, text if isinstance(text, str) else "")


def _from_mapping(value: Mapping[object, object]) -> VerifyRepairResult:
    models = _model_rows(value.get("models", ()))
    text = value.get("text", "")
    summary = text if isinstance(text, str) else ""
    code = value.get("exit_code", value.get("code"))
    if isinstance(code, int) and not isinstance(code, bool):
        return VerifyRepairResult(code, models, summary)
    ok = value.get("ok")
    if ok is True:
        return VerifyRepairResult(0, models, summary)
    if ok is False or any(_broken(row) for row in models):
        return VerifyRepairResult(EXIT_CHECKSUM, models, summary)
    return VerifyRepairResult(0, models, summary)


def _code(value: object, name: str) -> int | None:
    code = getattr(value, name, None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    return None


def _model_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


def _broken(row: Mapping[str, object]) -> bool:
    return str(row.get("status", "")).lower() in {"missing", "corrupt"}


def _empty_table() -> str:
    rows = (("model", "status", "checked", "files"), ("(none)", "OK", "0/0", "-"))
    widths = [max(len(row[index]) for row in rows) for index in range(4)]
    lines = [
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(rows[0])).rstrip(),
        "  ".join("-" * width for width in widths).rstrip(),
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(rows[1])).rstrip(),
    ]
    return "\n".join(lines) + "\n"


def _app_ids(manifest: Path, root: Path) -> list[str]:
    if not manifest.is_file():
        return []
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MdlAdapterError(f"{manifest.name} could not be read") from exc
    if not isinstance(document, dict):
        return []
    found: list[str] = []
    for model in document.get("models") or []:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        files = model.get("files")
        if not isinstance(model_id, str) or not model_id:
            continue
        rows = files if isinstance(files, list) else []
        if _app_installed(root, model_id, rows):
            found.append(model_id)
    return found


def _app_installed(root: Path, model_id: str, files: Sequence[object]) -> bool:
    if (root / APP_DIR / model_id).is_dir():
        return True
    for row in files:
        if not isinstance(row, dict):
            continue
        digest = row.get("sha256")
        if isinstance(digest, str) and (root / BLOBS_DIR / digest.strip().lower()).is_file():
            return True
    return False


def _plugin_ids(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in WALK_SKIP)
        if MANIFEST_NAME in filenames:
            paths.append(Path(dirpath) / MANIFEST_NAME)
    found: list[str] = []
    for path in sorted(paths):
        try:
            descriptor = parse_descriptor_text(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, DescriptorError):
            continue
        if descriptor.id:
            found.append(descriptor.id)
    return found


def _mdl_executable() -> list[str]:
    """``voxprint-mdl`` beside this interpreter, else on PATH, else ``python -m``.

    Same resolution as :func:`voxprint_forge.mdl_app.mdl_executable`. This
    module cannot import that one: ``mdl_app`` imports :func:`execute_verify`.
    """
    suffix = ".exe" if os.name == "nt" else ""
    bundled = Path(sys.executable).with_name(CLI_NAME + suffix)
    if bundled.is_file():
        return [str(bundled)]
    found = shutil.which(CLI_NAME)
    if found:
        return [found]
    return [sys.executable, "-m", "voxprint_common.mdl"]
