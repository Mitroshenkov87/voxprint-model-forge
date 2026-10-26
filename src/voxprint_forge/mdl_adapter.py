"""Calls into voxprint-common 0.5.0 for check, repair, and RAM staging.

The vendored pin is 0.5.0. This module is the only forge code that names
that surface:

* ``run_verify_repair(model_ids, repair=...)`` opens the shared progress
  window, hashes each model, reports OK / missing / corrupt, repairs only
  the bad ones when asked, then shows a summary.
* ``voxprint-mdl verify [--repair]`` is the table for a terminal.
* ``ModelPark.stage(name, models, device=...)`` from
  :func:`voxprint_common.model_park.for_process` is the shared RAM hot tier.

Nothing here imports the progress-window module. A missing name is an error.
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
from voxprint_common.model_park import for_process
from voxprint_common.paths import models_dir

from voxprint_forge.diffusion import moss_fp8
from voxprint_forge.errors import EXIT_CHECKSUM, EXIT_TOOL, ForgeError
from voxprint_forge.model_views import APP_DIR, BLOBS_DIR

APP_ID = "model-forge"
MANIFEST_NAME = "voxprint-model.json"
USER_DATA_DIRS = frozenset({"user-models", "projects", "lora"})
WALK_SKIP = USER_DATA_DIRS | frozenset({BLOBS_DIR, APP_DIR})


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


def run_verify_repair(
    model_ids: Sequence[str],
    *,
    repair: bool = False,
    manifest: Path | None = None,
) -> VerifyRepairResult:
    """Call ``voxprint_common.mdl.run_verify_repair(model_ids, repair=repair)``.

    ``repair`` re-downloads only missing or corrupt files. The call still
    happens when ``model_ids`` is empty, so the progress window opens.
    ``manifest`` is that app's ``mdl.json``. An empty id list is not ``None``,
    so the planned tier is not substituted for the installed set.
    """
    upstream = getattr(mdl, "run_verify_repair", None)
    if not callable(upstream):
        raise MdlAdapterError(
            "voxprint-common 0.5.0 is required for models verify "
            "(voxprint_common.mdl.run_verify_repair). The installed package does not provide it."
        )
    manifests = None if manifest is None else [manifest]
    return _normalize(upstream(list(model_ids), repair=repair, manifests=manifests, show=True))


def stage(name: str, models: Sequence[str], *, device: object = "cuda") -> object:
    """Unpark ``models`` onto ``device`` for the step named ``name``.

    This is ``for_process().stage(name, models, device=device)``. The returned
    context manager yields each model's state dict and moves those weights
    back to RAM on exit.
    """
    park = for_process(app="model-forge")
    fn = getattr(park, "stage", None)
    if not callable(fn):
        raise MdlAdapterError(
            "voxprint-common 0.5.0 is required to stage a model "
            "(voxprint_common.model_park.stage). The installed package does not provide it."
        )
    return fn(name, models, device=device)


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

    MOSS FP8 ids are left out of that call. They are hashed here, and ``repair``
    rebuilds a missing or corrupt file from the pinned bf16 snapshot (or from
    the mirror, once that revision is published).
    """
    ids = installed_model_ids(manifest, root=root)
    base = root if root is not None else models_dir()
    fp8_rows = [dict(row) for row in moss_fp8.cover(base, repair=repair)]
    upstream = [model_id for model_id in ids if model_id not in moss_fp8.model_ids()]
    if gui:
        outcome = run_verify_repair(upstream, repair=repair, manifest=manifest)
        return _with_fp8(
            {
                "ok": outcome.exit_code == 0,
                "exit_code": outcome.exit_code,
                "models": [dict(row) for row in outcome.models],
                "text": outcome.text,
            },
            fp8_rows,
        )
    if not upstream and not fp8_rows:
        return {"ok": True, "exit_code": 0, "models": [], "text": _empty_table()}
    if not upstream:
        return _with_fp8({"ok": True, "exit_code": 0, "models": [], "text": ""}, fp8_rows)
    command = verify_argv(repair=repair, model_ids=upstream, manifest=manifest, executable=executable)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    text = completed.stdout
    if text and not text.endswith("\n"):
        text += "\n"
    return _with_fp8(
        {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "models": [],
            "text": text,
        },
        fp8_rows,
    )


def _with_fp8(result: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    """Merge FP8 rows. A missing, corrupt or stale FP8 file fails a run the upstream call passed."""
    if not rows:
        return result
    models = list(result["models"]) if isinstance(result["models"], list) else []
    models.extend(rows)
    text = result["text"] if isinstance(result["text"], str) else ""
    if text and not text.endswith("\n"):
        text += "\n"
    text += "".join(f"{row['id']}  {row['status']}\n" for row in rows)
    code = result["exit_code"]
    exit_code = code if isinstance(code, int) and not isinstance(code, bool) else EXIT_CHECKSUM
    broken = {"missing", "corrupt", "stale"}
    if exit_code == 0 and any(str(row.get("status", "")).lower() in broken for row in rows):
        exit_code = EXIT_CHECKSUM
    return {"ok": exit_code == 0, "exit_code": exit_code, "models": models, "text": text}


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
    if not models:
        models = _rows_from_status(getattr(value, "status", None))
    text = _text(value)
    if code is None:
        raise MdlAdapterError("run_verify_repair returned an unusable result")
    return VerifyRepairResult(code, models, text)


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


def _text(value: object) -> str:
    text = getattr(value, "text", None)
    if isinstance(text, str) and text:
        return text
    message = getattr(value, "message", "")
    if isinstance(message, str) and message:
        return message
    return text if isinstance(text, str) else ""


def _rows_from_status(status: object) -> tuple[Mapping[str, object], ...]:
    """One row per step on an ``Outcome`` status. ``item_id`` is the model id."""
    steps = getattr(status, "steps", None)
    if not isinstance(steps, (list, tuple)):
        return ()
    rows: list[Mapping[str, object]] = []
    for step in steps:
        model_id = getattr(step, "item_id", "") or getattr(step, "model", "")
        state = getattr(step, "state", "")
        if isinstance(model_id, str) and model_id:
            rows.append({"id": model_id, "status": state if isinstance(state, str) else ""})
    return tuple(rows)


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
