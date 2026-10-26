"""Calls into voxprint-common 0.5.2 for check, repair, and RAM staging.

The vendored pin is 0.5.2. This module is the only forge code that names
that surface:

* ``voxprint-mdl verify --app model-forge --gui --repair`` opens the
  check-and-repair window. ``voxprint-mdl-gui --app`` opens Manage models and
  is not used here. The launch passes ``CREATE_NO_WINDOW`` so a console does
  not flash. A missing ``voxprint-mdl`` is an error and is not replaced with
  ``python -m``.
* A terminal check calls :func:`voxprint_common.mdl.job.verify_repair`. The
  ``voxprint-mdl --json verify`` result line in this pin is ``ready`` plus
  ``ready``, ``state``, and ``jobs``, with no per-status counts. The counts
  are ``status.extra["counts"]``.
* ``ModelPark.stage(name, models, device=...)`` from
  :func:`voxprint_common.model_park.for_process` is the shared RAM hot tier.

Nothing here imports the progress-window module. A missing executable is an error.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from voxprint_common.mdl.descriptor import DescriptorError, parse_descriptor_text
from voxprint_common.mdl.job import verify_repair
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
MISSING_MDL = (
    "voxprint-common 0.5.2 is required for models verify "
    "(voxprint-mdl verify --gui). The installed package does not provide it."
)
_OK_STATES = frozenset({"ok", "repaired", "verified"})
_MISSING_STATES = frozenset({"missing"})
_DAMAGED_STATES = frozenset({"corrupt", "damaged", "failed", "stale"})


class MdlAdapterError(ForgeError):
    """The installed voxprint-common does not provide a 0.5.2 name this command needs."""

    exit_code = EXIT_TOOL


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


def _is_windows() -> bool:
    """True when this process should use the Windows launch flags."""
    return os.name == "nt"


def mdl_script() -> list[str] | None:
    """``voxprint-mdl`` beside this interpreter, else on ``PATH``.

    A missing file is not replaced with ``python -m``. That interpreter would
    open a console in front of the check window.
    """
    suffix = ".exe" if _is_windows() else ""
    bundled = Path(sys.executable).with_name(CLI_NAME + suffix)
    if bundled.is_file():
        return [str(bundled)]
    found = shutil.which(CLI_NAME)
    if found:
        return [found]
    if suffix:
        found = shutil.which(CLI_NAME + suffix)
        if found:
            return [found]
    return None


def verify_window_argv(
    *,
    repair: bool,
    manifest: Path,
    root: Path | None = None,
    executable: Sequence[str] | None = None,
) -> list[str]:
    """``voxprint-mdl verify --app model-forge --manifest <path> [--model id] --gui [--repair]``.

    Installed catalog ids are passed with ``--model``. Plug-in ids that are
    not in this manifest are left out: 0.5.2 rejects them. MOSS FP8 ids are
    left out so the window does not download those files. ``repair`` adds
    ``--repair``.

    Raises:
        MdlAdapterError: ``voxprint-mdl`` is not installed.
    """
    if executable is not None:
        command = [str(part) for part in executable]
    else:
        found = mdl_script()
        if found is None:
            raise MdlAdapterError(MISSING_MDL)
        command = list(found)
    command += ["verify", "--app", APP_ID, "--manifest", str(manifest)]
    for model_id in _check_ids(manifest, root):
        command += ["--model", model_id]
    command.append("--gui")
    if repair:
        command.append("--repair")
    return command


def spawn(argv: Sequence[str], *, wait: bool = False) -> int:
    """Start ``argv`` with no console window. Return the exit code when ``wait`` is set."""
    command = [str(part) for part in argv]
    options = _no_console_kwargs()
    if wait:
        completed = subprocess.run(command, check=False, **options)
        return int(completed.returncode)
    subprocess.Popen(command, **options)
    return 0


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
            "voxprint-common 0.5.2 is required to stage a model "
            "(voxprint_common.model_park.stage). The installed package does not provide it."
        )
    return fn(name, models, device=device)


def execute_verify(
    *,
    manifest: Path,
    repair: bool,
    gui: bool,
    root: Path | None = None,
    executable: Sequence[str] | None = None,
) -> dict[str, object]:
    """Check installed models and return ``ok``, ``exit_code``, ``models``, and ``text``.

    ``gui`` launches :func:`verify_window_argv` and waits for that window,
    including when every installed model is already OK. Without ``gui``,
    :func:`verify_repair` checks the same ids and ``text`` is
    ``N models: X OK, Y damaged, Z missing``. An empty install without ``gui``
    is success and does not ask MDL to plan the whole catalog.

    MOSS FP8 ids are left out of that call. They are hashed here, and ``repair``
    rebuilds a missing or corrupt file from the pinned bf16 snapshot (or from
    the mirror, once that revision is published).
    """
    base = root if root is not None else models_dir()
    fp8_rows = [dict(row) for row in moss_fp8.cover(base, repair=repair)]
    if gui:
        try:
            code = spawn(
                verify_window_argv(repair=repair, manifest=manifest, root=root, executable=executable),
                wait=True,
            )
        except OSError as exc:
            raise MdlAdapterError(MISSING_MDL) from exc
        return _with_fp8(
            {"ok": code == 0, "exit_code": code, "models": [], "text": ""},
            fp8_rows,
        )
    upstream = _check_ids(manifest, root)
    if not upstream and not fp8_rows:
        return {"ok": True, "exit_code": 0, "models": [], "text": format_verify_counts(0, 0, 0)}
    if not upstream:
        merged = _with_fp8({"ok": True, "exit_code": 0, "models": [], "text": ""}, fp8_rows)
        return _with_summary(merged, counts=None, fp8_rows=fp8_rows)
    outcome = verify_repair(list(upstream), repair=repair, manifests=[manifest])
    counts = _outcome_counts(outcome)
    raw_message = getattr(outcome, "message", "")
    message = raw_message if isinstance(raw_message, str) else ""
    merged = _with_fp8(
        {
            "ok": outcome.code == 0,
            "exit_code": outcome.code,
            "models": _rows_from_steps(getattr(outcome.status, "steps", ())),
            "text": "",
        },
        fp8_rows,
    )
    if counts is None and outcome.code != 0:
        text = message if message.endswith("\n") else f"{message}\n"
        merged["text"] = text
        return merged
    return _with_summary(merged, counts=counts, fp8_rows=fp8_rows)


def format_verify_counts(ok: int, damaged: int, missing: int) -> str:
    """``N models: X OK, Y damaged, Z missing`` including the trailing newline."""
    total = ok + damaged + missing
    return f"{total} models: {ok} OK, {damaged} damaged, {missing} missing\n"


def _with_fp8(result: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    """Merge FP8 rows. A missing, corrupt or stale FP8 file fails a run the upstream call passed."""
    if not rows:
        return result
    models = list(result["models"]) if isinstance(result["models"], list) else []
    models.extend(rows)
    text = result["text"] if isinstance(result["text"], str) else ""
    code = result["exit_code"]
    exit_code = code if isinstance(code, int) and not isinstance(code, bool) else EXIT_CHECKSUM
    broken = {"missing", "corrupt", "stale", "failed", "damaged"}
    if exit_code == 0 and any(str(row.get("status", "")).lower() in broken for row in rows):
        exit_code = EXIT_CHECKSUM
    return {"ok": exit_code == 0, "exit_code": exit_code, "models": models, "text": text}


def _with_summary(
    result: dict[str, object],
    *,
    counts: Mapping[str, object] | None,
    fp8_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Replace ``text`` with the per-status counts line, including FP8 rows."""
    if counts is not None:
        ok, damaged, missing = _tally_counts(counts)
    else:
        models = result["models"] if isinstance(result["models"], list) else []
        upstream = [row for row in models if row not in fp8_rows]
        ok, damaged, missing = _tally_rows(upstream)
    fp8_ok, fp8_damaged, fp8_missing = _tally_rows(fp8_rows)
    result["text"] = format_verify_counts(ok + fp8_ok, damaged + fp8_damaged, missing + fp8_missing)
    return result


def _no_console_kwargs() -> dict[str, object]:
    """``CREATE_NO_WINDOW`` on Windows, and detached standard streams everywhere.

    The flag is how a ``pythonw``-style child avoids a flashing command window.
    Other platforms pass ``creationflags=0``.
    """
    if _is_windows():
        flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    else:
        flags = 0
    return {
        "start_new_session": True,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "creationflags": flags,
    }


def _check_ids(manifest: Path, root: Path | None) -> list[str]:
    """Installed catalog ids, without MOSS FP8 and without ids 0.5.2 would reject."""
    listed = _catalog_ids(manifest)
    skip = moss_fp8.model_ids()
    return [
        model_id
        for model_id in installed_model_ids(manifest, root=root)
        if model_id in listed and model_id not in skip
    ]


def _catalog_ids(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MdlAdapterError(f"{manifest.name} could not be read") from exc
    if not isinstance(document, dict):
        return set()
    found: set[str] = set()
    for model in document.get("models") or []:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            found.add(model_id)
    return found


def _outcome_counts(outcome: object) -> Mapping[str, object] | None:
    extra = getattr(getattr(outcome, "status", None), "extra", None)
    if not isinstance(extra, Mapping):
        return None
    counts = extra.get("counts")
    if isinstance(counts, Mapping):
        return counts
    return None


def _rows_from_steps(steps: object) -> list[dict[str, object]]:
    if not isinstance(steps, (list, tuple)):
        return []
    rows: list[dict[str, object]] = []
    for step in steps:
        model_id = getattr(step, "item_id", "") or getattr(step, "model", "")
        state = getattr(step, "state", "")
        if isinstance(model_id, str) and model_id:
            rows.append({"id": model_id, "status": state if isinstance(state, str) else ""})
    return rows


def _tally_counts(counts: Mapping[str, object]) -> tuple[int, int, int]:
    ok = _count(counts, "ok") + _count(counts, "repaired") + _count(counts, "verified")
    damaged = (
        _count(counts, "corrupt")
        + _count(counts, "damaged")
        + _count(counts, "failed")
        + _count(counts, "stale")
    )
    return ok, damaged, _count(counts, "missing")


def _tally_rows(rows: Sequence[Mapping[str, object]]) -> tuple[int, int, int]:
    ok = damaged = missing = 0
    for row in rows:
        status = str(row.get("status", "")).lower()
        if status in _OK_STATES:
            ok += 1
        elif status in _MISSING_STATES:
            missing += 1
        elif status in _DAMAGED_STATES:
            damaged += 1
    return ok, damaged, missing


def _count(counts: Mapping[str, object], key: str) -> int:
    value = counts.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


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
