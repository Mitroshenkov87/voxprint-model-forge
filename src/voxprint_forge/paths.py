"""Default project and photo locations (the suite Documents rule, 2026-11-15).

The default Forge project root is ``<Documents>/VOXPrint Projects/Model Forge``
and the default photos root is ``<Documents>/VOXPrint Projects/Model Forge/Photos``,
holding one folder per person. The Documents folder is resolved, never hard-coded
in a localized spelling:

1. the ``XDG_DOCUMENTS_DIR`` environment variable, when set and non-empty;
2. ``xdg-user-dir DOCUMENTS`` on Linux, or the ``XDG_DOCUMENTS_DIR`` value in
   ``~/.config/user-dirs.dirs`` (``$HOME`` is expanded);
3. the Windows Known Folder "Documents";
4. ``~/Documents``.

The old ``~/VOXPrint/projects`` layout is only ever *listed* (leftover projects
stay visible); nothing is moved or copied to the new root.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SUITE_PROJECTS_FOLDER = "VOXPrint Projects"
APP_FOLDER = "Model Forge"
PHOTOS_FOLDER = "Photos"
LEGACY_CLI_PROJECTS = Path("VOXPrint") / "projects"


class PhotosResolutionError(Exception):
    """``--photos`` could not be resolved to one person folder. Carries what the CLI prints."""

    def __init__(self, message: str, *, names: list[str] | None = None) -> None:
        super().__init__(message)
        self.names = names or []


def documents_dir() -> Path:
    """The user's Documents folder (see the module docstring for the resolution order)."""
    env = os.environ.get("XDG_DOCUMENTS_DIR")
    if env:
        return Path(env).expanduser()
    configured = _configured_documents_dir()
    if configured is not None:
        return configured
    if sys.platform == "win32":
        known = _windows_documents_dir()
        if known is not None:
            return known
    return Path.home() / "Documents"


def default_projects_dir() -> Path:
    """``<Documents>/VOXPrint Projects/Model Forge``. The directory is not created."""
    return documents_dir() / SUITE_PROJECTS_FOLDER / APP_FOLDER


def default_photos_root() -> Path:
    """``<Documents>/VOXPrint Projects/Model Forge/Photos``. The directory is not created."""
    return default_projects_dir() / PHOTOS_FOLDER


def legacy_cli_projects_dir() -> Path:
    """``~/VOXPrint/projects``: where the laptop wrapper used to create projects. Listed only."""
    return Path.home() / LEGACY_CLI_PROJECTS


def person_folders(root: Path | None = None) -> list[Path]:
    """The person folders directly under the photos root, sorted by name; empty when the root is missing."""
    root = Path(root) if root is not None else default_photos_root()
    if not root.is_dir():
        return []
    return sorted((folder for folder in root.iterdir() if folder.is_dir()), key=lambda folder: folder.name)


def resolve_photos_arg(value: str | Path | None, *, root: Path | None = None, cwd: Path | None = None) -> Path:
    """Resolve a ``--photos`` value to a concrete folder.

    An absolute path is used as it is. A person-folder name or a relative path is
    resolved under the photos root; a relative path that already exists from the
    current directory (for example ``Photos/Ada Example`` run inside the
    ``Model Forge`` projects folder) is accepted too. ``None`` asks for the
    single-folder rule: the only person folder under the root wins; with several,
    their names are listed and the user must pass ``--photos``.

    Raises:
        PhotosResolutionError: The value is missing, ambiguous or does not exist.
    """
    root = Path(root) if root is not None else default_photos_root()
    if value is None:
        return _single_person_folder(root)
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if candidate.is_dir():
            return candidate
        raise PhotosResolutionError(
            f"Photo folder not found: {candidate}. Pass an absolute path to a folder of photos, "
            f"or the name of a person folder under the photos root ({root})."
        )
    under_root = root / candidate
    if under_root.is_dir():
        return under_root
    from_cwd = (Path.cwd() if cwd is None else Path(cwd)) / candidate
    if from_cwd.is_dir():
        return from_cwd
    raise PhotosResolutionError(
        f"Photo folder not found: {value}. Expected a person folder under the photos root ({root}) "
        f"such as {root / candidate.name}, or an absolute path."
    )


def _single_person_folder(root: Path) -> Path:
    """The one person folder under ``root``; lists the names and refuses when there are several."""
    folders = person_folders(root)
    if len(folders) == 1:
        return folders[0]
    if not folders:
        raise PhotosResolutionError(
            f"No person folders were found under the photos root ({root}). "
            "Add one folder of photos per person there, or pass --photos with a photo folder."
        )
    names = [folder.name for folder in folders]
    raise PhotosResolutionError(
        "Several person folders exist under the photos root "
        f"({root}). Pass one with --photos:\n" + "\n".join(f"  {name}" for name in names),
        names=names,
    )


def _configured_documents_dir() -> Path | None:
    """Step 2 of the rule: user-dirs.dirs first (cheap, no subprocess), then ``xdg-user-dir DOCUMENTS``."""
    from_file = _user_dirs_file_documents()
    if from_file is not None:
        return from_file
    return _xdg_user_dir_documents()


def _user_dirs_file_documents() -> Path | None:
    """``XDG_DOCUMENTS_DIR`` from ``~/.config/user-dirs.dirs`` with ``$HOME`` expanded, when it is set there."""
    try:
        text = (Path.home() / ".config" / "user-dirs.dirs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        name, _, raw = line.partition("=")
        if name.strip() != "XDG_DOCUMENTS_DIR":
            continue
        value = raw.strip().strip('"').strip("'")
        if value.startswith("$HOME"):
            value = str(Path.home()) + value[len("$HOME"):]
        if value:
            return Path(value)
    return None


def _xdg_user_dir_documents() -> Path | None:
    """``xdg-user-dir DOCUMENTS`` when the tool exists and answers; ``None`` otherwise."""
    if sys.platform == "win32":
        return None
    try:
        answer = subprocess.run(
            ["xdg-user-dir", "DOCUMENTS"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    folder = Path(answer.stdout.strip())
    if answer.returncode == 0 and str(folder):
        return folder
    return None


def _windows_documents_dir() -> Path | None:
    """The Known Folder "Documents" on Windows; ``None`` when it cannot be read."""
    if sys.platform != "win32":
        return None
    try:  # pragma: no cover - exercised on Windows only
        import ctypes

        class _Guid(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_char * 8),
            ]

        documents = _Guid(0xFDD39AD0, 0x238F, 0x46AF, b"\xad\xb4\x6c\x85\x48\x03\x69\xc7")
        pointer = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(documents), 0, None, ctypes.byref(pointer)
        )
        if result != 0 or not pointer.value:
            return None
        return Path(pointer.value)
    except Exception:  # noqa: BLE001 - any failure falls back to ~/Documents
        return None
