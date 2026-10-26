"""Windows taskbar identity and the installed icon for VOXPrint AI Model Forge.

The Start menu shortcut and this process share one AppUserModelID. The window
icon is the same ``.ico`` the setup, the shortcut and the uninstall entry use.
``voxprint_common`` 0.5.0 provides ``branding``; it is used when the import
succeeds.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

try:
    from voxprint_common import branding as suite_branding
except ImportError:
    suite_branding = None  # type: ignore[assignment]

APP_USER_MODEL_ID = "VOXPrint.AISuite.ModelForge"
ICON_NAME = "voxprint-model-forge.ico"


def set_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Give this process the taskbar id of its Start menu shortcut.

    Call it before the first window opens. Off Windows this does nothing.

    Args:
        app_id: ``VOXPrint.AISuite.ModelForge`` unless a test passes another id.

    Returns:
        True when Windows accepted the id. False elsewhere or on failure.
    """
    if sys.platform != "win32":
        return False
    setter = getattr(suite_branding, "set_app_user_model_id", None)
    if setter is not None:
        return bool(setter(app_id))
    try:
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        return shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id)) == 0
    except (AttributeError, OSError):
        return False


def icon_path(executable: Path | None = None, source_root: Path | None = None) -> Path | None:
    """Return the app icon, or None when neither the install nor the repo has it.

    The installed file is ``<install dir>/assets/voxprint-model-forge.ico``.
    The install dir is the parent of ``venv/Scripts``
    (``Path(executable).resolve().parents[2]``). A source checkout falls back
    to the repository ``assets/`` file.

    Args:
        executable: Interpreter path. Defaults to ``sys.executable``.
        source_root: Repository root used for the fallback. Defaults to this
            tree.

    Returns:
        The icon path that exists, or None.
    """
    exe = Path(sys.executable if executable is None else executable).resolve()
    try:
        installed = exe.parents[2] / "assets" / ICON_NAME
    except IndexError:
        installed = None
    if installed is not None and installed.is_file():
        return installed
    root = Path(__file__).resolve().parents[3] if source_root is None else source_root
    bundled = root / "assets" / ICON_NAME
    if bundled.is_file():
        return bundled
    return None
