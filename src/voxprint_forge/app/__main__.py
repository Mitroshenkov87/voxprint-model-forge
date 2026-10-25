"""Start VOXPrint AI Model Forge (``voxprint-model-forge``): the suite splash, then the window."""

from __future__ import annotations

import sys

from voxprint_forge.app.branding import icon_path, set_app_user_model_id
from voxprint_forge.version import version_line

_CONSOLE_FLAGS = frozenset({"--version", "-V", "--help", "-h"})


def main(argv: list[str] | None = None) -> int:
    """Open the window, or print ``--version`` / ``--help`` and return without Qt."""
    args = list(sys.argv if argv is None else argv)
    if any(arg in _CONSOLE_FLAGS for arg in args[1:]):
        return _console(args)
    set_app_user_model_id()
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(list(argv or sys.argv))
    path = icon_path()
    if path is not None:
        app.setWindowIcon(QtGui.QIcon(str(path)))
    from voxprint_forge.gui.__main__ import open_splash

    splash = open_splash()
    from voxprint_forge.app.window import ForgeApp

    forge = ForgeApp()
    forge.window.show()
    splash.set_progress(1)
    splash.finish()
    return int(app.exec())


def _console(args: list[str]) -> int:
    """Print the version or a short usage line. The window stays closed."""
    if any(arg in {"--version", "-V"} for arg in args[1:]):
        print(version_line())
        return 0
    print(
        "usage: voxprint-model-forge [--version] [--help]\n"
        "\n"
        "VOXPrint AI Model Forge. With no flags, open the window.\n"
        "\n"
        "  --version, -V   print the version and exit\n"
        "  --help, -h      print this help and exit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
