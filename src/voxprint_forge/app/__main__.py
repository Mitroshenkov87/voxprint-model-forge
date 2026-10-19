"""Start VOXPrint AI Model Forge (``voxprint-model-forge``): the suite splash, then the window."""

from __future__ import annotations

import sys

from voxprint_forge.app.branding import icon_path, set_app_user_model_id


def main(argv: list[str] | None = None) -> int:
    """Open the window on the shared shell and run the Qt loop."""
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


if __name__ == "__main__":
    raise SystemExit(main())
