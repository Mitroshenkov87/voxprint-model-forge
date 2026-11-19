"""Start VOXPrint AI Model Forge (``voxprint-model-forge``): the suite splash, then the window."""

from __future__ import annotations

import sys

from voxprint_forge.app.branding import DESKTOP_FILE_NAME, icon_path, set_app_user_model_id
from voxprint_forge.version import version_line

_CONSOLE_FLAGS = frozenset({"--version", "-V", "--help", "-h", "--print-selection"})


def main(argv: list[str] | None = None) -> int:
    """Open the window, or print ``--version`` / ``--help`` / the selection and return without Qt."""
    args = list(sys.argv if argv is None else argv)
    if any(arg in _CONSOLE_FLAGS for arg in args[1:]):
        return _console(args)
    set_app_user_model_id()
    from PySide6 import QtGui, QtWidgets

    QtGui.QGuiApplication.setDesktopFileName(DESKTOP_FILE_NAME)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(list(argv or sys.argv))
    path = icon_path()
    if path is not None:
        app.setWindowIcon(QtGui.QIcon(str(path)))
    from voxprint_forge.gui.__main__ import open_splash

    splash = open_splash()
    from voxprint_forge.app.launch import parse_launch_args
    from voxprint_forge.app.window import ForgeApp

    forge = ForgeApp(launch=parse_launch_args(args[1:]))
    app.aboutToQuit.connect(forge.shutdown)
    forge.window.show()
    splash.set_progress(1)
    splash.finish()
    return int(app.exec())


def _console(args: list[str]) -> int:
    """Print the version, a short usage line or the picked photo folder. The window stays closed."""
    if any(arg in {"--version", "-V"} for arg in args[1:]):
        print(version_line())
        return 0
    if "--print-selection" in args[1:]:
        from voxprint_forge.app.launch import main as print_selection

        return print_selection(["--print-selection"])
    print(
        "usage: voxprint-model-forge [--version] [--help] [--crop MODE] [--rank N] [--steps N]\n"
        "                            [--base ID] [--adult | --no-adult] [--photos DIR]\n"
        "\n"
        "VOXPrint AI Model Forge. With no flags, open the window.\n"
        "\n"
        "  --version, -V      print the version and exit\n"
        "  --help, -h         print this help and exit\n"
        "  --print-selection  print the photo folder picked in the Forge window and exit\n"
        "                     (exit 3 when nothing is picked yet)\n"
        "  --crop MODE        preload the crop mode: both, subject or whole (default both)\n"
        "  --rank N           preload the LoRA rank; alpha follows rank (default 64)\n"
        "  --steps N          preload the training steps (default 1600)\n"
        "  --base ID          preload the base model MDL id (default z-image)\n"
        "  --adult            preload 18+ and adult content on (default)\n"
        "  --no-adult         preload them off\n"
        "  --photos DIR       load photos from DIR on startup\n"
        "\n"
        "Defaults come from <home>/model-forge/gui-defaults.json when that file exists;"
        " the flags above override it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
