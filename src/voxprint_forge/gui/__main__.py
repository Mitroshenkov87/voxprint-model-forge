"""`python -m voxprint_forge.gui` opens the Model Forge window.

The splash is ``voxprint_common.ui.splash.Splash``. This module does not draw
its own window. The art is the approved suite splash. The progress fill is the
warm bronze of that art: ``PALETTE.md`` names identity colours for Audiobook
Builder, Movie Dubber, Enlarge, and Clarity, and does not name Model Forge.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QApplication
from voxprint_common.buildinfo import load
from voxprint_common.ui.splash import Splash

from voxprint_forge.gui.window import DISPLAY_NAME, MainWindow

# Sampled from the warm shards of assets/splash.jpg. Not a theme token.
IDENTITY_COLOUR = "#a06020"
_ART = Path(__file__).resolve().parent / "assets" / "splash.jpg"


def splash_art() -> Path:
    """Path of the approved splash image shipped with the GUI."""
    return _ART


def open_splash() -> Splash:
    """Show the suite splash before the main window is built.

    ``voxprint_common`` 0.1.1 maps the id ``model-forge`` to the old spelling.
    The splash prints ``BuildInfo.app`` as written when that string is not a
    known id, so the window passes the display name and keeps the id for load.
    """
    info = load(package_dir=Path(__file__).resolve().parents[1], app="model-forge")
    splash = Splash(_ART, IDENTITY_COLOUR, replace(info, app=DISPLAY_NAME))
    splash.set_stage("Opening")
    return splash


def main() -> int:
    """Start the Model Forge window and return the process exit code."""
    app = QApplication([])
    splash = open_splash()
    window = MainWindow()
    window.show()
    splash.set_progress(1)
    splash.finish()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
