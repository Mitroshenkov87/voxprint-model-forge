"""`python -m voxprint_forge.gui` opens the Model Forge window."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from voxprint_forge.gui.window import MainWindow


def main() -> int:
    """Start the Model Forge window and return the process exit code."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
