"""Grab the finished Model Forge window against the demo fixture (WebP only).

Usage: QT_QPA_PLATFORM=offscreen python scripts/forge_gui_shots.py --out DIR [--root DIR]

The window opens offscreen with its projects root pointed at a temporary demo
fixture (never the user's Documents) and saves five grabs:

* ``01-empty`` — the wizard before anything is picked;
* ``02-photos`` — the photos step with the demo folder picked;
* ``03-settings`` — the Train step with Expert mode on;
* ``04-progress`` — the Train step following the running fixture;
* ``05-result`` — the result step of the finished fixture.

Nothing is trained: the runner is the test stub and no Start button is pressed.
``--root`` keeps the fixture for inspection; without it a temp dir is used.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

WINDOW_SIZE = (1280, 860)
SHOTS = ("01-empty", "02-photos", "03-settings", "04-progress", "05-result")


def grab(window, path: Path) -> None:
    """One WebP grab; the script refuses to write any other format."""
    if path.suffix.lower() != ".webp":
        raise SystemExit(f"screenshots must be WebP: {path}")
    image = window.grab()
    if not image.save(str(path), "WEBP", 90):
        raise SystemExit(f"Qt could not write {path} (WebP plugin missing?)")


def check_webp() -> None:
    from PySide6 import QtGui

    if b"webp" not in QtGui.QImageWriter.supportedImageFormats():
        raise SystemExit("this Qt build has no WebP writer; screenshots must be WebP")


def main(out: Path, root: Path | None = None) -> None:
    root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="forge-gui-shots-"))
    os.environ["VOXPRINT_HOME"] = str(root / "home")
    from PySide6 import QtWidgets

    from forge_gui_demo import build_demo
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, fake_image_checks
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.draft import STEPS
    from voxprint_forge.app.window import ForgeApp

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    check_webp()
    set_language("en")
    out.mkdir(parents=True, exist_ok=True)
    demo = build_demo(root)

    forge = ForgeApp(
        runner=StubRunner(),
        home=root / "state",
        projects_root=demo["root"] / "projects",
        commands=ALL_COMMANDS,
        open_path=lambda _p: None,
        detached=lambda _a: None,
        image_checks=fake_image_checks(root / "worker"),
    )
    forge.window.resize(*WINDOW_SIZE)
    forge.window.show()
    app.processEvents()

    def shot(name: str) -> None:
        app.processEvents()
        forge.window.repaint()
        app.processEvents()
        grab(forge.window, out / f"{name}.webp")

    def scroll_to(name: str, margin: int = 200) -> None:
        """Show the widget (and the content above it) before the grab."""
        widget = forge.window.findChild(QtWidgets.QWidget, name)
        if widget is None:
            return
        area = widget.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        if area is not None:
            area.ensureWidgetVisible(widget, 0, margin)
            app.processEvents()

    shot("01-empty")
    forge.draft.add_photos([demo["photos"]])
    forge.repaint()
    scroll_to("forge.photos.release")
    shot("02-photos")
    forge.draft.kind = "person"
    forge.draft.set_names("Demo Model", "")
    forge.draft.consent = forge.draft.adult = True
    forge.draft.set_verbal(True)
    forge.expert.set_enabled(True)
    forge.layout.sections._index = STEPS.index("train")
    forge.repaint()
    scroll_to("forge.train.strength")
    shot("03-settings")
    forge.expert.set_enabled(False)
    forge.open_project(demo["running"])
    shot("04-progress")
    forge.open_project(demo["done"])
    shot("05-result")
    forge.window.close()
    for name in SHOTS:
        print(out / f"{name}.webp")


def _argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="where the WebP grabs are written")
    parser.add_argument("--root", type=Path, default=None, help="fixture root (default: a temp dir)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _argv()
    main(args.out, args.root)
