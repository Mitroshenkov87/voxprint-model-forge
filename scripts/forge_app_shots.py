"""Grab each Model Forge step (Simple and Expert, en and ru) with a stub engine.

Usage: QT_QPA_PLATFORM=offscreen python scripts/forge_app_shots.py OUT_DIR
Photos are generated gradients. Nothing is trained and no GPU is used.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Narrow window widths in px (Simple, Expert); the subject rows wrap there without cutting a label.
# Expert mode is wider because the Expert panel takes the right-hand side.
NARROW = {False: 420, True: 900}


def main(out: Path) -> None:
    home = Path(tempfile.mkdtemp(prefix="forge-shots-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos, write_prepared
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.engine import EngineEvent, Outcome
    from voxprint_forge.app.window import ForgeApp

    out.mkdir(parents=True, exist_ok=True)
    photos = make_photos(home / "photos", 14)
    for lang in ("en", "ru"):
        for expert in (False, True):
            set_language(lang)
            runner = StubRunner({
                "prepare": ([EngineEvent("prepare.safety", "lora.progress.safety", {"n": 14, "total": 14},
                                         "lora.reason.safety", 100.0, None, "done")],
                            Outcome(True, {"state": "done", "prepared": 12})),
                "train": ([EngineEvent("train", "lora.progress.train", {"step": 450, "steps": 1500},
                                       "lora.reason.train", 30.0, 1260)], Outcome(True, {"state": "done"})),
            })
            drops = [(photos[12], "lora.drop.blurry"), (photos[13], "lora.drop.safety")]
            runner.effects["prepare"] = lambda project: write_prepared(project, photos[:12], drops)
            forge = ForgeApp(runner=runner, home=home / f"state-{lang}-{expert}", commands=ALL_COMMANDS,
                             open_path=lambda _p: None, launch=lambda _a: None)
            forge.expert.set_enabled(expert)
            forge.window.show()
            mode = "expert" if expert else "simple"

            def shot(name: str) -> None:
                app.processEvents()
                forge.window.grab().save(str(out / f"{lang}-{mode}-{name}.png"))

            next_button = forge.widget._parts()["next"]
            forge.draft.add_photos(photos)
            forge.draft.kind = "person"
            forge.repaint()
            shot("1-photos")
            next_button.click()
            forge.draft.set_names("Mira Demo", "")
            forge.draft.consent = forge.draft.adult = True
            forge.draft.set_verbal(True)
            forge.draft.adult_content = True
            forge.repaint()
            shot("2-subject")
            wide = forge.window.size()
            forge.window.resize(NARROW[expert], wide.height())
            shot("2-subject-narrow")
            forge.window.resize(wide)
            next_button.click()
            forge.prepare()
            shot("3-prepare")
            next_button.click()
            shot("4-train")
            shot_open_base(forge, app, out / f"{lang}-{mode}-4-train-base-open.png")
            from voxprint_forge.app.details import Details

            for details in forge.window.findChildren(Details):
                details.button.setChecked(True)
            shot("4-train-details-open")
            base_details = forge.window.findChild(Details, "forge.details.train.base")
            scroll = base_details.parentWidget().parentWidget().parentWidget()  # page -> viewport -> scroll area
            scroll.ensureWidgetVisible(base_details.body)
            shot("4-train-details-open-base")
            for details in forge.window.findChildren(Details):
                details.button.setChecked(False)
            runner.hold = True
            forge.train()
            shot("4-train-running")
            runner.release()
            next_button.click()
            shot("5-result")
            forge.widget._layout.sections.select("check")
            forge.widget.refresh()
            shot("6-check")
            forge.window.close()
            forge.expert.set_enabled(False)


def shot_open_base(forge, app, path: Path) -> None:
    """The Train step with the base-model list open (the popup is its own window, so paint it in)."""
    from PySide6 import QtCore, QtGui, QtWidgets

    combo = forge.window.findChild(QtWidgets.QComboBox, "forge.train.base")
    combo.showPopup()
    app.processEvents()
    image = forge.window.grab()
    popup = combo.view().window()
    painter = QtGui.QPainter(image)
    corner = forge.window.mapFromGlobal(popup.mapToGlobal(QtCore.QPoint(0, 0)))
    painter.drawPixmap(corner, popup.grab())
    painter.end()
    image.save(str(path))
    combo.hidePopup()
    app.processEvents()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
