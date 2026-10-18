"""Grab each Model Forge step (Simple and Expert, en and ru) with a stub engine.

Usage: QT_QPA_PLATFORM=offscreen python scripts/forge_app_shots.py OUT_DIR [--manual]
Photos are generated neutral placeholders (no people). Nothing is trained and no GPU is used.
``--manual`` writes only the pictures the user manual uses (docs/manual/img/) plus the model downloader.
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
# Pictures docs/manual uses, per language and mode.
MANUAL = {
    "simple": ("1-photos", "2-subject", "3-build", "4-prepare", "5-train", "5-train-base-open",
               "5-train-details-open", "5-train-running", "6-result", "check"),
    "expert": ("5-train", "5-train-running"),
}
# Model Forge rows in the downloader picture: (id, display name, size in bytes, received share, state).
MDL_ROWS = (
    ("z-image", "Z-Image", 20_540_000_000, 1.0, "ready"),
    ("qwen3-vl-4b-instruct", "Qwen3-VL-4B-Instruct", 8_890_000_000, 1.0, "ready"),
    ("mivolo-v2", "MiVOLO v2", 120_000_000, 1.0, "ready"),
    ("yunet-2023mar", "YuNet (2023mar)", 400_000, 1.0, "ready"),
    ("florence2-large", "Florence-2-large", 1_560_000_000, 0.62, "downloading"),
    ("moss-image-base", "MOSS-Image Base", 27_220_000_000, 0.0, "pending"),
)


def placeholder_photos(folder: Path, count: int) -> list[Path]:
    """Neutral placeholder pictures: soft backgrounds with simple shapes, a number, no people, no metadata."""
    from PySide6 import QtCore, QtGui

    folder.mkdir(parents=True, exist_ok=True)
    hues = (24, 200, 140, 280, 48, 330, 180, 100, 220, 10, 260, 160, 60, 300)
    paths = []
    for index in range(count):
        side = 512
        image = QtGui.QImage(side, side, QtGui.QImage.Format.Format_RGB32)
        hue = hues[index % len(hues)]
        gradient = QtGui.QLinearGradient(0, 0, 0, side)
        gradient.setColorAt(0, QtGui.QColor.fromHsv(hue, 40, 245))
        gradient.setColorAt(1, QtGui.QColor.fromHsv(hue, 90, 200))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(image.rect(), gradient)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor.fromHsv((hue + 30) % 360, 120, 170))
        painter.drawRoundedRect(QtCore.QRectF(150, 210 + (index % 3) * 10, 212, 190), 28, 28)
        painter.setBrush(QtGui.QColor.fromHsv((hue + 180) % 360, 70, 235))
        painter.drawEllipse(QtCore.QPointF(256, 170), 70 + (index % 4) * 6, 70 + (index % 4) * 6)
        painter.setPen(QtGui.QColor(60, 60, 60))
        font = painter.font()
        font.setPixelSize(40)
        painter.setFont(font)
        painter.drawText(QtCore.QRectF(0, side - 70, side, 60), QtCore.Qt.AlignmentFlag.AlignCenter, f"{index + 1:02d}")
        painter.end()
        path = folder / f"demo-{index + 1:02d}.png"
        image.save(str(path))
        paths.append(path)
    return paths


def demo_prepared(project: Path, photos: list[Path], drops: list[tuple[Path, str]]) -> None:
    """Like the test stub's ``write_prepared``, with descriptions that read naturally in the manual."""
    import json

    project.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"p{i}", "file": str(photo), "long": f"mira_demo, placeholder picture {i + 1:02d}, a circle above "
             "a rounded square on a soft background", "short": f"mira_demo, placeholder picture {i + 1:02d}",
             "round": 0} for i, photo in enumerate(photos)]
    (project / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    lines = [json.dumps({"id": f"d{i}", "name": path.name, "reason": reason}) for i, (path, reason) in enumerate(drops)]
    (project / "drops.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def shot_downloader(out: Path, lang: str) -> None:
    """The model downloader (Manage models) with the Model Forge models, part-way through."""
    from voxprint_common.mdl.present import AppSlot, ItemState, QueueItem, Snapshot, view_for
    from voxprint_common.ui.mdl_window import grab_view

    items = [
        QueueItem(key, {"en": name, "ru": name}, name, size, int(size * share), ItemState(state), "huggingface",
                  42_000_000.0 if state == "downloading" else 0.0, 600.0 if state == "downloading" else None,
                  ["model-forge"])
        for key, name, size, share, state in MDL_ROWS
    ]
    snapshot = Snapshot(items=items, apps=[AppSlot("model-forge")])
    snapshot.recompute()
    grab_view(view_for(snapshot, mode="queue", focus_app="model-forge", lang=lang), out / f"{lang}-mdl-download.png")


def main(out: Path, manual: bool = False) -> None:
    home = Path(tempfile.mkdtemp(prefix="forge-shots-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos, write_prepared
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.engine import EngineEvent, Outcome
    from voxprint_forge.app.window import ForgeApp

    out.mkdir(parents=True, exist_ok=True)
    photos = placeholder_photos(home / "photos", 14) if manual else make_photos(home / "photos", 14)
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
            second = "lora.drop.duplicate" if manual else "lora.drop.safety"
            drops = [(photos[12], "lora.drop.blurry"), (photos[13], second)]
            write = demo_prepared if manual else write_prepared
            runner.effects["prepare"] = lambda project, write=write: write(project, photos[:12], drops)
            forge = ForgeApp(runner=runner, home=home / f"state-{lang}-{expert}", commands=ALL_COMMANDS,
                             open_path=lambda _p: None, launch=lambda _a: None)
            forge.expert.set_enabled(expert)
            forge.window.show()
            mode = "expert" if expert else "simple"

            def shot(name: str) -> None:
                if manual and name not in MANUAL[mode]:
                    return
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
            shot("3-build")
            next_button.click()
            forge.prepare()
            shot("4-prepare")
            next_button.click()
            shot("5-train")
            if not manual or "5-train-base-open" in MANUAL[mode]:
                shot_open_base(forge, app, out / f"{lang}-{mode}-5-train-base-open.png")
            from voxprint_forge.app.details import Details

            for details in forge.window.findChildren(Details):
                details.button.setChecked(True)
            shot("5-train-details-open")
            base_details = forge.window.findChild(Details, "forge.details.train.base")
            scroll = base_details.parentWidget().parentWidget().parentWidget()  # page -> viewport -> scroll area
            scroll.ensureWidgetVisible(base_details.body)
            shot("5-train-details-open-base")
            forge.window.resize(1024, 768)  # the smallest window we check; both Details open, scrolled to Start
            scroll_to_start(forge, app, scroll)
            shot("5-train-details-open-start")
            forge.window.resize(wide)
            for details in forge.window.findChildren(Details):
                details.button.setChecked(False)
            runner.hold = True
            forge.train()
            shot("5-train-running")
            runner.release()
            next_button.click()
            shot("6-result")
            forge.widget._layout.sections.select("check")
            forge.widget.refresh()
            shot("check")
            forge.window.close()
            forge.expert.set_enabled(False)
        if manual:
            shot_downloader(out, lang)


def scroll_to_start(forge, app, scroll) -> None:
    """Scroll the Train step so the Estimate/Start/Pause/Stop row sits at the bottom of the view."""
    from PySide6 import QtCore, QtWidgets
    from voxprint_common.i18n import tr

    start = next(b for b in forge.window.findChildren(QtWidgets.QPushButton) if b.text() == tr("forge.app.train.start"))
    app.processEvents()
    y = start.mapTo(scroll.widget(), QtCore.QPoint(0, start.height())).y()
    scroll.verticalScrollBar().setValue(max(0, y + 8 - scroll.viewport().height()))


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
    main(Path(sys.argv[1]), manual="--manual" in sys.argv[2:])
