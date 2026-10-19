"""Grab each Model Forge step (Simple and Expert, en and ru) with a stub engine.

Usage: QT_QPA_PLATFORM=offscreen python scripts/forge_app_shots.py OUT_DIR [--manual] [--photos DIR]
Photos are generated neutral placeholders (no people), or the first 14 pictures in ``--photos DIR`` by file name,
downscaled and saved again without metadata. Nothing is trained and no GPU is used.
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


def sample_photos(source: Path, folder: Path, count: int, side: int = 512) -> list[Path]:
    """The first ``count`` pictures in ``source``, at most ``side`` px on the long edge, saved as PNG, no metadata."""
    from PySide6 import QtCore, QtGui

    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:count]
    if len(files) < count:
        raise SystemExit(f"--photos needs at least {count} pictures, found {len(files)} in {source}")
    paths = []
    for index, file in enumerate(files):
        image = QtGui.QImage(str(file))
        if image.isNull():
            raise SystemExit(f"cannot read {file}")
        image = image.scaled(side, side, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
        clean = QtGui.QImage(image.size(), QtGui.QImage.Format.Format_RGB32)
        painter = QtGui.QPainter(clean)
        painter.drawImage(0, 0, image)
        painter.end()
        path = folder / f"demo-{index + 1:02d}.png"
        clean.save(str(path))
        paths.append(path)
    return paths


def demo_prepared(project: Path, photos: list[Path], drops: list[tuple[Path, str]], samples: bool = False) -> None:
    """Like the test stub's ``write_prepared``, with descriptions that read naturally in the manual."""
    import json

    project.mkdir(parents=True, exist_ok=True)
    if samples:
        texts = [(f"mira_demo, sample photo {i + 1:02d}, a woman with long blonde hair",
                  f"mira_demo, sample photo {i + 1:02d}") for i in range(len(photos))]
    else:
        texts = [(f"mira_demo, placeholder picture {i + 1:02d}, a circle above a rounded square on a soft background",
                  f"mira_demo, placeholder picture {i + 1:02d}") for i in range(len(photos))]
    rows = [{"id": f"p{i}", "file": str(photo), "long": texts[i][0], "short": texts[i][1], "round": 0}
            for i, photo in enumerate(photos)]
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


def shot_review_rounds(out: Path, home: Path, photos: list[Path]) -> None:
    """1280 px grabs: the photos switch on (ru) and a review with mixed marks (en and ru)."""
    import json

    from PySide6 import QtCore, QtGui, QtWidgets
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.draft import STEPS
    from voxprint_forge.app.window import ForgeApp
    from voxprint_forge.lora.project import Project
    from voxprint_forge.lora.review import save_state, transition

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    out.mkdir(parents=True, exist_ok=True)
    chosen = photos[:4]

    def seed(project: Path) -> None:
        root = Project(project)
        root.prepared.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, photo in enumerate(chosen):
            name = f"prepared/p{index:02d}.png"
            (project / name).write_bytes(photo.read_bytes())
            rows.append({"id": f"p{index:02d}", "file": name, "long": "a soft shape", "short": "shape", "round": 0})
        (project / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        root.write({
            "kind": "object", "gate_passed": True,
            "subject": {"name": "Soft shape", "trigger": "soft_shape", "rights_confirmed": True, "kind": "object"},
        })
        state = transition(
            {}, "start", rounds=3, samples_per_round=8, first_round_steps=450, first_round_samples=6,
            sample_steps=28, total_steps=1500, prompts=["a soft shape"], seed=3,
            base="draw-test", config={"steps": 1500},
        )
        verdicts = ("like", "dislike", "unmarked", "like", "dislike", "unmarked")
        samples = []
        folder = project / "review" / "round-0001"
        folder.mkdir(parents=True, exist_ok=True)
        for index, verdict in enumerate(verdicts):
            image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
            image.fill(QtGui.qRgb(30 + index * 36, 90 + (index % 3) * 40, 150))
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QColor(40, 40, 40))
            font = painter.font()
            font.setPixelSize(72)
            painter.setFont(font)
            painter.drawText(image.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, f"{index + 1}")
            painter.end()
            image.save(str(folder / f"s{index:02d}.png"))
            samples.append({
                "id": f"r1-s{index:02d}",
                "file": f"review/round-0001/s{index:02d}.png",
                "seed": index,
                "prompt": "a soft shape",
                "round": 1,
                "verdict": verdict,
            })
        save_state(root, transition(
            state, "pause", checkpoint="checkpoints/step-000001.safetensors", samples=samples,
        ))

    for lang in ("ru", "en"):
        set_language(lang)
        forge = ForgeApp(
            runner=StubRunner(), home=home / f"review-{lang}", commands=ALL_COMMANDS,
            open_path=lambda _p: None, launch=lambda _a: None,
        )
        forge.window.resize(1280, 960)
        forge.window.show()
        app.processEvents()
        if lang == "ru":
            forge.draft.kind = "object"
            forge.draft.add_photos(chosen)
            forge.repaint()
            forge.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").click()
            app.processEvents()
            forge.window.grab().save(str(out / "ru-simple-1-photos-extend.png"))
        seed(forge.project)
        forge.layout.sections._index = STEPS.index("train")
        forge.repaint()
        app.processEvents()
        button = forge.window.findChild(QtWidgets.QToolButton, "forge.review.continue")
        note = forge.window.findChild(QtWidgets.QLabel, "forge.review.note")
        anchor = note if note is not None else button
        area = anchor.parentWidget() if anchor is not None else None
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        if area is not None and anchor is not None:
            area.ensureWidgetVisible(anchor, 0, 120)
            app.processEvents()
        if lang == "ru" and button is not None and button.menu() is not None:
            menu = button.menu()
            menu.popup(button.mapToGlobal(QtCore.QPoint(0, button.height())))
            app.processEvents()
            image = forge.window.grab()
            popup = menu.grab()
            point = button.mapTo(forge.window, QtCore.QPoint(0, button.height()))
            painter = QtGui.QPainter(image)
            painter.drawPixmap(point, popup)
            painter.end()
            image.save(str(out / "ru-simple-5-review.png"))
            menu.close()
            app.processEvents()

            def grab_confirm() -> None:
                box = QtWidgets.QApplication.activeModalWidget()
                if not isinstance(box, QtWidgets.QMessageBox):
                    QtCore.QTimer.singleShot(0, grab_confirm)
                    return
                base = forge.window.grab()
                dialog = box.grab()
                painter = QtGui.QPainter(base)
                x = max(0, (base.width() - dialog.width()) // 2)
                y = max(0, (base.height() - dialog.height()) // 2)
                painter.drawPixmap(x, y, dialog)
                painter.end()
                base.save(str(out / "ru-simple-5-finish-confirm.png"))
                box.reject()

            QtCore.QTimer.singleShot(0, grab_confirm)
            forge.finish_review()
            app.processEvents()
        else:
            forge.window.grab().save(str(out / f"{lang}-simple-5-review.png"))
        forge.window.close()
    shot_train_live(out, home)
    set_language("en")


def shot_train_live(out: Path, home: Path) -> None:
    """1280 px: the training step while a stage is running, with ``train.json`` on the bar."""
    import json

    from PySide6 import QtWidgets
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.draft import Prepared, STEPS
    from voxprint_forge.app.window import ForgeApp
    from voxprint_forge.lora.project import Project
    from voxprint_forge.lora.review import save_state, transition

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    set_language("ru")
    photos = placeholder_photos(home / "live-photos", 12)
    runner = StubRunner()
    runner.busy = True
    forge = ForgeApp(
        runner=runner, home=home / "train-live", commands=ALL_COMMANDS,
        open_path=lambda _p: None, launch=lambda _a: None,
    )
    forge.draft.kind = "object"
    forge.draft.add_photos(photos)
    forge.draft.set_names("Soft shape", "")
    forge.draft.rights = True
    forge.draft.prepared = Prepared(
        [{"id": f"p{i:02d}", "file": str(photos[i]), "long": "a soft shape", "short": "shape"} for i in range(12)],
        [],
        False,
    )
    forge.project.mkdir(parents=True, exist_ok=True)
    save_state(Project(forge.project), transition(
        {}, "start", rounds=3, samples_per_round=8, first_round_steps=200, first_round_samples=6,
        sample_steps=28, total_steps=1500, prompts=["a soft shape"], seed=1,
        base="draw-test", config={"steps": 1500},
    ))
    (forge.project / "train.json").write_text(json.dumps({
        "elapsed_seconds": 90,
        "eta_seconds": 400,
        "loss": 0.21,
        "stage": "step",
        "step": 40,
        "total": 200,
        "round": 1,
        "first_pictures_seconds": 480,
        "seconds_per_sample": 18,
        "sample_seconds_estimated": False,
        "seconds_per_step": 2.2,
    }), encoding="utf-8")
    forge.window.resize(1280, 960)
    forge.window.show()
    forge.layout.sections._index = STEPS.index("train")
    forge.repaint()
    app.processEvents()
    forge.window.grab().save(str(out / "ru-simple-5-train-live.png"))
    forge.window.close()


def main(out: Path, manual: bool = False, samples: Path | None = None) -> None:
    home = Path(tempfile.mkdtemp(prefix="forge-shots-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos, write_prepared
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.engine import EngineEvent, Outcome
    from voxprint_forge.app.window import ForgeApp

    out.mkdir(parents=True, exist_ok=True)
    if samples is not None:
        photos = sample_photos(samples, home / "photos", 14)
    elif manual:
        photos = placeholder_photos(home / "photos", 14)
    else:
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
            second = "lora.drop.duplicate" if manual else "lora.drop.safety"
            drops = [(photos[12], "lora.drop.blurry"), (photos[13], second)]
            if manual or samples is not None:
                with_samples = samples is not None
                runner.effects["prepare"] = lambda project: demo_prepared(project, photos[:12], drops, with_samples)
            else:
                runner.effects["prepare"] = lambda project: write_prepared(project, photos[:12], drops)
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
    if manual:
        shot_review_rounds(out, home, photos)


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
    flags = sys.argv[2:]
    given = Path(flags[flags.index("--photos") + 1]) if "--photos" in flags else None
    if "--review-only" in flags:
        home = Path(tempfile.mkdtemp(prefix="forge-shots-"))
        os.environ["VOXPRINT_HOME"] = str(home)
        from PySide6 import QtWidgets

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pictures = placeholder_photos(home / "photos", 6)
        shot_review_rounds(Path(sys.argv[1]), home, pictures)
    else:
        main(Path(sys.argv[1]), manual="--manual" in flags, samples=given)
