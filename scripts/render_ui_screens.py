"""Render the main Model Forge screens offscreen, in English and Russian, for UI review.

Usage:
    QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu \
        python scripts/render_ui_screens.py OUT_DIR [--scale 2]

Each screen is grabbed with the window at 1920x1080 (``--scale 2`` grabs a 1280x720 window at
devicePixelRatio 2, a HiDPI check, and only the screens in ``HIDPI``). Pictures are saved as
8-bit palette PNGs to keep them small. A stub engine answers every ``lora`` command; the photo
check passes through a stub job, so no model is loaded and nothing is trained.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZE = (1920, 1080)
HIDPI_SIZE = (1280, 720)
# Screens grabbed again at devicePixelRatio 2.
HIDPI = ("1-photos-red", "5-train-expert")


def main(out: Path, scale: int) -> None:
    if scale != 1:
        os.environ["QT_SCALE_FACTOR"] = str(scale)
    home = Path(tempfile.mkdtemp(prefix="forge-ui-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos, write_prepared
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.engine import EngineEvent, Outcome
    from voxprint_forge.app.window import ForgeApp
    from voxprint_forge.ui.check_gate import GateState, ItemResult, join_workers

    out.mkdir(parents=True, exist_ok=True)
    photos = make_photos(home / "photos", 14)
    size = HIDPI_SIZE if scale != 1 else SIZE
    suffix = f"@{scale}x" if scale != 1 else ""
    for lang in ("en", "ru"):
        set_language(lang)
        runner = StubRunner({
            "prepare": ([], Outcome(True, {"state": "done", "prepared": 12})),
            "train": ([EngineEvent("train", "lora.progress.train", {"step": 450, "steps": 1500},
                                   "lora.reason.train", 30.0, 1260)], Outcome(True, {"state": "done"})),
        })
        drops = [(photos[12], "lora.drop.blurry"), (photos[13], "lora.drop.safety")]
        runner.effects["prepare"] = lambda project: write_prepared(project, photos[:12], drops)
        forge = ForgeApp(runner=runner, home=home / f"state-{lang}-{scale}", commands=ALL_COMMANDS,
                         open_path=lambda _p: None, launch=lambda _a: None)
        forge.window.resize(*size)
        forge.window.show()
        next_button = forge.widget._parts()["next"]

        def shot(name: str) -> None:
            if scale != 1 and name not in HIDPI:
                return
            for _ in range(3):
                app.processEvents()
            image = forge.window.grab().toImage()
            image = image.convertToFormat(QtGui.QImage.Format.Format_Indexed8,
                                          QtGui.Qt.ImageConversionFlag.ThresholdDither)
            image.save(str(out / f"{lang}-{name}{suffix}.png"), "PNG", 9)

        forge.draft.add_photos(photos)
        forge.draft.kind = "person"
        forge.repaint()
        shot("1-photos-red")
        gate = forge.photo_gate
        gate._job = lambda item, _token: ItemResult(item.id, True)
        gate.start()
        for _ in range(400):
            app.processEvents()
            if gate.state is GateState.PASSED:
                break
            time.sleep(0.01)
        forge.repaint()
        shot("1-photos-green")
        next_button.click()
        forge.draft.set_names("Mira Demo", "")
        forge.draft.consent = forge.draft.adult = True
        forge.draft.set_verbal(True)
        forge.repaint()
        shot("2-subject")
        next_button.click()
        forge.prepare()
        shot("4-prepare")
        next_button.click()
        shot("5-train")
        forge.expert.set_enabled(True)
        forge.repaint()
        shot("5-train-expert")
        forge.expert.set_enabled(False)
        forge.repaint()
        runner.hold = True
        forge.train()
        shot("5-train-running")
        runner.release()
        next_button.click()
        shot("6-result")
        forge.widget._layout.sections.select("check")
        forge.widget.refresh()
        shot("check")
        forge.photo_gate.shutdown()
        forge.window.close()
    join_workers()


if __name__ == "__main__":
    flags = sys.argv[2:]
    main(Path(sys.argv[1]), int(flags[flags.index("--scale") + 1]) if "--scale" in flags else 1)
