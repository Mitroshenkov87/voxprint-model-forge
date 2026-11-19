"""Build the demo fixture the GUI tests and the screenshot script use.

Usage: QT_QPA_PLATFORM=offscreen python scripts/forge_gui_shots.py OUT_DIR
   or: QT_QPA_PLATFORM=offscreen python scripts/forge_gui_demo.py --root DIR

Under ``--root`` it creates:

* ``photos/`` — 12 generated abstract pictures (gradients and shapes, no people,
  no real photos) plus a complete ``release.txt`` for the fictional adult
  "Demo Model";
* ``projects/running/`` — a project folder mid-training: ``train.json`` at stage
  ``train``, step 640/1600 with loss, elapsed time and ETA, ``vram.csv`` samples
  and two abstract preview pictures;
* ``projects/done/`` — a finished project: ``train.json`` stage ``done``, a small
  dummy ``lora.safetensors``, its ``voxprint-model.json`` (trigger, default
  strength 1.3, the file's real sha256) and previews.

Nothing is trained, no GPU is used and no network is touched; the pictures are
drawn by the script itself. The tests drive the real window offscreen against
these folders; nothing here ever starts a training job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHOTO_COUNT = 12
TRIGGER = "vxpdmdlz"  # a rare vxp token, the same shape the trainer generates
DEFAULT_STRENGTH = 1.3

RELEASE_TEXT = (
    'Model name: "Demo Model"\n'
    "Adult (18+): yes\n"
    "Consent to training: yes\n"
    "Consent to generation: yes\n"
    "LoRA training allowed: yes\n"
    "Adult content in training: no\n"
    "Adult content in generation: no\n"
    "No minors in photos: yes\n"
    "Date: 2026-11-16\n"
)

RUNNING_ROW = {
    "step": 640,
    "total": 1600,
    "loss": 0.1847,
    "elapsed_seconds": 1500.0,
    "eta_seconds": 2250,
    "stage": "train",
}

VRAM_ROWS = (
    "elapsed_seconds,allocated_gb,reserved_gb\n"
    "300.000,8.375,9.125\n"
    "900.000,9.500,10.250\n"
    "1500.000,9.500,10.250\n"
)

# Deterministic bytes stand in for trained weights; the sidecar records their real hash.
LORA_BYTES = b"VOXPRINT-DEMO-LORA-DO-NOT-LOAD\n"


def abstract_image(path: Path, index: int, side: int = 512) -> Path:
    """One generated abstract picture: a gradient with simple shapes. No people, no metadata."""
    from PySide6 import QtCore, QtGui

    hues = (24, 200, 140, 280, 48, 330, 180, 100, 220, 10, 260, 160)
    hue = hues[index % len(hues)]
    image = QtGui.QImage(side, side, QtGui.QImage.Format.Format_RGB32)
    gradient = QtGui.QLinearGradient(0, 0, side, side)
    gradient.setColorAt(0, QtGui.QColor.fromHsv(hue, 60, 235))
    gradient.setColorAt(1, QtGui.QColor.fromHsv((hue + 40) % 360, 110, 190))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.fillRect(image.rect(), gradient)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor.fromHsv((hue + 160) % 360, 90, 220))
    painter.drawEllipse(QtCore.QPointF(side * 0.32, side * 0.38), side * 0.18, side * 0.18)
    painter.setBrush(QtGui.QColor.fromHsv((hue + 220) % 360, 120, 170))
    painter.drawRect(QtCore.QRectF(side * 0.52, side * 0.42, side * 0.26, side * 0.26))
    painter.setBrush(QtGui.QColor.fromHsv((hue + 60) % 360, 70, 240))
    points = [
        QtCore.QPointF(side * 0.30, side * 0.78),
        QtCore.QPointF(side * 0.44, side * 0.60),
        QtCore.QPointF(side * 0.58, side * 0.78),
    ]
    painter.drawPolygon(points)
    painter.setPen(QtGui.QColor(50, 50, 56))
    font = painter.font()
    font.setPixelSize(max(14, side // 8))
    painter.setFont(font)
    painter.drawText(
        QtCore.QRectF(0, side - side // 4, side, side // 6),
        QtCore.Qt.AlignmentFlag.AlignCenter,
        f"{index + 1:02d}",
    )
    painter.end()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    return path


def write_photos(folder: Path, count: int = PHOTO_COUNT) -> list[Path]:
    """The demo photo folder: generated abstract pictures and a complete release.txt."""
    photos = [abstract_image(folder / f"demo-{index + 1:02d}.png", index) for index in range(count)]
    (folder / "release.txt").write_text(RELEASE_TEXT, encoding="utf-8", newline="\n")
    return photos


def write_running_project(project: Path) -> Path:
    """A project folder as a running training leaves it: fresh train.json, vram.csv, previews."""
    project.mkdir(parents=True, exist_ok=True)
    subject = {
        "kind": "person",
        "name": "Demo Model",
        "stage_name": "",
        "trigger": TRIGGER,
        "consent_obtained": True,
        "adult_confirmed": True,
        "consent_method": "file",
    }
    (project / "project.json").write_text(
        json.dumps({"schema": "voxprint-lora-project/1", "kind": "person", "subject": subject}, indent=2) + "\n",
        encoding="utf-8",
    )
    (project / "train.json").write_text(json.dumps(RUNNING_ROW, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (project / "vram.csv").write_text(VRAM_ROWS, encoding="utf-8")
    previews = project / "previews"
    abstract_image(previews / "step-000640-0.png", 0, side=256)
    abstract_image(previews / "step-000640-1.png", 1, side=256)
    return project


def write_done_project(project: Path) -> Path:
    """A finished project: dummy lora.safetensors, sidecar with trigger/strength/sha256, previews."""
    project.mkdir(parents=True, exist_ok=True)
    subject = {
        "kind": "person",
        "name": "Demo Model",
        "stage_name": "",
        "trigger": TRIGGER,
        "consent_obtained": True,
        "adult_confirmed": True,
        "consent_method": "file",
    }
    train = {"state": "done", "step": 1600, "steps_planned": 1600, "checkpoint": "lora.safetensors"}
    (project / "project.json").write_text(
        json.dumps(
            {"schema": "voxprint-lora-project/1", "kind": "person", "gate_passed": True,
             "subject": subject, "train": train},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (project / "train.json").write_text(
        json.dumps({"step": 1600, "total": 1600, "loss": 0.0912, "elapsed_seconds": 3700.0,
                    "eta_seconds": 0, "stage": "done"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lora = project / "lora.safetensors"
    lora.write_bytes(LORA_BYTES)
    sidecar = {
        "schema": "voxprint-model/1",
        "min_schema": 1,
        "id": f"{TRIGGER}-demodem",
        "type": "lora",
        "file": lora.name,
        "trigger": TRIGGER,
        "sha256": hashlib.sha256(LORA_BYTES).hexdigest(),
        "size": len(LORA_BYTES),
        "default_strength": DEFAULT_STRENGTH,
        "strength_range": [0.0, 1.5],
    }
    (project / "voxprint-model.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    previews = project / "previews"
    abstract_image(previews / "step-001600-0.png", 2, side=256)
    return project


def build_demo(root: Path) -> dict[str, Path]:
    """Create the whole fixture under ``root`` and return the created paths by name."""
    root = Path(root)
    write_photos(root / "photos")
    running = write_running_project(root / "projects" / "running")
    done = write_done_project(root / "projects" / "done")
    return {"root": root, "photos": root / "photos", "running": running, "done": done}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="fixture folder to create")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    made = build_demo(args.root)
    for name in ("photos", "running", "done"):
        print(f"{name}: {made[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
