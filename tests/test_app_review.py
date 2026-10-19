"""Review screen, the sample-set switch, and the shared button radius."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtGui, QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos  # noqa: E402
from voxprint_forge.app.draft import STEPS  # noqa: E402
from voxprint_forge.app.text import tr  # noqa: E402
from voxprint_forge.app.window import FORGE_STYLE, button_radius_px  # noqa: E402
from voxprint_forge.lora.project import SCHEMA, Project  # noqa: E402
from voxprint_forge.lora.review import save_state, transition  # noqa: E402


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made = []

    def build(lang="en"):
        set_language(lang)
        app = ForgeApp(
            runner=StubRunner(), home=tmp_path / "state", commands=ALL_COMMANDS,
            open_path=lambda _p: None, launch=lambda _a: None,
        )
        app.window.show()
        qapp.processEvents()
        made.append(app)
        return app

    yield build
    for app in made:
        app.expert.set_enabled(False)
        app.window.close()
    set_language("en")


def test_square_buttons_use_the_shared_radius() -> None:
    radius = button_radius_px()
    assert radius == 4
    assert "border-radius: 11px" not in FORGE_STYLE
    assert f"border-radius: {radius}px" in FORGE_STYLE
    assert 'QPushButton[keepMark="true"]' in FORGE_STYLE
    assert 'QToolButton[photoRemove="true"]' in FORGE_STYLE
    assert 'QToolButton[selectionRemove="true"]' in FORGE_STYLE
    assert "QFrame#card" in FORGE_STYLE and "border-radius: 6px" in FORGE_STYLE


def test_the_sample_switch_adds_review_rounds_and_off_keeps_a_straight_train(make_app, tmp_path) -> None:
    app = make_app()
    photos = make_photos(tmp_path / "photos", 2)
    app.draft.kind = "object"
    app.draft.add_photos(photos)
    app.draft.set_source("refs")
    flagged = app._train_args()
    assert flagged[flagged.index("--review-rounds") + 1] == "3"
    assert flagged[flagged.index("--samples-per-round") + 1] == "8"
    app.draft.set_source("photos")
    plain = app._train_args()
    assert "--review-rounds" not in plain
    assert "--samples-per-round" not in plain
    dry = app._train_args(dry_run=True)
    assert "--review-rounds" not in dry


def _seed_review(project: Path, photos: list[Path]) -> None:
    root = Project(project)
    root.prepared.mkdir(parents=True, exist_ok=True)
    items = []
    for index, photo in enumerate(photos):
        name = f"prepared/p{index:02d}.png"
        target = project / name
        target.write_bytes(photo.read_bytes())
        items.append({"id": f"p{index:02d}", "file": name, "long": "a gradient", "short": "gradient", "round": 0})
    (project / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items), encoding="utf-8")
    root.write({
        "schema": SCHEMA,
        "kind": "object",
        "gate_passed": True,
        "subject": {"name": "Test cube", "trigger": "tcube", "rights_confirmed": True, "kind": "object"},
    })
    state = transition(
        {}, "start", rounds=3, samples_per_round=3, prompts=["a cube"], seed=1, base="draw-test", config={"steps": 1},
    )
    samples = []
    for index, verdict in enumerate(("like", "dislike", "unmarked")):
        folder = project / "review" / "round-0001"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"s{index:02d}.png"
        image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.qRgb(40 + index * 70, 80, 160))
        image.save(str(path))
        samples.append({
            "id": f"r1-s{index:02d}",
            "file": f"review/round-0001/s{index:02d}.png",
            "seed": index,
            "prompt": "a cube",
            "round": 1,
            "verdict": verdict,
        })
    state = transition(state, "pause", checkpoint="checkpoints/step-000001.safetensors", samples=samples)
    save_state(root, state)


def test_the_review_screen_counts_marks_and_offers_continue_and_finish(make_app, tmp_path, qapp) -> None:
    app = make_app("ru")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.round").text()
    counts = app.window.findChild(QtWidgets.QLabel, "forge.review.counts")
    assert counts.text() == tr("forge.app.review.counts", liked=1, disliked=1, unmarked=1)
    assert len(app.window.findChildren(QtWidgets.QLabel, "forge.review.original")) == 2
    like = app.window.findChild(QtWidgets.QPushButton, "forge.review.like.r1-s02")
    dislike = app.window.findChild(QtWidgets.QPushButton, "forge.review.dislike.r1-s00")
    assert like is not None and dislike is not None
    like.click()
    qapp.processEvents()
    assert counts.text() == tr("forge.app.review.counts", liked=2, disliked=1, unmarked=0)
    dislike.click()
    qapp.processEvents()
    assert counts.text() == tr("forge.app.review.counts", liked=1, disliked=2, unmarked=0)
    saved = json.loads((app.project / "review" / "state.json").read_text(encoding="utf-8"))
    verdicts = {sample["id"]: sample["verdict"] for sample in saved["samples"]}
    assert verdicts == {"r1-s00": "dislike", "r1-s01": "dislike", "r1-s02": "like"}
    app.window.findChild(QtWidgets.QPushButton, "forge.review.continue").click()
    qapp.processEvents()
    assert app.runner.calls[-1][:3] == ["--json", "lora", "continue"]
    assert app.runner.calls[-1][-1] == str(app.project)
    app.window.findChild(QtWidgets.QPushButton, "forge.review.finish").click()
    qapp.processEvents()
    assert app.runner.calls[-1][:3] == ["--json", "lora", "finish"]


def test_opening_a_paused_review_returns_to_the_training_step(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "paused"
    photos = make_photos(folder / "photos", 2)
    _seed_review(folder, photos)
    app.open_project(folder)
    qapp.processEvents()
    assert app.layout.sections.step.step_id == "train"
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.counts") is not None
    assert app.draft.rights is True
    assert app.draft.prepared is not None
