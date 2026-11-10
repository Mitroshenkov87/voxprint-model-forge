"""``lora prepare --base`` stores a base list. Captions stay base-free.

Legacy projects (no ``bases`` list) keep ``checkpoints/`` and ``export/`` at the
project root. A new project records Z-Image unless a base was chosen.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.cli import build_parser
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import dataset_digest
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project, bases_for_choice, train_bases

CLEAN = Verdict("no", "no", None, 0.9, "no_person", False, {key: "absent" for key in INDICATORS})
TRIGGER = "vxpbcdfgh"


class _Judge:
    name = "fake-judge"

    def judge(self, image):
        return CLEAN

    def unload(self):
        return None


class _Estimator:
    name = "fake-estimator"

    def estimate(self, image):
        return AgeReading(0, ())


class _Captioner:
    name = "fake-captioner"

    def caption(self, image):
        return {"long": "a textured square on a plain background", "short": "textured square"}

    def unload(self):
        return None


def _subject():
    return check_subject("object", name="Blue mug", rights_confirmed=True, trigger=TRIGGER)


def _photos(folder, count: int = 5):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"img{index:02d}.png"
        rng = np.random.default_rng(index + 3)
        Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def _run(root, files, **kw):
    return prepare(
        Project(root),
        files,
        kw.pop("subject", _subject()),
        gate=Gate(_Judge(), _Estimator()),
        load_captioner=lambda _choice: _Captioner(),
        progress=Progress(),
        mode="expert",
        **kw,
    )


def test_prepare_flag_defaults_to_z_image_and_maps_names(tmp_path) -> None:
    omitted = build_parser().parse_args(["lora", "prepare", "--project", str(tmp_path)])
    assert omitted.base == "z-image" and omitted.base_explicit is False
    chosen = build_parser().parse_args(["lora", "prepare", "--project", str(tmp_path), "--base", "both"])
    assert chosen.base == "both" and chosen.base_explicit is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(["lora", "prepare", "--project", str(tmp_path), "--base", "tiny-test"])
    assert bases_for_choice("moss") == ["moss-image-base"]
    assert bases_for_choice("z-image") == ["z-image"]
    assert bases_for_choice("both") == ["z-image", "moss-image-base"]
    assert bases_for_choice("tiny-test") == ["tiny-test"]


def test_train_base_flag_follows_the_project_unless_it_was_passed(tmp_path) -> None:
    omitted = build_parser().parse_args(["lora", "train", "--project", str(tmp_path)])
    assert omitted.base == "z-image" and omitted.base_explicit is False
    both = build_parser().parse_args(["lora", "train", "--project", str(tmp_path), "--base", "both"])
    assert both.base == "both" and both.base_explicit is True
    assert train_bases("moss-image-base", ["z-image"], explicit=False) == ["z-image"]
    assert train_bases("both", ["moss-image-base", "z-image"], explicit=True) == ["moss-image-base", "z-image"]
    assert train_bases("both", ["z-image"], explicit=True) == ["z-image", "moss-image-base"]
    assert train_bases("moss", ["z-image"], explicit=True) == ["moss-image-base"]
    assert train_bases("Z-Image", ["moss-image-base"], explicit=True) == ["z-image"]
    assert train_bases("tiny-test", ["z-image"], explicit=True) == ["tiny-test"]


def test_prepare_stores_bases_and_keeps_one_base_free_dataset(tmp_path) -> None:
    digests = {}
    choices = {
        "moss": ["moss-image-base"],
        "z-image": ["z-image"],
        "both": ["moss-image-base", "z-image"],
    }
    for name, bases in choices.items():
        root = tmp_path / name
        _run(root / "proj", _photos(root / "in"), bases=bases)
        project = Project(root / "proj")
        assert project.read()["bases"] == bases
        rows = [json.loads(line) for line in project.captions.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows and all("base" not in row for row in rows)
        digests[name] = dataset_digest(project.items())
    assert digests["moss"] == digests["z-image"] == digests["both"]


def test_a_new_prepare_records_z_image_and_a_legacy_folder_keeps_its_paths(tmp_path) -> None:
    fresh = tmp_path / "fresh"
    _run(fresh, _photos(fresh / "in"))
    fresh_project = Project(fresh)
    assert fresh_project.read()["bases"] == ["z-image"]
    assert fresh_project.checkpoints == fresh / "checkpoints" / "z-image"
    assert fresh_project.export == fresh / "export" / "z-image"

    legacy = Project(tmp_path / "legacy")
    legacy.write({"kind": "object", "mode": "expert", "subject": _subject().to_dict()})
    _run(legacy.root, _photos(tmp_path / "legacy-in"))
    data = legacy.read()
    assert "bases" not in data
    assert legacy.legacy_layout() is True
    assert legacy.recorded_bases() == ["z-image"]
    assert legacy.checkpoints == legacy.root / "checkpoints"
    assert legacy.export == legacy.root / "export"

    switched = Project(tmp_path / "switched")
    switched.write({"kind": "object", "train": {"base": "moss-image-base"}})
    assert switched.legacy_layout() is True
    assert switched.recorded_bases() == ["moss-image-base"]
    assert switched.checkpoints == switched.root / "checkpoints"

    _run(switched.root, _photos(tmp_path / "switched-in"), bases=["z-image"])
    assert switched.read()["bases"] == ["z-image"]
    assert switched.legacy_layout() is False
    # A recorded train.base still names the cold folder. A new run calls activate_base.
    assert switched.checkpoints == switched.root / "checkpoints" / "moss-image-base"
    switched.activate_base("z-image")
    assert switched.checkpoints == switched.root / "checkpoints" / "z-image"
    assert switched.export == switched.root / "export" / "z-image"

    multi = Project(tmp_path / "multi")
    multi.write({"bases": ["moss-image-base", "z-image"], "train": {"base": "z-image"}})
    assert multi.checkpoints == multi.root / "checkpoints" / "z-image"
    multi.activate_base("moss-image-base")
    assert multi.checkpoints == multi.root / "checkpoints" / "moss-image-base"
    assert multi.export == multi.root / "export" / "moss-image-base"
    cold = Project(multi.root)
    assert cold.checkpoints == multi.root / "checkpoints" / "z-image"
