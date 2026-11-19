"""The safety check cuts photos into size-aware chunks but writes rows in input order. Fakes only, no models."""

from __future__ import annotations

import numpy as np
from PIL import Image

from tests.test_lora_prepare_batch import BatchJudge, ScriptEstimator, _run, _trace, key, keys_of
from voxprint_forge.lora.batching import size_class
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import Gate
from voxprint_forge.lora.photos import photo_id
from voxprint_forge.lora.project import Project

LIGHT = (600, 600)
MEDIUM = (1600, 1200)
HEAVY = (4000, 3000)
# Input order (sorted file names): light chunks have to wait for the medium and heavy photos
# between them, so the trace is written out of chunk order.
LAYOUT = [LIGHT, LIGHT, MEDIUM, LIGHT, LIGHT, HEAVY, LIGHT, MEDIUM, LIGHT, LIGHT, HEAVY, LIGHT]


def _block_noise(seed: int, size: tuple[int, int], block: int = 4) -> Image.Image:
    """Sharp random blocks: passes the blur filter like the existing noise photos, compresses better."""
    width, height = size
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (height // block, width // block, 3), dtype=np.uint8)
    return Image.fromarray(np.repeat(np.repeat(small, block, axis=0), block, axis=1))


def _mixed(folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, size in enumerate(LAYOUT):
        path = folder / f"img{index:02d}.png"
        _block_noise(100 + index, size).save(path)
        paths.append(path)
    return paths


def _untriggered(caption: str) -> str:
    return caption.split(", ", 1)[-1]


def _prepared(root):
    """Items (in order, with the per-project random trigger word cut from the captions) and drops."""
    project = Project(root / "proj")
    items = [(item.id, item.file, _untriggered(item.long), _untriggered(item.short)) for item in project.items()]
    drops = project.drops.read_text() if project.drops.is_file() else ""
    return items, drops


class ChunkGate(Gate):
    """A real gate that records the pixel keys of every chunk prepare hands it."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.chunks: list[list[str]] = []

    def check_batch(self, images, **kw):
        self.chunks.append([key(image) for image in images])
        return super().check_batch(images, **kw)


def _go(root, files, monkeypatch, batch):
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", str(batch))
    judge, estimator = BatchJudge(), ScriptEstimator()
    gate = ChunkGate(judge, estimator)
    progress = Progress()
    _run(root, files, judge, estimator, progress=progress, gate=gate)
    return gate, judge, estimator, progress


def test_size_aware_chunks_keep_the_input_order_and_one_event_per_chunk(tmp_path, monkeypatch, capsys) -> None:
    files = _mixed(tmp_path / "in")
    expected = keys_of(files)
    kind = {k: size_class(*size) for k, size in zip(expected, LAYOUT, strict=True)}

    gate4, judge4, estimator4, progress4 = _go(tmp_path / "b4", files, monkeypatch, 4)
    err = capsys.readouterr().err
    gate1, judge1, estimator1, _ = _go(tmp_path / "b1", files, monkeypatch, 1)

    ids = [photo_id(path) for path in files]
    assert [row["id"] for row in _trace(tmp_path / "b4")] == ids
    assert _trace(tmp_path / "b4") == _trace(tmp_path / "b1")
    assert _prepared(tmp_path / "b4") == _prepared(tmp_path / "b1")

    # Every photo is checked exactly once (a one-photo chunk goes to ``judge``, not ``judge_batch``).
    for gate, judge, estimator in ((gate4, judge4, estimator4), (gate1, judge1, estimator1)):
        judged = judge.single + [k for batch in judge.batches for k in batch]
        assert sorted(judged) == sorted(expected)
        assert sorted(estimator.seen) == sorted(expected)
        assert sorted(k for chunk in gate.chunks for k in chunk) == sorted(expected)
    assert gate1.chunks == [[k] for k in expected]

    # Batch 4: one size class per chunk, at most 4 light, 2 medium and 1 heavy photo, in plan order.
    limit = {"light": 4, "medium": 2, "heavy": 1}
    for chunk in gate4.chunks:
        classes = {kind[k] for k in chunk}
        assert len(classes) == 1
        assert len(chunk) <= limit[classes.pop()]
    order = [[0, 1, 3, 4], [2, 7], [5], [6, 8, 9, 11], [10]]
    assert gate4.chunks == [[expected[i] for i in chunk] for chunk in order]
    assert [kind[k] for k in judge4.single] == ["heavy", "heavy"]

    safety = [event for event in progress4.events if event.step == "prepare.safety"]
    assert len(safety) == len(gate4.chunks)
    assert [event.params["n"] for event in safety] == [1, 5, 7, 8, 12]
    percents = [event.percent for event in safety]
    assert percents[0] == 0.0 and percents == sorted(percents)
    assert "prepare: batch size 4 (check)" in err
    assert "prepare: check chunks 5 (light 8, medium 2, heavy 2 photos)" in err
