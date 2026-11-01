"""``drops.jsonl`` holds one line per set-aside photo, also after a re-run on the same project."""

from __future__ import annotations

import json

from PIL import Image

from tests.test_lora_prepare import noise, photos, run
from voxprint_forge.lora.photos import collect

# ``dup.jpg`` is a near-duplicate; it is only set aside with ``drop_near_duplicates``.


def _ids(tmp_path) -> list[str]:
    lines = (tmp_path / "proj" / "drops.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["id"] for line in lines]


def _inputs(tmp_path):
    photos(tmp_path / "in", 11)
    Image.new("RGB", (600, 600), (128, 128, 128)).save(tmp_path / "in" / "flat.png")
    noise(99, 200).save(tmp_path / "in" / "small.png")
    noise(3).save(tmp_path / "in" / "dup.jpg", quality=98)
    (tmp_path / "in" / "broken.png").write_bytes(b"not an image")
    return collect(tmp_path / "in", [], [])


def test_each_dropped_photo_is_written_once(tmp_path) -> None:
    result = run(tmp_path, _inputs(tmp_path), drop_near_duplicates=True)
    ids = _ids(tmp_path)
    assert result["set_aside"] == 4
    assert len(ids) == 4 and len(set(ids)) == 4


def test_rerun_does_not_repeat_drop_lines(tmp_path) -> None:
    files = _inputs(tmp_path)
    run(tmp_path, files, drop_near_duplicates=True)
    first = (tmp_path / "proj" / "drops.jsonl").read_text(encoding="utf-8")
    run(tmp_path, files, drop_near_duplicates=True)
    ids = _ids(tmp_path)
    assert len(ids) == 4 and len(set(ids)) == 4
    assert (tmp_path / "proj" / "drops.jsonl").read_text(encoding="utf-8") == first


def test_rerun_keeps_job_rows(tmp_path) -> None:
    files = _inputs(tmp_path)
    run(tmp_path, files, drop_near_duplicates=True)
    path = tmp_path / "proj" / "drops.jsonl"
    job = json.dumps({"id": "job", "name": "", "reason": "lora.drop.safety", "scope": "job"})
    path.write_text(path.read_text(encoding="utf-8") + job + "\n", encoding="utf-8")
    run(tmp_path, files, drop_near_duplicates=True)
    ids = _ids(tmp_path)
    assert ids.count("job") == 1 and len(ids) == 5 and len(set(ids)) == 5
