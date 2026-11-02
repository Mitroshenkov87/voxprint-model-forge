"""Likeness score (SFace): thresholds, no-face accounting, report files, model card, preview prompts
and preview PNG metadata. Everything here is synthetic: random unit vectors stand in for face
embeddings and procedural gradients stand in for pictures. No photos of people."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import zipfile
from pathlib import Path

import jsonschema
import pytest
from PIL import Image

from tests.lora_fakes import TinyAdapter, TinyNet, gradient, make_project
from voxprint_forge.lora import bases, likeness, likeness_worker
from voxprint_forge.lora import train as train_mod
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import export_package
from voxprint_forge.lora.layers import inject
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.sample_prompts import PERSON_PREVIEW, preview_prompt
from voxprint_forge.lora.train import PreviewStamp, read_preview_text, save_preview, train

ROOT = Path(__file__).resolve().parents[1]
DIM = 128


def _unit(seed: int) -> list[float]:
    rnd = random.Random(seed)
    return likeness.normalise([rnd.gauss(0, 1) for _ in range(DIM)])


def _near(center: list[float], spread: float, seed: int) -> list[float]:
    """A unit vector at a controlled distance from ``center``."""
    noise = _unit(seed)
    return likeness.normalise([c + spread * n for c, n in zip(center, noise, strict=True)])


def _row(vector: list[float] | None) -> dict[str, object]:
    return {"embedding": vector, "face_px": 120 if vector else 0}


# --- thresholds ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("mean", "baseline", "status"),
    [
        (0.30, None, "fail"),
        (0.3629, 0.40, "fail"),
        (0.363, None, "warn"),  # at the SFace line: not a fail, still below 0.50
        (0.45, 0.50, "warn"),
        (0.55, 0.95, "warn"),  # 0.55 < 0.6 x 0.95
        (0.55, 0.80, "pass"),  # 0.55 >= 0.48 and >= 0.50
        (0.50, None, "pass"),
        (None, 0.80, "warn"),  # no face in any picture is never a pass
    ],
)
def test_status_thresholds(mean, baseline, status) -> None:
    assert likeness.status_for(mean, baseline) == status


def test_reference_mean_and_leave_one_out_baseline() -> None:
    center = _unit(1)
    refs = [_near(center, 0.6, 100 + i) for i in range(8)]
    mean, baseline, loo = likeness.reference(refs)
    assert math.isclose(sum(v * v for v in mean), 1.0, rel_tol=1e-9)
    assert likeness.cosine(mean, center) > 0.9
    assert len(loo) == 8 and baseline is not None and 0.5 < baseline < 1.0
    expected = [likeness.cosine(refs[i], [sum(c) for c in zip(*(r for j, r in enumerate(refs) if j != i))])
                for i in range(8)]
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(loo, expected, strict=True))


def test_score_counts_no_face_apart_and_never_as_a_pass() -> None:
    center = _unit(2)
    refs = [_row(_near(center, 0.5, 200 + i)) for i in range(6)] + [_row(None)]
    images = [
        ("same-1", _row(_near(center, 0.4, 300))),
        ("same-2", _row(_near(center, 0.4, 301))),
        ("other", _row(_unit(999))),
        ("blank", _row(None)),
    ]
    result = likeness.score(refs, images)
    summary, rows = result["summary"], result["rows"]
    assert summary["n"] == 3 and summary["no_face"] == 1
    assert summary["refs"] == 6 and summary["refs_no_face"] == 1
    assert summary["threshold"] == likeness.FAIL_BELOW
    blank = next(r for r in rows if r["image"] == "blank")
    assert blank == {"image": "blank", "cos": None, "face": False, "face_px": 0, "pass": False}
    assert next(r for r in rows if r["image"] == "other")["pass"] is False
    assert all(next(r for r in rows if r["image"] == n)["pass"] for n in ("same-1", "same-2"))
    cosines = [r["cos"] for r in rows if r["cos"] is not None]
    assert summary["median"] == pytest.approx(sorted(cosines)[1], abs=1e-4)


def test_all_pictures_without_a_face_is_a_warning_not_a_pass() -> None:
    refs = [_row(_unit(i)) for i in range(3)]
    summary = likeness.score(refs, [("a", _row(None)), ("b", _row(None))])["summary"]
    assert summary["n"] == 0 and summary["no_face"] == 2 and summary["mean"] is None
    assert summary["status"] == "warn"


def test_no_reference_face_cannot_run() -> None:
    with pytest.raises(likeness.LikenessUnavailable):
        likeness.score([_row(None)], [("a", _row(_unit(1)))])


# --- the "no face" path on a picture with no person ---------------------------------------------

class _NoFaceDetector:
    """Stands in for cv2.FaceDetectorYN: a smooth gradient has no face at any score."""

    def __init__(self) -> None:
        self.scores: list[float] = []

    def setInputSize(self, size) -> None:  # noqa: N802 - OpenCV method name
        self.size = size

    def setScoreThreshold(self, score: float) -> None:  # noqa: N802 - OpenCV method name
        self.scores.append(score)

    def detect(self, image):
        return 1, None


class _FakeCv2:
    """Enough of cv2 for :func:`likeness_worker.embed_one`."""

    COLOR_RGB2BGR = 4

    @staticmethod
    def imread(path):
        import numpy as np

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()

    @staticmethod
    def resize(image, size, fx=1.0, fy=1.0):
        return image


def test_a_gradient_picture_has_no_face_and_is_retried_once_at_a_lower_score(tmp_path: Path) -> None:
    path = tmp_path / "gradient.png"
    gradient(7, 96).save(path)
    detector = _NoFaceDetector()
    row = likeness_worker.embed_one(_FakeCv2, detector, None, str(path))
    assert row == {"embedding": None, "face_px": 0}
    assert detector.scores == [likeness_worker.FACE_SCORE, likeness_worker.RETRY_SCORE, likeness_worker.FACE_SCORE]


def test_an_unreadable_file_is_a_row_with_no_face(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a picture")

    class Cv2(_FakeCv2):
        @staticmethod
        def imread(path):
            return None

    assert likeness_worker.embed_one(Cv2, _NoFaceDetector(), None, str(path)) == {"embedding": None, "face_px": 0}


@pytest.mark.skipif(
    not os.environ.get("VOXPRINT_LIKENESS_MODELS"),
    reason="set VOXPRINT_LIKENESS_MODELS to a folder with the YuNet and SFace ONNX files to run the real detector",
)
def test_real_yunet_finds_no_face_on_a_gradient(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    folder = Path(os.environ["VOXPRINT_LIKENESS_MODELS"])
    path = tmp_path / "gradient.png"
    gradient(3, 320).save(path)
    rows = likeness_worker.embed_paths(
        str(folder / likeness.DETECTOR_FILE), str(folder / likeness.RECOGNIZER_FILE), [str(path)],
    )
    assert rows == [{"embedding": None, "face_px": 0}]


# --- evaluate: files, project record, model card ------------------------------------------------

def _fake_embedder(project, *, image_vector):
    """Training photos near one direction; previews get ``image_vector(path)``."""
    center = _unit(42)
    refs = {str(project.root / item.file): _near(center, 0.5, 500 + i) for i, item in enumerate(project.items())}

    def embed(paths):
        return [_row(refs[str(p)]) if str(p) in refs else _row(image_vector(Path(p), center)) for p in paths]

    return embed


def _previews(project, step: int, count: int) -> list[Path]:
    project.previews.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = project.previews / f"step-{step:06d}-{index}.png"
        gradient(step + index).save(path)
        paths.append(path)
    return paths


def test_evaluate_writes_rows_report_and_project_record(tmp_path: Path) -> None:
    project = make_project(tmp_path / "proj", count=10)
    _previews(project, 2, 3)
    _previews(project, 4, 3)

    def vector(path, center):
        if path.name.endswith("-2.png"):
            return None  # one picture per step without a face
        spread = 0.4 if "000004" in path.name else 3.0
        return _near(center, spread, hash(path.name) % 10_000)

    groups = {likeness.step_label(s): p for s, p in sorted(likeness.preview_groups(project).items())}
    report = likeness.evaluate(project, groups, embedder=_fake_embedder(project, image_vector=vector), record_step=4)
    assert set(report["groups"]) == {"step-000002", "step-000004"}
    assert report["recorded"] == "step-000004" and report["best"] == "step-000004"
    assert report["groups"]["step-000004"]["status"] in ("pass", "warn")
    assert report["groups"]["step-000002"]["status"] == "fail"
    rows = [json.loads(line) for line in (project.root / "eval" / "likeness.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and {r["group"] for r in rows} == {"step-000002", "step-000004"}
    assert sum(1 for r in rows if not r["face"]) == 2
    assert all(r["image"].startswith("previews/") for r in rows)
    saved = json.loads((project.root / "eval" / "report.json").read_text())["likeness"]
    assert saved["summary"] == report["summary"]
    markdown = (project.root / "eval" / "report.md").read_text()
    assert "| step-000002 |" in markdown and "| step-000004 |" in markdown and "Quality score only" in markdown
    stored = project.read()["likeness"]
    assert stored["step"] == 4 and set(likeness.CARD_KEYS) <= set(stored)
    assert stored["n"] == 2 and stored["no_face"] == 1


def test_card_fields_only_for_the_exported_step() -> None:
    data = {"likeness": {"mean": 0.3, "median": 0.3, "ref_baseline": 0.7, "threshold": 0.363,
                         "status": "fail", "n": 4, "no_face": 0, "step": 4}}
    assert likeness.card_fields(data, 4) == {k: data["likeness"][k] for k in likeness.CARD_KEYS}
    assert likeness.card_fields(data, 2) is None
    assert likeness.card_fields({"likeness": {"status": "skipped", "reason": "x"}}, 4) is None


def test_a_failing_likeness_does_not_block_export_and_is_on_the_card(tmp_path: Path) -> None:
    project = make_project(tmp_path / "proj", count=10)
    config = build_config("quick", "balanced", 24, steps=2, resolution=64, save_every=1, batch=1, previews=False)
    train(project, config, TinyAdapter(), Progress())
    _previews(project, 2, 3)
    embed = _fake_embedder(project, image_vector=lambda path, center: _unit(hash(path.name) % 10_000))
    summary = likeness.score_after_finish(project, 2, embedder=embed)
    assert summary["status"] == "fail"
    assert likeness.warning_line(summary).startswith("Likeness fail")
    bases.register(TinyAdapter)
    try:
        result = export_package(project, Progress())
    finally:
        bases.unregister(TinyAdapter.info.mdl_id)
    with zipfile.ZipFile(result["file"]) as archive:
        doc = json.loads(archive.read("voxprint-model.json"))
    assert doc["likeness"]["status"] == "fail" and doc["likeness"]["threshold"] == likeness.FAIL_BELOW
    assert set(doc["likeness"]) == set(likeness.CARD_KEYS)
    jsonschema.validate(doc, json.loads((ROOT / "docs" / "schemas" / "voxprint-model-1.json").read_text()))
    assert project.read()["gate_passed"] is True


def test_finish_scores_the_final_previews_and_still_exports_on_a_fail(tmp_path: Path, monkeypatch, capsys) -> None:
    from voxprint_forge.cli import main

    class DrawingAdapter(TinyAdapter):
        info = bases.BaseInfo("draw-like", "example/draw", "0" * 40, "Apache-2.0", "Draw")

        def preview(self, model, prompt, seed, config, steps=None):
            return gradient(seed, 64)

    project = make_project(tmp_path / "proj", count=10)
    embed = _fake_embedder(project, image_vector=lambda path, center: _unit(hash(path.name) % 10_000))
    monkeypatch.setattr(likeness, "missing_models", lambda: [])
    monkeypatch.setattr(likeness, "model_files", lambda: (Path("det.onnx"), Path("rec.onnx")))
    monkeypatch.setattr(likeness, "default_embedder", lambda det, rec: embed)
    monkeypatch.setattr(train_mod, "_device", lambda adapter, config, plan=None: "cpu")
    bases.register(DrawingAdapter)
    try:
        assert main([
            "--json", "lora", "train", "--project", str(project.root), "--base", "draw-like", "--no-fetch",
            "--review-rounds", "0", "--steps", "2", "--resolution", "64", "--batch", "1", "--save-every", "1",
            "--no-previews", "--vram", "24",
        ]) == 0
        capsys.readouterr()
        assert main(["--json", "lora", "finish", str(project.root)]) == 0
        captured = capsys.readouterr()
    finally:
        bases.unregister("draw-like")
    result = json.loads(captured.out[captured.out.rindex("\n{") + 1:] if "\n{" in captured.out else captured.out)
    result = result["result"]
    assert result["state"] == "done" and Path(result["voxlora"]).is_file()
    assert result["likeness"]["status"] == "fail"
    assert "Likeness fail" in captured.err and "nothing was blocked" in captured.err
    assert (project.root / "eval" / "likeness.jsonl").is_file()


def test_finish_without_the_face_models_records_skipped_and_exports(tmp_path: Path, monkeypatch) -> None:
    project = make_project(tmp_path / "proj", count=10)
    _previews(project, 2, 1)
    monkeypatch.setattr(likeness, "model_files", lambda: None)
    summary = likeness.score_after_finish(project, 2)
    assert summary["status"] == "skipped" and "models missing" in summary["reason"]
    assert project.read()["likeness"]["status"] == "skipped"


def test_a_worker_that_cannot_start_is_skipped_not_raised(tmp_path: Path, monkeypatch) -> None:
    project = make_project(tmp_path / "proj", count=10)
    _previews(project, 2, 1)

    def broken(paths):
        raise likeness.LikenessUnavailable("worker exited rc=1")

    assert likeness.score_after_finish(project, 2, embedder=broken)["status"] == "skipped"


def test_the_common_worker_is_used_when_importable_else_a_subprocess(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(likeness, "_via_worker", lambda payload: calls.append("worker") or [])
    monkeypatch.setattr(likeness, "_via_subprocess", lambda payload: calls.append("subprocess") or [])
    embed = likeness.default_embedder(Path("d.onnx"), Path("r.onnx"))
    monkeypatch.setattr(likeness, "_has_common_worker", lambda: True)
    monkeypatch.setattr(likeness, "_has_cv2", lambda: True)
    embed([])
    monkeypatch.setattr(likeness, "_has_common_worker", lambda: False)
    embed([])
    monkeypatch.setattr(likeness, "_has_common_worker", lambda: True)
    monkeypatch.setattr(likeness, "_has_cv2", lambda: False)
    embed([])
    assert calls == ["worker", "subprocess", "subprocess"]


def test_a_bad_worker_reply_is_unavailable() -> None:
    with pytest.raises(likeness.LikenessUnavailable):
        likeness._rows({"rows": [{}]}, 2)
    assert likeness._rows({"rows": [{"embedding": None}]}, 1) == [{"embedding": None}]


def test_the_score_never_touches_the_gate_record(tmp_path: Path) -> None:
    project = make_project(tmp_path / "proj", count=10)
    before = {k: v for k, v in project.read().items() if k != "likeness"}
    _previews(project, 2, 2)
    likeness.score_after_finish(project, 2, embedder=_fake_embedder(project, image_vector=lambda p, c: None))
    after = {k: v for k, v in project.read().items() if k != "likeness"}
    assert after == before


# --- default preview prompts --------------------------------------------------------------------

def test_person_preview_prompt_shows_the_face() -> None:
    captions = [{"short": "A woman standing in a park"}, {"short": "A woman sitting on a bench"}]
    assert preview_prompt("zq7", "person", captions) == (
        "zq7 woman, head-and-shoulders portrait, face clearly visible, looking at camera"
    )
    assert preview_prompt("zq7", "person", []) == f"zq7 person, {PERSON_PREVIEW}"


def test_object_and_style_preview_prompts_are_kind_aware() -> None:
    cube = preview_prompt("tcube", "object", [{"short": "A red cube on a table"}])
    assert cube.startswith("tcube red cube, centered") and "face" not in cube
    assert preview_prompt("tstyle", "style", []).startswith("tstyle style, ")


def test_training_previews_default_to_the_face_prompt_and_keep_the_override(tmp_path: Path) -> None:
    class Recording(TinyAdapter):
        drawn: list[str] = []

        def preview(self, model, prompt, seed, config, steps=None):
            type(self).drawn.append(prompt)
            return gradient(seed, 32)

    project = make_project(tmp_path / "proj", count=10)
    config = build_config("quick", "balanced", 24, steps=2, resolution=64, save_every=1, batch=1,
                          previews=True, preview_every=2)
    Recording.drawn = []
    # The bare trigger is what the CLI passes without --preview-prompts.
    train(project, config, Recording(), Progress(), preview_prompts=["tcube"])
    assert Recording.drawn == [train_mod.default_preview_prompts(project, project.read())[0]]
    assert Recording.drawn[0].startswith("tcube ") and "clear view" in Recording.drawn[0]
    Recording.drawn = []
    train(project, config, Recording(), Progress(), preview_prompts=["tcube on a shelf"])
    assert Recording.drawn == ["tcube on a shelf"]


# --- preview PNG metadata -----------------------------------------------------------------------

def test_preview_png_metadata_round_trip(tmp_path: Path) -> None:
    stamp = PreviewStamp("z-image", "ab" * 32, 500)
    path = tmp_path / "step-000500-0.png"
    save_preview(gradient(1, 32), path, stamp.text(1234))
    assert read_preview_text(path) == {
        "voxprint.seed": "1234",
        "voxprint.lora_scale": "1",
        "voxprint.base_model": "z-image",
        "voxprint.lora_sha256": "ab" * 32,
        "voxprint.step": "500",
    }


def test_preview_lora_hash_is_the_checkpoint_file_hash(tmp_path: Path) -> None:
    layers = inject(TinyNet(), ("to_q", "to_v"), 4, 4)
    project = make_project(tmp_path / "proj", count=10)

    class Opt:
        def state_dict(self):
            return {}

    path = train_mod._save_checkpoint(project, layers, 3, Opt(), random.Random(0), {"base": "tiny-test"})
    assert train_mod._weights_sha256(layers, "tiny-test", 3) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_previews_carry_seed_scale_base_and_hash(tmp_path: Path) -> None:
    class Drawing(TinyAdapter):
        def preview(self, model, prompt, seed, config, steps=None):
            return gradient(seed, 32)

    project = make_project(tmp_path / "proj", count=10)
    config = build_config("quick", "balanced", 24, steps=2, resolution=64, save_every=2, batch=1,
                          previews=True, preview_every=2)
    train(project, config, Drawing(), Progress())
    text = read_preview_text(project.previews / "step-000002-0.png")
    checkpoint = project.checkpoints / "step-000002.safetensors"
    assert text["voxprint.seed"] == str(config.seed)
    assert text["voxprint.lora_scale"] == "1" and text["voxprint.base_model"] == "tiny-test"
    assert text["voxprint.lora_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
