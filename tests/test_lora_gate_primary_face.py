"""LoRA gate: face_rule (primary / all). Stubbed judge and estimator, synthetic images only."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.lora.backends import MivoloEstimator
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import (
    INDICATORS,
    PRIMARY_MIN_SCORE,
    RULE_VERSION,
    STRAY_MAX_AREA,
    AgeReading,
    Face,
    Gate,
    Verdict,
    decide,
)
from voxprint_forge.lora.job_gate import MANIFEST, preflight_refuse, verify
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

ADULT = Verdict("yes", "yes", 30.0, 0.9, "clearly_adult", False, {k: "absent" for k in INDICATORS})
BIG = (100.0, 100.0, 200.0, 200.0)
SMALL = (400.0, 50.0, 40.0, 40.0)
NEAR = (350.0, 100.0, 190.0, 190.0)


def reading(*faces: tuple[tuple[float, ...], float, float]) -> AgeReading:
    return AgeReading(len(faces), tuple(age for _, _, age in faces), tuple(Face(box, score) for box, score, _ in faces))


def unboxed(r: AgeReading) -> AgeReading:
    return AgeReading(r.n_faces, r.ages)


def v4(verdict: Verdict, r: AgeReading):
    return decide(verdict, r, estimator_ok=True, rule="v4")


def same(a, b) -> bool:
    return (a.reasons, a.minor, a.path, a.tier, a.trace["estimate"]) == (
        b.reasons,
        b.minor,
        b.path,
        b.tier,
        b.trace["estimate"],
    )


def test_constants() -> None:
    assert RULE_VERSION == 5
    assert PRIMARY_MIN_SCORE == 0.9
    assert STRAY_MAX_AREA == 0.25


def test_primary_26_small_15_clear() -> None:
    r = reading((BIG, 0.95, 26.0), (SMALL, 0.8, 15.0))
    d = v4(ADULT, r)
    assert d.reasons == () and d.tier == "clear" and not d.minor
    assert d.trace["estimate"] == pytest.approx(28.0)
    faces = d.trace["faces"]
    assert faces["rule"] == "primary" and faces["why"] == "stray"
    assert faces["primary"] == 0 and faces["stray"] == [1]
    assert [f["area"] for f in faces["faces"]] == [40000.0, 1600.0]
    assert [f["age"] for f in faces["faces"]] == [26.0, 15.0]
    assert v4(ADULT, unboxed(r)).tier != "clear"


def test_primary_order_independent() -> None:
    d = v4(ADULT, reading((SMALL, 0.8, 15.0), (BIG, 0.95, 26.0)))
    assert d.tier == "clear"
    assert d.trace["faces"]["primary"] == 1 and d.trace["faces"]["stray"] == [0]


@pytest.mark.parametrize(
    ("r", "why"),
    [
        (reading((BIG, 0.95, 15.0), (SMALL, 0.8, 26.0)), "primary_age"),
        (reading((BIG, 0.95, 26.0), (NEAR, 0.95, 15.0)), "similar_size"),
        (reading((NEAR, 0.95, 15.0), (BIG, 0.95, 26.0)), "similar_size"),
        (reading((BIG, 0.95, 26.0), (NEAR, 0.95, 24.0), (SMALL, 0.8, 15.0)), "similar_size"),
        (reading((BIG, PRIMARY_MIN_SCORE - 0.01, 26.0), (SMALL, 0.8, 15.0)), "low_score"),
        (reading((BIG, 0.95, 19.0), (SMALL, 0.8, 15.0)), "primary_age"),
        (
            reading(
                (BIG, 0.95, 15.0),
            ),
            "n_faces",
        ),
    ],
)
def test_keeps_current_result(r, why) -> None:
    d = v4(ADULT, r)
    assert d.trace["faces"]["rule"] == "all" and d.trace["faces"]["why"] == why
    assert d.trace["faces"]["stray"] == []
    assert same(d, v4(ADULT, unboxed(r)))


def test_area_at_limit_is_not_stray() -> None:
    side = (STRAY_MAX_AREA * 200.0 * 200.0) ** 0.5
    r = reading((BIG, 0.95, 26.0), ((0.0, 0.0, side, side), 0.95, 15.0))
    d = v4(ADULT, r)
    assert d.trace["faces"]["why"] == "similar_size"
    assert same(d, v4(ADULT, unboxed(r)))


def test_primary_minor_still_strong() -> None:
    r = reading((BIG, 0.95, 15.0), (SMALL, 0.8, 26.0))
    d = v4(replace(ADULT, min_age=15.0), r)
    assert d.tier == "strong" and d.minor
    assert same(d, v4(replace(ADULT, min_age=15.0), unboxed(r)))


@pytest.mark.parametrize(
    "verdict",
    [
        replace(ADULT, confidence=0.4),
        replace(ADULT, min_age=19.0),
        replace(ADULT, min_age=None),
        replace(ADULT, could_be_under_20=True),
        replace(ADULT, reason_code="child_or_adolescent"),
        replace(ADULT, indicators={**ADULT.indicators, "child_face": "present"}),
        replace(ADULT, indicators={**ADULT.indicators, "child_face": "unsure"}),
    ],
)
def test_judge_not_clear_keeps_current_result(verdict) -> None:
    r = reading((BIG, 0.95, 26.0), (SMALL, 0.8, 15.0))
    d = v4(verdict, r)
    assert d.trace["faces"]["rule"] == "all" and d.trace["faces"]["why"] == "judge"
    assert d.tier != "clear"
    assert same(d, v4(verdict, unboxed(r)))


@pytest.mark.parametrize(
    "r",
    [
        AgeReading(2, (26.0, 15.0), (Face(BIG, 0.95),)),
        AgeReading(2, (26.0, 15.0), (Face(BIG, 0.95), Face((0.0, 0.0, 0.0, 0.0), 0.9))),
        AgeReading(2, (26.0, 15.0), (Face(BIG, 0.95), Face(SMALL, None))),  # type: ignore[arg-type]
        AgeReading(2, (26.0, None), (Face(BIG, 0.95), Face(SMALL, 0.9))),
    ],
)
def test_bad_faces_keep_current_result(r) -> None:
    d = v4(ADULT, r)
    assert d.trace["faces"]["rule"] == "all"
    assert same(d, v4(ADULT, unboxed(r)))


class Judge:
    name = "stub-judge"

    def judge(self, image):
        return ADULT


class Estimator:
    name = "stub-estimator"

    def __init__(self, r: AgeReading):
        self.r = r

    def estimate(self, image):
        return self.r


class Captioner:
    name = "stub-captioner"

    def caption(self, image):
        return {"long": "a person on a plain background", "short": "person"}

    def unload(self):
        pass


def photos(folder, count: int, stem: str = "img"):
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for index in range(count):
        path = folder / f"{stem}{index:02d}.png"
        rng = np.random.default_rng(index)
        Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)).save(path)
        out.append(path)
    return out


def run(tmp_path, files, r: AgeReading) -> Project:
    project = Project(tmp_path / "proj")
    prepare(
        project,
        files,
        check_subject("person", name="Demo", consent_obtained=True, subject_adult=True),
        gate=Gate(Judge(), Estimator(r)),
        load_captioner=lambda choice: Captioner(),
        progress=Progress(),
        mode="expert",
    )
    return project


def trace(project: Project) -> list[dict]:
    return [json.loads(line) for line in project.gate_trace.read_text("utf-8").splitlines()]


def test_prepare_trace_and_pin(tmp_path) -> None:
    project = run(tmp_path, photos(tmp_path / "in", 10), reading((BIG, 0.95, 26.0), (SMALL, 0.8, 15.0)))
    assert project.read()["gate_passed"] is True
    rows = trace(project)
    assert len(rows) == 10
    for row in rows:
        assert row["tier"] == "clear" and row["decision"] == "keep"
        assert row["face_rule"] == "primary" and row["face_why"] == "stray"
        assert row["primary"] == 0 and row["stray"] == [1]
        assert row["mivolo_ages"] == [26.0, 15.0]
        assert row["faces"] == [
            {"i": 0, "box": list(BIG), "area": 40000.0, "score": 0.95, "age": 26.0},
            {"i": 1, "box": list(SMALL), "area": 1600.0, "score": 0.8, "age": 15.0},
        ]
    stored = json.loads((project.root / MANIFEST).read_text("utf-8"))
    assert stored["rule"] == RULE_VERSION
    assert preflight_refuse(project).rule == RULE_VERSION


def test_old_pin_rule_refused(tmp_path) -> None:
    project = run(tmp_path, photos(tmp_path / "in", 10), reading((BIG, 0.95, 26.0), (SMALL, 0.8, 15.0)))
    path = project.root / MANIFEST
    stored = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps({**stored, "rule": 4}), encoding="utf-8")
    with pytest.raises(LoraBlocked) as info:
        verify(project)
    assert info.value.code == "lora.err.dataset_changed"


def test_similar_size_prepare_unchanged(tmp_path) -> None:
    r = reading((BIG, 0.95, 26.0), (NEAR, 0.95, 15.0))
    with pytest.raises(LoraBlocked):
        run(tmp_path / "a", photos(tmp_path / "in", 10), r)
    with pytest.raises(LoraBlocked):
        run(tmp_path / "b", photos(tmp_path / "in", 10), unboxed(r))
    a, b = trace(Project(tmp_path / "a" / "proj")), trace(Project(tmp_path / "b" / "proj"))
    assert [(x["tier"], x["reasons"], x["estimate"]) for x in a] == [
        (x["tier"], x["reasons"], x["estimate"]) for x in b
    ]
    assert {x["face_why"] for x in a} == {"similar_size"}


def test_keyword_refusal_unchanged(tmp_path) -> None:
    files = photos(tmp_path / "in", 9) + photos(tmp_path / "kw", 1, stem="teen")
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, reading((BIG, 0.95, 26.0), (SMALL, 0.8, 15.0)))
    assert info.value.code == "lora.err.blocked"
    project = Project(tmp_path / "proj")
    assert project.read()["gate_passed"] is False
    assert not (project.root / MANIFEST).exists()
    assert any(row["tier"] == "strong" and "child_keyword" in row["reasons"] for row in trace(project))


class _Proc:
    def __init__(self, reply: dict):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(json.dumps(reply) + "\n")

    def poll(self):
        return None


def _estimator(tmp_path, reply: dict) -> MivoloEstimator:
    est = MivoloEstimator.__new__(MivoloEstimator)
    est.tmp = tmp_path
    est.proc = _Proc(reply)  # type: ignore[assignment]
    return est


def test_worker_reply_faces(tmp_path) -> None:
    reply = {
        "n_faces": 2,
        "ages": [26.0, 15.0],
        "faces": [{"box": list(BIG), "score": 0.95}, {"box": list(SMALL), "score": 0.8}],
    }
    r = _estimator(tmp_path, reply).estimate(Image.new("RGB", (8, 8)))
    assert r.faces == (Face(BIG, 0.95), Face(SMALL, 0.8))


@pytest.mark.parametrize("faces", [None, [{"box": [1, 2, 3]}], [{"score": 0.9}], "x"])
def test_worker_reply_bad_faces(tmp_path, faces) -> None:
    reply = {"n_faces": 2, "ages": [26.0, 15.0]}
    if faces is not None:
        reply["faces"] = faces
    r = _estimator(tmp_path, reply).estimate(Image.new("RGB", (8, 8)))
    d = v4(ADULT, r)
    assert d.trace["faces"]["rule"] == "all"
    assert same(d, v4(ADULT, unboxed(r)))
