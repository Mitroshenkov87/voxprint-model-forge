"""LoRA input gate config v4."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from voxprint_forge.lora.consent import Subject, check_subject
from voxprint_forge.lora.errors import LoraBlocked, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict, child_keyword, decide
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

ABSENT = {k: "absent" for k in INDICATORS}
FACE = Verdict("yes", "yes", 19.0, 0.9, "youthful_adult", True, dict(ABSENT))
BODY = Verdict("no", "yes", None, 0.6, "youthful_adult", True, dict(ABSENT))


def v4(verdict, ages=None, ok=True):
    reading = None if ages is None else AgeReading(len(ages), tuple(ages))
    return decide(verdict, reading, estimator_ok=ok, rule="v4")


class J:
    name = "j"

    def __init__(self, verdict, fail=False):
        self.verdict, self.fail = verdict, fail

    def judge(self, image):
        if self.fail:
            raise RuntimeError("/private/path boom")
        return self.verdict


class E:
    name = "e"

    def __init__(self, ages):
        self.ages = ages

    def estimate(self, image):
        return AgeReading(len(self.ages), tuple(self.ages))


class Cap:
    name = "cap"

    def caption(self, image):
        return {"long": "a square", "short": "square"}

    def unload(self):
        pass


def person(**kw) -> Subject:
    return check_subject("person", name="Demo", consent_obtained=True, subject_adult=True, **kw)


def photos(folder, count, stem="img", meta=None):
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(count):
        path = folder / f"{stem}{i:02d}.png"
        info = None
        if meta:
            info = PngInfo()
            info.add_text("Description", meta)
        rng = np.random.default_rng(i)
        Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)).save(path, pnginfo=info)
        out.append(path)
    return out


def run(tmp_path, files, gate, subject=None):
    return prepare(
        Project(tmp_path / "proj"),
        files,
        subject or person(),
        gate=gate,
        load_captioner=lambda c: Cap(),
        progress=Progress(),
        mode="expert",
    )


@pytest.mark.parametrize(("judge_age", "est"), [(18.0, 19.0), (19.0, 20.0), (20.0, 18.0), (17.0, 21.0)])
def test_cfg_a(judge_age, est) -> None:
    d = v4(replace(FACE, min_age=judge_age), [est])
    assert d.keep, d.reasons


def test_cfg_b() -> None:
    d = v4(replace(FACE, min_age=17.0), [18.0])
    assert not d.keep and "average_under_18" in d.reasons


def test_cfg_c() -> None:
    d = v4(replace(FACE, min_age=25.0), [15.0])
    assert not d.keep and d.reasons == ("single_under_16",) and not d.minor


@pytest.mark.parametrize("marker", ["child_face", "child_body", "school_context", "lacks_adult_features"])
def test_cfg_d(marker) -> None:
    d = v4(replace(FACE, min_age=30.0, indicators={**ABSENT, marker: "present"}), [30.0])
    assert not d.keep and "markers" in d.reasons
    assert not v4(replace(FACE, min_age=30.0, reason_code="child_or_adolescent"), [30.0]).keep


def test_cfg_e() -> None:
    assert v4(replace(FACE, min_age=24.0, indicators={**ABSENT, "child_face": "unsure"}, confidence=0.3), [24.0]).keep


def test_cfg_f() -> None:
    assert child_keyword("cupid_03.png") and child_keyword("putti") and child_keyword("Schoolgirl set")
    assert child_keyword("фото_ребенка") and child_keyword("амур")
    assert not child_keyword("babyface_portrait.png") and not child_keyword("eighteen kidney details детали")


def test_cfg_g(tmp_path) -> None:
    gate = Gate(J(replace(FACE, min_age=30.0)), E([30.0]))
    with pytest.raises(LoraBlocked):
        run(tmp_path / "a", photos(tmp_path / "a" / "in", 6, stem="cupid"), gate)
    with pytest.raises(LoraBlocked):
        run(tmp_path / "b", photos(tmp_path / "b" / "in", 6, meta="teen"), gate)


def test_cfg_h() -> None:
    assert v4(BODY, []).keep and v4(BODY, []).path == "body"
    assert v4(replace(BODY, indicators={**ABSENT, "child_face": "present"}), []).keep
    assert not v4(replace(BODY, indicators={**ABSENT, "childlike_body": "present"}), []).keep


def test_cfg_i(tmp_path) -> None:
    files = photos(tmp_path / "in", 6)
    subject = Subject(kind="person", name="Demo", consent_obtained=True, adult_confirmed=False)
    with pytest.raises(LoraUsageError) as info:
        run(tmp_path, files, Gate(J(FACE), E([20.0])), subject)
    assert info.value.code == "lora.err.no_consent"
    with pytest.raises(LoraUsageError):
        check_subject("person", name="Demo", consent_obtained=True, subject_adult=False)


def test_cfg_j(tmp_path) -> None:
    assert not v4(replace(FACE, min_age=30.0), None, ok=False).keep
    assert not v4(replace(FACE, min_age=None), [30.0]).keep
    assert not v4(replace(FACE, indicators={}), [30.0]).keep
    assert not decide(None, None, estimator_ok=True, rule="v4").keep
    files = photos(tmp_path / "in", 6)
    with pytest.raises(LoraBlocked):
        run(tmp_path, files, Gate(None, E([30.0])))
    with pytest.raises(LoraUsageError):
        run(tmp_path / "x", files, Gate(J(FACE, fail=True), E([30.0])))


def test_cfg_k(tmp_path) -> None:
    files = photos(tmp_path / "in", 6)
    ages = iter([17.0, 22.0, 22.0, 22.0, 22.0, 22.0])

    class Varying(E):
        def estimate(self, image):
            return AgeReading(1, (next(ages),))

    gate = Gate(J(replace(FACE, min_age=18.0)), Varying([]), identify=lambda image: "p1")
    assert run(tmp_path, files, gate)["prepared"] == 6


def test_cfg_l(tmp_path, monkeypatch, capsys) -> None:
    files = photos(tmp_path / "in", 6)
    gate = Gate(J(FACE), E([19.0]))
    assert not gate.internal and gate.check(object(), attested=True).keep
    prepare(
        Project(tmp_path / "proj"),
        files,
        person(),
        gate=gate,
        load_captioner=lambda c: Cap(),
        progress=Progress(),
        mode="expert",
    )
    assert not (tmp_path / "proj" / "gate_v3_shadow.jsonl").exists()
    for path in (tmp_path / "proj").rglob("*.json*"):
        text = path.read_text()
        assert "19.0" not in text and "estimate" not in text and "audit" not in text
    assert "img" not in capsys.readouterr().err


def test_cfg_m(monkeypatch) -> None:
    from voxprint_forge.lora import backends, cli

    for value in ("1", "true", "internal"):
        monkeypatch.setenv("VOXPRINT_FORGE_INTERNAL_GATE", value)
        monkeypatch.setattr(backends, "installed_dir", lambda mdl_id: None)
        for gate in (backends.load_gate("cpu"), cli._load_gate()):
            assert not gate.internal and gate.rule == "v4"
        assert Gate(J(FACE), E([19.0])).rule == "v4"
    parser_text = json.dumps(sorted(vars(cli).keys()))
    assert "internal" not in parser_text


def test_cfg_n(tmp_path, monkeypatch) -> None:
    gate = Gate(J(FACE), E([19.0]))
    assert not gate.check(object()).keep
    assert gate.check(object(), attested=True).keep
    seen = []

    class Spy(Gate):
        def check(self, image, *, attested=False):
            seen.append(attested)
            return super().check(image, attested=attested)

    files = photos(tmp_path / "in", 6)
    subject = check_subject("person", name="Demo", consent_obtained=True, subject_adult=True)
    assert run(tmp_path, files, Spy(J(FACE), E([19.0])), subject)["prepared"] == 6
    assert seen and all(seen)
    obj = check_subject("object", name="Mug", rights_confirmed=True)
    seen.clear()
    with pytest.raises(LoraUsageError):
        run(tmp_path / "o", files, Spy(J(FACE), E([19.0])), obj)
    assert seen and not any(seen)


@pytest.mark.parametrize(
    ("judge_age", "ages", "keep"),
    [
        (21.0, [17.0], True),
        (18.0, [18.5], True),
        (20.0, [20.0], True),
        (19.0, [19.0, 30.0], True),
        (16.0, [17.0], False),
        (14.0, [21.0], False),
        (30.0, [15.0], False),
        (12.0, [13.0], False),
    ],
)
def test_cfg_probe(judge_age, ages, keep) -> None:
    v = replace(FACE, min_age=judge_age, could_be_under_20=True, reason_code="youthful_adult")
    assert Gate(J(v), E(ages), internal=False).check(object(), attested=True).keep is keep


def test_cfg_o() -> None:
    unsure = replace(FACE, min_age=22.0, indicators={k: "unsure" for k in INDICATORS})
    assert v4(unsure, [22.0]).keep
    assert not v4(replace(FACE, min_age=22.0, reason_code="child_or_adolescent"), [22.0]).keep
