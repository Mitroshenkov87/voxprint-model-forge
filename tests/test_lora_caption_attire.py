"""Captions keep the captioner's attire wording, and the unclothed flag follows it. Stub captioners only."""

from __future__ import annotations

import json

from tests.test_lora_prepare import FakeCaptioner, photos, run
from voxprint_forge.lora import captioner
from voxprint_forge.lora.project import Project


def _rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "proj" / "captions.jsonl").read_text().splitlines()]


class _Stub(FakeCaptioner):
    def __init__(self, long, short, flag=None):
        super().__init__()
        self.text = {"long": long, "short": short}
        if flag is not None:
            self.text["unclothed"] = flag

    def caption(self, image):
        return dict(self.text)


def test_unclothed_wording_is_stored_as_written_and_sets_the_flag(tmp_path) -> None:
    long = "An adult woman, nude, standing in a bright studio against a grey backdrop."
    short = "unclothed adult woman in a studio"
    run(tmp_path, photos(tmp_path / "in", 10), captioner=_Stub(long, short, True))
    rows = _rows(tmp_path)
    trigger = str(Project(tmp_path / "proj").read()["subject"]["trigger"])  # type: ignore[index]
    assert rows and all(row["long"] == f"{trigger}, {long}" for row in rows)
    assert all(row["short"] == f"{trigger}, {short}" for row in rows)
    assert all(row["unclothed"] is True for row in rows)


def test_unclothed_wording_sets_the_flag_even_when_the_captioner_said_no(tmp_path) -> None:
    long = "An adult man, topless and unclothed, sitting on a wooden chair."
    run(tmp_path, photos(tmp_path / "in", 10), captioner=_Stub(long, "unclothed adult man", False))
    rows = _rows(tmp_path)
    assert all(row["long"].endswith(long) for row in rows)
    assert all(row["unclothed"] is True for row in rows)


def test_clothed_wording_stays_clothed(tmp_path) -> None:
    long = "An adult woman in a wool coat and jeans standing on a station platform."
    short = "adult woman in a wool coat"
    run(tmp_path, photos(tmp_path / "in", 10), captioner=_Stub(long, short, False))
    rows = _rows(tmp_path)
    assert all(row["long"].endswith(long) and row["short"].endswith(short) for row in rows)
    assert all(row["unclothed"] is False for row in rows)
    assert not any("unclothed" in row["long"] or "nude" in row["long"] for row in rows)


def test_clothed_wording_without_a_flag_stores_no_flag(tmp_path) -> None:
    run(tmp_path, photos(tmp_path / "in", 10), captioner=_Stub("a man in a grey suit", "man in a suit"))
    assert all("unclothed" not in row for row in _rows(tmp_path))


def test_the_description_prompts_ask_for_attire_as_it_is() -> None:
    for prompt in (captioner.LONG, captioner.SHORT):
        assert captioner.ATTIRE in prompt
    assert "unclothed" in captioner.ATTIRE and "Do not name clothing that is not in the photo" in captioner.ATTIRE


class _Vlm:
    """Answers each prompt the way a captioning model might, with no model loaded."""

    def __init__(self, long, short, answer):
        self.answers = {captioner.LONG: long, captioner.SHORT: short, captioner._UNCLOTHED_ASK: answer}
        self.systems = []

    def ask(self, image, system, user, max_new_tokens):
        self.systems.append(system)
        return self.answers[user]


def _qwen(vlm):
    stub = captioner.QwenCaptioner.__new__(captioner.QwenCaptioner)
    stub.vlm = vlm
    return stub


def test_qwen_captioner_returns_unclothed_text_unchanged() -> None:
    vlm = _Vlm(" An adult woman, nude, on a sofa. ", "nude adult woman", "Yes.")
    out = _qwen(vlm).caption(object())
    assert out == {"long": "An adult woman, nude, on a sofa.", "short": "nude adult woman", "unclothed": True}
    assert set(vlm.systems) == {captioner.SYSTEM}


def test_qwen_captioner_keeps_clothed_text_clothed() -> None:
    out = _qwen(_Vlm("An adult man in a red jacket.", "man in a red jacket", "No")).caption(object())
    assert out == {"long": "An adult man in a red jacket.", "short": "man in a red jacket", "unclothed": False}


def test_unclothed_flag_helper() -> None:
    assert captioner.unclothed_flag(None, "an adult, nude, on a beach") is True
    assert captioner.unclothed_flag(False, "an unclothed adult") is True
    assert captioner.unclothed_flag(True, "an adult on a beach") is True
    assert captioner.unclothed_flag(False, "a fully clothed adult") is False
    assert captioner.unclothed_flag(None, "a fully clothed adult") is None
    assert captioner.unclothed_flag("yes", "a fully clothed adult") is None
