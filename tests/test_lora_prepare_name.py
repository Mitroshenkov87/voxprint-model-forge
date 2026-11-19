"""``lora prepare`` without --name: one clear line with an example, still a usage error (exit 2)."""

from __future__ import annotations

import json

import pytest

from voxprint_forge.cli import build_parser, main

BASE = ["lora", "prepare", "--project", "proj", "--photos", "photos", "--no-fetch"]
PERSON = ["--kind", "person", "--consent-obtained", "--subject-adult"]


MISSING = [PERSON, ["--kind", "object", "--rights-confirmed"], ["--kind", "person", "--name", " "]]


@pytest.mark.parametrize("extra", MISSING)
def test_missing_name_says_what_and_shows_an_example(tmp_path, monkeypatch, capsys, extra) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photos").mkdir()
    assert main(["--json", *BASE, *extra]) == 2
    payload = json.loads(capsys.readouterr().out)
    error = payload["error"]
    assert error["code"] == "lora.err.no_name"
    message = error["message"]
    assert "--name is required" in message and "name or stage name" in message
    assert "--name Garnis_AI --consent-method verbal" in message
    assert "\n" not in message


def test_missing_name_plain_output_is_one_line(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photos").mkdir()
    assert main([*BASE, *PERSON]) == 2
    out = capsys.readouterr()
    lines = [line for line in (out.out + out.err).splitlines() if line.strip()]
    assert len(lines) == 1 and "--name Garnis_AI --consent-method verbal" in lines[0]


def test_stage_name_alone_still_passes_the_name_check(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photos").mkdir()
    code = main(["--json", *BASE, *PERSON, "--stage-name", "Garnis_AI"])
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("error", {}).get("code") != "lora.err.no_name", (code, payload)


def test_help_documents_name_consent_method_and_an_example() -> None:
    parser = build_parser()
    lora = next(a for a in parser._actions if hasattr(a, "choices") and a.choices and "lora" in a.choices)
    prepare = next(a for a in lora.choices["lora"]._actions if hasattr(a, "choices") and a.choices
                   and "prepare" in a.choices).choices["prepare"]
    text = " ".join(prepare.format_help().split())
    assert "the subject's name or stage name" in text
    assert "{verbal,file}" in text and "verbal: in person, no file" in text and "needs --release FILE" in text
    assert "--name Garnis_AI" in text and "--consent-method verbal --fetch-missing" in text
