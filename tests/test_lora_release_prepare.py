"""release.txt records on prepare: the adult hint, caption fixes, clothed-only training, export keys. Fakes only."""

from __future__ import annotations

import json

import pytest

from tests.test_lora_gate_rule6 import ScriptedCaptioner, drops, run
from tests.test_lora_job_gate import person
from tests.test_lora_prepare import FakeCaptioner
from voxprint_forge.cli import build_parser
from voxprint_forge.lora import captioner as captioner_mod
from voxprint_forge.lora import cli as lora_cli
from voxprint_forge.lora.consent import Subject, check_subject
from voxprint_forge.lora.gate import child_keyword
from voxprint_forge.lora.i18n import text
from voxprint_forge.lora.prepare import _replace_word
from voxprint_forge.lora.release_file import TEMPLATE, parse_release


def release(tmp_path, **changes: str):
    """A release.txt from the template with ``Key: value`` lines replaced (``""`` drops the line)."""
    lines = []
    for line in TEMPLATE.splitlines():
        key = line.partition(":")[0]
        if key in changes:
            if changes[key]:
                lines.append(f"{key}: {changes[key]}")
            continue
        lines.append(line)
    path = tmp_path / "release.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def released(tmp_path, **changes: str) -> Subject:
    path = release(tmp_path, **changes)
    return person(release=path, release_info=parse_release(path))


class HintCaptioner(ScriptedCaptioner):
    """Records the ``adult_hint`` of every call."""

    def __init__(self, script=None, unclothed=()) -> None:
        super().__init__(script or {})
        self.hints: list[bool] = []
        self.unclothed = set(unclothed)

    def caption(self, image, adult_hint=False):
        self.hints.append(adult_hint)
        out = dict(super().caption(image))
        out["unclothed"] = self.calls in self.unclothed
        return out


class OldCaptioner(FakeCaptioner):
    """A captioner with the old one-argument signature."""

    def caption(self, image):
        return {"long": "a textured square", "short": "square"}


def gate_run(project) -> dict:
    return json.loads(project.gate_run.read_text("utf-8"))


# --- the attestation is recorded ---------------------------------------------------------


def test_release_fields_are_recorded_on_the_subject(tmp_path) -> None:
    subject = released(tmp_path, **{"Consent to generation": "no", "Adult content in generation": ""})
    assert subject.release_name == "Tatiana Model" and subject.release_date == "2026-11-15"
    assert subject.no_minors_attested is True and subject.adult_attested
    assert subject.consent_generation is False and subject.adult_generation is None
    assert subject.adult_content_training is True
    assert Subject(**subject.to_dict()) == subject


def test_an_old_project_subject_without_the_new_fields_still_loads() -> None:
    old = {
        "kind": "person", "name": "Demo", "stage_name": "", "trigger": "vxpdemo", "consent_obtained": True,
        "adult_confirmed": True, "rights_confirmed": False, "provenance_note": "", "release_attached": False,
        "release_sha256": "", "consent_method": "verbal", "adult_content": False, "attr": None,
    }
    subject = Subject(**old)
    assert subject.no_minors_attested is None and subject.consent_generation is None
    assert subject.adult_generation is None and subject.adult_content_training is None
    assert subject.release_name == "" and not subject.adult_attested
    assert Subject(**subject.to_dict()) == subject


def test_validation_is_unchanged_by_the_release(tmp_path) -> None:
    from voxprint_forge.lora.errors import LoraUsageError

    path = release(tmp_path)
    with pytest.raises(LoraUsageError):
        check_subject("person", name="Demo", release=path, release_info=parse_release(path))


def test_the_cli_parses_release_txt_and_ignores_other_release_files(tmp_path) -> None:
    def subject_for(path):
        args = build_parser().parse_args([
            "lora", "prepare", "--project", str(tmp_path / "p"), "--kind", "person", "--name", "Demo",
            "--consent-obtained", "--subject-adult", "--release", str(path),
        ])
        return lora_cli.subject_from(args)

    parsed = subject_for(release(tmp_path, **{"No minors in photos": "yes", "Adult content in training": "no"}))
    assert parsed.no_minors_attested is True and parsed.adult_content_training is False
    assert parsed.release_attached and parsed.consent_method == "file"
    other = tmp_path / "signed.pdf"
    other.write_bytes(b"%PDF-1.4 signed")
    plain = subject_for(other)
    assert plain.release_attached and plain.no_minors_attested is None and plain.release_name == ""


# --- the captioner hint ------------------------------------------------------------------


def test_the_hint_goes_to_the_captioner_only_when_adult_and_attested(tmp_path) -> None:
    attested = HintCaptioner()
    project, _ = run(tmp_path / "a", attested, subject=released(tmp_path))
    assert attested.hints and all(attested.hints)
    assert gate_run(project)["no_minors_attested"] is True
    assert project.read()["subject"]["no_minors_attested"] is True
    not_attested = HintCaptioner()
    project, _ = run(tmp_path / "b", not_attested, subject=released(tmp_path, **{"No minors in photos": "no"}))
    assert not_attested.hints and not any(not_attested.hints)
    assert gate_run(project)["no_minors_attested"] is False
    # Without a release.txt the captioner is called as before: the one-argument fake still works.
    project, summary = run(tmp_path / "c", OldCaptioner())
    assert summary["prepared"] == 10 and gate_run(project)["no_minors_attested"] is None


def test_qwen_prompts_carry_the_adult_sentence_only_with_the_hint() -> None:
    asked: list[tuple[str, str]] = []

    class Vlm:
        def ask(self, image, system, prompt, tokens):
            asked.append((system, prompt))
            return "no"

    qwen = object.__new__(captioner_mod.QwenCaptioner)
    qwen.vlm = Vlm()
    qwen.caption(None)
    assert not any(captioner_mod.ADULT_HINT in s or captioner_mod.ADULT_HINT in p for s, p in asked)
    asked.clear()
    qwen.caption(None, adult_hint=True)
    assert len(asked) == 3 and all(s.endswith(captioner_mod.ADULT_HINT) for s, _ in asked)
    assert asked[0][1].endswith(captioner_mod.ADULT_HINT) and asked[1][1].endswith(captioner_mod.ADULT_HINT)


# --- caption fixes with the attestation --------------------------------------------------


def test_attested_two_retries_then_the_word_is_replaced_and_the_frame_kept(tmp_path, capsys) -> None:
    cap = HintCaptioner({3: "a teen by a wall", 4: "a teen by a wall", 5: "a teen by a wall"})
    project, summary = run(tmp_path, cap, subject=released(tmp_path))
    assert summary["prepared"] == 10 and cap.calls == 12
    assert project.read()["gate_passed"] is True and not drops(project)
    assert not (project.root / "refusals.jsonl").exists()
    fixed = [item for item in project.items() if "by a wall" in item.long]
    assert len(fixed) == 1 and "an adult by a wall" in fixed[0].long
    assert not any(child_keyword(item.long, item.short) for item in project.items())
    fixes = project.read()["caption_fixes"]
    assert [row["action"] for row in fixes] == ["recaptioned", "recaptioned", "replaced"]
    assert all(row["match"] == "teen" for row in fixes)
    assert gate_run(project)["caption_fixes"] == 3 and gate_run(project)["state"] == "complete"
    err = capsys.readouterr().err
    assert err.count("caption keyword 'teen': re-captioning") == 2
    assert "caption keyword 'teen': replaced with 'adult'" in err and "dropped this frame" not in err


def test_attested_a_replacement_still_dirty_drops_only_that_frame(tmp_path, capsys) -> None:
    dirty = "a teen with a kid"
    cap = HintCaptioner({3: dirty, 4: dirty, 5: dirty})
    project, summary = run(tmp_path, cap, subject=released(tmp_path))
    assert summary["prepared"] == 9 and cap.calls == 12
    assert project.read()["gate_passed"] is True and (project.root / "gate.json").is_file()
    rows = drops(project)
    assert len(rows) == 1 and rows[0]["reason"] == "lora.drop.caption_keyword"
    assert rows[0]["match"] == "teen" and rows[0]["action"] == "dropped_frame"
    assert [row["action"] for row in project.read()["caption_fixes"]] == ["recaptioned", "recaptioned", "dropped_frame"]
    assert not (project.root / "refusals.jsonl").exists()
    assert "caption keyword 'teen': dropped this frame" in capsys.readouterr().err


def test_attested_a_clean_retry_needs_no_replacement(tmp_path) -> None:
    cap = HintCaptioner({2: "a child"})
    project, summary = run(tmp_path, cap, subject=released(tmp_path))
    assert summary["prepared"] == 10 and cap.calls == 11
    assert [row["action"] for row in project.read()["caption_fixes"]] == ["recaptioned"]


def test_not_attested_keeps_one_retry_then_drops_the_frame(tmp_path, capsys) -> None:
    cap = HintCaptioner({3: "a teen", 4: "a teen"})
    project, summary = run(tmp_path, cap, subject=released(tmp_path, **{"No minors in photos": ""}))
    assert summary["prepared"] == 9 and cap.calls == 11
    assert drops(project)[0]["reason"] == "lora.drop.caption_keyword"
    assert [row["action"] for row in project.read()["caption_fixes"]] == ["recaptioned", "dropped_frame"]
    assert "replaced" not in capsys.readouterr().err


def test_replace_word_keeps_case_and_fixes_the_article() -> None:
    assert _replace_word("A girl in a park. Girl smiling.", "girl") == "An adult in a park. Adult smiling."
    assert _replace_word("kidney beans and a kid", "kid") == "kidney beans and an adult"


# --- Adult content in training ------------------------------------------------------------


@pytest.mark.parametrize(("value", "kept"), [("no", 8), ("yes", 10), ("", 10)])
def test_unclothed_frames_are_left_out_only_when_training_is_clothed_only(tmp_path, value, kept) -> None:
    cap = HintCaptioner(unclothed={2, 5})
    project, summary = run(tmp_path, cap, subject=released(tmp_path, **{"Adult content in training": value}))
    assert summary["prepared"] == kept and len(list(project.prepared.glob("*.webp"))) == kept
    reasons = [row["reason"] for row in drops(project)]
    assert reasons == (["lora.drop.unclothed_release"] * 2 if value == "no" else [])
    assert not any(item.unclothed for item in project.items()) or value != "no"


def test_unclothed_release_drop_has_a_message() -> None:
    assert "clothed" in text("lora.drop.unclothed_release")
