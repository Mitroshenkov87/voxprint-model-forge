"""Model Forge app: catalogs, step validators and the CLI adapter (no Qt window)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from voxprint_forge.app import engine
from voxprint_forge.app.draft import EXPERT_CAP, Draft, Prepared, suggest_trigger
from voxprint_forge.lora.trigger import is_rare_trigger, looks_like_words

LOCALES = Path(__file__).resolve().parents[1] / "src" / "voxprint_forge" / "app" / "locales"


def _catalog(lang: str) -> dict[str, str]:
    return json.loads((LOCALES / f"{lang}.json").read_text(encoding="utf-8"))


def test_catalogs_have_the_same_keys_and_placeholders() -> None:
    english, russian = _catalog("en"), _catalog("ru")
    assert english.keys() == russian.keys()
    def forms(value):  # a count string holds one text per plural form
        return list(value.values()) if isinstance(value, dict) else [value]

    for key, text in english.items():
        wanted = {tuple(re.findall(r"{(\w+)}", form)) for form in forms(text)}
        assert wanted == {tuple(re.findall(r"{(\w+)}", form)) for form in forms(russian[key])}, key
        assert all(form.strip() for form in forms(russian[key])), key


def test_catalogs_hold_no_safety_or_age_settings() -> None:
    assert not [key for key in _catalog("en") if re.search(r"\.(minor|age|guard|safety)(\.|$)", key)]


def _photos(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"p{index}.png"
        path.write_bytes(b"x")
        paths.append(path)
    return paths


def test_simple_mode_is_a_hard_10_to_30(tmp_path: Path) -> None:
    draft = Draft(kind="object")
    draft.add_photos(_photos(tmp_path, 9))
    assert draft.check("photos").key == "lora.err.too_few"
    assert draft.check("photos").params == {"need": 1}
    (tmp_path / "more").mkdir()
    draft.add_photos(_photos(tmp_path / "more", 1))
    assert draft.check("photos").ok
    (tmp_path / "big").mkdir()
    draft.add_photos(_photos(tmp_path / "big", 21))
    result = draft.check("photos")
    assert not result.ok and result.key == "lora.err.too_many" and result.params == {"extra": 1, "high": 30}


def test_expert_mode_warns_outside_10_30_with_floor_5(tmp_path: Path) -> None:
    draft = Draft(kind="style", expert=True)
    draft.add_photos(_photos(tmp_path, 4))
    assert draft.check("photos").key == "lora.err.below_floor"
    (tmp_path / "b").mkdir()
    draft.add_photos(_photos(tmp_path / "b", 1))
    result = draft.check("photos")
    assert result.ok and result.warning and result.key == "lora.warn.count"
    (tmp_path / "c").mkdir()
    draft.add_photos(_photos(tmp_path / "c", EXPERT_CAP))
    assert draft.check("photos").key == "lora.err.above_cap"


def test_adding_a_folder_skips_other_files_and_duplicates(tmp_path: Path) -> None:
    _photos(tmp_path, 3)
    (tmp_path / "notes.txt").write_text("x")
    draft = Draft()
    assert draft.add_photos([tmp_path]) == 3
    assert draft.add_photos([tmp_path / "p0.png"]) == 0


@pytest.mark.parametrize(
    ("kind", "fields", "missing"),
    [
        ("person", {"consent": True}, "lora.err.no_consent"),
        ("person", {"adult": True}, "lora.err.no_consent"),
        ("person", {"consent": True, "adult": True}, "lora.err.no_release"),
        ("object", {}, "lora.err.no_rights"),
        ("style", {}, "lora.err.no_rights"),
    ],
)
def test_consent_per_kind(tmp_path: Path, kind: str, fields: dict[str, bool], missing: str) -> None:
    draft = Draft(kind=kind)
    draft.add_photos(_photos(tmp_path, 12))
    assert draft.check("subject").key == "lora.err.no_name"
    draft.set_names("", "Demo Shape")
    assert is_rare_trigger(draft.trigger) and "demo" not in draft.trigger
    for name, value in fields.items():
        setattr(draft, name, value)
    assert draft.check("subject").key == missing
    if kind == "person":
        draft.consent = draft.adult = True
        draft.set_verbal(True)
    else:
        draft.rights = True
    assert draft.check("subject").ok


def test_consent_is_verbal_or_a_file_and_adult_content_needs_18(tmp_path: Path) -> None:
    draft = Draft(kind="person", consent=True, adult=True)
    draft.add_photos(_photos(tmp_path, 12))
    draft.set_names("Demo Person", "")
    assert draft.check("subject").key == "lora.err.no_release"
    draft.release = tmp_path / "release.pdf"
    assert draft.check("subject").ok and draft.consent_method() == "file"
    draft.set_verbal(True)
    assert draft.release is None and draft.consent_method() == "verbal" and draft.check("subject").ok
    draft.adult_content = True
    assert draft.check("subject").ok
    draft.set_adult(False)
    assert draft.adult_content is False and draft.check("subject").key == "lora.err.no_consent"
    draft.adult = True
    draft.adult_content, draft.kind = True, "object"
    draft.rights = True
    assert draft.check("subject").key == "lora.err.adult_content"


def test_a_safety_block_is_final(tmp_path: Path) -> None:
    draft = Draft(kind="object", rights=True)
    draft.add_photos(_photos(tmp_path, 12))
    draft.set_names("Cup", "")
    assert draft.check("prepare").key == "forge.app.err.not_prepared"
    draft.prepared = Prepared([{"id": str(i)} for i in range(12)], [], blocked=True)
    assert draft.check("prepare").key == "lora.err.blocked"
    assert draft.check("train").key == "lora.err.blocked"
    draft.prepared = Prepared([{"id": str(i)} for i in range(9)], [{"reason": "lora.drop.safety"}] * 3)
    assert draft.check("prepare").key == "lora.err.too_few"


def test_trigger_suggestion() -> None:
    token = suggest_trigger(" Mira  Stone ")
    other = suggest_trigger("Мира")
    assert is_rare_trigger(token) and is_rare_trigger(other)
    assert "mira" not in token and "stone" not in token and "мира" not in other
    assert token != other


def test_a_dictionary_trigger_warns_without_blocking(tmp_path: Path) -> None:
    draft = _ready_draft(tmp_path)
    assert draft.check("subject").ok and not draft.check("subject").warning
    draft.trigger = "pg_synthetic_a"
    draft.trigger_edited = True
    result = draft.check("subject")
    assert result.ok and result.warning and result.key == "lora.warn.trigger_word"
    assert looks_like_words("pg_synthetic_a") and looks_like_words("synthetic")
    assert not looks_like_words(suggest_trigger("PG Synthetic A"))


def _ready_draft(tmp_path: Path) -> Draft:
    draft = Draft(kind="person", consent=True, adult=True, note="studio set")
    draft.add_photos(_photos(tmp_path, 10))
    draft.set_names("Mira Demo", "Mira")
    draft.release = tmp_path / "release.pdf"
    return draft


def test_prepare_args_follow_the_flag_table(tmp_path: Path) -> None:
    args = engine.prepare_args(_ready_draft(tmp_path), tmp_path / "proj")
    assert args[:5] == ["--json", "lora", "prepare", "--project", str(tmp_path / "proj")]
    assert args.count("--photo") == 10
    for flag in ("--json", "--consent-obtained", "--subject-adult", "--release", "--provenance-note"):
        assert flag in args
    trigger = args[args.index("--trigger") + 1]
    assert is_rare_trigger(trigger) and "mira" not in trigger
    assert args[args.index("--mode") + 1] == "simple"
    assert "--rights-confirmed" not in args and "--allow-count-outside-range" not in args


def test_expert_values_are_sent_only_in_expert_mode(tmp_path: Path) -> None:
    values = {"lora.preset": "quick", "lora.rank": 8, "lora.grad_checkpointing": True, "lora.mix_old": "0.5"}
    simple = engine.train_args(tmp_path, values)
    assert "--rank" not in simple and "--grad-checkpointing" not in simple
    expert = engine.train_args(tmp_path, values, expert=True, resume=True)
    assert expert[expert.index("--rank") + 1] == "8" and "--grad-checkpointing" in expert and "--resume" in expert
    assert "--replay" not in expert
    off = engine.train_args(tmp_path, {**values, "lora.grad_checkpointing": False}, expert=True)
    assert "--no-grad-checkpointing" in off and "--grad-checkpointing" not in off
    extend = engine.train_args(tmp_path, values, expert=True, extend_from=tmp_path / "old.safetensors")
    assert extend[2] == "extend" and "--rank" not in extend and extend[extend.index("--replay") + 1] == "0.5"


def test_every_builder_parses_with_the_real_cli(tmp_path: Path) -> None:
    from voxprint_forge.cli import build_parser

    draft = _ready_draft(tmp_path)
    built = [
        engine.prepare_args(draft, tmp_path),
        engine.train_args(tmp_path, {}, extend_from=None, resume=False, dry_run=True, expert=False),
        engine.export_args(draft, tmp_path, tmp_path / "out.safetensors", checkpoint=500, install=True),
    ]
    for args in built:
        assert build_parser().parse_args(args).json is True, args


@pytest.mark.parametrize("verbal", [False, True])
@pytest.mark.parametrize("adult_content", [False, True])
def test_gui_subject_flags_mean_the_same_to_the_cli(tmp_path: Path, verbal: bool, adult_content: bool) -> None:
    from voxprint_forge.cli import build_parser
    from voxprint_forge.lora.cli import subject_from

    draft = _ready_draft(tmp_path)
    (tmp_path / "release.pdf").write_bytes(b"signed")
    draft.set_verbal(verbal)
    draft.adult_content = adult_content
    assert draft.check("subject").ok
    args = engine.prepare_args(draft, tmp_path / "proj")
    assert args[args.index("--consent-method") + 1] == ("verbal" if verbal else "file")
    assert ("--release" in args) is (not verbal)
    assert ("--adult-content" in args) is adult_content
    subject = subject_from(build_parser().parse_args(args))
    assert subject.consent_method == draft.consent_method()
    assert subject.adult_content is adult_content and subject.adult_confirmed and subject.consent_obtained
    assert subject.release_attached is (not verbal) and (subject.release_sha256 == "") is verbal


def test_adult_content_is_not_sent_without_18(tmp_path: Path) -> None:
    draft = _ready_draft(tmp_path)
    draft.adult_content = True
    draft.set_adult(False)
    args = engine.prepare_args(draft, tmp_path / "proj")
    assert "--adult-content" not in args and "--subject-adult" not in args


def test_export_args(tmp_path: Path) -> None:
    args = engine.export_args(_ready_draft(tmp_path), tmp_path, None, checkpoint=500, install=True)
    assert "--install-to-mdl" in args and args[args.index("--checkpoint") + 1] == "500" and "--out" not in args


def test_line_reader_reads_events_then_the_envelope() -> None:
    reader = engine.LineReader()
    event = {"step": "train", "message_key": "lora.progress.train", "params": {"step": 3, "steps": 9},
             "reason_key": "lora.reason.train", "percent": 33.3, "eta_seconds": 60, "state": "running"}
    line = json.dumps({"event": event})
    assert reader.feed(line[:20]) == []
    events = reader.feed(line[20:] + "\n" + '{\n  "ok": true,\n  "result": {"state": "done"}\n}')
    assert events == [engine.EngineEvent.from_dict(event)]
    outcome = reader.finish(0)
    assert outcome.ok and outcome.result == {"state": "done"}


def test_a_missing_or_failed_envelope_is_never_success() -> None:
    assert not engine.LineReader().finish(0).ok
    reader = engine.LineReader()
    reader.feed('{"ok": false, "exit_code": 6, "error": {"code": "lora.err.blocked", "message": "x"}}')
    outcome = reader.finish(6)
    assert not outcome.ok and outcome.error_code == "lora.err.blocked"
    reader = engine.LineReader()
    reader.feed('{"ok": true, "result": {}}')
    assert not reader.finish(1).ok


def test_commands_lists_the_lora_subcommands_of_this_build() -> None:
    found = engine.commands()
    assert {"prepare", "train", "extend"} <= found


def test_prepared_from_reads_the_project_files(tmp_path: Path) -> None:
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared" / "a.png").write_bytes(b"x")
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"id": "a", "file": "prepared/a.png", "long": "l", "short": "s", "round": 0}) + "\n"
    )
    drop = json.dumps({"id": "b", "name": "b.png", "reason": "lora.drop.blurry"})
    (tmp_path / "drops.jsonl").write_text(drop + "\n" + drop + "\nnot json\n")
    kept, drops = engine.prepared_from(tmp_path)
    assert kept == [{"id": "a", "file": str(tmp_path / "prepared/a.png"), "long": "l", "short": "s"}]
    assert drops == [{"id": "b", "name": "b.png", "reason": "lora.drop.blurry"}]
