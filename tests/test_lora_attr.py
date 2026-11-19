"""An optional typed subject attribute stored with integrity binding."""

from __future__ import annotations

import base64

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.app.draft import Draft
from voxprint_forge.app.engine import prepare_args
from voxprint_forge.cli import build_parser
from voxprint_forge.lora import attr, bases
from voxprint_forge.lora.attr import KEY, encode, read_declared
from voxprint_forge.lora.cli import subject_from
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import export, gate_like
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import train

DECLARED = 73
EXISTING = {
    "voxprint.subject.name",
    "voxprint.subject.stage_name",
    "voxprint.subject.trigger",
    "voxprint.subject.consent_obtained",
    "voxprint.subject.adult_confirmed",
    "voxprint.subject.provenance_note",
    "voxprint.subject.release_attached",
    "voxprint.subject.release_sha256",
    "voxprint.subject.consent_method",
    "voxprint.subject.rights_confirmed",
    "voxprint.lora.base",
    "voxprint.lora.base_revision",
    "voxprint.lora.rank",
    "voxprint.lora.alpha",
    "voxprint.lora.steps",
    "voxprint.lora.kind",
    "voxprint.lora.trained_on",
    "voxprint.lora.adult_content",
}


@pytest.fixture(autouse=True)
def tiny_base():
    bases.register(TinyAdapter)
    yield
    bases.unregister("tiny-test")


def _header(path):
    with safe_open(str(path), framework="pt") as handle:
        return handle.metadata()


def _person(**kwargs):
    return check_subject(
        "person",
        name="Demo",
        consent_obtained=True,
        subject_adult=True,
        consent_method="verbal",
        **kwargs,
    )


@pytest.mark.parametrize("value", [17, 100, True, False, 18.0, "30"])
def test_out_of_range_or_non_int_is_rejected(value) -> None:
    with pytest.raises(LoraUsageError) as raised:
        _person(subject_age=value)
    assert raised.value.code == "lora.err.subject_attr"


def test_value_without_adult_attestation_is_rejected() -> None:
    with pytest.raises(LoraUsageError) as raised:
        check_subject(
            "person",
            name="Demo",
            consent_obtained=True,
            subject_adult=False,
            consent_method="verbal",
            subject_age=30,
        )
    assert raised.value.code == "lora.err.subject_attr"


def test_non_person_drops_the_value() -> None:
    assert check_subject("object", name="Mug", rights_confirmed=True, subject_age=15).attr is None
    assert check_subject("style", name="Ink", rights_confirmed=True, subject_age="nope").attr is None


@pytest.mark.parametrize("number", [18, 99])
def test_bounds_roundtrip(number: int) -> None:
    tensors = {"w": torch.tensor([1.0, 2.0, 3.0])}
    stored = encode(number, tensors)
    assert stored is not None and str(number) not in stored
    assert read_declared({KEY: stored}, tensors) == number
    assert _person(subject_age=number).attr == number


def test_absent_input_stays_absent() -> None:
    assert _person().attr is None
    assert encode(None, {"w": torch.zeros(1)}) is None
    assert read_declared({}, {"w": torch.zeros(1)}) is None
    assert read_declared({"w": torch.zeros(1)}) is None


def test_cli_forwards_the_value_and_clears_it_with_the_attestation(tmp_path) -> None:
    draft = Draft(kind="person", consent=True, adult=True, attr=DECLARED)
    draft.set_names("Demo", "")
    draft.set_verbal(True)
    args = prepare_args(draft, tmp_path)
    assert args[args.index("--subject-age") + 1] == str(DECLARED)
    assert subject_from(build_parser().parse_args(args)).attr == DECLARED
    draft.set_adult(False)
    assert draft.attr is None and "--subject-age" not in prepare_args(draft, tmp_path)
    draft.kind, draft.attr, draft.rights = "object", DECLARED, True
    assert "--subject-age" not in prepare_args(draft, tmp_path)


def test_export_binds_the_value_and_rejects_tampering(tmp_path) -> None:
    project = make_project(tmp_path / "proj")
    config = build_config("quick", "balanced", 24, steps=4, resolution=64, save_every=2, previews=False)
    train(project, config, TinyAdapter(), Progress())
    plain = export(project, Progress(), out=tmp_path / "plain.safetensors")
    plain_meta = _header(plain["file"])
    assert KEY not in plain_meta and read_declared(plain["file"]) is None

    data = project.read()
    data["kind"] = "person"
    data["subject"] = _person(subject_age=DECLARED).to_dict()
    project.write(data)
    bound = export(project, Progress(), out=tmp_path / "bound.safetensors")
    meta = _header(bound["file"])
    tensors = load_file(bound["file"])
    assert meta[KEY] and KEY not in EXISTING and not gate_like(KEY)
    assert not any(gate_like(key) for key in meta)
    assert all(str(DECLARED) not in value for value in meta.values())
    assert read_declared(bound["file"]) == DECLARED
    assert read_declared(meta, tensors) == DECLARED

    flipped = bytearray(base64.b64decode(meta[KEY]))
    flipped[0] ^= 0xFF
    tampered_meta = {**meta, KEY: base64.b64encode(bytes(flipped)).decode("ascii")}
    assert read_declared(tampered_meta, tensors) is None
    tampered_path = tmp_path / "tampered.safetensors"
    save_file(tensors, str(tampered_path), metadata=tampered_meta)
    assert read_declared(tampered_path) is None

    changed = {key: value.clone() for key, value in tensors.items()}
    first = next(iter(changed))
    changed[first] = changed[first] + 1
    edited = tmp_path / "edited.safetensors"
    save_file(changed, str(edited), metadata=meta)
    assert read_declared(edited) is None
    assert read_declared(meta, changed) is None

    copied = tmp_path / "copied.safetensors"
    save_file({key: value + 3 for key, value in tensors.items()}, str(copied), metadata={KEY: meta[KEY]})
    assert read_declared(copied) is None
    assert read_declared(bound["file"]) == DECLARED

    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert read_declared(junk) is None
    assert read_declared(tmp_path / "missing.safetensors") is None
    assert read_declared({KEY: DECLARED}, tensors) is None
    assert attr.accepted(True) is None
