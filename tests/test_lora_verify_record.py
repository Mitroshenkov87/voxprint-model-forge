"""Signed adult-verification record beside saved LoRA files (Canvas's SCHEMA, voxprint-lora-verify/1).

Every test runs under a temporary ``VOXPRINT_HOME`` (conftest ``_suite_home``), so the key is created
there and never in the real suite folder. No private key is stored in this repository.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import sys
from datetime import datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.cli import main
from voxprint_forge.lora import bases
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import export, export_package
from voxprint_forge.lora.gate import RULE_VERSION
from voxprint_forge.lora.job_gate import MANIFEST
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import train
from voxprint_forge.lora.verify_record import (
    AGE_GATE_VERSION,
    SIGNED_FIELDS,
    RecordInvalid,
    RecordSignFailed,
    SignRefused,
    canonical_message,
    key_path,
    record_path_for,
    sign_checkpoints,
    sign_record,
    verify_lora,
    verify_record,
    write_record,
)

CANVAS_TREE = Path("/workspace/canvas/wt-verified-lora")
TOOL = Path(__file__).resolve().parents[1] / "tools" / "forge_sign_lora.py"

# The SCHEMA worked example's eleven signed fields (design note, canvas-verified-lora).
EXAMPLE = {
    "schema": "voxprint-lora-verify/1",
    "record_id": "forge-546bb146-2e12-4611-8dbe-8348991ad109",
    "lora_file": "step-001600.safetensors",
    "lora_sha256": "056dada6ce9cfe341e0d6ae477eee5d1d6592eac45ec6fc69e7552adecf49190",
    "project_id": "zhenya-belaya-20261114-2202",
    "dataset_image_count": 42,
    "age_gate_version": "0.1.0-rc.45",
    "verified": True,
    "result": "pass",
    "verified_at": "2026-11-15T00:30:00+03:00",
    "algorithm": "ed25519",
}
# Computed with Canvas's canonical_message (commit 1d9c389).
EXAMPLE_CANONICAL = (
    b'{"age_gate_version":"0.1.0-rc.45","algorithm":"ed25519","dataset_image_count":42,'
    b'"lora_file":"step-001600.safetensors",'
    b'"lora_sha256":"056dada6ce9cfe341e0d6ae477eee5d1d6592eac45ec6fc69e7552adecf49190",'
    b'"project_id":"zhenya-belaya-20261114-2202","record_id":"forge-546bb146-2e12-4611-8dbe-8348991ad109",'
    b'"result":"pass","schema":"voxprint-lora-verify/1","verified":true,"verified_at":"2026-11-15T00:30:00+03:00"}'
)


@pytest.fixture(autouse=True)
def tiny_base():
    bases.register(TinyAdapter)
    yield
    bases.unregister("tiny-test")


@pytest.fixture
def suite_home() -> Path:
    return Path(os.environ["VOXPRINT_HOME"])


def person_project(root, *, gate_passed: bool = True):
    project = make_project(root, gate_passed=gate_passed)
    data = project.read()
    data["kind"] = "person"
    data["subject"] = {
        "kind": "person", "name": "Demo", "trigger": "vxpdemo", "consent_obtained": True, "adult_confirmed": True,
    }
    project.write(data)
    return project


def write_trace(project, decisions=("keep",)) -> None:
    rows = [{"file": f"p{i:02d}.png", "id": f"p{i:02d}", "decision": d} for i, d in enumerate(decisions)]
    project.gate_trace.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fake_lora(project, name: str = "step-000002.safetensors", blob: bytes = b"fake lora bytes"):
    project.checkpoints.mkdir(parents=True, exist_ok=True)
    path = project.checkpoints / name
    path.write_bytes(blob)
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rewrite(path, record):
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


# --- worked example round trip --------------------------------------------------------


def test_worked_example_canonical_bytes_are_golden() -> None:
    assert canonical_message(EXAMPLE) == EXAMPLE_CANONICAL
    # Extra keys and the signature are never part of the signed bytes.
    assert canonical_message({**EXAMPLE, "signature": "ab" * 64, "extra": 1}) == EXAMPLE_CANONICAL
    with pytest.raises(KeyError):
        canonical_message({k: v for k, v in EXAMPLE.items() if k != "verified_at"})


def test_worked_example_signs_and_verifies_with_forge(suite_home) -> None:
    signature = sign_record(EXAMPLE)
    assert re.fullmatch(r"[0-9a-f]{128}", signature)
    assert key_path() == suite_home / "keys" / "lora-verify-ed25519.key"
    seed = key_path().read_bytes()
    assert len(seed) == 32
    public = Ed25519PrivateKey.from_private_bytes(seed).public_key()
    public.verify(bytes.fromhex(signature), EXAMPLE_CANONICAL)  # raises on a bad signature
    assert sign_record(EXAMPLE) == signature  # Ed25519 is deterministic with the same key


@pytest.fixture
def canvas_lora_verify():
    if not (CANVAS_TREE / "canvas" / "lora_verify.py").is_file():
        pytest.skip("Canvas verified-LoRA worktree is not on this machine")
    sys.path.insert(0, str(CANVAS_TREE))
    try:
        module = importlib.import_module("canvas.lora_verify")
        yield module
    finally:
        sys.path.remove(str(CANVAS_TREE))
        for name in [n for n in sys.modules if n == "canvas" or n.startswith("canvas.")]:
            del sys.modules[name]


def test_worked_example_matches_canvas(tmp_path, suite_home, canvas_lora_verify) -> None:
    assert canvas_lora_verify.canonical_message(EXAMPLE) == canonical_message(EXAMPLE) == EXAMPLE_CANONICAL
    assert canvas_lora_verify.key_path() == key_path()
    # Signature path only: Forge signs the example, Canvas's key verifies it.
    signature = sign_record(EXAMPLE)
    assert canvas_lora_verify.sign_record(EXAMPLE) == signature
    assert canvas_lora_verify._verify_signature(EXAMPLE_CANONICAL, signature)

    # A record Forge writes beside a real temp weights file is accepted by Canvas's verify_lora.
    project = person_project(tmp_path / "zhenya-belaya-20261114-2202")
    lora = fake_lora(project, "step-001600.safetensors")
    write_record(project, lora)
    result = canvas_lora_verify.verify_lora(lora)
    assert result.verified and result.reason == "verified"
    assert result.lora_sha256 == hashlib.sha256(lora.read_bytes()).hexdigest()

    # And a tampered one is refused by Canvas too.
    record = load(record_path_for(lora))
    rewrite(record_path_for(lora), {**record, "dataset_image_count": 99})
    assert canvas_lora_verify.verify_lora(lora).reason == "bad-signature"


# --- valid record ----------------------------------------------------------------------


def test_gate_passed_record_is_signed_and_accepted(tmp_path, suite_home, capsys) -> None:
    project = person_project(tmp_path / "my-project")
    lora = fake_lora(project)
    path = write_record(project, lora)
    assert path == record_path_for(lora) and path.name == "step-000002.safetensors.voxprint-verify.json"
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("}\n") and raw.startswith("{\n  ")
    record = json.loads(raw)
    assert set(record) == {*SIGNED_FIELDS, "signature"}
    assert record["schema"] == "voxprint-lora-verify/1"
    assert re.fullmatch(r"forge-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}", record["record_id"])
    assert record["lora_file"] == lora.name
    assert record["lora_sha256"] == hashlib.sha256(lora.read_bytes()).hexdigest()
    assert record["project_id"] == "my-project"
    assert record["dataset_image_count"] == 10  # prepared photos, not captions.jsonl
    assert len(load(project.root / MANIFEST)["files"]) == 11
    assert record["age_gate_version"] == AGE_GATE_VERSION == f"voxprint-lora-gate/1@{RULE_VERSION}"
    assert record["verified"] is True and record["result"] == "pass" and record["algorithm"] == "ed25519"
    assert datetime.fromisoformat(record["verified_at"]).utcoffset() is not None
    assert re.fullmatch(r"[0-9a-f]{128}", record["signature"])

    assert verify_lora(lora).verified
    result = verify_record(lora)
    assert result["state"] == "verified" and result["project_id"] == "my-project"
    assert "signature" not in result
    assert verify_record(path)["lora_sha256"] == record["lora_sha256"]
    assert main(["lora", "verify-record", str(lora)]) == 0
    assert main(["lora", "verify-record", str(path)]) == 0
    out = capsys.readouterr().out
    assert key_path().read_bytes().hex() not in out


def test_key_is_shared_seed_created_once_with_owner_only_mode(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    first = load(write_record(project, fake_lora(project)))
    key = suite_home / "keys" / "lora-verify-ed25519.key"
    assert key.is_file() and len(key.read_bytes()) == 32
    if os.name == "posix":
        assert stat.S_IMODE(key.stat().st_mode) == 0o600
        assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    seed = key.read_bytes()
    write_record(project, fake_lora(project, "step-000004.safetensors"))
    assert key.read_bytes() == seed
    assert "public_key" not in first and seed.hex() not in json.dumps(first)


def test_create_race_loser_reads_the_winner_key(suite_home, monkeypatch) -> None:
    from voxprint_forge.lora import verify_record as module

    winner = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization

    seed = winner.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    path = key_path()
    path.write_bytes(seed)
    loser = module._create_private_key(path)
    assert path.read_bytes() == seed
    assert loser.sign(b"x") == winner.sign(b"x")


def test_pretty_printing_does_not_change_validity(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    record = load(path)
    path.write_text(json.dumps(dict(reversed(list(record.items()))), indent=4), encoding="utf-8")
    assert verify_record(lora)["state"] == "verified"


# --- tamper ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_id", "forge-00000000-0000-4000-8000-000000000000"),
        ("lora_file", "other.safetensors"),
        ("project_id", "other"),
        ("dataset_image_count", 99),
        ("age_gate_version", "voxprint-lora-gate/1@1"),
        ("verified_at", "2020-01-01T00:00:00+00:00"),
    ],
)
def test_changed_signed_field_fails(tmp_path, suite_home, field, value) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    record = load(path)
    assert record[field] != value
    rewrite(path, {**record, field: value})
    assert verify_lora(lora).reason == "bad-signature"
    with pytest.raises(RecordInvalid):
        verify_record(lora)
    assert main(["lora", "verify-record", str(lora)]) != 0


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda r: {**r, "schema": "voxprint-lora-verify/2"}, "schema-mismatch"),
        (lambda r: {**r, "algorithm": "rsa"}, "algorithm-mismatch"),
        (lambda r: {**r, "verified": False}, "not-a-pass"),
        (lambda r: {**r, "result": "fail"}, "not-a-pass"),
        (lambda r: {k: v for k, v in r.items() if k != "signature"}, "signature-missing"),
        (lambda r: {k: v for k, v in r.items() if k != "project_id"}, "record-incomplete"),
        (lambda r: {**r, "signature": "not hex!"}, "bad-signature"),
        (lambda r: {**r, "signature": "00" * 64}, "bad-signature"),
        (lambda r: {**r, "signature": r["signature"][:-2] + ("00" if r["signature"][-2:] != "00" else "11")},
         "bad-signature"),
    ],
)
def test_malformed_or_bad_signature_fails(tmp_path, suite_home, change, reason) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    rewrite(path, change(load(path)))
    assert verify_lora(lora).reason == reason
    with pytest.raises(RecordInvalid):
        verify_record(lora)
    assert main(["lora", "verify-record", str(path)]) != 0


def test_unsigned_extra_field_is_ignored_not_trusted(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    rewrite(path, {**load(path), "note": "added later"})
    assert "note" not in verify_record(lora)


def test_changed_lora_bytes_fail_hash_check(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    write_record(project, lora)
    lora.write_bytes(b"fake lora bytes, merged")
    assert verify_lora(lora).reason == "hash-mismatch"
    with pytest.raises(RecordInvalid, match="sha256"):
        verify_record(lora)
    assert main(["lora", "verify-record", str(lora)]) != 0


def test_unreadable_record_fails(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    write_record(project, lora).write_text("{not json", encoding="utf-8")
    with pytest.raises(RecordInvalid, match="cannot be read"):
        verify_record(lora)


# --- wrong key -----------------------------------------------------------------------------


def test_record_from_another_install_fails(tmp_path, suite_home, monkeypatch) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    assert verify_record(lora)["state"] == "verified"

    other = tmp_path / "other-install"
    monkeypatch.setenv("VOXPRINT_HOME", str(other))
    assert verify_lora(lora).reason == "bad-signature"  # this install has no key yet
    (other / "keys").mkdir(parents=True)
    (other / "keys" / "lora-verify-ed25519.key").write_bytes(os.urandom(32))
    assert verify_lora(lora).reason == "bad-signature"
    with pytest.raises(RecordInvalid, match="this install's key"):
        verify_record(path)
    assert main(["lora", "verify-record", str(path)]) != 0


def test_signature_from_another_key_fails(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    record = load(path)
    other = Ed25519PrivateKey.generate()
    rewrite(path, {**record, "signature": other.sign(canonical_message(record)).hex()})
    assert verify_lora(lora).reason == "bad-signature"
    assert main(["lora", "verify-record", str(lora)]) != 0


# --- missing record ------------------------------------------------------------------------


def test_missing_record_fails(tmp_path, suite_home, capsys) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    with pytest.raises(RecordInvalid, match="No verification record"):
        verify_record(lora)
    assert main(["lora", "verify-record", str(lora)]) == 1
    assert main(["lora", "verify-record", str(record_path_for(lora))]) == 1
    captured = capsys.readouterr()
    assert "No verification record" in captured.out + captured.err


def test_missing_lora_beside_record_fails(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    path = write_record(project, lora)
    lora.unlink()
    with pytest.raises(RecordInvalid, match="not found"):
        verify_record(path)


# --- no record without a passed gate ---------------------------------------------------------


def test_no_record_when_gate_not_passed(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj", gate_passed=False)
    lora = fake_lora(project)
    assert write_record(project, lora) is None
    assert not record_path_for(lora).exists()
    assert not (suite_home / "keys" / "lora-verify-ed25519.key").exists()


def test_no_record_when_gate_flag_is_not_exactly_true(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    project.write({**project.read(), "gate_passed": "true"})
    lora = fake_lora(project)
    assert write_record(project, lora) is None and not record_path_for(lora).exists()


def test_no_record_when_manifest_missing_or_stale(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    (project.root / MANIFEST).unlink()
    assert write_record(project, lora) is None and not record_path_for(lora).exists()

    other = person_project(tmp_path / "other")
    stored = load(other.root / MANIFEST)
    (other.root / MANIFEST).write_text(json.dumps({**stored, "rule": RULE_VERSION - 1}), encoding="utf-8")
    assert write_record(other, fake_lora(other)) is None


def test_no_record_when_dataset_changed_and_gate_state_untouched(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    assert write_record(project, lora) is not None
    (project.prepared / "p00.png").write_bytes(b"changed after the gate")
    before = (project.root / MANIFEST).read_bytes()
    assert write_record(project, lora) is None
    assert not record_path_for(lora).exists()  # the stale record is removed
    # The record writer only reads the gate: no refusal, no unpin.
    assert project.read()["gate_passed"] is True
    assert (project.root / MANIFEST).read_bytes() == before
    assert not (project.root / "refusals.jsonl").exists()


def test_sign_failure_is_raised_not_hidden(tmp_path, suite_home) -> None:
    keys = suite_home / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    (keys / "lora-verify-ed25519.key").write_bytes(b"not a 32-byte seed")
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    with pytest.raises(RecordSignFailed):
        write_record(project, lora)
    assert not record_path_for(lora).exists()


def test_sign_failure_when_key_folder_cannot_be_made(tmp_path, monkeypatch) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_HOME", str(blocker))
    project = person_project(tmp_path / "proj")
    lora = fake_lora(project)
    with pytest.raises(RecordSignFailed):
        write_record(project, lora)
    assert not record_path_for(lora).exists()


# --- hooks: training checkpoints and export --------------------------------------------------


def tiny_config(steps: int = 4):
    return build_config("quick", "balanced", 24, steps=steps, resolution=64, save_every=2, previews=False)


def test_training_writes_a_record_beside_every_checkpoint(tmp_path, suite_home) -> None:
    project = make_project(tmp_path / "proj")
    train(project, tiny_config(), TinyAdapter(), Progress())
    saved = sorted(project.checkpoints.glob("step-*.safetensors"))
    assert [p.name for p in saved] == ["step-000002.safetensors", "step-000004.safetensors"]
    for lora in saved:
        assert verify_record(lora)["lora_file"] == lora.name
        assert main(["lora", "verify-record", str(lora)]) == 0


def test_export_writes_records_beside_outputs_and_install_copy(tmp_path, suite_home, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "models"))
    project = make_project(tmp_path / "proj")
    train(project, tiny_config(), TinyAdapter(), Progress())
    result = export(project, Progress(), install_to_mdl=True)
    target = Path(str(result["file"]))
    assert result["verify_record"] == str(record_path_for(target))
    assert verify_record(target)["lora_file"] == target.name
    installed = next((tmp_path / "models" / "user-models" / "lora").glob("*/*.safetensors"))
    assert verify_record(installed)["lora_sha256"] == hashlib.sha256(installed.read_bytes()).hexdigest()
    package = export_package(project, Progress())
    assert package["verify_record"] == str(record_path_for(package["file"]))
    assert verify_record(package["file"])["lora_file"].endswith(".voxlora")


def test_export_after_dataset_change_writes_no_record(tmp_path, suite_home) -> None:
    project = make_project(tmp_path / "proj")
    train(project, tiny_config(), TinyAdapter(), Progress())
    (project.prepared / "p00.png").write_bytes(b"changed after the gate")
    result = export_package(project, Progress())
    assert "verify_record" not in result
    assert not record_path_for(result["file"]).exists()


# --- sign-checkpoints ------------------------------------------------------------------------


def test_sign_checkpoints_signs_every_checkpoint_after_a_pass(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    write_trace(project, ("keep", "drop", "keep"))
    loras = [fake_lora(project, f"step-00000{n}.safetensors", f"bytes {n}".encode()) for n in (2, 4)]
    result = sign_checkpoints(project.root)
    assert result["signed"] == 2
    for lora in loras:
        assert verify_record(lora)["lora_file"] == lora.name
    for lora in loras:
        record_path_for(lora).unlink()
    assert main(["lora", "sign-checkpoints", str(project.root)]) == 0
    assert all(verify_lora(lora).verified for lora in loras)
    assert project.read()["gate_passed"] is True


@pytest.mark.parametrize(
    "setup",
    [
        lambda p: None,  # no gate_trace.jsonl
        lambda p: p.gate_trace.write_text("", encoding="utf-8"),
        lambda p: write_trace(p, ("drop", "minor")),
        lambda p: (write_trace(p), p.write({**p.read(), "gate_passed": False})),
        lambda p: (write_trace(p), (p.root / MANIFEST).unlink()),
        lambda p: (write_trace(p), (p.prepared / "p00.png").write_bytes(b"changed after the gate")),
    ],
    ids=["no-trace", "empty-trace", "no-keep", "gate-not-passed", "no-manifest", "files-changed"],
)
def test_sign_checkpoints_refuses_without_a_pass(tmp_path, suite_home, setup, capsys) -> None:
    project = person_project(tmp_path / "proj")
    setup(project)
    lora = fake_lora(project)
    manifest = project.root / MANIFEST
    before = manifest.read_bytes() if manifest.is_file() else None
    passed = project.read()["gate_passed"]
    with pytest.raises(SignRefused):
        sign_checkpoints(project.root)
    assert main(["lora", "sign-checkpoints", str(project.root)]) != 0
    assert not record_path_for(lora).exists()
    assert not (suite_home / "keys" / "lora-verify-ed25519.key").exists()
    # Read only: the gate state is exactly as it was.
    assert (manifest.read_bytes() if manifest.is_file() else None) == before
    assert project.read()["gate_passed"] is passed
    assert not (project.root / "refusals.jsonl").exists()


def test_sign_checkpoints_refuses_without_checkpoints(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj")
    write_trace(project)
    with pytest.raises(SignRefused, match="No checkpoint"):
        sign_checkpoints(project.root)


# --- tools/forge_sign_lora.py -----------------------------------------------------------------


def _tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("forge_sign_lora_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sign_tool_writes_a_verified_record(tmp_path, suite_home) -> None:
    weights = tmp_path / "loose.safetensors"
    weights.write_bytes(b"loose lora")
    argv = [str(weights), "--project-id", "p", "--dataset-image-count", "12", "--age-gate-version", AGE_GATE_VERSION]
    assert _tool().main(argv) == 0
    assert verify_lora(weights).verified


def test_sign_tool_refuses_inside_a_project_without_a_pass(tmp_path, suite_home) -> None:
    project = person_project(tmp_path / "proj", gate_passed=False)
    lora = fake_lora(project)
    argv = [str(lora), "--project-id", "proj", "--dataset-image-count", "10", "--age-gate-version", AGE_GATE_VERSION]
    assert _tool().main(argv) == 2
    assert not record_path_for(lora).exists()
