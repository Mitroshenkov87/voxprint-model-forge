"""Signed adult-verification record beside a saved LoRA (``voxprint-lora-verify/1``).

After a LoRA file is saved for a project whose minors gate passed, Forge writes
``<weights file name>.voxprint-verify.json`` beside it. Canvas reads the record
and may skip only its per-frame output age check for that exact file.

The format is Canvas's SCHEMA (``canvas/lora_verify.py``), byte-compatible with
Canvas's verifier:

* Key: one raw 32-byte Ed25519 seed per install, shared with Canvas, at
  ``$VOXPRINT_HOME/keys/lora-verify-ed25519.key`` (folder 0700, file 0600,
  created on first use with ``O_EXCL``; a loser of the create race reads the
  winner's key). It is never printed. It is not the MDL index key.
* Signed bytes: compact JSON (sorted keys, ``separators=(",", ":")``, UTF-8) of
  exactly the eleven :data:`SIGNED_FIELDS`. ``signature`` is the lower-case hex
  Ed25519 signature over those bytes and is not signed itself. Extra keys are
  never signed and never trusted.

This module only reads the gate. It never changes ``gate.json``, ``gate_passed``
or the audit log, and it writes no record when the gate state is not a pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from voxprint_common.paths import home

from voxprint_forge.errors import ForgeError, StageError
from voxprint_forge.lora.gate import RULE_VERSION
from voxprint_forge.lora.job_gate import MANIFEST, _hashes
from voxprint_forge.lora.job_gate import SCHEMA as GATE_SCHEMA
from voxprint_forge.lora.project import Project

RECORD_SUFFIX = ".voxprint-verify.json"
SCHEMA_ID = "voxprint-lora-verify/1"
ALGORITHM = "ed25519"
AGE_GATE_VERSION = f"{GATE_SCHEMA}@{RULE_VERSION}"
_KEY_NAME = "lora-verify-ed25519.key"
_READ_CHUNK = 1024 * 1024
_CAPTIONS = "captions.jsonl"

# Fields covered by the signature, as in Canvas's canvas/lora_verify.py.
SIGNED_FIELDS = (
    "schema",
    "record_id",
    "lora_file",
    "lora_sha256",
    "project_id",
    "dataset_image_count",
    "age_gate_version",
    "verified",
    "result",
    "verified_at",
    "algorithm",
)


class RecordInvalid(ForgeError):
    """The record is missing, unreadable, tampered with, or does not match the LoRA. Exit code 1."""

    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message, code="lora.verify_record.invalid")


class RecordSignFailed(StageError):
    """The gate passed but the record could not be signed or written. Exit code 6."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="lora.verify_record.sign_failed")


class SignRefused(ForgeError):
    """``sign-checkpoints`` found no gate pass for the project, or nothing to sign. Exit code 1."""

    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message, code="lora.verify_record.sign_refused")


@dataclass(frozen=True)
class VerifyResult:
    """The outcome of checking one LoRA's record."""

    path: Path
    verified: bool
    reason: str
    record_id: str = ""
    lora_sha256: str = ""


# --- key and signing (ported from canvas/lora_verify.py) ----------------------------


def keys_dir() -> Path:
    """The per-install key folder inside the shared suite folder (mode 0700)."""
    path = home() / "keys"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def key_path() -> Path:
    """Absolute path of this install's Ed25519 signing key, shared with Canvas."""
    return keys_dir() / _KEY_NAME


def _load_private_key():
    """Load this install's signing key, creating it (0600) on first use."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = key_path()
    if not path.exists():
        return _create_private_key(path)
    raw = path.read_bytes()
    if len(raw) != 32:
        raise ValueError("lora-verify key is not a 32-byte Ed25519 seed")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _create_private_key(path: Path):
    """Generate and persist a fresh Ed25519 key with 0600 permissions.

    The write is exclusive. A second process that loses the race reads the key
    the winner wrote instead of overwriting it.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys_dir()
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:
        existing = path.read_bytes()
        if len(existing) != 32:
            raise ValueError("lora-verify key is not a 32-byte Ed25519 seed") from None
        return Ed25519PrivateKey.from_private_bytes(existing)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def canonical_message(record: dict) -> bytes:
    """The exact bytes signed and verified. Raises ``KeyError`` if incomplete."""
    payload = {field: record[field] for field in SIGNED_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_record(record: dict) -> str:
    """Return the hex Ed25519 signature over ``record``'s signed fields."""
    key = _load_private_key()
    return key.sign(canonical_message(record)).hex()


def _verify_signature(message: bytes, signature_hex: str) -> bool:
    """Check ``signature_hex`` against this install's key. No key on this install means not verified."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    path = home() / "keys" / _KEY_NAME
    if not path.is_file():
        return False
    raw = path.read_bytes()
    if len(raw) != 32:
        return False
    try:
        Ed25519PrivateKey.from_private_bytes(raw).public_key().verify(signature, message)
        return True
    except InvalidSignature:
        return False


def file_sha256(path: Path) -> str:
    """Streaming sha256 of a file, lower-case hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_path_for(weights: Path) -> Path:
    """The ``<weights>.voxprint-verify.json`` path next to a LoRA file."""
    weights = Path(weights)
    return weights.with_name(weights.name + RECORD_SUFFIX)


def verify_lora(weights: Path) -> VerifyResult:
    """Check one LoRA's record the way Canvas does. Fail-closed: any problem means not verified."""
    weights = Path(weights)
    try:
        if not weights.is_file():
            return VerifyResult(weights, False, "weights-missing")
        record_path = record_path_for(weights)
        if not record_path.is_file():
            return VerifyResult(weights, False, "record-missing")
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return VerifyResult(weights, False, "record-unreadable")
        if not isinstance(record, dict):
            return VerifyResult(weights, False, "record-malformed")
        if record.get("schema") != SCHEMA_ID:
            return VerifyResult(weights, False, "schema-mismatch")
        if record.get("algorithm") != ALGORITHM:
            return VerifyResult(weights, False, "algorithm-mismatch")
        if record.get("verified") is not True or record.get("result") != "pass":
            return VerifyResult(weights, False, "not-a-pass")
        signature = record.get("signature")
        if not isinstance(signature, str) or not signature:
            return VerifyResult(weights, False, "signature-missing")
        record_id = str(record.get("record_id") or "")
        try:
            message = canonical_message(record)
        except KeyError:
            return VerifyResult(weights, False, "record-incomplete", record_id=record_id)
        if not _verify_signature(message, signature):
            return VerifyResult(weights, False, "bad-signature", record_id=record_id)
        actual = file_sha256(weights)
        claimed = str(record.get("lora_sha256") or "").lower()
        if actual.lower() != claimed:
            return VerifyResult(weights, False, "hash-mismatch", record_id=record_id, lora_sha256=actual)
        return VerifyResult(weights, True, "verified", record_id=record_id, lora_sha256=actual)
    except Exception:  # noqa: BLE001 - fail-closed on anything unexpected
        return VerifyResult(weights, False, "error")


# --- Forge side: the gate, the writer, the CLI -----------------------------------------


def gate_fields(project: Project) -> dict[str, object] | None:
    """``project_id`` and ``dataset_image_count`` when the project's gate passed, else ``None``.

    Read only: ``gate_passed`` must be exactly true and ``gate.json`` must exist,
    have the current schema and rule, and match the files on disk. Anything else,
    or any error while reading, gives ``None``. Nothing is refused or unpinned here.
    """
    try:
        data = project.read()
        if data.get("gate_passed") is not True:
            return None
        manifest = project.root / MANIFEST
        if not manifest.is_file():
            return None
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(stored, dict) or stored.get("schema") != GATE_SCHEMA or stored.get("rule") != RULE_VERSION:
            return None
        pinned = stored.get("files")
        if not isinstance(pinned, dict) or not pinned or _hashes(project) != pinned:
            return None
        images = sum(1 for name in pinned if name != _CAPTIONS)
        if images < 1:
            return None
        return {"project_id": project.root.resolve().name, "dataset_image_count": images}
    except Exception:  # noqa: BLE001 - an unreadable gate state is not a pass
        return None


def write_record(project: Project, lora: Path) -> Path | None:
    """Sign and write the record beside ``lora`` when the project's gate passed.

    Returns the record path, or ``None`` when the gate did not pass (no record, and
    any stale record beside ``lora`` is removed). Raises :class:`RecordSignFailed`
    when the gate passed but the key or the record could not be written.
    """
    lora = Path(lora)
    target = record_path_for(lora)
    gate = gate_fields(project)
    if gate is None:
        target.unlink(missing_ok=True)
        return None
    try:
        record: dict[str, object] = {
            "schema": SCHEMA_ID,
            "record_id": f"forge-{uuid.uuid4()}",
            "lora_file": lora.name,
            "lora_sha256": file_sha256(lora),
            "project_id": gate["project_id"],
            "dataset_image_count": gate["dataset_image_count"],
            "age_gate_version": AGE_GATE_VERSION,
            "verified": True,
            "result": "pass",
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "algorithm": ALGORITHM,
        }
        record["signature"] = sign_record(record)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(target)
        check = verify_lora(lora)
        if not check.verified:
            raise ValueError(f"the written record did not verify ({check.reason})")
    except Exception as exc:  # noqa: BLE001 - re-raised visibly; never a silent unsigned LoRA
        target.unlink(missing_ok=True)
        target.with_name(target.name + ".tmp").unlink(missing_ok=True)
        raise RecordSignFailed(
            f"Could not write the signed verification record for {lora.name}: {type(exc).__name__}"
        ) from exc
    return target


def verify_record(path: Path) -> dict[str, object]:
    """Check a record, or the record beside a LoRA file, against this install's key.

    Returns the signed fields for display. Raises :class:`RecordInvalid` when
    :func:`verify_lora` does not verify it.
    """
    path = Path(path)
    if path.name.endswith(RECORD_SUFFIX) and len(path.name) > len(RECORD_SUFFIX):
        weights = path.with_name(path.name[: -len(RECORD_SUFFIX)])
    else:
        weights = path
    result = verify_lora(weights)
    if not result.verified:
        message = _REASONS.get(result.reason, "The verification record does not verify.")
        raise RecordInvalid(f"{message} ({result.reason}: {weights.name})")
    record = json.loads(record_path_for(weights).read_text(encoding="utf-8"))
    shown = {field: record[field] for field in SIGNED_FIELDS}
    return {"state": "verified", "record": str(record_path_for(weights)), "file": str(weights), **shown}


_REASONS = {
    "weights-missing": "The LoRA file beside the record was not found.",
    "record-missing": "No verification record was found beside the LoRA.",
    "record-unreadable": "The verification record cannot be read.",
    "record-malformed": "The verification record is not a JSON object.",
    "schema-mismatch": f"Unknown verification record schema; expected {SCHEMA_ID}.",
    "algorithm-mismatch": f"Unknown signature algorithm; expected {ALGORITHM}.",
    "not-a-pass": "The record does not state a verified pass.",
    "signature-missing": "The record has no signature.",
    "record-incomplete": "The record is missing a signed field.",
    "bad-signature": (
        "The record signature does not verify with this install's key: it was changed or signed elsewhere."
    ),
    "hash-mismatch": "The sha256 of the LoRA does not match the record: the file was changed.",
}


def trace_shows_pass(project: Project) -> bool:
    """``gate_trace.jsonl`` exists, is not empty, and at least one row kept a photo."""
    path = project.gate_trace
    try:
        if not path.is_file():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("decision") == "keep":
                return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def sign_checkpoints(project_root: Path) -> dict[str, object]:
    """Sign every existing checkpoint ``*.safetensors`` of a project whose gate passed.

    Refuses, writing nothing, unless ``gate_trace.jsonl`` shows a kept photo,
    ``gate_passed`` is true, and ``gate.json`` is current and matches the files.
    Read only on the gate: nothing is refused or unpinned.
    """
    project = Project(Path(project_root))
    if not project.file.is_file():
        raise SignRefused(f"{project.root.name} is not a prepared Forge project (no project.json).")
    if not trace_shows_pass(project):
        raise SignRefused("No signed records: gate_trace.jsonl is missing, empty, or shows no kept photo.")
    if gate_fields(project) is None:
        raise SignRefused(
            "No signed records: the project's safety check did not pass, or gate.json no longer matches the files."
        )
    root = project.root / "checkpoints"
    files = sorted(path for path in root.rglob("*.safetensors") if path.is_file()) if root.is_dir() else []
    if not files:
        raise SignRefused("No checkpoint .safetensors files were found to sign.")
    records = []
    for weights in files:
        record = write_record(project, weights)
        if record is None:
            raise RecordSignFailed(f"The safety check state changed while signing {weights.name}.")
        records.append(str(record))
    return {"state": "done", "project": str(project.root), "signed": len(records), "records": records}
