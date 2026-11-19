#!/usr/bin/env python3
"""Sign a LoRA so Canvas may skip the per-frame output age check for it.

Port of Canvas's reference signer (``tools/forge_sign_lora.py`` in
voxprint-canvas) with the same flags. It writes
``<weights>.voxprint-verify.json`` next to the weights file, signed with this
install's shared key (``voxprint_forge.lora.verify_record.key_path()``, i.e.
``$VOXPRINT_HOME/keys/lora-verify-ed25519.key``). Canvas trusts the record only
when the signature checks out *and* the recorded sha256 matches the exact
weights file it loads.

Prefer ``voxprint-forge-cli lora sign-checkpoints PROJECT``: it reads the
project's gate itself. This tool takes the field values from its flags. When
the weights file lies inside a Forge project folder (a parent holds
``project.json``), it refuses unless that project's safety check passed and
``gate.json`` still matches the files.

Usage:
    forge_sign_lora.py WEIGHTS.safetensors \\
        --project-id my-project \\
        --dataset-image-count 42 \\
        --age-gate-version voxprint-lora-gate/1@5 \\
        [--record-id forge-...] [--verified-at ISO8601] [--result pass]

Exit codes: 0 wrote a record, 2 bad arguments, the weights file is missing, or
the project's safety check did not pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Allow running straight from a checkout without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxprint_forge.lora.project import Project  # noqa: E402
from voxprint_forge.lora.verify_record import (  # noqa: E402
    ALGORITHM,
    SCHEMA_ID,
    file_sha256,
    gate_fields,
    key_path,
    record_path_for,
    sign_record,
    verify_lora,
)


def build_record(args: argparse.Namespace, weights: Path) -> dict:
    """Assemble the record dict, including the signature over the signed fields."""
    record = {
        "schema": SCHEMA_ID,
        "record_id": args.record_id or f"forge-{uuid.uuid4()}",
        "lora_file": weights.name,
        "lora_sha256": file_sha256(weights),
        "project_id": args.project_id,
        "dataset_image_count": int(args.dataset_image_count),
        "age_gate_version": args.age_gate_version,
        "verified": True,
        "result": args.result,
        "verified_at": args.verified_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "algorithm": ALGORITHM,
    }
    record["signature"] = sign_record(record)
    return record


def enclosing_project(weights: Path) -> Project | None:
    """The Forge project folder that holds ``weights``, if any."""
    for parent in weights.resolve().parents:
        if (parent / "project.json").is_file():
            return Project(parent)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign a LoRA verification record for Canvas.")
    parser.add_argument("weights", type=Path, help="Path to the LoRA .safetensors file.")
    parser.add_argument("--project-id", required=True, help="Forge project id for the training run.")
    parser.add_argument("--dataset-image-count", type=int, required=True, help="Number of verified training images.")
    parser.add_argument("--age-gate-version", required=True, help="Age-gate version that verified the set.")
    parser.add_argument("--record-id", default="", help="Record id; a forge-<uuid> is generated when omitted.")
    parser.add_argument("--verified-at", default="", help="ISO 8601 timestamp; now when omitted.")
    parser.add_argument("--result", default="pass", help="Verification result; only 'pass' is trusted by Canvas.")
    args = parser.parse_args(argv)

    weights = args.weights
    if not weights.is_file():
        print(f"weights file not found: {weights}", file=sys.stderr)
        return 2
    project = enclosing_project(weights)
    if project is not None and gate_fields(project) is None:
        print(f"the safety check of project {project.root.name} did not pass; no record written", file=sys.stderr)
        return 2

    record = build_record(args, weights)
    out_path = record_path_for(weights)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    check = verify_lora(weights)
    if not check.verified:
        print(f"wrote {out_path} but it did not verify: {check.reason}", file=sys.stderr)
        return 2
    print(f"wrote {out_path.name} (key {key_path()}, record {record['record_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
