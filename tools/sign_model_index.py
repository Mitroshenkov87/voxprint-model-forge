#!/usr/bin/env python3
"""Build a signed model index from a Components handover and sign it.

Does not upload. Refuses to sign while any ``TODO`` remains in the document
(for example ``z-image-turbo`` revision ``TODO-pending-upload``). Pass the
real 40-hex turbo revision when Components has uploaded the descriptor.

One command (set the key file in the environment; the key is never printed):

    VOXPRINT_MDL_INDEX_KEY_FILE=/path/to/mdl-index-key.pem \\
      python tools/sign_model_index.py \\
        --handover /path/to/index-entries-for-forge-0.7.0.json \\
        --turbo-revision <40-hex-commit> \\
        --out signed-index/index.json

The bundled pin (serial 10, head 15edeab0) is left unchanged. The new index
is serial 11 with the same head unless ``--head`` or ``--serial`` is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TODO = re.compile(r"TODO", re.IGNORECASE)


def _todo_paths(value: object, prefix: str = "$") -> list[str]:
    """JSON Pointer-like paths whose string values contain TODO."""
    found: list[str] = []
    if isinstance(value, str):
        if _TODO.search(value):
            found.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_todo_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_todo_paths(item, f"{prefix}[{index}]"))
    return found


def _apply_turbo_revision(handover: dict[str, object], revision: str) -> dict[str, object]:
    """Replace ``TODO-pending-upload`` (or any TODO revision) on z-image-turbo."""
    entries = handover.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("handover has no entries list")
    for row in entries:
        if not isinstance(row, dict) or row.get("id") != "z-image-turbo":
            continue
        row["revision"] = revision
        note = row.get("note")
        if isinstance(note, str):
            # Drop the pending-upload marker from this row once a real revision is supplied.
            cleaned = re.sub(r"TODO-pending-upload:\s*", "", note).strip()
            if cleaned:
                row["note"] = cleaned
            else:
                row.pop("note", None)
    return handover


def _bundled_index() -> dict[str, object]:
    from importlib import resources

    raw = resources.files("voxprint_common.mdl.data").joinpath("model-index.json").read_text(encoding="utf-8")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise SystemExit("bundled model-index.json is not an object")
    return document


def _retired_ids(handover: dict[str, object]) -> set[str]:
    """Index ids that must not appear in the signed document."""
    retired = {"moss-image-compressed"}
    rows = handover.get("retired_rows")
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict) and item.get("id"):
                retired.add(str(item["id"]))
    return retired


def build_index(
    handover: dict[str, object],
    *,
    serial: int,
    head: str,
    generated_at: str,
) -> dict[str, object]:
    """Overlay handover entries onto the bundled serial-10 models and bump the serial."""
    base = _bundled_index()
    by_id = {str(row["id"]): dict(row) for row in base["models"] if isinstance(row, dict)}
    for entry in handover.get("entries") or []:
        if not isinstance(entry, dict) or "id" not in entry:
            raise SystemExit("handover entry is missing id")
        mid = str(entry["id"])
        row = dict(by_id.get(mid, {}))
        for key in ("id", "repo", "revision", "sha256", "default", "optional", "note", "size", "license"):
            if key in entry:
                row[key] = entry[key]
        by_id[mid] = row
    models = [by_id[key] for key in by_id if key not in _retired_ids(handover)]
    models.sort(key=lambda row: str(row["id"]))
    document = {
        "head": head,
        "serial": serial,
        "generated_at": generated_at,
        "models": models,
    }
    for key, value in base.items():
        if key not in document:
            document[key] = value
    return document


def main(argv: list[str] | None = None) -> int:
    """Build, refuse on TODO, then sign and verify. Never prints the private key."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handover", type=Path, required=True, help="Components index-entries JSON.")
    parser.add_argument(
        "--turbo-revision",
        required=True,
        help="40 lowercase hex commit for the z-image-turbo index row (replaces TODO-pending-upload).",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "signed-index" / "index.json")
    parser.add_argument("--serial", type=int, default=11)
    parser.add_argument("--head", default="", help="Defaults to the handover or bundled pin head.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the document, write nothing, and do not sign.",
    )
    args = parser.parse_args(argv)

    if not args.handover.is_file():
        print(f"missing handover {args.handover}", file=sys.stderr)
        return 2
    if _COMMIT.fullmatch(args.turbo_revision) is None:
        print("--turbo-revision must be a 40-character lowercase hex commit", file=sys.stderr)
        return 2

    handover = json.loads(args.handover.read_text(encoding="utf-8"))
    if not isinstance(handover, dict):
        print("handover is not an object", file=sys.stderr)
        return 2
    _apply_turbo_revision(handover, args.turbo_revision)
    leftover = _todo_paths(handover)
    if leftover:
        print("refuse to sign: TODO remains at " + ", ".join(leftover), file=sys.stderr)
        return 2

    head = args.head or str(handover.get("head") or "")
    if not head:
        print("missing index head", file=sys.stderr)
        return 2
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    document = build_index(handover, serial=args.serial, head=head, generated_at=generated_at)
    leftover = _todo_paths(document)
    if leftover:
        print("refuse to sign: TODO remains at " + ", ".join(leftover), file=sys.stderr)
        return 2

    # Late import so --help works without the package on PYTHONPATH in odd layouts.
    sys.path.insert(0, str(ROOT / "src"))
    from voxprint_forge.mdl import check_index, sign_index
    from voxprint_common.mdl.index import verify_index_file

    check_index(document)
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if args.dry_run:
        print(f"dry-run ok: serial={document['serial']} head={document['head']} models={len(document['models'])}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    sign_index(args.out)
    public = ROOT / "keys" / "index-public.pem"
    ok = verify_index_file(args.out, public_key_path=public if public.is_file() else None)
    if not ok:
        print("signature verify failed", file=sys.stderr)
        return 1
    print(f"signed {args.out} serial={document['serial']} verified={ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
