"""Compare src/voxprint_forge/data/mdl.json with the signed model index.

    python tools/check_model_index.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from voxprint_common.mdl.index import load_index, parse_index, verify_index_file

ROOT = Path(__file__).resolve().parents[1]
MDL = ROOT / "src" / "voxprint_forge" / "data" / "mdl.json"
INDEX_REPO = "Mitroshenkov87/voxprint-model-index"
INDEX_REVISION = "main"
MIN_SERIAL = 10

# Forge ids with no index row, and the index id recorded beside each one.
NOT_IN_INDEX = {"z-image-turbo": "mirror-z-image-turbo"}


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-model-forge-ci"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return bytes(response.read())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dataset(path: str, revision: str = INDEX_REVISION) -> bytes:
    return _get(f"https://huggingface.co/datasets/{INDEX_REPO}/resolve/{revision}/{path}")


def _descriptor(row: dict[str, str]) -> bytes:
    repo, revision = row["repo"], row["revision"]
    if repo == f"datasets/{INDEX_REPO}":
        return _dataset(f"descriptors/{row['id']}/voxprint-model.json", revision)
    return _get(f"https://huggingface.co/{repo}/resolve/{revision}/voxprint-model.json")


def _verify(payload: bytes, signature: bytes) -> bool:
    with tempfile.TemporaryDirectory() as folder:
        index = Path(folder) / "index.json"
        index.write_bytes(payload)
        (Path(folder) / "index.json.sig").write_bytes(signature)
        return verify_index_file(index)


def check(
    fetch_index: Callable[[str], bytes] = _dataset,
    fetch_descriptor: Callable[[dict[str, str]], bytes] = _descriptor,
    manifest: dict[str, object] | None = None,
    *,
    verify: Callable[[bytes, bytes], bool] = _verify,
    min_serial: int = MIN_SERIAL,
) -> list[str]:
    """Return problem lines. Empty when mdl.json matches the signed index."""
    problems: list[str] = []
    raw = fetch_index("index.json")
    try:
        signature = fetch_index("index.json.sig")
    except OSError:
        signature = b""
    if not signature or not verify(raw, signature):
        return ["index.json.sig"]
    index = json.loads(raw.decode("utf-8"))
    parse_index(index)
    bundled = load_index(remote=False).serial
    floor = max(min_serial, bundled)
    serial = index.get("serial")
    if not isinstance(serial, int) or serial < floor:
        problems.append(f"index.json serial {serial}")
    rows = {row["id"]: row for row in index["models"] if isinstance(row, dict) and "id" in row}
    descriptors: dict[str, dict[str, object]] = {}
    for row in rows.values():
        try:
            blob = fetch_descriptor(row)
        except OSError as exc:
            problems.append(f"{row['id']}: {exc}")
            continue
        if _sha(blob) != row.get("sha256"):
            problems.append(f"{row['id']}: sha256")
            continue
        parsed = json.loads(blob.decode("utf-8"))
        if isinstance(parsed, dict):
            descriptors[str(row["id"])] = parsed
    manifest = manifest or json.loads(MDL.read_text(encoding="utf-8"))
    models = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(models, list):
        return [*problems, "mdl.json"]
    for model in models:
        if not isinstance(model, dict):
            problems.append("mdl.json")
            continue
        mdl_id = str(model.get("id") or "")
        if mdl_id in NOT_IN_INDEX:
            mapped = NOT_IN_INDEX[mdl_id]
            if mdl_id in rows:
                problems.append(f"{mdl_id}: {mapped}")
            elif mapped not in rows:
                problems.append(mapped)
            continue
        row = rows.get(mdl_id)
        if row is None:
            problems.append(mdl_id)
            continue
        descriptor = descriptors.get(mdl_id)
        if descriptor is None:
            problems.append(mdl_id)
            continue
        theirs = {
            str(item["path"]): item
            for item in descriptor.get("files") or []
            if isinstance(item, dict) and item.get("path")
        }
        for item in model.get("files") or []:
            if not isinstance(item, dict):
                problems.append(f"{mdl_id}")
                continue
            path = str(item.get("path") or "")
            want = theirs.get(path)
            if want is None:
                problems.append(f"{mdl_id}/{path}")
            elif (str(want.get("sha256")), int(want.get("size") or -1)) != (
                str(item.get("sha256")),
                int(item.get("size") or -2),
            ):
                problems.append(f"{mdl_id}/{path}")
    return problems


def main() -> int:
    """Print problem lines and return 1 when any are found."""
    problems = check()
    for line in problems:
        print(line, file=sys.stderr)
    if not problems:
        print("mdl.json index.json")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
