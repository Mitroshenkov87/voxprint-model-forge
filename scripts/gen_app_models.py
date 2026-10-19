"""Regenerate manifests/model-forge.models.json from the Hugging Face API (no token).

Each entry is pinned to a full commit; sizes and sha256 come from the API
(LFS files) or from the bytes of the small non-LFS files at that commit.
Card art, notebooks and duplicate .bin weights are left out. The packaged
copy src/voxprint_forge/data/app-models.json is written too, and
src/voxprint_forge/data/mdl.json is written from those same entries.

    python scripts/gen_app_models.py
    python scripts/gen_app_models.py --check
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests" / "model-forge.models.json"
PACKAGED = ROOT / "src" / "voxprint_forge" / "data" / "app-models.json"
MDL = ROOT / "src" / "voxprint_forge" / "data" / "mdl.json"
SCHEMA = "voxprint-forge-models/1"
# tier.min_vram_gib from the signed index when it is greater than 0.
MIN_VRAM_GIB = {
    "z-image": 15.5,
    "qwen3-vl-4b-instruct": 8.0,
    "florence2-large": 2.0,
    "moss-image-base": 23.5,
}
VRAM_TIER = {
    "z-image": "16gb",
    "z-image-turbo": "16gb",
    "moss-image-base": "24gb",
}

# (role, MDL id, display name, repo, revision prefix)
HF_ENTRIES = [
    ("required", "z-image", "Z-Image base (BF16)", "Tongyi-MAI/Z-Image", "04cc4abb"),
    ("required", "qwen3-vl-4b-instruct", "Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec"),
    ("required", "mivolo-v2", "MiVOLO v2", "iitolstykh/mivolo_v2", "53393526"),
    ("optional", "florence2-large", "Florence-2-large", "microsoft/Florence-2-large", "21a599d4"),
    ("optional", "moss-image-base", "MOSS-Image Base", "OpenMOSS-Team/MOSS-Image-Base", "1ceede80"),
    ("optional", "z-image-turbo", "Z-Image-Turbo", "Tongyi-MAI/Z-Image-Turbo", "f332072a"),
    ("optional", "dinov2-base", "DINOv2-base", "facebook/dinov2-base", "f9e44c81"),
]
# opencv_zoo has no HF repo; the files are fetched from the pinned GitHub commit.
YUNET = {
    "role": "required",
    "id": "yunet-2023mar",
    "display_name": "YuNet face detector (2023mar)",
    "url": "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/",
    "revision": "47534e27c9851bb1128ccc0102f1145e27f23f98",
    "license": "mit",
    "files": ["LICENSE", "face_detection_yunet_2023mar.onnx"],
}
SKIP = ("README.md", ".gitattributes", ".gitignore", "assets/*", "*.jpg", "*.png", "*.webp", "*.ipynb",
        "CODE_OF_CONDUCT.md", "SECURITY.md", "SUPPORT.md", "requirements.txt")


def _skip(path: str, names: set[str]) -> bool:
    if any(fnmatch.fnmatch(path, pattern) for pattern in SKIP):
        return True
    # pytorch_model.bin is a duplicate of model.safetensors in the same folder.
    return path.endswith("pytorch_model.bin") and path.replace("pytorch_model.bin", "model.safetensors") in names


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hf_entry(api: HfApi, role: str, mdl_id: str, name: str, repo: str, prefix: str) -> dict[str, object]:
    commits = [c.commit_id for c in api.list_repo_commits(repo, token=False) if c.commit_id.startswith(prefix)]
    if len(commits) != 1:
        raise SystemExit(f"{repo}@{prefix}: {len(commits)} matching commits")
    revision = commits[0]
    info = api.model_info(repo, revision=revision, files_metadata=True, token=False)
    licence = str((info.card_data or {}).get("license") or "")
    names = {s.rfilename for s in info.siblings}
    files = []
    for s in sorted(info.siblings, key=lambda item: item.rfilename):
        if _skip(s.rfilename, names):
            continue
        if s.lfs:
            digest = s.lfs.sha256
        else:
            local = hf_hub_download(repo, s.rfilename, revision=revision, token=False)
            digest = _sha(Path(local).read_bytes())
        files.append({"path": s.rfilename, "size": int(s.size), "sha256": digest})
    return _entry(role, mdl_id, name, {"repo": repo}, revision, licence, files)


def url_entry(spec: dict[str, object]) -> dict[str, object]:
    files = []
    for path in spec["files"]:  # type: ignore[union-attr]
        data = urllib.request.urlopen(f"{spec['url']}{path}", timeout=60).read()
        files.append({"path": path, "size": len(data), "sha256": _sha(data)})
    return _entry(str(spec["role"]), str(spec["id"]), str(spec["display_name"]), {"url": spec["url"]},
                  str(spec["revision"]), str(spec["license"]), files)


def _entry(role, mdl_id, name, source, revision, licence, files) -> dict[str, object]:
    return {"id": mdl_id, "display_name": name, "role": role, **source, "revision": revision, "license": licence,
            "size": sum(item["size"] for item in files), "files": files}


def app_document(entries: list[dict[str, object]]) -> dict[str, object]:
    """``voxprint-forge-models/1`` document for the packaged app list."""
    return {
        "schema": SCHEMA,
        "app": "model-forge",
        "required": [e["id"] for e in entries if e["role"] == "required"],
        "optional": [e["id"] for e in entries if e["role"] == "optional"],
        "models": [{k: v for k, v in e.items() if k != "role"} for e in entries],
    }


def _mirror(model: dict[str, object], path: str) -> str:
    revision = str(model["revision"])
    repo = model.get("repo")
    if repo:
        return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    return f"{model['url']}{path}"


def mdl_document(document: dict[str, object]) -> dict[str, object]:
    """Schema 1 ``mdl.json`` built from one app-models document."""
    required = set(document["required"])  # type: ignore[arg-type]
    models = []
    for model in document["models"]:  # type: ignore[union-attr]
        if not isinstance(model, dict):
            continue
        mdl_id = str(model["id"])
        row: dict[str, object] = {
            "id": mdl_id,
            "version": model["revision"],
            "vram_tier": VRAM_TIER.get(mdl_id, "any"),
        }
        min_vram = MIN_VRAM_GIB.get(mdl_id)
        if min_vram is not None and min_vram > 0:
            row["min_vram_gib"] = min_vram
        row["optional"] = mdl_id not in required
        row["display_name"] = model["display_name"]
        row["licence"] = model["license"]
        row["files"] = [
            {
                "path": item["path"],
                "size": item["size"],
                "sha256": item["sha256"],
                "mirrors": [_mirror(model, str(item["path"]))],
            }
            for item in model["files"]  # type: ignore[union-attr]
        ]
        models.append(row)
    return {
        "schema": 1,
        "app": "model-forge",
        "generated_from": "src/voxprint_forge/data/app-models.json",
        "models": models,
    }


def _dump(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def check() -> int:
    """Exit 1 when the committed ``mdl.json`` differs from the app-models list."""
    document = json.loads(PACKAGED.read_text(encoding="utf-8"))
    current = json.loads(MDL.read_text(encoding="utf-8")) if MDL.is_file() else None
    if current != mdl_document(document):
        print("src/voxprint_forge/data/mdl.json differs", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Write both model lists, or check ``mdl.json`` with ``--check``."""
    parser = argparse.ArgumentParser(prog="gen_app_models")
    parser.add_argument("--check", action="store_true", help="Fail when mdl.json differs.")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    api = HfApi()
    entries = [hf_entry(api, *spec) for spec in HF_ENTRIES]
    entries.insert(3, url_entry(YUNET))
    document = app_document(entries)
    text = _dump(document)
    for path in (OUT, PACKAGED):
        path.write_text(text, encoding="utf-8")
    MDL.write_text(_dump(mdl_document(document)), encoding="utf-8")
    print(OUT, len(entries), "models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
