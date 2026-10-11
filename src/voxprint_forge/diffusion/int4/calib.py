"""Calibration prompts and image refs under ``data/diffusion_calib``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from voxprint_forge.errors import PlanError

REQUIRED_KINDS = frozenset({"edit", "people", "text", "hands"})
REQUIRED_LANGS = frozenset({"ru", "en"})
GATE_CASES = ("identity", "hands", "text")
_PRIVATE_NOTICES: set[str] = set()


def calib_dir(repo: Path) -> Path:
    """Directory of the committed calibration set."""
    return repo / "data" / "diffusion_calib"


def image_root(repo: Path) -> Path:
    """Root for image refs. ``FORGE_CALIB_IMAGE_DIR`` replaces the committed swatches."""
    env = os.environ.get("FORGE_CALIB_IMAGE_DIR", "").strip()
    if env:
        return Path(env)
    return calib_dir(repo)


def load_prompts(repo: Path) -> list[dict[str, object]]:
    """Load prompts and check languages, kinds, and image refs."""
    path = calib_dir(repo) / "prompts.json"
    if not path.is_file():
        raise PlanError(f"missing calibration set {path}", code="calib_missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise PlanError(f"{path} has no prompts", code="calib_shape")
    kinds: set[str] = set()
    langs: set[str] = set()
    for item in prompts:
        if not isinstance(item, dict):
            raise PlanError(f"{path} has a prompt that is not an object", code="calib_shape")
        lang = str(item.get("lang", ""))
        item_kinds = item.get("kinds")
        if lang not in REQUIRED_LANGS or not isinstance(item_kinds, list):
            raise PlanError(f"{path} prompt {item.get('id')} is missing lang or kinds", code="calib_shape")
        langs.add(lang)
        kinds.update(str(kind) for kind in item_kinds)
        image = image_root(repo) / str(item.get("image", ""))
        if not image.is_file():
            raise PlanError(f"missing image ref {image}", code="calib_image")
    missing_kinds = REQUIRED_KINDS - kinds
    missing_langs = REQUIRED_LANGS - langs
    if missing_kinds or missing_langs:
        raise PlanError(
            f"calibration set is missing kinds {sorted(missing_kinds)} langs {sorted(missing_langs)}",
            code="calib_coverage",
        )
    return [dict(item) for item in prompts if isinstance(item, dict)]


def calibration_prompts(repo: Path) -> list[dict[str, object]]:
    """Public prompts plus the optional ``FORGE_PRIVATE_CALIB`` set.

    The variable is a directory or a private Hugging Face dataset id. When it
    is unset or cannot be loaded, that set is skipped and the public prompts
    still return.
    """
    return load_prompts(repo) + _private_prompts()


def _warn_private(message: str) -> None:
    if message in _PRIVATE_NOTICES:
        return
    _PRIVATE_NOTICES.add(message)
    print(message, file=sys.stderr)


def _private_prompts() -> list[dict[str, object]]:
    raw = os.environ.get("FORGE_PRIVATE_CALIB", "").strip()
    if not raw:
        _warn_private("FORGE_PRIVATE_CALIB is unset; that calibration set is skipped.")
        return []
    try:
        return _read_private(_private_root(raw))
    except Exception:
        _warn_private("FORGE_PRIVATE_CALIB could not be loaded; that calibration set is skipped.")
        return []


def _private_root(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        return path
    if _dataset_id(raw):
        folder = _private_cache()
        snapshot_download(repo_id=raw, repo_type="dataset", local_dir=str(folder))
        if not folder.is_dir():
            raise PlanError("private calibration download produced no directory", code="calib_private")
        return folder
    raise PlanError("private calibration path is not a directory", code="calib_private")


def _dataset_id(raw: str) -> bool:
    if Path(raw).is_absolute() or raw.startswith("."):
        return False
    owner, _slash, name = raw.partition("/")
    return bool(owner) and bool(name) and "/" not in name and "\\" not in raw


def _private_cache() -> Path:
    root = os.environ.get("FORGE_OUT", "").strip()
    base = Path(root) if root else Path.home() / ".cache" / "voxprint-forge"
    return base / "private-calib"


def _read_private(root: Path) -> list[dict[str, object]]:
    path = root / "prompts.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise PlanError(f"{path} has no prompts", code="calib_private")
    rows: list[dict[str, object]] = []
    for item in prompts:
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt") or not item.get("image"):
            raise PlanError(f"{path} has an incomplete prompt", code="calib_private")
        row = dict(item)
        image = Path(str(row["image"]))
        if not image.is_absolute():
            image = root / image
        if not image.is_file():
            raise PlanError("private calibration image is missing", code="calib_private")
        row["image"] = str(image.resolve())
        rows.append(row)
    return rows


def gate_prompts(prompts: list[dict[str, object]]) -> list[dict[str, object]]:
    """English hard cases used for the winner: identity, hands, and text."""
    chosen: list[dict[str, object]] = []
    for case in GATE_CASES:
        match = next(
            (
                item
                for item in prompts
                if item.get("case") == case and item.get("lang") == "en"
            ),
            None,
        )
        if match is None:
            raise PlanError(f"calibration set has no English {case} case", code="calib_coverage")
        chosen.append(match)
    return chosen


def write_prompt_yaml(prompts: list[dict[str, object]], dest: Path, *, image_root: Path) -> None:
    """Write the deepcompressor prompt map. Image paths are absolute."""
    lines = ["# Generated from data/diffusion_calib. Apache-2.0."]
    for item in prompts:
        image = (image_root / str(item["image"])).resolve()
        prompt = str(item["prompt"]).replace('"', '\\"')
        lines.append(f'{item["id"]}:')
        lines.append(f'  prompt: "{prompt}"')
        lines.append(f'  image: "{image.as_posix()}"')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
