"""Calibration prompts and image refs under ``data/diffusion_calib``."""

from __future__ import annotations

import json
import os
from pathlib import Path

from voxprint_forge.errors import PlanError

REQUIRED_KINDS = frozenset({"edit", "people", "text", "hands"})
REQUIRED_LANGS = frozenset({"ru", "en"})
GATE_CASES = ("identity", "hands", "text")


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
