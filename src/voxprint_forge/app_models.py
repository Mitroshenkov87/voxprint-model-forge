"""Model Forge's required and optional models (``voxprint-app-models/1``).

The list lives in ``manifests/model-forge.models.json``; the wheel carries the
same bytes as ``voxprint_forge/data/app-models.json``. Each model is pinned to a
full commit with per-file size and sha256 from the Hugging Face API, so the
first-run check reads this file instead of a hardcoded list.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from voxprint_forge.errors import StageError

SCHEMA = "voxprint-app-models/1"
PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"})


@dataclass(frozen=True)
class AppModel:
    """One pinned model of the app."""

    id: str
    display_name: str
    source: str  # Hugging Face repo id, or a URL folder ending in "/"
    revision: str
    license: str
    size: int
    files: tuple[dict[str, object], ...]
    required: bool


def load(path: Path | None = None) -> list[AppModel]:
    """Read and check the app-models file; the packaged copy by default."""
    if path is None:
        text = resources.files("voxprint_forge").joinpath("data/app-models.json").read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return parse(json.loads(text))


def parse(document: object) -> list[AppModel]:
    """Check the structure the app relies on and return the models, required first."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise StageError(f"app models file is not {SCHEMA}", code="mdl")
    required, optional = list(document.get("required") or []), list(document.get("optional") or [])
    models = {str(item.get("id")): item for item in document.get("models") or [] if isinstance(item, dict)}
    if set(required) & set(optional):
        raise StageError("a model is both required and optional", code="mdl")
    if set(required) | set(optional) != set(models):
        raise StageError("required + optional must list every model exactly once", code="mdl")
    out = []
    for mdl_id in required + optional:
        item = models[mdl_id]
        revision = str(item.get("revision", ""))
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise StageError(f"{mdl_id} revision must be a full commit", code="mdl")
        files = tuple(item.get("files") or ())
        if not files or sum(int(f["size"]) for f in files) != int(item.get("size", -1)):
            raise StageError(f"{mdl_id} size does not match its files", code="mdl")
        out.append(
            AppModel(
                id=mdl_id,
                display_name=str(item["display_name"]),
                source=str(item.get("repo") or item.get("url")),
                revision=revision,
                license=str(item["license"]),
                size=int(item["size"]),
                files=files,
                required=mdl_id in required,
            )
        )
    return out


def missing_required(installed: Callable[[str], bool], models: list[AppModel] | None = None) -> list[str]:
    """Required model ids that ``installed`` reports as absent, in file order."""
    return [m.id for m in (models if models is not None else load()) if m.required and not installed(m.id)]
