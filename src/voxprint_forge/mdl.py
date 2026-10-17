"""Shared ``voxprint-model.json`` manifest (schema ``voxprint-model/1``).

The schema is provisional until VOXPrint Components reviews its MDL change.
The Plotweaver run export in ``manifest.py`` is a different file and stays.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Never

from huggingface_hub import HfApi, hf_hub_download

from voxprint_forge import __version__
from voxprint_forge.cards import hub_title
from voxprint_forge.errors import StageError, UsageError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.licence import normalize
from voxprint_forge.llama import TAG as LLAMA_CPP_TAG
from voxprint_forge.scrub import active_secrets, scrub
from voxprint_forge.upload import Artifact

SCHEMA_ID = "voxprint-model/1"
MANIFEST_NAME = "voxprint-model.json"
INDEX_REPO = "Mitroshenkov87/voxprint-model-index"
INDEX_FILE = "index.json"
INDEX_SIG_FILE = "index.json.sig"
INDEX_KEY_ENV = "VOXPRINT_MDL_INDEX_KEY_FILE"
# Official-source models have no forge repo. Their descriptor lives in the
# index dataset at descriptors/<id>/voxprint-model.json, and the index row's
# repo is this string (the Hub URL prefix of the dataset).
HOSTED_REPO = f"datasets/{INDEX_REPO}"
HOSTED_DIR = "descriptors"
# Serial and head of the index bundled in voxprint-common 0.2.5. A published
# index must be newer, so the first signed one is serial 2.
BUNDLED_INDEX_SERIAL = 1
BUNDLED_INDEX_HEAD = "d2325ed8de2eec808b22a88381bfb35055f84203"
INDEX_MIN_GENERATED_AT = "2026-10-17T10:42:07Z"
HF_AUTHOR = "Mitroshenkov87"
DIFFUSERS_VERSION_REQ = ">=0.37.1,<0.38"
NUNCHAKU_VERSION_REQ = ">=1.0"
WAN_LLAMA_TAG = "b3962"
LLAMA_CC = 6.0
CUDA_CC = 8.0

Engine = Literal["llama.cpp", "diffusers", "nunchaku", "onnxruntime"]

_TOP = (
    "schema",
    "id",
    "display_name",
    "app_ids",
    "task",
    "license",
    "files",
    "mirrors",
    "tier",
    "runtime",
    "base_model",
    "card_url",
    "flags",
    "on_demand",
    "build",
)
# Optional voxprint-model/1 fields. mirror_policy, file_groups and per-file
# sources come from voxprint-common MDL PR #15. prebuilt, post_install and
# built_for are the 0.3.0 additions; they are accepted, not interpreted.
_TOP_OPTIONAL = ("mirror_policy", "file_groups", "prebuilt", "post_install")
_FILE_KEYS = ("path", "size", "sha256")
_FILE_OPTIONAL = ("mirror_policy", "sources", "built_for")
_GROUP_OPTIONAL = ("mirror_policy", "sources")
_POLICIES = frozenset({"race", "strict-priority"})
_INDEX_KEYS = ("head", "serial", "generated_at", "models")
_HEAD = re.compile(r"^[0-9a-f]{7,64}$")
_MIRROR_KEYS = ("kind", "ref", "revision")
_TIER_KEYS = ("min_vram_gib", "recommended_vram_gib", "gpu_min_cc")
_RUNTIME_KEYS = ("engine", "version_req")
_BUILD_KEYS = ("forge_version", "tag", "date")
_KINDS = frozenset({"hf", "kaggle", "gdrive", "modelscope", "url"})
_SPDX = re.compile(r"^[a-z0-9][a-z0-9.+-]*$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CARD = re.compile(r"^(?:|https://\S+)$")
_INLINE_MAX = 8 * 1024 * 1024
_INDEX_REPO_NAME = "voxprint-model-index"


def schema_path() -> Path:
    """Path of ``schemas/voxprint-model.schema.json`` in a repo checkout."""
    return Path(__file__).resolve().parents[2] / "schemas" / "voxprint-model.schema.json"


def gb_to_gib(gb: float) -> float:
    """Convert a decimal GB card budget to the GiB value this manifest records.

    MDL treats a 24 GB card as about 23.5 GiB. The value is ``gb * 1000 / 1024``,
    rounded to the nearest 0.5, so 24 becomes 23.5 and 16 becomes 15.5.
    """
    raw = float(gb) * 1000.0 / 1024.0
    return round(raw * 2.0) / 2.0


def default_cc(engine: Engine) -> float:
    """Minimum CUDA compute capability for one runtime engine."""
    match engine:
        case "llama.cpp":
            return LLAMA_CC
        case "diffusers":
            return CUDA_CC
        case "nunchaku":
            return CUDA_CC
        case _ as unreachable:
            _never: Never = unreachable
            raise StageError(f"unknown engine {_never}", code="mdl")


def build_document(
    *,
    model_id: str,
    display_name: str,
    app_ids: Sequence[str],
    task: str,
    license: str,
    files: Sequence[Mapping[str, object]],
    mirrors: Sequence[Mapping[str, str]],
    min_vram_gb: float,
    recommended_vram_gb: float,
    gpu_min_cc: float,
    engine: str,
    version_req: str,
    base_model: str,
    card_url: str,
    flags: Sequence[str] = (),
    on_demand: bool = True,
    forge_version: str = "",
    tag: str = "",
    date: str = "",
) -> dict[str, object]:
    """Build one ``voxprint-model/1`` document. VRAM arguments are decimal GB."""
    copied = [{"path": str(item["path"]), "size": item["size"], "sha256": str(item["sha256"])} for item in files]
    copied.sort(key=lambda item: str(item["path"]))
    document: dict[str, object] = {
        "schema": SCHEMA_ID,
        "id": model_id,
        "display_name": display_name,
        "app_ids": list(app_ids),
        "task": task,
        "license": normalize(license),
        "files": copied,
        "mirrors": [dict(item) for item in mirrors],
        "tier": {
            "min_vram_gib": gb_to_gib(min_vram_gb),
            "recommended_vram_gib": gb_to_gib(recommended_vram_gb),
            "gpu_min_cc": float(gpu_min_cc),
        },
        "runtime": {"engine": engine, "version_req": version_req},
        "base_model": base_model,
        "card_url": card_url,
        "flags": list(flags),
        "on_demand": on_demand,
        "build": {
            "forge_version": forge_version or __version__,
            "tag": tag,
            "date": date or _today(),
        },
    }
    check_document(document)
    return document


def write_manifest(directory: Path, document: Mapping[str, object]) -> Path:
    """Write ``voxprint-model.json`` into ``directory`` and return its path."""
    check_document(document)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(_dump(document), encoding="utf-8")
    return path


def check_document(document: object) -> None:
    """Raise ``StageError`` when ``document`` is not a ``voxprint-model/1`` object."""
    if not isinstance(document, dict):
        raise StageError("voxprint-model document is not an object", code="mdl")
    _keys(document, _TOP, _TOP_OPTIONAL, "document")
    if document["schema"] != SCHEMA_ID:
        raise StageError(f"schema must be {SCHEMA_ID}", code="mdl")
    _ident(document["id"])
    _text(document["display_name"], "display_name")
    _str_list(document["app_ids"], "app_ids", minimum=1)
    _text(document["task"], "task")
    _licence(document["license"])
    _files(document["files"])
    _mirrors(document["mirrors"])
    _tier(document["tier"])
    _runtime(document["runtime"])
    if not isinstance(document["base_model"], str):
        raise StageError("base_model must be a string", code="mdl")
    _card(document["card_url"])
    _str_list(document["flags"], "flags", minimum=0)
    if not isinstance(document["on_demand"], bool):
        raise StageError("on_demand must be a boolean", code="mdl")
    _build(document["build"])
    if "mirror_policy" in document:
        _policy(document["mirror_policy"])
    if "file_groups" in document:
        _file_groups(document["file_groups"])
    if "prebuilt" in document:
        _prebuilt(document["prebuilt"])
    if "post_install" in document and not isinstance(document["post_install"], (dict, list)):
        raise StageError("post_install must be an object or a list", code="mdl")


def hash_tree(root: Path) -> list[dict[str, object]]:
    """Hash files under ``root``. ``voxprint-model.json`` is omitted."""
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if MANIFEST_NAME == relative.as_posix() or "kaggle-staging" in relative.parts:
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def rows_from_artifacts(artifacts: Sequence[Artifact]) -> list[dict[str, object]]:
    """File rows for a manifest. The manifest artifact itself is omitted."""
    rows = [
        {"path": item.name, "size": item.size, "sha256": item.sha256}
        for item in artifacts
        if item.name != MANIFEST_NAME
    ]
    rows.sort(key=lambda item: str(item["path"]))
    return rows


def manifest_artifact(path: Path) -> Artifact:
    """Artifact for an already written ``voxprint-model.json``."""
    return Artifact(path, MANIFEST_NAME, sha256_file(path), path.stat().st_size, "shared", "", "mdl")


def refresh_manifest(artifacts: list[Artifact], path: Path) -> list[Artifact]:
    """Replace the manifest artifact after its bytes change."""
    fresh = manifest_artifact(path)
    return [fresh if item.name == MANIFEST_NAME else item for item in artifacts]


def hub_mirrors(
    *,
    hf_repo: str = "",
    kaggle_ref: str = "",
    hf_revision: str = "main",
    kaggle_revision: str = "private",
) -> list[dict[str, str]]:
    """Hugging Face and Kaggle mirror entries. Empty refs are left out."""
    found: list[dict[str, str]] = []
    if hf_repo:
        found.append({"kind": "hf", "ref": hf_repo, "revision": hf_revision})
    if kaggle_ref:
        found.append({"kind": "kaggle", "ref": kaggle_ref, "revision": kaggle_revision})
    return found


def drive_mirror(model_id: str, date: str) -> dict[str, str]:
    """Drive mirror for ``Voxprint/model-forge/<date>/<model>``."""
    return {"kind": "gdrive", "ref": f"Voxprint/model-forge/{date}/{model_id}", "revision": ""}


def card_url_for(mirrors: Sequence[Mapping[str, str]]) -> str:
    """Hub card URL for the mirror that holds the descriptor.

    An empty string when there is no ``hf`` mirror. A marked ``ours`` mirror
    or the ``Mitroshenkov87`` mirror wins over an upstream author listed first.
    One ``hf`` mirror is that repo: publish writes the card there.
    """
    if not any(item.get("kind") == "hf" and item.get("ref") for item in mirrors):
        return ""
    chosen = _descriptor_mirror(mirrors, label="mirrors", sole_repo=True)
    return f"https://huggingface.co/{chosen['ref']}"


def retarget_mirrors(path: Path, mirrors: Sequence[Mapping[str, str]], *, card_url: str | None = None) -> None:
    """Rewrite mirrors on an existing manifest and refresh its digest.

    The default card URL is our repo, the mirror that holds the descriptor.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise StageError(f"{path} is not a manifest object", code="mdl")
    chosen = [dict(item) for item in mirrors]
    document["mirrors"] = chosen
    document["card_url"] = card_url_for(chosen) if card_url is None else card_url
    check_document(document)
    path.write_text(_dump(document), encoding="utf-8")


def infer_task(repo_id: str, paths: Sequence[str]) -> str:
    """Pick a task string from a repo id and its file paths."""
    lower = repo_id.lower()
    if "edit" in lower or "control" in lower:
        return "image-to-image"
    if "wan" in lower or "t2v" in lower or "video" in lower:
        return "text-to-video"
    if any(path.endswith(".gguf") for path in paths):
        return "text-generation"
    return "text-to-image"


def infer_app_ids(task: str) -> tuple[str, ...]:
    """Suite app ids for a task. Text models serve Plotweaver and Audiobook Builder."""
    if task == "text-generation":
        return ("plotweaver", "audiobook-builder")
    return ("canvas",)


def infer_manifest(slug: str, licence: str, *, title: str = "") -> ManifestInput:
    """Best-effort manifest fields when a publish step has only a slug."""
    model_id = slug.removeprefix("voxprint-")
    engine, version_req, cc, task = _infer_from_slug(slug)
    return ManifestInput(
        model_id=model_id,
        display_name=title or hub_title(model_id),
        app_ids=infer_app_ids(task),
        task=task,
        license=licence,
        base_model="",
        engine=engine,
        version_req=version_req,
        min_vram_gb=16,
        recommended_vram_gb=24,
        gpu_min_cc=cc,
    )


@dataclass(frozen=True)
class ManifestInput:
    """Fields known before the published file list is hashed."""

    model_id: str
    display_name: str
    app_ids: tuple[str, ...]
    task: str
    license: str
    base_model: str
    engine: str
    version_req: str
    min_vram_gb: float
    recommended_vram_gb: float
    gpu_min_cc: float
    flags: tuple[str, ...] = ()
    on_demand: bool = True
    tag: str = ""
    date: str = ""


@dataclass(frozen=True)
class RemoteFile:
    """One file listed on a Hub repo."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RemoteRepo:
    """One published ``voxprint-*`` repo the backfill can turn into a manifest."""

    repo_id: str
    revision: str
    license: str
    files: tuple[RemoteFile, ...]
    base_model: str = ""
    task: str = ""
    display_name: str = ""
    min_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    engine: str = ""
    version_req: str = ""
    flags: tuple[str, ...] = ()
    app_ids: tuple[str, ...] = ()
    gpu_min_cc: float | None = None


@dataclass(frozen=True)
class Listing:
    """Repos the backfill can write, and repos it skipped."""

    repos: tuple[RemoteRepo, ...]
    skipped: tuple[tuple[str, str], ...]


def document_from_repo(repo: RemoteRepo, *, date: str = "") -> dict[str, object]:
    """Build a manifest for one already published repo. Nothing is uploaded."""
    name = repo.repo_id.split("/")[-1]
    model_id = name.removeprefix("voxprint-")
    paths = [item.path for item in repo.files if item.path != MANIFEST_NAME]
    engine, version_req, cc = _infer_runtime(repo.repo_id, paths)
    task = repo.task or infer_task(repo.repo_id, paths)
    chosen_engine = repo.engine or engine
    mirrors = hub_mirrors(hf_repo=repo.repo_id, hf_revision=repo.revision or "main")
    min_gb, rec_gb = _vram_pair(repo, paths)
    files = [
        {"path": item.path, "size": item.size, "sha256": item.sha256}
        for item in repo.files
        if item.path != MANIFEST_NAME
    ]
    flags = repo.flags or (("least-censored",) if "abliterat" in repo.repo_id.lower() else ())
    return build_document(
        model_id=model_id,
        display_name=repo.display_name or hub_title(model_id),
        app_ids=repo.app_ids or infer_app_ids(task),
        task=task,
        license=repo.license,
        files=files,
        mirrors=mirrors,
        min_vram_gb=min_gb,
        recommended_vram_gb=rec_gb,
        gpu_min_cc=cc if repo.gpu_min_cc is None else repo.gpu_min_cc,
        engine=chosen_engine,
        version_req=repo.version_req or version_req,
        base_model=repo.base_model,
        card_url=card_url_for(mirrors),
        flags=flags,
        on_demand=True,
        tag="",
        date=date,
    )


def backfill(repos: Sequence[RemoteRepo], dest: Path) -> list[Path]:
    """Write ``voxprint-model.json`` for each repo under ``dest``. Does not upload."""
    written: list[Path] = []
    for repo in repos:
        name = repo.repo_id.split("/")[-1]
        if not name.startswith("voxprint-") or name == _INDEX_REPO_NAME:
            continue
        written.append(write_manifest(dest / repo.repo_id, document_from_repo(repo)))
    return written


def list_published_repos(user: str = HF_AUTHOR) -> Listing:
    """Read ``user/voxprint-*`` model repos from the Hub. Does not upload.

    The index dataset is not a model repo and is skipped. A repo with no
    licence tag, or a file that has no sha256, is returned in ``skipped``.
    """
    api = HfApi()
    found: list[RemoteRepo] = []
    skipped: list[tuple[str, str]] = []
    for info in api.list_models(author=user, cardData=True):
        repo_id = str(info.id)
        name = repo_id.split("/")[-1]
        if not name.startswith("voxprint-") or name == _INDEX_REPO_NAME:
            continue
        try:
            found.append(_snapshot(api, info))
        except StageError as exc:
            skipped.append((repo_id, str(exc)))
    found.sort(key=lambda item: item.repo_id)
    return Listing(tuple(found), tuple(skipped))


def find_manifests(root: Path) -> list[Path]:
    """Every ``voxprint-model.json`` under ``root``."""
    if not root.is_dir():
        raise UsageError(f"{root} is not a directory", code="usage")
    return sorted(path for path in root.rglob(MANIFEST_NAME) if path.is_file())


def write_index(
    paths: Sequence[Path],
    dest: Path,
    *,
    revisions: Mapping[str, str] | None = None,
    commits: Callable[[str], str] | None = None,
    serial: int | None = None,
    head: str | None = None,
    generated_at: str | None = None,
) -> list[dict[str, str]]:
    """Write ``index.json`` and return its rows. This does not upload.

    The file is the signed-index object ``{head, serial, generated_at,
    models}``. ``serial`` defaults to the published serial plus one (at
    least 2), and ``head`` to the dataset commit this index replaces.

    ``revisions`` pins the descriptor commit for a repo. Every other repo is
    resolved with ``commits`` (the Hugging Face commit of ``voxprint-model.json``
    by default). The mirror's ``revision`` is not a fallback.
    """
    pinned = _pins(revisions)
    lookup = descriptor_commit if commits is None else commits
    rows = [_index_row(path, revisions=pinned, commits=lookup) for path in paths]
    rows.sort(key=lambda item: item["id"])
    ids = [item["id"] for item in rows]
    if len(ids) != len(set(ids)):
        raise StageError("index ids are not unique", code="mdl")
    if serial is None or head is None:
        published_serial, published_head = published_index_state()
        serial = next_serial(published_serial) if serial is None else serial
        head = published_head if head is None else head
    document = index_document(rows, serial=serial, head=head, generated_at=generated_at)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return rows


def index_document(
    rows: Sequence[Mapping[str, str]],
    *,
    serial: int,
    head: str,
    generated_at: str | None = None,
) -> dict[str, object]:
    """The signed-index object. ``generated_at`` defaults to now, in UTC."""
    stamp = generated_at or datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    document: dict[str, object] = {"head": head, "serial": serial, "generated_at": stamp, "models": list(rows)}
    check_index(document)
    return document


def next_serial(published: int | None) -> int:
    """One more than the published serial, never below the bundled pin plus one."""
    return max(published or 0, BUNDLED_INDEX_SERIAL) + 1


def check_index(document: object) -> None:
    """Raise ``StageError`` unless ``document`` is a signable index object."""
    if not isinstance(document, dict):
        raise StageError("index.json must be an object with head, serial, generated_at and models", code="mdl")
    _exact_keys(document, _INDEX_KEYS, "index")
    head = document["head"]
    if not isinstance(head, str) or _HEAD.fullmatch(head) is None:
        raise StageError("index head must be a lowercase commit sha", code="mdl")
    serial = document["serial"]
    if isinstance(serial, bool) or not isinstance(serial, int) or serial <= BUNDLED_INDEX_SERIAL:
        raise StageError(f"index serial must be an integer above {BUNDLED_INDEX_SERIAL}", code="mdl")
    stamp = document["generated_at"]
    if not isinstance(stamp, str) or _utc(stamp) < _utc(INDEX_MIN_GENERATED_AT):
        raise StageError(f"index generated_at must be UTC ISO-8601 and not before {INDEX_MIN_GENERATED_AT}", code="mdl")
    models = document["models"]
    if not isinstance(models, list) or not models:
        raise StageError("index models must be a non-empty list", code="mdl")
    for row in models:
        if not isinstance(row, dict):
            raise StageError("index row is not an object", code="mdl")
        _exact_keys(row, ("id", "repo", "revision", "sha256"), "index row")
        if not isinstance(row["revision"], str) or _COMMIT.fullmatch(row["revision"]) is None:
            raise StageError("index row revision must be a 40-hex commit", code="mdl")
        if not isinstance(row["sha256"], str) or _SHA.fullmatch(row["sha256"]) is None:
            raise StageError("index row sha256 must be 64 lowercase hex characters", code="mdl")


def per_file_sources(
    document: Mapping[str, object],
    upstream_paths: Sequence[str],
    *,
    policy: Literal["race", "strict-priority"] = "strict-priority",
) -> dict[str, object]:
    """Return a copy of ``document`` with ``mirror_policy`` and per-file ``sources``.

    Each file gets our forge mirror as its source. A path in
    ``upstream_paths`` (same bytes upstream) lists the author's mirror first.
    Paths, sizes, hashes and mirror revisions are unchanged.
    """
    mirrors = [dict(item) for item in document["mirrors"]]  # type: ignore[union-attr]
    ours = dict(_descriptor_mirror(mirrors, label=str(document.get("id"))))
    upstream = [item for item in mirrors if item.get("kind") == "hf" and item.get("ref") != ours.get("ref")]
    known = {str(item["path"]) for item in document["files"]}  # type: ignore[union-attr,index]
    unknown = sorted(set(upstream_paths) - known)
    if unknown:
        raise StageError(f"upstream paths not in files: {unknown}", code="mdl")
    if upstream_paths and len(upstream) != 1:
        raise StageError("per-file upstream sources need exactly one upstream hf mirror", code="mdl")
    files = []
    for item in document["files"]:  # type: ignore[union-attr]
        entry = {key: item[key] for key in _FILE_KEYS}  # type: ignore[index]
        entry["sources"] = ([upstream[0]] if entry["path"] in upstream_paths else []) + [ours]
        files.append(entry)
    result = dict(document)
    result["files"] = files
    result["mirror_policy"] = policy
    check_document(result)
    return result


def published_index_state() -> tuple[int | None, str]:
    """Serial and dataset commit of the published index. No token is passed.

    A bare-list index (before signing) has no serial and returns ``None``.
    """
    api = HfApi()
    try:
        info = api.dataset_info(INDEX_REPO, token=False)
        commit = str(info.sha)
        local = hf_hub_download(INDEX_REPO, INDEX_FILE, repo_type="dataset", revision=commit, token=False)
        payload = json.loads(Path(local).read_text(encoding="utf-8"))
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        raise StageError(f"published index could not be read: {message}", code="mdl") from exc
    serial = payload.get("serial") if isinstance(payload, dict) else None
    if isinstance(serial, bool) or not isinstance(serial, int):
        serial = None
    return serial, commit


def sign_index(path: Path, *, key_file: Path | None = None) -> Path:
    """Write ``index.json.sig`` beside ``path``: base64 Ed25519 over the exact bytes.

    The format matches ``voxprint-mdl index sign``. The key is a PEM private
    key read from ``key_file`` or ``VOXPRINT_MDL_INDEX_KEY_FILE``; it is
    never printed. The signature is checked against the derived public key
    before the file is written.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    source = key_file or (Path(os.environ[INDEX_KEY_ENV]) if os.environ.get(INDEX_KEY_ENV) else None)
    if source is None or not source.is_file():
        raise UsageError(f"set {INDEX_KEY_ENV} to the Ed25519 private key file", code="usage")
    key = load_pem_private_key(source.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise UsageError("index signing key is not Ed25519", code="usage")
    payload = path.read_bytes()
    signature = key.sign(payload)
    key.public_key().verify(signature, payload)
    target = path.with_name(path.name + ".sig")
    target.write_bytes(base64.b64encode(signature) + b"\n")
    return target


def run_build(
    source: Path,
    dest: Path,
    *,
    revisions: Mapping[str, str] | None = None,
    commits: Callable[[str], str] | None = None,
    serial: int | None = None,
) -> dict[str, object]:
    """Build ``index.json`` from local manifests. Does not upload."""
    rows = write_index(find_manifests(source), dest, revisions=revisions, commits=commits, serial=serial)
    document = json.loads(dest.read_text(encoding="utf-8"))
    return {
        "index": str(dest),
        "count": len(rows),
        "serial": document["serial"],
        "head": document["head"],
        "generated_at": document["generated_at"],
        "pushed": False,
    }


def revision_map(items: Sequence[str]) -> dict[str, str]:
    """Parse repeatable ``REPO=SHA`` pins. SHA is 40 lowercase hex digits."""
    found: dict[str, str] = {}
    for item in items:
        repo, sep, sha = item.partition("=")
        if sep != "=" or not repo or _COMMIT.fullmatch(sha) is None:
            raise UsageError("index revision must be REPO=40-hex commit", code="usage")
        if repo in found:
            raise UsageError(f"index revision repeated for {repo}", code="usage")
        found[repo] = sha
    return found


def descriptor_commit(repo: str) -> str:
    """Last commit that touches ``voxprint-model.json`` on the published ref.

    ``repo`` is a model repo, or a :func:`hosted_path` in the index dataset.

    No token is passed. A missing file or a value that is not a full commit
    id is an error. This does not read a mirror revision.
    """
    api = HfApi()
    if repo.startswith(HOSTED_REPO + "/"):
        target, kind, files = INDEX_REPO, "dataset", [repo[len(HOSTED_REPO) + 1 :]]
    else:
        target, kind, files = repo, "model", [MANIFEST_NAME]
    try:
        infos = api.get_paths_info(target, files, repo_type=kind, expand=True, token=False)
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        raise StageError(f"{repo} descriptor commit could not be resolved: {message}", code="mdl") from exc
    return _commit_oid(repo, infos)


def push_index(
    path: Path,
    *,
    upload: Callable[[str, Path, Path], None] | None = None,
    key_file: Path | None = None,
    published_serial: int | None | Literal["fetch"] = "fetch",
) -> dict[str, object]:
    """Sign ``index.json`` and upload it with ``index.json.sig`` in one commit.

    The CLI calls this only after ``--push``. There is no upload on the
    default path. A bare list, or a serial that is not above the published
    one, is refused before anything is signed.
    """
    if not path.is_file():
        raise UsageError(f"missing {path}", code="usage")
    document = json.loads(path.read_text(encoding="utf-8"))
    check_index(document)
    current = published_index_state()[0] if published_serial == "fetch" else published_serial
    if current is not None and document["serial"] <= current:
        raise StageError(f"index serial {document['serial']} is not above published serial {current}", code="mdl")
    signature = sign_index(path, key_file=key_file)
    sender = _upload_index if upload is None else upload
    sender(INDEX_REPO, path, signature)
    return {
        "repo": INDEX_REPO,
        "path": INDEX_FILE,
        "signature": INDEX_SIG_FILE,
        "serial": document["serial"],
        "generated_at": document["generated_at"],
        "pushed": True,
    }


def run_backfill(dest: Path, user: str = HF_AUTHOR) -> dict[str, object]:
    """Write local manifests for published repos. Does not upload."""
    listing = list_published_repos(user)
    written = backfill(listing.repos, dest)
    return {
        "written": [str(path) for path in written],
        "skipped": [{"repo": repo, "reason": reason} for repo, reason in listing.skipped],
        "uploaded": False,
    }


def is_hosted(document: Mapping[str, object]) -> bool:
    """True when no forge repo holds the descriptor, so the index dataset does.

    That is an official-source model: it has an ``hf`` or ``url`` mirror,
    none is marked ``ours``, and none is under ``Mitroshenkov87``.
    """
    mirrors = document.get("mirrors")
    items = [item for item in mirrors if isinstance(item, Mapping)] if isinstance(mirrors, list) else []
    if any(item.get("ours") is True or _forge_repo(str(item.get("ref", ""))) for item in items):
        return False
    return any(item.get("kind") in ("hf", "url") for item in items)


def hosted_path(model_id: str) -> str:
    """Row-side name of a hosted descriptor: ``datasets/<index>/descriptors/<id>/voxprint-model.json``."""
    if _ID.fullmatch(model_id) is None:
        raise StageError(f"model id {model_id!r} is not valid", code="mdl")
    return f"{HOSTED_REPO}/{HOSTED_DIR}/{model_id}/{MANIFEST_NAME}"


def host_descriptors(
    source: Path,
    *,
    upload: Callable[[list[tuple[str, Path]], str], str] | None = None,
) -> dict[str, object]:
    """Upload official-source descriptors into the index dataset in one commit.

    Only documents that :func:`is_hosted` accepts are sent, to
    ``descriptors/<id>/voxprint-model.json``. The CLI calls this only after
    ``--push``. Returns the dataset commit to pin with
    ``index build --revision datasets/Mitroshenkov87/voxprint-model-index=SHA``.
    """
    items: list[tuple[str, Path]] = []
    for path in find_manifests(source):
        document = json.loads(path.read_text(encoding="utf-8"))
        check_document(document)
        if isinstance(document, dict) and is_hosted(document):
            items.append((hosted_path(str(document["id"]))[len(HOSTED_REPO) + 1 :], path))
    if not items:
        raise UsageError(f"no official-source voxprint-model.json under {source}", code="usage")
    names = [name for name, _ in items]
    if len(names) != len(set(names)):
        raise StageError("hosted descriptor ids are not unique", code="mdl")
    sender = _upload_descriptors if upload is None else upload
    commit = sender(items, f"Official-source descriptors: {', '.join(sorted(n.split('/')[1] for n in names))}")
    if _COMMIT.fullmatch(commit) is None:
        raise StageError("descriptor upload returned no commit", code="mdl")
    return {"repo": HOSTED_REPO, "commit": commit, "paths": sorted(names), "pushed": True}


def _upload_descriptors(items: list[tuple[str, Path]], message: str) -> str:
    from huggingface_hub import CommitOperationAdd

    api = HfApi()
    info = api.create_commit(
        repo_id=INDEX_REPO,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in items],
        commit_message=message,
    )
    return str(info.oid)


def _upload_index(repo_id: str, path: Path, signature: Path) -> None:
    from huggingface_hub import CommitOperationAdd

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    serial = json.loads(path.read_text(encoding="utf-8"))["serial"]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=INDEX_FILE, path_or_fileobj=str(path)),
            CommitOperationAdd(path_in_repo=INDEX_SIG_FILE, path_or_fileobj=str(signature)),
        ],
        commit_message=f"Signed model index, serial {serial}",
    )


def _utc(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StageError(f"{value!r} is not an ISO-8601 timestamp", code="mdl") from exc
    if parsed.tzinfo is None:
        raise StageError(f"{value!r} has no timezone", code="mdl")
    return parsed.astimezone(timezone.utc)


def _index_row(
    path: Path,
    *,
    revisions: Mapping[str, str],
    commits: Callable[[str], str],
) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    check_document(document)
    if not isinstance(document, dict):
        raise StageError(f"{path} is not a manifest object", code="mdl")
    mirrors = document["mirrors"]
    if not isinstance(mirrors, list):
        raise StageError(f"{path} has no mirrors", code="mdl")
    model_id = str(document["id"])
    if is_hosted(document):
        repo = HOSTED_REPO
        revision = revisions[repo] if repo in revisions else commits(hosted_path(model_id))
    else:
        chosen = _descriptor_mirror(mirrors, label=str(path))
        repo = str(chosen["ref"])
        # chosen["revision"] is the weights commit. The row needs the commit that
        # contains voxprint-model.json, and there is no fallback to the mirror.
        revision = revisions[repo] if repo in revisions else commits(repo)
    if _COMMIT.fullmatch(revision) is None:
        raise StageError(f"{repo} descriptor commit could not be resolved", code="mdl")
    return {
        "id": model_id,
        "repo": repo,
        "revision": revision,
        "sha256": sha256_file(path),
    }


def _pins(revisions: Mapping[str, str] | None) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for repo, sha in (revisions or {}).items():
        if _COMMIT.fullmatch(sha) is None:
            raise StageError(f"{repo} revision must be a 40-hex commit", code="mdl")
        pinned[str(repo)] = sha
    return pinned


def _commit_oid(repo: str, infos: object) -> str:
    if not isinstance(infos, list) or len(infos) != 1:
        raise StageError(f"{repo} descriptor commit could not be resolved", code="mdl")
    last = getattr(infos[0], "last_commit", None)
    oid = getattr(last, "oid", None) if last is not None else None
    if not isinstance(oid, str) or _COMMIT.fullmatch(oid) is None:
        raise StageError(f"{repo} descriptor commit could not be resolved", code="mdl")
    return oid


def _descriptor_mirror(
    mirrors: Sequence[object],
    *,
    label: str,
    sole_repo: bool = False,
) -> Mapping[str, object]:
    """The ``hf`` mirror that holds ``voxprint-model.json``.

    An explicit ``ours`` flag wins. Otherwise the single mirror in the
    forge-owned namespace ``Mitroshenkov87`` holds the descriptor. No
    candidate, or more than one, is an error. ``sole_repo`` accepts one
    ``hf`` mirror when a publish is writing the only copy.
    """
    hf = [item for item in mirrors if isinstance(item, Mapping) and item.get("kind") == "hf" and item.get("ref")]
    if not hf:
        raise StageError(f"{label} has no hf mirror", code="mdl")
    marked = [item for item in hf if item.get("ours") is True]
    if len(marked) > 1:
        raise StageError(f"{label} marks more than one mirror as ours", code="mdl")
    if len(marked) == 1:
        return marked[0]
    owned = [item for item in hf if _forge_repo(str(item.get("ref")))]
    if len(owned) == 1:
        return owned[0]
    if len(owned) > 1:
        raise StageError(f"{label} has more than one forge-owned hf mirror", code="mdl")
    if sole_repo and len(hf) == 1:
        return hf[0]
    raise StageError(f"{label} has no mirror that holds voxprint-model.json", code="mdl")


def _forge_repo(ref: str) -> bool:
    owner, sep, name = ref.partition("/")
    return sep == "/" and owner == HF_AUTHOR and bool(name)


def _snapshot(api: HfApi, info: object) -> RemoteRepo:
    repo_id = str(getattr(info, "id"))
    revision = str(getattr(info, "sha", "") or "main")
    card = getattr(info, "card_data", None)
    licence = _card_text(card, "license")
    if not licence:
        raise StageError(f"{repo_id} has no licence tag", code="mdl")
    files: list[RemoteFile] = []
    for item in api.list_repo_tree(repo_id, recursive=True, expand=True, revision=revision):
        if getattr(item, "size", None) is None:
            continue
        path = str(getattr(item, "path"))
        if path == MANIFEST_NAME:
            continue
        size = int(getattr(item, "size"))
        files.append(RemoteFile(path, size, _file_sha(api, repo_id, revision, item, size)))
    if not files:
        raise StageError(f"{repo_id} has no files", code="mdl")
    task = str(getattr(info, "pipeline_tag", "") or "")
    return RemoteRepo(
        repo_id=repo_id,
        revision=revision,
        license=licence,
        files=tuple(files),
        base_model=_card_text(card, "base_model"),
        task=task,
        flags=("least-censored",) if "abliterat" in repo_id.lower() else (),
    )


def _file_sha(api: HfApi, repo_id: str, revision: str, item: object, size: int) -> str:
    del api
    lfs = getattr(item, "lfs", None)
    raw = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if isinstance(raw, str) and _SHA.fullmatch(raw):
        return raw
    path = str(getattr(item, "path"))
    if size > _INLINE_MAX:
        raise StageError(f"{repo_id} {path} has no sha256", code="mdl")
    local = hf_hub_download(repo_id=repo_id, filename=path, revision=revision)
    return sha256_file(Path(local))


def _card_text(card: object, key: str) -> str:
    if card is None:
        return ""
    value = card.get(key, "") if isinstance(card, dict) else getattr(card, key, "")
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _infer_from_slug(slug: str) -> tuple[Engine, str, float, str]:
    lower = slug.lower()
    if "svdquant" in lower or "int4" in lower or "nunchaku" in lower:
        task = "image-to-image" if "edit" in lower else "text-to-image"
        return "nunchaku", NUNCHAKU_VERSION_REQ, CUDA_CC, task
    if "wan" in lower or "t2v" in lower:
        return "llama.cpp", WAN_LLAMA_TAG, LLAMA_CC, "text-to-video"
    if "gguf" in lower:
        return "llama.cpp", LLAMA_CPP_TAG, LLAMA_CC, "text-generation"
    task = infer_task(slug, ())
    return "diffusers", DIFFUSERS_VERSION_REQ, CUDA_CC, task


def _infer_runtime(repo_id: str, paths: Sequence[str]) -> tuple[str, str, float]:
    engine, version_req, cc, _task = _infer_from_slug(repo_id)
    if any(path.endswith(".gguf") for path in paths) and engine == "diffusers":
        return "llama.cpp", LLAMA_CPP_TAG, LLAMA_CC
    return engine, version_req, cc


def _vram_pair(repo: RemoteRepo, paths: Sequence[str]) -> tuple[float, float]:
    if repo.min_vram_gb is not None and repo.recommended_vram_gb is not None:
        return repo.min_vram_gb, repo.recommended_vram_gb
    found: list[float] = []
    for path in paths:
        lower = path.lower()
        for token, gb in (("24gb", 24.0), ("16gb", 16.0), ("12gb", 12.0), ("8gb", 8.0)):
            if token in lower:
                found.append(gb)
    if repo.min_vram_gb is not None:
        found.append(repo.min_vram_gb)
    if repo.recommended_vram_gb is not None:
        found.append(repo.recommended_vram_gb)
    if not found:
        return 16.0, 24.0
    return min(found), max(found)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _dump(document: object) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _exact_keys(value: dict[str, object], keys: tuple[str, ...], label: str) -> None:
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        extra = sorted(set(value) - set(keys))
        raise StageError(f"{label} keys mismatch missing={missing} extra={extra}", code="mdl")


def _keys(value: dict[str, object], required: tuple[str, ...], optional: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(value))
    extra = sorted(set(value) - set(required) - set(optional))
    if missing or extra:
        raise StageError(f"{label} keys mismatch missing={missing} extra={extra}", code="mdl")


def _policy(value: object) -> None:
    if value not in _POLICIES:
        raise StageError(f"mirror_policy {value!r} must be race or strict-priority", code="mdl")


def _file_groups(value: object) -> None:
    if not isinstance(value, list):
        raise StageError("file_groups must be a list", code="mdl")
    for group in value:
        if not isinstance(group, dict):
            raise StageError("file group is not an object", code="mdl")
        _keys(group, ("paths",), _GROUP_OPTIONAL, "file group")
        _str_list(group["paths"], "file group paths", minimum=1)
        if "mirror_policy" in group:
            _policy(group["mirror_policy"])
        if "sources" in group:
            _mirrors(group["sources"])


def _prebuilt(value: object) -> None:
    if not isinstance(value, list):
        raise StageError("prebuilt must be a list", code="mdl")
    for item in value:
        if not isinstance(item, dict):
            raise StageError("prebuilt entry is not an object", code="mdl")
        if "built_for" in item:
            _built_for(item["built_for"])


def _built_for(value: object) -> None:
    if not isinstance(value, dict):
        raise StageError("built_for must be an object", code="mdl")
    if "onnxruntime" in value and not isinstance(value["onnxruntime"], str):
        raise StageError("built_for.onnxruntime must be a string", code="mdl")
    opset = value.get("opset")
    if opset is not None and (isinstance(opset, bool) or not isinstance(opset, int)):
        raise StageError("built_for.opset must be an integer", code="mdl")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise StageError(f"{label} must be a non-empty string", code="mdl")


def _ident(value: object) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise StageError("id must be a lowercase model id", code="mdl")


def _licence(value: object) -> None:
    if not isinstance(value, str) or _SPDX.fullmatch(value) is None:
        raise StageError("license must be a lowercase SPDX id", code="mdl")


def _card(value: object) -> None:
    if not isinstance(value, str) or _CARD.fullmatch(value) is None:
        raise StageError("card_url must be empty or an https URL", code="mdl")


def _str_list(value: object, label: str, *, minimum: int) -> None:
    if not isinstance(value, list) or len(value) < minimum:
        raise StageError(f"{label} must have at least {minimum} strings", code="mdl")
    if any(not isinstance(item, str) or not item for item in value):
        raise StageError(f"{label} entries must be non-empty strings", code="mdl")
    if len(set(value)) != len(value):
        raise StageError(f"{label} entries must be unique", code="mdl")


def _files(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise StageError("files must list at least one published file", code="mdl")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise StageError("file entry is not an object", code="mdl")
        _keys(item, _FILE_KEYS, _FILE_OPTIONAL, "file")
        if "mirror_policy" in item:
            _policy(item["mirror_policy"])
        if "sources" in item:
            _mirrors(item["sources"])
        if "built_for" in item:
            _built_for(item["built_for"])
        path = item["path"]
        if not isinstance(path, str) or not _repo_path(path):
            raise StageError(f"path {path!r} is not relative to the repo root", code="mdl")
        if path in seen:
            raise StageError(f"duplicate path {path}", code="mdl")
        seen.add(path)
        size = item["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StageError("file size must be a non-negative integer", code="mdl")
        digest = item["sha256"]
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise StageError("file sha256 must be 64 lowercase hex characters", code="mdl")


def _repo_path(path: str) -> bool:
    if path == MANIFEST_NAME or path.startswith(("/", "\\")) or "\\" in path:
        return False
    parts = path.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _mirrors(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise StageError("mirrors must be a non-empty list", code="mdl")
    for item in value:
        if not isinstance(item, dict):
            raise StageError("mirror is not an object", code="mdl")
        _mirror_keys(item)
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in _KINDS:
            raise StageError(f"mirror kind {kind!r} is not supported", code="mdl")
        _text(item["ref"], "mirror ref")
        if not isinstance(item["revision"], str):
            raise StageError("mirror revision must be a string", code="mdl")
        if "ours" in item and not isinstance(item["ours"], bool):
            raise StageError("mirror ours must be a boolean", code="mdl")


def _mirror_keys(item: dict[str, object]) -> None:
    allowed = set(_MIRROR_KEYS)
    if "ours" in item:
        allowed.add("ours")
    if set(item) != allowed:
        missing = sorted(set(_MIRROR_KEYS) - set(item))
        extra = sorted(set(item) - allowed)
        raise StageError(f"mirror keys mismatch missing={missing} extra={extra}", code="mdl")


def _tier(value: object) -> None:
    if not isinstance(value, dict):
        raise StageError("tier is not an object", code="mdl")
    _exact_keys(value, _TIER_KEYS, "tier")
    minimum = _number(value["min_vram_gib"], "min_vram_gib")
    recommended = _number(value["recommended_vram_gib"], "recommended_vram_gib")
    capability = _number(value["gpu_min_cc"], "gpu_min_cc")
    if recommended < minimum:
        raise StageError("recommended_vram_gib is below min_vram_gib", code="mdl")
    if capability < 0:
        raise StageError("gpu_min_cc must be >= 0", code="mdl")


def _runtime(value: object) -> None:
    if not isinstance(value, dict):
        raise StageError("runtime is not an object", code="mdl")
    _exact_keys(value, _RUNTIME_KEYS, "runtime")
    engine = value["engine"]
    if not isinstance(engine, str):
        raise StageError("runtime engine must be a string", code="mdl")
    _as_engine(engine)
    _text(value["version_req"], "version_req")


def _as_engine(engine: str) -> Engine:
    if engine == "llama.cpp":
        return "llama.cpp"
    if engine == "diffusers":
        return "diffusers"
    if engine == "nunchaku":
        return "nunchaku"
    if engine == "onnxruntime":
        return "onnxruntime"
    raise StageError(f"unknown engine {engine}", code="mdl")


def _build(value: object) -> None:
    if not isinstance(value, dict):
        raise StageError("build is not an object", code="mdl")
    _exact_keys(value, _BUILD_KEYS, "build")
    _text(value["forge_version"], "forge_version")
    if not isinstance(value["tag"], str):
        raise StageError("build tag must be a string", code="mdl")
    date = value["date"]
    if not isinstance(date, str) or _DATE.fullmatch(date) is None:
        raise StageError("build date must be YYYY-MM-DD", code="mdl")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageError(f"{label} must be a number", code="mdl")
    return float(value)
