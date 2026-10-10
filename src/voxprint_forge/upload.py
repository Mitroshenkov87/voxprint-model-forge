"""Private Hugging Face and Kaggle uploads. A stage is done only after sha256 and size match."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download

from voxprint_forge.errors import StageError, ToolMissingError, UsageError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.scrub import scrub

try:
    import hf_transfer as _hf_transfer
except ImportError:
    _hf_transfer = None

DELAYS = (2, 4, 8, 16)
FRAMEWORK = "gguf"
_HF_GIT_SHA_LIMIT = 64 * 1024 * 1024
_TARGET_NAMES = ("hf", "kaggle", "drive")


def selected_targets() -> tuple[str, ...]:
    """Upload destinations from FORGE_UPLOAD_TARGETS. Default is Hugging Face, Kaggle, then Drive."""
    raw = os.environ.get("FORGE_UPLOAD_TARGETS", "hf,kaggle,drive")
    names = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    unknown = [name for name in names if name not in _TARGET_NAMES]
    if not names or unknown:
        raise UsageError("FORGE_UPLOAD_TARGETS must list hf, kaggle, and/or drive", code="usage")
    return names


@dataclass(frozen=True)
class Artifact:
    """One file to upload: path, name, sha256, size, tier, quant, and kind."""
    path: Path
    name: str
    sha256: str
    size: int
    tier: str
    quant: str
    kind: str


class Remote(Protocol):
    """Hub and Kaggle operations the upload stage needs."""
    def check_hf(self) -> str:
        """Return the signed-in Hugging Face user name."""
        ...

    def check_kaggle(self) -> str:
        """Return the signed-in Kaggle user name."""
        ...

    def hf_private_repo(self, repo_id: str) -> str:
        """Create or reuse a private Hub repo and return its URL."""
        ...

    def hf_upload(self, repo_id: str, path: Path, name: str) -> None:
        """Upload one file into the Hub repo."""
        ...

    def hf_stat(self, repo_id: str, name: str) -> tuple[str | None, int]:
        """Return the Hub file's sha256 and size."""
        ...

    def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
        """Create or reuse a private Kaggle model and return its URL."""
        ...

    def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
        """Upload one Kaggle variation and return its handle."""
        ...

    def kaggle_stat(self, handle: str, name: str) -> tuple[str | None, int]:
        """Return the Kaggle file's sha256 and size."""
        ...


def hf_user() -> str:
    """Hugging Face user from ``HF_USER``."""
    user = os.environ.get("HF_USER", "").strip()
    if not user:
        raise UsageError("HF_USER is not set", code="hf_user")
    return user


def kaggle_user() -> str:
    """Kaggle user from ``KAGGLE_USER``."""
    return os.environ.get("KAGGLE_USER", "mitroshenkov87").strip() or "mitroshenkov87"


def repo_slug(model_id: str) -> str:
    """Private repo slug for a model id."""
    return f"voxprint-{model_id}"


def kaggle_model_slug(model_id: str) -> str:
    """Kaggle slugs allow [a-z0-9-]. Hugging Face keeps the dotted repo name."""
    return re.sub(r"[^a-z0-9]+", "-", repo_slug(model_id).lower()).strip("-")


def variation_slug(tier: str, quant: str) -> str:
    """Kaggle variation slug for a tier and quant."""
    return f"{tier}-{quant.lower().replace('_', '-')}"


def hf_file_url(repo_id: str, name: str) -> str:
    """Hub URL of one uploaded file."""
    return f"https://huggingface.co/{repo_id}/blob/main/{name}"


def kaggle_variation_url(owner: str, slug: str, variation: str) -> str:
    """Kaggle URL of one variation."""
    return f"https://www.kaggle.com/models/{owner}/{slug}/{FRAMEWORK}/{variation}"


def model_metadata(owner: str, slug: str, title: str, description: str) -> dict[str, object]:
    """Private Kaggle model metadata. ``isPrivate`` stays true."""
    return {
        "ownerSlug": owner,
        "title": title,
        "slug": slug,
        "subtitle": "Voxprint AI Plotweaver private GGUF",
        "isPrivate": True,
        "description": description,
        "publishTime": "",
        "provenanceSources": "",
        "licenseName": "Apache 2.0",
    }


def variation_metadata(owner: str, slug: str, variation: str, overview: str) -> dict[str, object]:
    """Private Kaggle variation metadata."""
    return {
        "ownerSlug": owner,
        "modelSlug": slug,
        "instanceSlug": variation,
        "framework": FRAMEWORK,
        "overview": overview,
        "usage": overview,
        "licenseName": "Apache 2.0",
        "fineTunable": False,
        "trainingData": [],
        "modelInstanceType": "Unspecified",
        "baseModelInstance": "",
        "externalBaseModelUrl": "",
    }


def assert_private_metadata(payload: dict[str, object]) -> None:
    """Raise unless the metadata says the model is private."""
    if payload.get("isPrivate") is not True:
        raise StageError("refusing a model that is not private", code="upload_public")


def parse_kaggle_version(text: str) -> tuple[int, int]:
    """Parse a ``kaggle --version`` string into a major, minor pair."""
    match = re.search(r"(\d+)\.(\d+)", text)
    if match is None:
        raise ToolMissingError(f"cannot read kaggle version from {text!r}", code="kaggle_version")
    return int(match.group(1)), int(match.group(2))


def require_kaggle_version(text: str) -> tuple[int, int]:
    """Parse the version and raise when it is older than the pin."""
    major, minor = parse_kaggle_version(text)
    if (major, minor) < (2, 2):
        raise ToolMissingError(f"kaggle CLI {major}.{minor} is older than 2.2", code="kaggle_version")
    return major, minor


def _matched(remote_sha: str | None, remote_size: int, local_sha: str, local_size: int) -> bool:
    return remote_size == local_size and remote_sha == local_sha


def upload_verified(
    action: Callable[[], None],
    stat: Callable[[], tuple[str | None, int]],
    *,
    local_sha: str,
    local_size: int,
    attempts: int,
    sleep: Callable[[float], None],
) -> None:
    """Upload, then retry until the remote sha256 and size match."""
    last = "remote sha256/size did not match"
    for attempt in range(attempts):
        action()
        remote_sha, remote_size = stat()
        if _matched(remote_sha, remote_size, local_sha, local_size):
            return
        last = f"remote sha256={remote_sha} size={remote_size} != {local_sha} {local_size}"
        if attempt + 1 < attempts:
            sleep(DELAYS[min(attempt, len(DELAYS) - 1)])
    raise StageError(scrub(last), code="upload_mismatch")


def publish_private(
    remote: Remote,
    artifacts: Sequence[Artifact],
    *,
    model_id: str,
    title: str,
    description: str,
    attempts: int,
    sleep: Callable[[float], None],
    work: Path,
    targets: tuple[str, ...] | None = None,
    on_verified: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Upload every artifact to the selected private hubs. Default is Hugging Face and Kaggle."""
    if not artifacts:
        raise StageError(f"{model_id} has nothing to upload", code="upload_empty")
    chosen = ("hf", "kaggle") if targets is None else tuple(targets)
    want_hf = "hf" in chosen
    want_kaggle = "kaggle" in chosen
    if not want_hf and not want_kaggle:
        raise UsageError("publish_private needs hf, kaggle, or both", code="usage")
    user = hf_user() if want_hf else ""
    owner = kaggle_user() if want_kaggle else ""
    hf_slug = repo_slug(model_id)
    slug = kaggle_model_slug(model_id)
    repo_id = f"{user}/{hf_slug}" if want_hf else ""
    if want_hf:
        remote.hf_private_repo(repo_id)
    if want_kaggle:
        model_dir = work / "kaggle-model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)
        meta = model_metadata(owner, slug, title, description)
        assert_private_metadata(meta)
        (model_dir / "model-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        remote.kaggle_private_model(owner, slug, model_dir)

    records: list[dict[str, object]] = []
    for artifact in artifacts:
        hf_url = ""
        if want_hf:
            upload_verified(
                lambda artifact=artifact: remote.hf_upload(repo_id, artifact.path, artifact.name),
                lambda artifact=artifact: remote.hf_stat(repo_id, artifact.name),
                local_sha=artifact.sha256,
                local_size=artifact.size,
                attempts=attempts,
                sleep=sleep,
            )
            hf_url = hf_file_url(repo_id, artifact.name)
        records.append(
            {
                "name": artifact.name,
                "sha256": artifact.sha256,
                "size": artifact.size,
                "tier": artifact.tier,
                "quant": artifact.quant,
                "kind": artifact.kind,
                "min_vram_gb": None,
                "sources": {
                    "hf": hf_url,
                    "kaggle": "",
                    "torrent": "placeholder",
                },
            }
        )
    if want_hf and on_verified is not None:
        on_verified("hf")

    if not want_kaggle:
        return records
    by_tier: dict[str, list[Artifact]] = {}
    shared = [item for item in artifacts if item.tier in ("", "shared")]
    for artifact in artifacts:
        if artifact.kind == "gguf":
            by_tier.setdefault(artifact.tier, []).append(artifact)
    for tier, group in by_tier.items():
        gguf = group[0]
        variation = variation_slug(tier, gguf.quant)
        folder = work / "kaggle" / variation
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        names = []
        for item in [gguf, *shared, *_sidecars(artifacts, gguf)]:
            target = folder / item.name
            if not target.exists():
                shutil.copy2(item.path, target)
            names.append(item.name)
        overview = description[:4000]
        instance = variation_metadata(owner, slug, variation, overview)
        (folder / "model-instance-metadata.json").write_text(
            json.dumps(instance, indent=2) + "\n",
            encoding="utf-8",
        )
        bundled = [item for item in artifacts if item.name in names]
        page = kaggle_variation_url(owner, slug, variation)

        def _send(folder: Path = folder, variation: str = variation) -> str:
            return remote.kaggle_upload_variation(owner, slug, variation, folder)

        _verify_bundle(
            _send,
            bundled,
            remote,
            attempts=attempts,
            sleep=sleep,
        )
        for name in names:
            for record in records:
                if record["name"] == name:
                    sources = record["sources"]
                    if isinstance(sources, dict):
                        sources["kaggle"] = page
    for record in records:
        sources = record["sources"]
        if isinstance(sources, dict) and not sources.get("kaggle"):
            raise StageError(f"{record['name']} has no verified Kaggle URL", code="upload_kaggle")
    if on_verified is not None:
        on_verified("kaggle")
    return records


def _verify_bundle(
    send: Callable[[], str],
    bundled: Sequence[Artifact],
    remote: Remote,
    *,
    attempts: int,
    sleep: Callable[[float], None],
) -> None:
    last = "remote sha256/size did not match"
    for attempt in range(attempts):
        handle = send()
        mismatched = False
        for artifact in bundled:
            remote_sha, remote_size = remote.kaggle_stat(handle, artifact.name)
            if not _matched(remote_sha, remote_size, artifact.sha256, artifact.size):
                mismatched = True
                last = (
                    f"{artifact.name} remote sha256={remote_sha} size={remote_size} "
                    f"!= {artifact.sha256} {artifact.size}"
                )
                break
        if not mismatched:
            return
        if attempt + 1 < attempts:
            sleep(DELAYS[min(attempt, len(DELAYS) - 1)])
    raise StageError(scrub(last), code="upload_mismatch")


def _sidecars(artifacts: Sequence[Artifact], gguf: Artifact) -> list[Artifact]:
    stem = gguf.name + ".sha256"
    return [item for item in artifacts if item.name == stem]


def enable_fast_transfer() -> None:
    """Turn on ``hf_transfer`` when the package is installed."""
    if _hf_transfer is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def kaggle_argv() -> list[str]:
    """Argv prefix for the kaggle CLI, preferring the interpreter's venv."""
    found = shutil.which("kaggle")
    if found:
        return [found]
    bindir = Path(sys.executable).resolve().parent
    for name in ("kaggle", "kaggle.exe"):
        candidate = bindir / name
        if candidate.is_file():
            return [str(candidate)]
    uvx = shutil.which("uvx")
    if uvx is None:
        raise ToolMissingError("kaggle CLI is not installed and uvx is missing", code="tool_missing")
    return [uvx, "kaggle"]


class HubRemote:
    """Live HF + Kaggle client. Tokens are read from the environment and never written."""

    def __init__(self, hf_token: str | None, kaggle_token: str | None) -> None:
        self._hf_token = hf_token
        self._kaggle_token = kaggle_token
        enable_fast_transfer()

    def check_hf(self) -> str:
        """Return the signed-in Hugging Face user name."""
        if not self._hf_token:
            raise UsageError("HF_TOKEN is not set", code="hf_token")
        try:
            who = HfApi(token=self._hf_token).whoami()
        except Exception as exc:
            raise UsageError(scrub(f"Hugging Face login failed: {exc}"), code="hf_login") from None
        name = str(who.get("name", "")) if isinstance(who, dict) else ""
        if not name:
            raise UsageError("Hugging Face login failed", code="hf_login")
        return name

    def check_kaggle(self) -> str:
        """Return the signed-in Kaggle user name."""
        if not self._kaggle_token:
            raise UsageError("KAGGLE_API_TOKEN is not set", code="kaggle_token")
        version = _kaggle(self._kaggle_token, ["--version"])
        require_kaggle_version(version.stdout + version.stderr)
        owner = kaggle_user()
        listed = _kaggle(
            self._kaggle_token,
            ["models", "list", "--owner", owner, "--page-size", "1"],
        )
        if listed.returncode != 0:
            raise UsageError(scrub(f"Kaggle login failed: {listed.stderr}"), code="kaggle_login")
        return owner

    def hf_private_repo(self, repo_id: str) -> str:
        """Create or reuse a private Hub repo and return its URL."""
        if not self._hf_token:
            raise UsageError("HF_TOKEN is not set", code="hf_token")
        api = HfApi(token=self._hf_token)
        try:
            api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
            info = api.repo_info(repo_id)
            if not info.private:
                api.update_repo_visibility(repo_id, private=True)
                info = api.repo_info(repo_id)
        except Exception as exc:
            raise StageError(scrub(str(exc)), code="upload_hf") from None
        if not info.private:
            raise StageError(f"{repo_id} is public; refusing to upload", code="upload_public")
        return f"https://huggingface.co/{repo_id}"

    def hf_upload(self, repo_id: str, path: Path, name: str) -> None:
        """Upload one file into the Hub repo."""
        api = HfApi(token=self._hf_token)
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo_id,
                commit_message=f"Add {name}",
            )
        except Exception as exc:
            raise StageError(scrub(str(exc)), code="upload_hf") from None

    def hf_stat(self, repo_id: str, name: str) -> tuple[str | None, int]:
        """Return the Hub file's sha256 and size."""
        api = HfApi(token=self._hf_token)
        try:
            infos = api.get_paths_info(repo_id, paths=[name])
        except Exception as exc:
            raise StageError(scrub(str(exc)), code="upload_hf") from None
        if not infos:
            return None, 0
        item = infos[0]
        lfs = getattr(item, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        size = int(getattr(item, "size", 0) or 0)
        if not sha and 0 < size <= _HF_GIT_SHA_LIMIT:
            sha = self._hash_small_file(repo_id, name)
        return (str(sha) if sha else None), size

    def _hash_small_file(self, repo_id: str, name: str) -> str:
        """Git blobs have no LFS sha256. Hash the downloaded bytes instead."""
        with tempfile.TemporaryDirectory(prefix="forge-hfstat-") as tmp:
            try:
                local = hf_hub_download(repo_id, name, token=self._hf_token, local_dir=tmp)
            except Exception as exc:
                raise StageError(scrub(str(exc)), code="upload_hf") from None
            return sha256_file(Path(local))

    def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
        """Create or reuse a private Kaggle model and return its URL."""
        meta_path = folder / "model-metadata.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise StageError("model metadata is not an object", code="upload_kaggle")
        assert_private_metadata(payload)
        created = _kaggle(self._require_kaggle(), ["models", "create", "-p", str(folder)])
        text = created.stdout + created.stderr
        if created.returncode != 0 and "exist" not in text.lower():
            raise StageError(scrub(text), code="upload_kaggle")
        if created.returncode != 0:
            updated = _kaggle(self._require_kaggle(), ["models", "update", "-p", str(folder)])
            if updated.returncode != 0:
                raise StageError(scrub(updated.stderr or updated.stdout), code="upload_kaggle")
        self._assert_kaggle_private(owner, slug)
        return f"https://www.kaggle.com/models/{owner}/{slug}"

    def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
        """Upload one Kaggle variation and return its handle."""
        created = _kaggle(
            self._require_kaggle(),
            [
                "models",
                "variations",
                "create",
                "-p",
                str(folder),
                "-q",
                "-r",
                "skip",
                "--ignore-patterns",
                "model-instance-metadata.json",
            ],
        )
        text = created.stdout + created.stderr
        handle = f"{owner}/{slug}/{FRAMEWORK}/{variation}"
        if created.returncode != 0:
            versioned = _kaggle(
                self._require_kaggle(),
                [
                    "models",
                    "variations",
                    "versions",
                    "create",
                    handle,
                    "-p",
                    str(folder),
                    "-n",
                    "voxprint forge upload",
                    "-q",
                    "-r",
                    "skip",
                    "--ignore-patterns",
                    "model-instance-metadata.json",
                ],
            )
            if versioned.returncode != 0:
                raise StageError(scrub(versioned.stderr or text), code="upload_kaggle")
        return handle

    def kaggle_stat(self, handle: str, name: str) -> tuple[str | None, int]:
        """Return the Kaggle file's sha256 and size."""
        listed = _kaggle(
            self._require_kaggle(),
            ["models", "variations", "files", handle, "-v", "--page-size", "100"],
        )
        if listed.returncode != 0:
            raise StageError(scrub(listed.stderr or listed.stdout), code="upload_kaggle")
        size = _csv_size(listed.stdout, name)
        digest = _kaggle_sha(self._require_kaggle(), handle, name)
        return digest, size

    def _require_kaggle(self) -> str:
        if not self._kaggle_token:
            raise UsageError("KAGGLE_API_TOKEN is not set", code="kaggle_token")
        return self._kaggle_token

    def _assert_kaggle_private(self, owner: str, slug: str) -> None:
        with tempfile.TemporaryDirectory(prefix="forge-kaggle-meta-") as tmp:
            fetched = _kaggle(self._require_kaggle(), ["models", "get", f"{owner}/{slug}", "-p", tmp])
            if fetched.returncode != 0:
                raise StageError(scrub(fetched.stderr or fetched.stdout), code="upload_kaggle")
            payload = json.loads((Path(tmp) / "model-metadata.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("isPrivate") is not True:
            raise StageError(f"{owner}/{slug} is not private; refusing to upload", code="upload_public")


def _kaggle(token: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    config = Path(tempfile.mkdtemp(prefix="forge-kaggle-"))
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["KAGGLE_CONFIG_DIR"] = str(config)
    try:
        return subprocess.run(
            [*kaggle_argv(), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        shutil.rmtree(config, ignore_errors=True)


def _csv_size(text: str, name: str) -> int:
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.lower().startswith("name,"):
            start = index
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    for row in reader:
        if row.get("name") == name:
            return int(row.get("total_bytes") or 0)
    return 0


def latest_version_number(text: str) -> int:
    """Highest versionNumber in a `kaggle models variations versions list -v` CSV."""
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if "versionnumber" in line.lower():
            start = index
            break
    numbers: list[int] = []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    for row in reader:
        raw = ""
        for key, value in row.items():
            if key and key.lower() == "versionnumber" and value:
                raw = value.strip()
                break
        if raw.isdigit():
            numbers.append(int(raw))
    return max(numbers) if numbers else 1


def _kaggle_sha(token: str, handle: str, name: str) -> str | None:
    listed = _kaggle(token, ["models", "variations", "versions", "list", handle, "-v"])
    version = latest_version_number(listed.stdout) if listed.returncode == 0 else 1
    with tempfile.TemporaryDirectory(prefix="forge-kaggle-hash-") as tmp:
        downloaded = _kaggle(
            token,
            [
                "models",
                "variations",
                "versions",
                "download",
                f"{handle}/{version}",
                "-p",
                tmp,
                "-q",
            ],
        )
        if downloaded.returncode != 0:
            return None
        matches = list(Path(tmp).rglob(name))
        if not matches:
            return None
        return sha256_file(matches[0])


def verify_logins(remote: Remote | None = None) -> dict[str, str]:
    """Check Hub and Kaggle logins and return both user names."""
    client = remote if remote is not None else HubRemote(
        os.environ.get("HF_TOKEN"),
        os.environ.get("KAGGLE_API_TOKEN"),
    )
    return {"hf": client.check_hf(), "kaggle": client.check_kaggle()}
