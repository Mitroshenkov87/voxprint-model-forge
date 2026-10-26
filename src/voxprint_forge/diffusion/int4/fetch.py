"""Download the Wan experts before a rental run.

``--dry-run`` does not call this module. A licence that is not Apache-2.0
stops the download.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download

from voxprint_forge.diffusion.int4.pins import WAN_WEIGHT_REPO, WAN_WEIGHT_REVISION, skip_wan
from voxprint_forge.diffusion.sources import assert_apache
from voxprint_forge.errors import DownloadError

LicenceCheck = Callable[[str, str, str | None], str]


def _filled(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.safetensors"))


def fetch_wan(dest: Path, token: str | None) -> Path:
    """Snapshot Wan2.2-T2V-A14B at the pinned revision."""
    folder = dest / "wan2.2-t2v-a14b"
    if _filled(folder / "high_noise_model") and _filled(folder / "low_noise_model"):
        return folder
    try:
        snapshot_download(
            repo_id=WAN_WEIGHT_REPO,
            revision=WAN_WEIGHT_REVISION,
            local_dir=str(folder),
            token=token,
        )
    except Exception as exc:
        raise DownloadError(f"Wan2.2-T2V-A14B: {exc}", code="download") from None
    return folder


def fetch_weights(
    dest: Path,
    token: str | None,
    *,
    forge_out: Path | None = None,
    licence: LicenceCheck | None = None,
) -> dict[str, str]:
    """Download the Wan experts and return the environment paths."""
    check = licence or assert_apache
    wan = ""
    if not skip_wan():
        check(WAN_WEIGHT_REPO, WAN_WEIGHT_REVISION, token)
        wan = str(fetch_wan(dest, token))
    paths = {"FORGE_WAN_CKPT": wan}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "paths.json").write_text(json.dumps(paths, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    lines = [f"export {key}={shlex.quote(value)}" for key, value in paths.items()]
    (dest / "exports.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return paths
