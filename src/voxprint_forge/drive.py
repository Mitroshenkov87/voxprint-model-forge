"""Google Drive fallback when Hugging Face or Kaggle refuses an upload.

rclone reads the remote from the environment only. Tokens are not written to disk.
The folder is Voxprint/model-forge/<date>/<model>.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from voxprint_forge.errors import StageError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.scrub import scrub
from voxprint_forge.upload import Artifact


def _remote_name() -> str:
    return os.environ.get("FORGE_DRIVE_REMOTE", "forge").strip() or "forge"


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    remote = _remote_name().upper()
    prefix = f"RCLONE_CONFIG_{remote}_"
    if env.get(prefix + "TYPE"):
        return env
    token = env.get("FORGE_DRIVE_TOKEN", "").strip()
    account = env.get("FORGE_DRIVE_SERVICE_ACCOUNT", "").strip()
    if token:
        env[prefix + "TYPE"] = "drive"
        env[prefix + "SCOPE"] = "drive"
        env[prefix + "TOKEN"] = token
        return env
    if account:
        env[prefix + "TYPE"] = "drive"
        env[prefix + "SCOPE"] = "drive"
        env[prefix + "SERVICE_ACCOUNT_CREDENTIALS"] = account
        return env
    raise StageError(
        "Drive fallback needs RCLONE_CONFIG_* or FORGE_DRIVE_TOKEN or FORGE_DRIVE_SERVICE_ACCOUNT",
        code="drive_config",
    )


def _run(argv: list[str], env: dict[str, str]) -> str:
    binary = shutil.which("rclone")
    if binary is None:
        raise StageError("rclone is not on PATH", code="drive_missing")
    try:
        done = subprocess.run([binary, *argv], check=False, capture_output=True, text=True, env=env)
    except OSError as exc:
        raise StageError(f"rclone failed to start: {exc}", code="drive_missing") from exc
    if done.returncode != 0:
        raise StageError(scrub((done.stderr or done.stdout)[-1500:]), code="drive_upload")
    return done.stdout


def _remote_hashes(stdout: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            found[parts[-1]] = parts[0]
    return found


def upload_drive(
    artifacts: list[Artifact],
    *,
    model_id: str,
    date: str,
) -> list[dict[str, object]]:
    """Upload artifacts to the configured Drive remote and return records."""
    env = _child_env()
    remote = _remote_name()
    folder = f"Voxprint/model-forge/{date}/{model_id}"
    records: list[dict[str, object]] = []
    for artifact in artifacts:
        if not artifact.path.is_file():
            raise StageError(f"missing {artifact.path} for Drive upload", code="drive_upload")
        dest = f"{remote}:{folder}/{artifact.name}"
        _run(["copyto", str(artifact.path), dest], env)
        listing = _run(["sha256sum", f"{remote}:{folder}/{artifact.name}"], env)
        hashes = _remote_hashes(listing)
        remote_sha = hashes.get(artifact.name) or hashes.get(Path(artifact.name).name)
        local = sha256_file(artifact.path)
        if remote_sha != local:
            raise StageError(
                f"Drive sha256 {remote_sha} != {local} for {artifact.name}",
                code="upload_mismatch",
            )
        records.append(
            {
                "name": artifact.name,
                "sha256": local,
                "size": artifact.path.stat().st_size,
                "tier": artifact.tier,
                "target": "drive",
                "path": f"{folder}/{artifact.name}",
            }
        )
    return records
