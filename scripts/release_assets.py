"""Collect the tag's installer and guide PDFs, then publish a GitHub release.

``python scripts/release_assets.py collect DEST`` waits for the installer
workflow on this commit, or starts it once when none appears, and writes
``SHA256SUMS.txt``. ``publish DEST`` does that and uploads the setup, both
guide PDFs, and the sums file. A tag whose name contains ``-rc`` is a
prerelease. An existing release is updated in place. Lists of installer
runs are polled with a sleep, including when a finished run is joined by
one that is still in progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ASSET_NAMES = (
    "VoxprintModelForgeSetup.exe",
    "VOXPrint-Model-Forge-Guide-en.pdf",
)
IN_PROGRESS = frozenset({"queued", "in_progress", "waiting", "pending", "requested", "action_required"})
GRACE_S = 120.0
POLL_S = 20.0
TIMEOUT_S = 70 * 60.0


@dataclass(frozen=True)
class CmdResult:
    """One finished ``gh`` invocation."""

    code: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[list[str]], CmdResult]


def is_prerelease(tag: str) -> bool:
    """Return whether ``tag`` contains ``-rc`` in its last path segment."""
    return "-rc" in tag.rsplit("/", 1)[-1]


def choose_run(runs: Sequence[Mapping[str, object]]) -> tuple[str, int | None]:
    """Return ``wait``, ``download``, ``fail``, or ``dispatch``, plus a run id.

    An in-progress run waits even when another run already succeeded. Otherwise
    the newest successful run is downloaded. Any other non-empty list fails.
    An empty list asks for one new installer run.
    """
    if any(run.get("status") in IN_PROGRESS for run in runs):
        return "wait", None
    done = [run for run in runs if run.get("status") == "completed" and run.get("conclusion") == "success"]
    if done:
        done.sort(key=lambda run: str(run.get("createdAt") or ""), reverse=True)
        return "download", _run_id(done[0])
    if runs:
        return "fail", None
    return "dispatch", None


def write_sha256sums(paths: Sequence[Path], dest: Path) -> None:
    """Write sha256sum lines for ``paths``, sorted by file name, with LF endings."""
    lines: list[str] = []
    for path in sorted(paths, key=lambda item: item.name):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def find_assets(dest: Path) -> list[Path]:
    """Return the setup and both guide PDFs under ``dest``, one of each."""
    found: list[Path] = []
    for name in ASSET_NAMES:
        matches = [path for path in dest.rglob(name) if path.is_file()]
        if len(matches) != 1:
            raise SystemExit(f"{name}: found {len(matches)}")
        found.append(matches[0])
    return found


def gh_release_commands(tag: str, files: Sequence[Path], *, sha: str, exists: bool) -> list[list[str]]:
    """Return the ``gh release`` commands that publish ``files`` for ``tag``."""
    paths = [str(path) for path in files]
    pre = f"--prerelease={'true' if is_prerelease(tag) else 'false'}"
    if exists:
        edit = ["gh", "release", "edit", tag, "--title", tag, pre]
        if not is_prerelease(tag):
            edit.append("--latest")
        return [["gh", "release", "upload", tag, *paths, "--clobber"], edit]
    create = ["gh", "release", "create", tag, *paths, "--verify-tag", "--target", sha, "--title", tag, pre]
    if not is_prerelease(tag):
        create.append("--latest")
    return [create]


def collect_assets(
    dest: Path,
    *,
    ref: str,
    sha: str,
    runner: Runner,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    grace_s: float = GRACE_S,
    poll_s: float = POLL_S,
    timeout_s: float = TIMEOUT_S,
) -> Path:
    """Download installer artifacts for ``sha`` into ``dest`` and write the sums file.

    When no installer run exists, wait ``grace_s`` and then start
    ``installer.yml`` on ``ref`` once.
    """
    started = now()
    dispatched_at: float | None = None
    while True:
        if now() - started > timeout_s:
            raise SystemExit("installer.yml timed out")
        action, run_id = choose_run(_runs(runner, sha))
        if action == "wait":
            sleep(poll_s)
            continue
        if action == "download":
            action, run_id = choose_run(_runs(runner, sha))
            if action != "download" or run_id is None:
                sleep(poll_s)
                continue
            _download(runner, run_id, dest)
            assets = find_assets(dest)
            sums = dest / "SHA256SUMS.txt"
            write_sha256sums(assets, sums)
            return sums
        if action == "fail":
            raise SystemExit("installer.yml failed")
        if dispatched_at is None and now() - started < grace_s:
            sleep(poll_s)
            continue
        if dispatched_at is not None:
            if now() - dispatched_at < grace_s:
                sleep(poll_s)
                continue
            raise SystemExit("installer.yml did not start")
        action, run_id = choose_run(_runs(runner, sha))
        if action != "dispatch":
            continue
        result = runner(["gh", "workflow", "run", "installer.yml", "--ref", ref])
        if result.code != 0:
            raise SystemExit(result.stderr.strip() or "gh workflow run failed")
        dispatched_at = now()
        sleep(poll_s)


def publish_release(dest: Path, tag: str, sha: str, runner: Runner) -> None:
    """Upload the collected assets. Create the release, or update it when it exists."""
    sums = dest / "SHA256SUMS.txt"
    files = [*find_assets(dest), sums]
    if not sums.is_file():
        raise SystemExit("SHA256SUMS.txt is missing")
    if runner(["gh", "release", "view", tag]).code == 0:
        _run_all(runner, gh_release_commands(tag, files, sha=sha, exists=True))
        return
    created = gh_release_commands(tag, files, sha=sha, exists=False)[0]
    if runner(created).code == 0:
        return
    _run_all(runner, gh_release_commands(tag, files, sha=sha, exists=True))


def _run_id(run: Mapping[str, object]) -> int:
    value = run.get("databaseId")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise SystemExit("installer run has no databaseId")


def _runs(runner: Runner, sha: str) -> list[dict[str, object]]:
    result = runner(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            "installer.yml",
            "--commit",
            sha,
            "--json",
            "databaseId,status,conclusion,event,headSha,url,createdAt",
            "--limit",
            "50",
        ]
    )
    if result.code != 0:
        raise SystemExit(result.stderr.strip() or "gh run list failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise SystemExit("gh run list returned unexpected JSON")
    return [item for item in payload if isinstance(item, dict)]


def _download(runner: Runner, run_id: int, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    result = runner(["gh", "run", "download", str(run_id), "--dir", str(dest)])
    if result.code != 0:
        raise SystemExit(result.stderr.strip() or "gh run download failed")


def _run_all(runner: Runner, commands: Sequence[Sequence[str]]) -> None:
    for command in commands:
        result = runner(list(command))
        if result.code != 0:
            raise SystemExit(result.stderr.strip() or "gh release failed")


def _gh(argv: list[str]) -> CmdResult:
    completed = subprocess.run(argv, check=False, text=True, capture_output=True)
    return CmdResult(completed.returncode, completed.stdout, completed.stderr)


def main(argv: list[str] | None = None) -> int:
    """Collect or publish release assets for the current GitHub Actions tag."""
    parser = argparse.ArgumentParser(description="Publish the installer and guide PDFs for a tag.")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="download artifacts and write SHA256SUMS.txt")
    collect.add_argument("dest")
    publish = sub.add_parser("publish", help="collect assets and create or update the GitHub release")
    publish.add_argument("dest")
    args = parser.parse_args(argv)
    dest = Path(args.dest)
    sha = os.environ["GITHUB_SHA"]
    ref = os.environ["GITHUB_REF_NAME"]
    if args.command == "collect":
        collect_assets(dest, ref=ref, sha=sha, runner=_gh)
        return 0
    if args.command == "publish":
        collect_assets(dest, ref=ref, sha=sha, runner=_gh)
        publish_release(dest, ref, sha, _gh)
        return 0
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
