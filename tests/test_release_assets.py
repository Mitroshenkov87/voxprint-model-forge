"""Release asset collection and GitHub publish commands."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from release_assets import (  # noqa: E402
    ASSET_NAMES,
    CmdResult,
    choose_run,
    collect_assets,
    gh_release_commands,
    is_prerelease,
    publish_release,
    write_sha256sums,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _run(run_id: int, status: str, conclusion: str | None, created: str) -> dict[str, object]:
    return {
        "databaseId": run_id,
        "number": 180 + run_id,
        "status": status,
        "conclusion": conclusion,
        "createdAt": created,
        "event": "push",
    }


def test_rc_tag_is_a_prerelease() -> None:
    assert is_prerelease("refs/tags/v0.1.0-rc.7")
    assert is_prerelease("v0.1.0-rc.1")
    assert not is_prerelease("v0.1.0")
    assert not is_prerelease("refs/tags/v0.1.0")


def test_choose_run_waits_downloads_fails_or_dispatches() -> None:
    success = _run(1, "completed", "success", "2026-01-01T00:00:00Z")
    newer = _run(2, "completed", "success", "2026-03-01T00:00:00Z")
    active = _run(3, "in_progress", None, "2026-04-01T00:00:00Z")
    failed = _run(4, "completed", "failure", "2026-05-01T00:00:00Z")
    assert choose_run([success, active]) == ("wait", None)
    assert choose_run([success, newer]) == ("download", 2)
    assert choose_run([failed]) == ("fail", None)
    assert choose_run([]) == ("dispatch", None)


def test_sha256sums_are_sorted_lf_lines(tmp_path: Path) -> None:
    paths = []
    for name in ("b.txt", "a.txt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)
    dest = tmp_path / "SHA256SUMS.txt"
    write_sha256sums(paths, dest)
    raw = dest.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    lines = raw.decode().splitlines()
    assert [line.split("  ")[1] for line in lines] == ["a.txt", "b.txt"]
    assert lines[0].startswith(hashlib.sha256(b"a.txt").hexdigest() + "  ")


def test_release_commands_create_prerelease_or_update_latest() -> None:
    files = [Path("VoxprintModelForgeSetup.exe"), Path("SHA256SUMS.txt")]
    created = gh_release_commands("v0.1.0-rc.7", files, sha="abc", exists=False)[0]
    assert created[:3] == ["gh", "release", "create"]
    assert "--prerelease=true" in created
    assert "--verify-tag" in created
    assert "--target" in created
    assert "--latest" not in created
    updated = gh_release_commands("v0.1.0", files, sha="abc", exists=True)
    assert updated[0][-1] == "--clobber"
    assert "--prerelease=false" in updated[1]
    assert updated[1][-1] == "--latest"


def _assets(dest: Path) -> None:
    folder = dest / "artifact"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        (folder / name).write_bytes(name.encode())


def test_collect_downloads_a_finished_run(tmp_path: Path) -> None:
    runs = [_run(9, "completed", "success", "2026-10-26T00:00:00Z")]
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> CmdResult:
        calls.append(argv)
        if argv[1:3] == ["run", "list"]:
            return CmdResult(0, json.dumps(runs))
        if argv[1:3] == ["run", "download"]:
            _assets(Path(argv[argv.index("--dir") + 1]))
            return CmdResult(0)
        return CmdResult(1, "", "unexpected")

    collected = collect_assets(tmp_path, ref="v0.1.0", sha="abc", runner=runner, grace_s=120, poll_s=20, timeout_s=500)
    assert collected.sums.is_file()
    assert b"\r" not in collected.sums.read_bytes()
    assert collected.run_number == 189
    assert not any(argv[1:3] == ["workflow", "run"] for argv in calls)
    assert any(argv[1:3] == ["run", "download"] and "9" in argv for argv in calls)


def test_collect_reports_the_installer_run_number_not_the_run_id(tmp_path: Path) -> None:
    """The release build is the installer counter (rc.8 = 183), never this workflow's own run number."""
    runs = [
        {**_run(9_000_000_001, "completed", "success", "2026-10-26T00:00:00Z"), "number": 183},
        {**_run(9_000_000_002, "completed", "success", "2026-10-27T00:00:00Z"), "number": 185},
    ]

    def runner(argv: list[str]) -> CmdResult:
        if argv[1:3] == ["run", "list"]:
            assert "number" in argv[argv.index("--json") + 1].split(",")
            return CmdResult(0, json.dumps(runs))
        if argv[1:3] == ["run", "download"]:
            _assets(Path(argv[argv.index("--dir") + 1]))
            return CmdResult(0)
        return CmdResult(1, "", "unexpected")

    assert collect_assets(tmp_path, ref="v0.1.0-rc.9", sha="abc", runner=runner).run_number == 185


def test_collect_refuses_a_run_without_a_number(tmp_path: Path) -> None:
    runs = [{**_run(9, "completed", "success", "2026-10-26T00:00:00Z"), "number": None}]

    def runner(argv: list[str]) -> CmdResult:
        return CmdResult(0, json.dumps(runs)) if argv[1:3] == ["run", "list"] else CmdResult(0)

    with pytest.raises(SystemExit, match="run number"):
        collect_assets(tmp_path, ref="v0.1.0", sha="abc", runner=runner, timeout_s=50)


def test_collect_starts_installer_once_when_no_run_appears(tmp_path: Path) -> None:
    runs: list[dict[str, object]] = []
    calls: list[list[str]] = []
    clock = Clock()

    def runner(argv: list[str]) -> CmdResult:
        calls.append(argv)
        if argv[1:3] == ["run", "list"]:
            return CmdResult(0, json.dumps(runs))
        if argv[1:3] == ["workflow", "run"]:
            runs.append(_run(4, "completed", "success", "2026-10-26T00:00:00Z"))
            return CmdResult(0)
        if argv[1:3] == ["run", "download"]:
            _assets(Path(argv[argv.index("--dir") + 1]))
            return CmdResult(0)
        return CmdResult(1, "", "unexpected")

    collect_assets(
        tmp_path,
        ref="v0.1.0-rc.7",
        sha="abc",
        runner=runner,
        sleep=clock.sleep,
        now=clock.now,
        grace_s=100,
        poll_s=25,
        timeout_s=1000,
    )
    dispatched = [argv for argv in calls if argv[1:3] == ["workflow", "run"]]
    assert dispatched == [["gh", "workflow", "run", "installer.yml", "--ref", "v0.1.0-rc.7"]]
    assert clock.t >= 100


def test_collect_fails_when_every_run_failed(tmp_path: Path) -> None:
    runs = [_run(3, "completed", "failure", "2026-10-26T00:00:00Z")]

    def runner(argv: list[str]) -> CmdResult:
        if argv[1:3] == ["run", "list"]:
            return CmdResult(0, json.dumps(runs))
        return CmdResult(0)

    with pytest.raises(SystemExit, match="failed"):
        collect_assets(tmp_path, ref="v0.1.0", sha="abc", runner=runner, timeout_s=50)


def test_publish_updates_an_existing_release_and_retries_a_lost_create(tmp_path: Path) -> None:
    _assets(tmp_path)
    write_sha256sums([tmp_path / "artifact" / name for name in ASSET_NAMES], tmp_path / "SHA256SUMS.txt")
    calls: list[list[str]] = []

    def existing(argv: list[str]) -> CmdResult:
        calls.append(argv)
        return CmdResult(0)

    publish_release(tmp_path, "v0.1.0", "abc", existing)
    assert calls[0][:3] == ["gh", "release", "view"]
    assert calls[1][1:3] == ["release", "upload"]
    assert calls[2][1:3] == ["release", "edit"]

    calls.clear()

    def race(argv: list[str]) -> CmdResult:
        calls.append(argv)
        if argv[1:3] == ["release", "view"] or argv[1:3] == ["release", "create"]:
            return CmdResult(1, "", "release already exists")
        return CmdResult(0)

    publish_release(tmp_path, "v0.1.0-rc.7", "abc", race)
    assert [argv[1:3] for argv in calls] == [
        ["release", "view"],
        ["release", "create"],
        ["release", "upload"],
        ["release", "edit"],
    ]
    assert "--prerelease=true" in calls[-1]
