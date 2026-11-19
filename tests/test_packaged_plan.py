"""The default plan ships with the package, so preflight works on an installed copy without a checkout."""

from __future__ import annotations

import glob
import tomllib
from pathlib import Path

import pytest

from voxprint_forge import cli
from voxprint_forge.preflight import converter_recognises_t_lite

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_plan_equals_the_repo_plan() -> None:
    assert cli.PACKAGED_PLAN.read_bytes() == (ROOT / "plans" / "plan.toml").read_bytes()


def test_packaged_plan_is_package_data() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = data["tool"]["setuptools"]["package-data"]["voxprint_forge"]
    package = ROOT / "src" / "voxprint_forge"
    rel = cli.PACKAGED_PLAN.relative_to(package).as_posix()
    # setuptools expands package-data patterns with recursive glob, where ** also matches no folder.
    # glob returns os.sep-separated paths (backslashes on Windows), so normalise them to POSIX form.
    matched = {
        Path(hit).as_posix() for pattern in patterns for hit in glob.glob(pattern, root_dir=package, recursive=True)
    }
    assert rel in matched


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    def fake(plan, root, forge_out):
        seen.update(plan=plan, root=root)
        return {"ok": True}

    real_load = cli.load_plan

    def fake_load(path):
        seen["plan_path"] = path
        return real_load(path)

    monkeypatch.setattr(cli, "run_preflight", fake)
    monkeypatch.setattr(cli, "load_plan", fake_load)
    monkeypatch.delenv("FORGE_ROOT", raising=False)
    return seen


def test_preflight_from_any_folder_uses_the_packaged_plan(tmp_path, monkeypatch, captured) -> None:
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "site-packages")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--json", "preflight"]) == 0
    assert captured["plan_path"] == cli.PACKAGED_PLAN
    assert captured["root"] == cli.PACKAGED_PLAN.parents[1]


def test_checkout_plan_wins_in_a_repo(monkeypatch, captured) -> None:
    monkeypatch.chdir(ROOT / "docs")
    assert cli.main(["--json", "preflight"]) == 0
    assert captured["plan_path"] == ROOT / "plans" / "plan.toml"
    assert captured["root"] == ROOT


def test_explicit_plan_wins(tmp_path, monkeypatch, captured) -> None:
    plan = tmp_path / "mine.toml"
    plan.write_bytes((ROOT / "plans" / "plan.toml").read_bytes())
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "site-packages")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--json", "--plan", str(plan), "preflight"]) == 0
    assert captured["plan_path"] == plan


def test_explicit_root_without_a_plan_still_reports_the_missing_plan(tmp_path, capsys, captured) -> None:
    assert cli.main(["--json", "--root", str(tmp_path), "preflight"]) != 0
    assert "missing plan" in capsys.readouterr().out


def test_tokenizer_check_reads_the_installed_registration_without_a_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FORGE_LLAMA_SRC", raising=False)
    assert converter_recognises_t_lite(tmp_path) is True
