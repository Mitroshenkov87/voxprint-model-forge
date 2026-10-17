"""Public Hub create, licence gate, and ControlNet link."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import torch

from tests.complete_card import APACHE_LINE, complete_card
from voxprint_forge.diffusion.controlnet_link import swap_controlnet
from voxprint_forge.diffusion.stage_free import free_eval_stage
from voxprint_forge.errors import LicenceError, StageError
from voxprint_forge.upload import HubRemote, assert_public_upload

ROOT = Path(__file__).resolve().parents[1]


def test_fp8_controlnet_swap_links_the_transformer() -> None:
    seen: list[tuple[object, ...]] = []

    class Net:
        def to(self, device: object) -> Net:
            seen.append(("to", device))
            return self

    class Pipe:
        def __init__(self) -> None:
            self.transformer = object()
            self.controlnet: object = None

    pipe = Pipe()
    net = object()

    def link(candidate: object, transformer: object) -> Net:
        seen.append(("link", candidate, transformer))
        return Net()

    placed = swap_controlnet(pipe, net, "cuda", link)
    assert seen[0] == ("link", net, pipe.transformer)
    assert seen[1] == ("to", "cuda")
    assert pipe.controlnet is placed


def test_hf_repo_is_created_public(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Api:
        def __init__(self, token: str | None = None) -> None:
            calls.append(("token", token))

        def create_repo(self, repo_id: str, private: bool, exist_ok: bool, repo_type: str) -> None:
            calls.append(("create", repo_id, private, exist_ok, repo_type))

        def update_repo_visibility(self, repo_id: str, private: bool) -> None:
            calls.append(("visibility", repo_id, private))

    monkeypatch.setattr("voxprint_forge.upload.HfApi", Api)
    url = HubRemote("hf-token", None).hf_private_repo("Mitroshenkov87/voxprint-example")
    assert url == "https://huggingface.co/Mitroshenkov87/voxprint-example"
    created = [item for item in calls if item[0] == "create"]
    assert created == [("create", "Mitroshenkov87/voxprint-example", False, True, "model")]
    assert not any(item[0] == "visibility" for item in calls)


def test_public_upload_refuses_nc_and_an_incomplete_card() -> None:
    repo = "Mitroshenkov87/voxprint-example"
    with pytest.raises(LicenceError) as nc:
        assert_public_upload(licence="cc-by-nc-4.0", card_text=complete_card(), repo=repo)
    assert nc.value.code == "non_commercial"
    with pytest.raises(LicenceError):
        assert_public_upload(licence="CC-BY-NC-SA-4.0", card_text=complete_card(), repo=repo)
    with pytest.raises(StageError) as short:
        assert_public_upload(licence="apache-2.0", card_text="A card", repo=repo)
    assert short.value.code == "card_incomplete"
    fenced = complete_card().replace(APACHE_LINE, "```\n" + APACHE_LINE + "\n```")
    with pytest.raises(StageError) as fence:
        assert_public_upload(licence="apache-2.0", card_text=fenced, repo=repo)
    assert fence.value.code == "card_incomplete"
    assert assert_public_upload(licence="apache-2.0", card_text=complete_card(), repo=repo) == "apache-2.0"
    mit = complete_card("MIT. See https://opensource.org/license/mit")
    assert assert_public_upload(licence="MIT", card_text=mit, repo=repo) == "mit"


def test_free_eval_stage_drops_mocked_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    allocated = {"bytes": 8_000_000_000}
    calls: list[str] = []

    def empty_cache() -> None:
        calls.append("empty")
        allocated["bytes"] = 64

    def synchronize() -> None:
        calls.append("sync")

    def memory_allocated() -> int:
        return allocated["bytes"]

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    before = torch.cuda.memory_allocated()
    pipe = free_eval_stage(object())
    after = torch.cuda.memory_allocated()
    assert pipe is None
    assert after < before
    assert calls == ["empty", "sync"]


def _git_bash() -> str:
    """Git bash on Windows. System32\\bash.exe is a stub and is not used."""
    if os.name != "nt":
        return "bash"
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    return str(git_bash) if git_bash.is_file() else "bash"


def test_rented_vm_logs_in_once_and_local_does_not(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "vm_logins.sh"
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    hf = venv_bin / "hf"
    hf.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$FORGE_OUT/hf-args\"\n", encoding="utf-8")
    hf.chmod(0o755)
    home = tmp_path / "home"
    out = tmp_path / "out"
    home.mkdir()
    out.mkdir()
    env = {
        "HOME": home.as_posix(),
        "VENV": (tmp_path / "venv").as_posix(),
        "FORGE_OUT": out.as_posix(),
        "HF_TOKEN": "hf-secret-value",
        "KAGGLE_API_TOKEN": "kaggle-secret-value",
        "FORGE_HOST_KIND": "local",
        "PATH": os.environ.get("PATH", ""),
    }
    command = [_git_bash(), "-c", f'source "{script.as_posix()}"; persist_rented_vm_logins']
    subprocess.run(command, check=True, env=env)
    assert not (out / "hf-args").exists()
    assert not (home / ".kaggle" / "access_token").exists()
    env["FORGE_HOST_KIND"] = "rented-vm"
    subprocess.run(command, check=True, env=env)
    assert (out / "hf-args").read_text(encoding="utf-8") == "auth login --token hf-secret-value\n"
    token_path = home / ".kaggle" / "access_token"
    assert token_path.read_text(encoding="utf-8").strip() == "kaggle-secret-value"
    if os.name != "nt":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    text = script.read_text(encoding="utf-8")
    assert "chmod 600" in text
    assert "set +x" in text
    assert "unset HF_TOKEN" not in text
    assert "huggingface-cli login" not in text
