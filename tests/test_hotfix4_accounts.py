"""Hotfix 4, source consumers, the process guard, and account commands."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from tests.complete_card import complete_card
from voxprint_forge.diffusion.controlnet_union import FULL_PLACES, LITE_PLACES, places_for_count
from voxprint_forge.diffusion.guard import cmdline_is_job
from voxprint_forge.diffusion.prompts import EDIT_PAIRS, edit_budget
from voxprint_forge.gui.client import ssh_upload
from voxprint_forge.gui.secrets import MemoryStore
from voxprint_forge.upload import kaggle_model_slug, repo_slug

ROOT = Path(__file__).resolve().parents[1]


def test_lite_places_match_videox_fun() -> None:
    """VideoX-Fun config/z_image/z_image_control_2.1_lite.yaml uses [0, 10, 20]."""
    assert places_for_count(3) == list(LITE_PLACES) == [0, 10, 20]
    assert places_for_count(15) == list(FULL_PLACES)
    assert places_for_count(1) == [0]


def test_kaggle_slug_drops_dots_and_hf_keeps_them() -> None:
    assert repo_slug("t-lite-it-2.1") == "voxprint-t-lite-it-2.1"
    assert kaggle_model_slug("t-lite-it-2.1") == "voxprint-t-lite-it-2-1"


def test_publish_uses_a_dashed_kaggle_slug(tmp_path: Path, monkeypatch) -> None:
    from voxprint_forge.upload import Artifact, publish_private

    monkeypatch.setenv("HF_USER", "tester")
    monkeypatch.setenv("KAGGLE_USER", "tester")
    payload = b"gguf-bytes"
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)
    artifact = Artifact(path, "model.gguf", hashlib.sha256(payload).hexdigest(), len(payload), "16gb", "IQ2_M", "gguf")
    seen: dict[str, str] = {}

    class Remote:
        def hf_private_repo(self, repo_id: str) -> str:
            seen["hf"] = repo_id
            return repo_id

        def hf_upload(self, repo_id: str, file_path: Path, name: str) -> None:
            del repo_id, file_path, name

        def hf_stat(self, repo_id: str, name: str) -> tuple[str, int]:
            del repo_id, name
            return hashlib.sha256(payload).hexdigest(), len(payload)

        def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
            del owner, folder
            seen["kaggle"] = slug
            return "https://www.kaggle.com/models/example"

        def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
            del owner, folder
            seen["kaggle"] = slug
            return f"tester/{slug}/gguf/{variation}"

        def kaggle_stat(self, handle: str, name: str) -> tuple[str, int]:
            del handle, name
            return hashlib.sha256(payload).hexdigest(), len(payload)

    publish_private(
        Remote(),
        [artifact],
        model_id="t-lite-it-2.1",
        title="T",
        description=complete_card(),
        licence="apache-2.0",
        attempts=1,
        sleep=lambda _seconds: None,
        work=tmp_path / "work",
    )
    assert seen["hf"] == "tester/voxprint-t-lite-it-2.1"
    assert seen["kaggle"] == "voxprint-t-lite-it-2-1"


def test_hf_stat_hashes_a_small_file_when_lfs_sha_is_missing(monkeypatch, tmp_path: Path) -> None:
    del tmp_path
    from voxprint_forge.upload import HubRemote

    blob = b"small-card"

    def download(repo_id: str, name: str, token: str | None = None, local_dir: str | None = None) -> str:
        del repo_id, token
        path = Path(local_dir or ".") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return str(path)

    class Info:
        lfs = None
        size = len(blob)

    class Api:
        def __init__(self, token: str | None = None) -> None:
            del token

        def get_paths_info(self, repo_id: str, paths: list[str]) -> list[Info]:
            del repo_id, paths
            return [Info()]

    monkeypatch.setattr("voxprint_forge.upload.hf_hub_download", download)
    monkeypatch.setattr("voxprint_forge.upload.HfApi", Api)
    digest, size = HubRemote("token", None).hf_stat("org/model", "README.md")
    assert size == len(blob)
    assert digest == hashlib.sha256(blob).hexdigest()


def test_edit_trim_gate(tmp_path: Path) -> None:
    assert edit_budget(tmp_path) == (EDIT_PAIRS, (0, 1))
    marker = tmp_path / "forge-out" / "EDIT_TRIM"
    marker.parent.mkdir(parents=True)
    marker.write_text("1\n", encoding="utf-8")
    pairs, seeds = edit_budget(tmp_path)
    assert pairs == EDIT_PAIRS[:8]
    assert seeds == (0,)


def test_tmux_server_is_not_a_job() -> None:
    assert cmdline_is_job("tmux new-session -s forge bash scripts/bootstrap.sh") is False
    assert cmdline_is_job("bash scripts/bootstrap.sh") is True
    assert cmdline_is_job("python -m voxprint_forge run-all") is True
    assert cmdline_is_job("/usr/bin/python3 -m voxprint_forge.diffusion run plan.toml") is True


def test_source_is_kept_until_the_last_consumer(tmp_path: Path) -> None:
    from voxprint_forge.diffusion.planload import load_canvas_plan
    from voxprint_forge.diffusion.retain import consumers_for, release_consumer, source_store

    plan = load_canvas_plan(ROOT / "plans" / "canvas-zimage-tiers-v1.toml")
    names = consumers_for(plan, "turbo")
    assert names == ("base", "controlnet", "qwen", "bake-off")
    tree = source_store(tmp_path, plan.id) / "turbo"
    tree.mkdir(parents=True)
    (tree / "weights").write_text("x", encoding="utf-8")
    for name in names:
        if name == "bake-off":
            continue
        assert release_consumer(tmp_path, plan.id, "turbo", name, names) is False
        assert tree.is_dir()
    assert release_consumer(tmp_path, plan.id, "turbo", "bake-off", names) is True
    assert not tree.exists()


def test_accounts_round_trip_keeps_the_token_out_of_the_result(monkeypatch) -> None:
    from voxprint_forge.accounts import execute_accounts
    from voxprint_forge.cli import main

    store = MemoryStore()
    monkeypatch.setattr("voxprint_forge.accounts.open_store", lambda: store)

    class Args:
        accounts_command = "add"
        target = "kaggle"
        login = "paste"
        token = "KGAT_test_token_value"

    added = execute_accounts(Args(), store=store)
    assert added["connected"] is True
    assert "KGAT_test_token_value" not in json.dumps(added)
    assert store.get("KAGGLE_API_TOKEN") == "KGAT_test_token_value"

    def kaggle_name(token: str) -> str:
        assert token == "KGAT_test_token_value"
        return "kaggle-owner"

    tested = execute_accounts(
        type("T", (), {"accounts_command": "test", "target": "kaggle"})(),
        store=store,
        probes={"hf": kaggle_name, "kaggle": kaggle_name, "drive": kaggle_name},
    )
    assert tested["name"] == "kaggle-owner"
    listed = execute_accounts(type("L", (), {"accounts_command": "list"})(), store=store)
    assert listed["accounts"][1]["name"] == "kaggle-owner"
    removed = execute_accounts(type("R", (), {"accounts_command": "remove", "target": "kaggle"})(), store=store)
    assert removed["connected"] is False
    assert store.get("KAGGLE_API_TOKEN") == ""

    code = main(["--json", "--root", str(ROOT), "accounts", "add", "--target", "hf", "--token", "hf-cli-token"])
    assert code == 0
    assert store.get("HF_TOKEN") == "hf-cli-token"


def test_hf_browser_login_stores_the_token_in_the_keyring() -> None:
    from voxprint_forge.accounts import login_browser

    store = MemoryStore()

    def request() -> dict[str, object]:
        return {
            "verification_uri_complete": "https://example.invalid/device",
            "verification_uri": "https://example.invalid/device",
            "user_code": "ABCD-1234",
            "device_code": "device",
            "interval": 0,
            "expires_in": 1,
        }

    def poll(info: dict[str, object]) -> dict[str, object]:
        assert info["user_code"] == "ABCD-1234"
        return {"access_token": "hf-device-access", "refresh_token": "hf-device-refresh"}

    shown: list[str] = []
    result = login_browser(
        store,
        "hf",
        on_code=lambda uri, code: shown.append(f"{uri} {code}"),
        request=request,
        poll=poll,
        identify=lambda token: "ada" if token == "hf-device-access" else "",
    )
    assert result["name"] == "ada"
    assert result["user_code"] == "ABCD-1234"
    assert "hf-device-access" not in json.dumps(result)
    assert store.get("HF_TOKEN") == "hf-device-access"
    assert store.get("HF_REFRESH") == "hf-device-refresh"
    assert shown == ["https://example.invalid/device ABCD-1234"]


def test_drive_browser_login_keeps_rclone_json_in_memory() -> None:
    from voxprint_forge.accounts import login_browser

    store = MemoryStore()
    token = json.dumps({"access_token": "drive-access", "refresh_token": "drive-refresh"})
    result = login_browser(store, "drive", authorize=lambda: token, identify=lambda value: "ada@example.com")
    assert result == {"target": "drive", "name": "ada@example.com", "connected": True}
    assert store.get("FORGE_DRIVE_TOKEN") == token
    assert "drive-access" not in json.dumps(result)


def test_ssh_upload_puts_tokens_on_stdin_and_unsets_them() -> None:
    seen: dict[str, object] = {}

    def run(argv: list[str], stdin: str) -> str:
        seen["argv"] = argv
        seen["stdin"] = stdin
        return "ok"

    secret = "hf-ssh-only-token"
    text = ssh_upload(
        "vm.example",
        "forge",
        "/keys/id_ed25519",
        "forge --json upload --model tiny",
        {"HF_TOKEN": secret},
        run=run,
    )
    assert text == "ok"
    argv = seen["argv"]
    assert isinstance(argv, list)
    joined = " ".join(argv)
    assert secret not in joined
    assert "unset HF_TOKEN" in joined
    assert "vm.example" in joined
    stdin = str(seen["stdin"])
    assert secret not in stdin
    assert base64.b64encode(secret.encode()).decode() in stdin
