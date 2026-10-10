"""CPU dry run of the documented canvas command with a tiny model and a fake remote.

The operator command is `python -m voxprint_forge.diffusion run PLAN --json --forge-out ... --root ...`
from scripts/canvas_job.sh. Diffusers is not installed in CI, so the quality
stage is replaced after the real convert. Uploads go through a stand-in HubRemote.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def _bash(script: Path) -> list[str]:
    """Git bash on Windows. System32\\bash.exe is a stub and is not used."""
    exe = "bash"
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        exe = str(git_bash) if git_bash.is_file() else "bash"
    return [exe, script.as_posix()]


def _install_fake_generate() -> types.ModuleType:
    """Quality imports diffusers. The dry run supplies the three evaluators instead."""
    passing = {"decision": "fp8", "lpips_mean": 0.01, "clip_drop": 0.001, "clip_i": 0.99}
    fake = types.ModuleType("voxprint_forge.diffusion.generate")
    fake.evaluate_base = lambda **kwargs: {  # type: ignore[attr-defined]
        "components": {
            "turbo-transformer": dict(passing),
            "zimage-transformer": dict(passing),
            "qwen3-4b-encoder": dict(passing),
        }
    }
    fake.evaluate_control = lambda **kwargs: {"components": {"controlnet-lite": dict(passing)}}  # type: ignore[attr-defined]
    fake.evaluate_edit = lambda **kwargs: {  # type: ignore[attr-defined]
        "components": {
            "edit-transformer": dict(passing),
            "qwen2.5-vl-encoder": dict(passing),
        }
    }
    return fake


def _weight(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "layers.0.attention.to_q.weight": torch.zeros(4, 4),
            "norm.weight": torch.zeros(4),
        },
        path,
    )


def _diffusion_tree(root: Path) -> None:
    _weight(root / "transformer" / "model.safetensors")
    (root / "transformer" / "config.json").write_text("{}\n", encoding="utf-8")
    for name in ("text_encoder", "vae", "tokenizer"):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}\n", encoding="utf-8")
        _weight(folder / "model.safetensors")


class FakeHub:
    def __init__(self, hf_token: str | None = None, kaggle_token: str | None = None) -> None:
        del hf_token, kaggle_token
        self.hf_bytes: dict[str, bytes] = {}
        self.kaggle_bytes: dict[str, bytes] = {}
        self.repos: list[str] = []

    def hf_private_repo(self, repo_id: str) -> str:
        self.repos.append(repo_id)
        return f"https://huggingface.co/{repo_id}"

    def hf_upload(self, repo_id: str, path: Path, name: str) -> None:
        del repo_id
        self.hf_bytes[name] = path.read_bytes()

    def hf_stat(self, repo_id: str, name: str) -> tuple[str, int]:
        del repo_id
        payload = self.hf_bytes[name]
        return hashlib.sha256(payload).hexdigest(), len(payload)

    def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
        del owner, slug, folder
        return "https://www.kaggle.com/models/example"

    def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
        for path in folder.iterdir():
            if path.is_file() and path.name not in {"model-instance-metadata.json", "model-metadata.json"}:
                self.kaggle_bytes[path.name] = path.read_bytes()
        return f"{owner}/{slug}/gguf/{variation}"

    def kaggle_stat(self, handle: str, name: str) -> tuple[str, int]:
        del handle
        payload = self.kaggle_bytes[name]
        return hashlib.sha256(payload).hexdigest(), len(payload)


def test_canvas_job_refuses_a_missing_token_when_stdin_is_not_a_tty(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_OUT"] = tmp_path.as_posix()
    proc = subprocess.run(
        _bash(ROOT / "scripts/canvas_job.sh"),
        input=b"",
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )
    assert proc.returncode == 2
    assert b"non-interactive" in proc.stderr


def test_documented_canvas_command_dry_runs_on_cpu(tmp_path: Path, monkeypatch) -> None:
    script = (ROOT / "scripts/canvas_job.sh").read_text(encoding="utf-8")
    assert '"$VENV/bin/python" -m voxprint_forge.diffusion' in script
    needle = 'args=(run "$ROOT/plans/canvas-zimage-tiers-v1.toml" --json --forge-out "$FORGE_OUT" --root "$ROOT")'
    assert needle in script

    from voxprint_forge.diffusion import part_base, part_control, part_edit, publish
    from voxprint_forge.diffusion.cli import main
    from voxprint_forge.diffusion.names import (
        CONTROL_FULL_UPSTREAM,
        CONTROL_LITE_UPSTREAM,
    )

    def fetch_sources(sources: tuple, dest: Path, token: str | None) -> dict[str, object]:
        del token
        dest.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {}
        for source in sources:
            folder = dest / source.id
            if source.id == "controlnet":
                folder.mkdir(parents=True, exist_ok=True)
                _weight(folder / CONTROL_FULL_UPSTREAM)
                _weight(folder / CONTROL_LITE_UPSTREAM)
            else:
                _diffusion_tree(folder)
            record[source.id] = {
                "repo": source.repo,
                "revision": source.revision,
                "licence": "apache-2.0",
                "files": {},
            }
        (dest / "upstream-sha256.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    part_base.fetch_sources = fetch_sources
    part_control.fetch_sources = fetch_sources
    part_edit.fetch_sources = fetch_sources
    previous = sys.modules.get("voxprint_forge.diffusion.generate")
    sys.modules["voxprint_forge.diffusion.generate"] = _install_fake_generate()
    monkeypatch.setattr(publish, "HubRemote", FakeHub)
    monkeypatch.setattr("voxprint_forge.diffusion.common.ensure_free", lambda path, need, what: 10**12)
    monkeypatch.setenv("HF_USER", "Mitroshenkov87")
    monkeypatch.setenv("KAGGLE_USER", "mitroshenkov87")
    monkeypatch.delenv("FORGE_DEADLINE", raising=False)

    forge_out = tmp_path / "forge-out"
    try:
        code = main(
            [
                "run",
                str(ROOT / "plans" / "canvas-zimage-tiers-v1.toml"),
                "--json",
                "--forge-out",
                str(forge_out),
                "--root",
                str(ROOT),
                "--force",
            ]
        )
    finally:
        if previous is None:
            sys.modules.pop("voxprint_forge.diffusion.generate", None)
        else:
            sys.modules["voxprint_forge.diffusion.generate"] = previous
        sys.modules.pop("voxprint_forge.diffusion.job", None)
    assert code == 0
    assert "diffusers" not in sys.modules
    manifests = list((forge_out / "manifests").glob("*-canvas-zimage.json"))
    assert len(manifests) == 1
    body = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert [part["id"] for part in body["parts"]] == ["base", "controlnet", "edit"]
    assert {part["status"] for part in body["parts"]} == {"uploaded"}
    for part in body["parts"]:
        assert part["files"]


class _RefuseRemote:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("deferred prepare must not open a remote")


def test_defer_upload_skips_the_token_prompt(tmp_path: Path) -> None:
    script = (ROOT / "scripts/canvas_job.sh").read_text(encoding="utf-8")
    assert "FORGE_DEFER_UPLOAD" in script
    assert "--defer-upload" in script
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_DEFER_UPLOAD"] = "1"
    env["FORGE_OUT"] = tmp_path.as_posix()
    proc = subprocess.run(
        _bash(ROOT / "scripts/canvas_job.sh"),
        input=b"",
        capture_output=True,
        env=env,
        timeout=8,
        check=False,
    )
    assert b"non-interactive" not in proc.stderr
    assert proc.returncode != 2


def test_canvas_upload_still_requires_tokens(tmp_path: Path) -> None:
    script = (ROOT / "scripts/canvas_upload.sh").read_text(encoding="utf-8")
    needle = 'args=(upload "$ROOT/plans/canvas-zimage-tiers-v1.toml" --json --forge-out "$FORGE_OUT" --root "$ROOT")'
    assert needle in script
    assert 'args=(upload "$ROOT/plans/canvas-qwen-image-2512-v1.toml"' in script
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_OUT"] = tmp_path.as_posix()
    proc = subprocess.run(
        [*_bash(ROOT / "scripts/canvas_upload.sh"), "--parts", "A,B,C,D"],
        input=b"",
        capture_output=True,
        env=env,
        timeout=8,
        check=False,
    )
    assert proc.returncode == 2
    assert b"non-interactive" in proc.stderr


def test_deferred_canvas_job_then_batch_upload(tmp_path: Path, monkeypatch) -> None:
    """Exact canvas_job.sh args with --defer-upload, then canvas_upload.sh's upload line."""
    from voxprint_forge.diffusion import part_base, part_control, part_edit, publish
    from voxprint_forge.diffusion.cli import main
    from voxprint_forge.diffusion.defer import defer_requested
    from voxprint_forge.diffusion.names import CONTROL_FULL_UPSTREAM, CONTROL_LITE_UPSTREAM

    monkeypatch.setenv("FORGE_DEFER_UPLOAD", "1")
    assert defer_requested(False) is True
    monkeypatch.delenv("FORGE_DEFER_UPLOAD")
    assert defer_requested(False) is False
    assert defer_requested(True) is True

    def fetch_sources(sources: tuple, dest: Path, token: str | None) -> dict[str, object]:
        del token
        dest.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {}
        for source in sources:
            folder = dest / source.id
            if source.id == "controlnet":
                folder.mkdir(parents=True, exist_ok=True)
                _weight(folder / CONTROL_FULL_UPSTREAM)
                _weight(folder / CONTROL_LITE_UPSTREAM)
            else:
                _diffusion_tree(folder)
            record[source.id] = {
                "repo": source.repo,
                "revision": source.revision,
                "licence": "apache-2.0",
                "files": {},
            }
        (dest / "upstream-sha256.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    part_base.fetch_sources = fetch_sources
    part_control.fetch_sources = fetch_sources
    part_edit.fetch_sources = fetch_sources
    previous = sys.modules.get("voxprint_forge.diffusion.generate")
    sys.modules["voxprint_forge.diffusion.generate"] = _install_fake_generate()
    monkeypatch.setattr(publish, "HubRemote", _RefuseRemote)
    monkeypatch.setattr("voxprint_forge.diffusion.common.ensure_free", lambda path, need, what: 10**12)
    monkeypatch.setenv("HF_USER", "Mitroshenkov87")
    monkeypatch.setenv("KAGGLE_USER", "mitroshenkov87")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("FORGE_DEADLINE", raising=False)
    forge_out = tmp_path / "forge-out"
    plan = ROOT / "plans" / "canvas-zimage-tiers-v1.toml"
    try:
        code = main(
            [
                "run",
                str(plan),
                "--json",
                "--forge-out",
                str(forge_out),
                "--root",
                str(ROOT),
                "--defer-upload",
                "--force",
            ]
        )
        assert code == 0
        manifest = json.loads(next((forge_out / "manifests").glob("*-canvas-zimage.json")).read_text(encoding="utf-8"))
        assert [part["status"] for part in manifest["parts"]] == [
            "upload-deferred",
            "upload-deferred",
            "upload-deferred",
        ]
        stages = forge_out / "stages" / "canvas-zimage-tiers-v1"
        for name in ("base", "controlnet", "edit"):
            assert (stages / f"{name}-upload-deferred.json").is_file()
            assert not (stages / f"{name}-upload.json").exists()
        output = forge_out / "work" / "canvas-zimage-tiers-v1" / "Tongyi-MAI--Z-Image-Turbo-voxprint"
        shards = list(output.rglob("*.safetensors"))
        assert shards
        for shard in shards:
            sidecar = Path(str(shard) + ".sha256")
            assert sidecar.is_file()
            assert sidecar.read_text(encoding="utf-8").split()[0]
        assert not (forge_out / "work" / "canvas-zimage-tiers-v1" / "upstream" / "base").exists()

        monkeypatch.setattr(publish, "HubRemote", FakeHub)
        code = main(
            [
                "upload",
                str(plan),
                "--json",
                "--forge-out",
                str(forge_out),
                "--root",
                str(ROOT),
                "--parts",
                "A,B,C",
                "--force",
            ]
        )
        assert code == 0
        for name in ("base", "controlnet", "edit"):
            marker = json.loads((stages / f"{name}-upload.json").read_text(encoding="utf-8"))
            assert marker["files"]
            assert not (stages / f"{name}-upload-deferred.json").exists()
        assert shards[0].is_file()

        monkeypatch.setattr(publish, "HubRemote", _RefuseRemote)
        from voxprint_forge.cli import main as forge_main

        code = forge_main(
            [
                "--json",
                "--root",
                str(ROOT),
                "--forge-out",
                str(forge_out),
                "canvas",
                "upload",
                str(plan),
                "--parts",
                "A,B,C",
                "--force",
            ]
        )
        assert code == 0
        assert (stages / "base-upload.json").is_file()
    finally:
        if previous is None:
            sys.modules.pop("voxprint_forge.diffusion.generate", None)
        else:
            sys.modules["voxprint_forge.diffusion.generate"] = previous
        sys.modules.pop("voxprint_forge.diffusion.job", None)
