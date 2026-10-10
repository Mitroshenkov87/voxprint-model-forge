"""CPU dry run of the part D operator command, then the C/E selectors.

The operator command is the python line in scripts/canvas_job_d.sh. Diffusers is
not installed here, so the quality stage is replaced after the real convert.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

from voxprint_forge.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "canvas-qwen-image-2512-v1.toml"


def _bash(script: Path) -> list[str]:
    """Git bash on Windows. System32\\bash.exe is a stub and is not used."""
    exe = "bash"
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        exe = str(git_bash) if git_bash.is_file() else "bash"
    return [exe, script.as_posix()]


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


def _install_fakes() -> None:
    passing = {"decision": "fp8", "lpips_mean": 0.01, "clip_drop": 0.001, "clip_i": 0.99, "samples": 64}
    qwen = types.ModuleType("voxprint_forge.diffusion.qwen_generate")

    def evaluate_qwen(*, model_dir: Path, transformer_fp8: list[Path], grid_path: Path, progress=None) -> dict:
        del model_dir, transformer_fp8
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_path.write_bytes(b"avif")
        if progress is not None:
            progress("qwen-bf16", 1, 1)
        return {"components": {"qwen-transformer": dict(passing)}}

    qwen.evaluate_qwen = evaluate_qwen  # type: ignore[attr-defined]
    sys.modules["voxprint_forge.diffusion.qwen_generate"] = qwen

    generate = types.ModuleType("voxprint_forge.diffusion.generate")
    generate.evaluate_base = lambda **kwargs: {"components": {"turbo-transformer": dict(passing)}}  # type: ignore[attr-defined]
    generate.evaluate_control = lambda **kwargs: {"components": {"controlnet-lite": dict(passing)}}  # type: ignore[attr-defined]
    generate.evaluate_edit = lambda **kwargs: {  # type: ignore[attr-defined]
        "components": {
            "edit-transformer": dict(passing),
            "qwen2.5-vl-encoder": dict(passing),
        }
    }
    sys.modules["voxprint_forge.diffusion.generate"] = generate


def _clear_fakes() -> None:
    sys.modules.pop("voxprint_forge.diffusion.qwen_generate", None)
    sys.modules.pop("voxprint_forge.diffusion.generate", None)
    sys.modules.pop("voxprint_forge.diffusion.job", None)


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


def _fetch_sources(sources: tuple, dest: Path, token: str | None) -> dict[str, object]:
    del token
    dest.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {}
    for source in sources:
        _diffusion_tree(dest / source.id)
        record[source.id] = {"repo": source.repo, "revision": source.revision, "licence": "apache-2.0", "files": {}}
    (dest / "upstream-sha256.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _forbid_hub(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("the CPU dry run must not download from Hugging Face")


def _patch_runtime(monkeypatch) -> None:
    from voxprint_forge.diffusion import part_base, part_control, part_edit, publish, sources

    qwen_job = importlib.import_module("voxprint_forge.diffusion.qwen_job")
    monkeypatch.setattr(qwen_job, "fetch_sources", _fetch_sources)
    monkeypatch.setattr(part_base, "fetch_sources", _fetch_sources)
    monkeypatch.setattr(part_control, "fetch_sources", _fetch_sources)
    monkeypatch.setattr(part_edit, "fetch_sources", _fetch_sources)
    monkeypatch.setattr(sources, "snapshot_download", _forbid_hub)
    monkeypatch.setattr(sources, "hf_hub_download", _forbid_hub)
    monkeypatch.setattr(publish, "HubRemote", FakeHub)
    monkeypatch.setattr("voxprint_forge.diffusion.qwen_common.ensure_free", lambda path, need, what, **_kwargs: 10**12)
    monkeypatch.setattr("voxprint_forge.diffusion.common.ensure_free", lambda path, need, what, **_kwargs: 10**12)
    monkeypatch.setenv("HF_USER", "Mitroshenkov87")
    monkeypatch.setenv("KAGGLE_USER", "mitroshenkov87")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("FORGE_DEADLINE", raising=False)


def _operator_argv(forge_out: Path, *extra: str) -> list[str]:
    """Same flag order as scripts/canvas_job_d.sh. --force is the script's opt-in."""
    return [
        "run",
        str(PLAN),
        "--json",
        "--forge-out",
        str(forge_out),
        "--root",
        str(ROOT),
        *extra,
        "--force",
    ]


def _no_weights(directory: Path) -> None:
    assert directory.is_dir()
    assert list(directory.rglob("*.safetensors")) == []
    assert list(directory.rglob("*.gguf")) == []


def test_qwen_script_refuses_a_missing_token_when_stdin_is_not_a_tty(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_OUT"] = tmp_path.as_posix()
    proc = subprocess.run(
        _bash(ROOT / "scripts" / "canvas_job_d.sh"),
        input=b"",
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )
    assert proc.returncode == 2
    assert b"non-interactive" in proc.stderr


def test_refuses_a_live_canvas_job_before_the_runner_imports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("voxprint_forge.diffusion.qwen_main.canvas_ac_running", lambda: True)
    monkeypatch.setattr("voxprint_forge.diffusion.qwen_main.llm_job_running", lambda forge_out: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    sys.modules.pop("voxprint_forge.diffusion.qwen_job", None)
    diffusion = importlib.import_module("voxprint_forge.diffusion")
    if hasattr(diffusion, "qwen_job"):
        delattr(diffusion, "qwen_job")
    from voxprint_forge.diffusion.qwen_main import main

    # No --force: that flag is what skips the live-job guard.
    argv = [item for item in _operator_argv(tmp_path / "forge-out") if item != "--force"]
    code = main(argv)
    assert code == 2
    assert "voxprint_forge.diffusion.qwen_job" not in sys.modules
    assert "diffusers" not in sys.modules


def test_documented_qwen_command_dry_runs_on_cpu(tmp_path: Path, monkeypatch) -> None:
    script = (ROOT / "scripts" / "canvas_job_d.sh").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "CANVAS.md").read_text(encoding="utf-8")
    needle = (
        'args=(run "$ROOT/plans/canvas-qwen-image-2512-v1.toml" '
        '--json --forge-out "$FORGE_OUT" --root "$ROOT")'
    )
    assert needle in script
    assert '"$VENV/bin/python" -m voxprint_forge.diffusion.qwen_main' in script
    assert "set +x" in script
    assert "set -x" not in script
    assert "hf auth login" not in script
    assert "cleanup_secrets" not in script
    assert "unset HF_TOKEN" not in script
    assert "persist_rented_vm_logins" in script
    assert 'export FORGE_HOST_KIND="${FORGE_HOST_KIND:-local}"' in script
    assert "huggingface-cli login" not in script
    assert "github.com" not in script
    assert "GH_TOKEN" not in script
    assert "python -m voxprint_forge.diffusion.qwen_main run plans/canvas-qwen-image-2512-v1.toml" in docs
    assert "estimate part qwen 120 min" in script
    assert "estimate part encoder 15 min" in script

    _install_fakes()
    _patch_runtime(monkeypatch)
    from voxprint_forge.diffusion.qwen_main import main

    forge_out = tmp_path / "forge-out"
    try:
        code = main(_operator_argv(forge_out))
    finally:
        _clear_fakes()
    assert code == 0
    assert "diffusers" not in sys.modules
    assert "voxprint_forge.diffusion.generate" not in sys.modules

    manifests = list((forge_out / "manifests").glob("*-canvas-qwen-image-2512.json"))
    assert len(manifests) == 1
    body = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert body["int4"] == "deferred"
    assert [part["id"] for part in body["parts"]] == ["qwen"]
    assert body["parts"][0]["status"] == "uploaded"
    assert body["retired"] == ["qwen", "encoder"]
    assert body["encoder_owner"] == "edit"
    assert body["parts"][0]["encoder"] == "reuse-pending"
    assert body["parts"][0]["copied_weights"] is False
    assert body["parts"][0]["files"]
    models = manifests[0].with_name(manifests[0].name.replace(".json", ".MODELS.md")).read_text(encoding="utf-8")
    assert "Retired" in models
    assert "part C" in models

    output = forge_out / "work" / "canvas-qwen-image-2512" / "Qwen--Qwen-Image-2512-voxprint"
    shards = list(output.glob("qwen-image-2512-transformer-fp8-*.safetensors"))
    assert shards
    assert shards[0].stat().st_size < 1_000_000
    assert not (output / "qwen2.5-vl-7b-text-encoder-fp8.safetensors").exists()
    assert not (output / "text_encoder").exists()
    _no_weights(output / "int4")
    assert "no weights" in (output / "int4" / "DEFERRED.txt").read_text(encoding="utf-8")
    assert "nunchaku" in (output / "int4" / "DEFERRED.txt").read_text(encoding="utf-8")
    stable = forge_out / "canvas" / "qwen-image-2512"
    _no_weights(stable / "int4")
    assert list((stable / "fp8").glob("qwen-image-2512-transformer-fp8-*.safetensors"))
    assert (stable / "encoder").is_dir()
    _no_weights(stable / "encoder")
    assert (stable / "encoder" / "PENDING.txt").is_file()
    record = json.loads((output / "encoder-reuse.json").read_text(encoding="utf-8"))
    assert record == {
        "copied_weights": False,
        "edit_fp8": str(
            forge_out
            / "work"
            / "canvas-zimage-tiers-v1"
            / "Qwen--Qwen-Image-Edit-2511-voxprint"
            / "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
        ),
        "status": "reuse-pending",
    }


def test_parts_c_d_e_run_qwen_then_edit_then_encoder(tmp_path: Path, monkeypatch) -> None:
    _install_fakes()
    _patch_runtime(monkeypatch)
    from voxprint_forge.diffusion.qwen_main import main

    forge_out = tmp_path / "forge-out"
    try:
        code = main(_operator_argv(forge_out, "--parts", "C,D,E"))
    finally:
        _clear_fakes()
    assert code == 0
    assert "diffusers" not in sys.modules
    body = json.loads(next((forge_out / "manifests").glob("*-canvas-qwen-image-2512.json")).read_text(encoding="utf-8"))
    assert [part["id"] for part in body["parts"]] == ["qwen", "edit", "encoder"]
    assert body["parts"][0]["status"] == "uploaded"
    assert body["parts"][0]["encoder"] == "reuse-pending"
    assert body["parts"][1]["status"] == "uploaded"
    assert body["parts"][2]["status"] == "reuse-pending"
    assert body["parts"][2]["copied_weights"] is False
    qwen_output = forge_out / "work" / "canvas-qwen-image-2512" / "Qwen--Qwen-Image-2512-voxprint"
    assert not (qwen_output / "qwen2.5-vl-7b-text-encoder-fp8.safetensors").exists()
    edit_fp8 = (
        forge_out
        / "work"
        / "canvas-zimage-tiers-v1"
        / "Qwen--Qwen-Image-Edit-2511-voxprint"
        / "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
    )
    assert edit_fp8.is_file()


def test_encoder_selector_reuses_part_c_without_copying_weights(tmp_path: Path, monkeypatch) -> None:
    """Pins match a tiny encoder. The second E run links the part C file that appeared."""
    from voxprint_forge.diffusion.qwen_plan import load_qwen_plan
    from voxprint_forge.diffusion.qwen_spec import ENCODER_REUSE

    source = tmp_path / "src-encoder"
    _weight(source / "model.safetensors")
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    pins = {f"text_encoder/{path.name}": sha256_file(path) for path in sorted(source.iterdir())}
    plan_path = tmp_path / "qwen.toml"
    text = PLAN.read_text(encoding="utf-8")
    start = text.index("[encoder_pins]")
    lines = ["[encoder_pins]"]
    for name, digest in pins.items():
        lines.append(f'"{name}" = "{digest}"')
    plan_path.write_text(text[:start] + "\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_qwen_plan(plan_path)
    assert loaded.encoder_pins == pins

    def fetch_sources(sources: tuple, dest: Path, token: str | None) -> dict[str, object]:
        del token
        dest.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {}
        for source_row in sources:
            folder = dest / source_row.id / "text_encoder"
            folder.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                target = folder / path.name
                target.write_bytes(path.read_bytes())
            record[source_row.id] = {"repo": source_row.repo, "revision": source_row.revision, "licence": "apache-2.0"}
        (dest / "upstream-sha256.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    _install_fakes()
    _patch_runtime(monkeypatch)
    qwen_job = importlib.import_module("voxprint_forge.diffusion.qwen_job")
    from voxprint_forge.diffusion.qwen_main import main

    monkeypatch.setattr(qwen_job, "fetch_sources", fetch_sources)
    forge_out = tmp_path / "forge-out"
    argv = [
        "run",
        str(plan_path),
        "--json",
        "--forge-out",
        str(forge_out),
        "--root",
        str(ROOT),
        "--parts",
        "E",
        "--force",
    ]
    try:
        assert main(argv) == 0
        output = forge_out / "work" / "canvas-qwen-image-2512" / "Qwen--Qwen-Image-2512-voxprint"
        pending = json.loads((output / ENCODER_REUSE).read_text(encoding="utf-8"))
        assert pending["status"] == "reuse-pending"
        assert pending["copied_weights"] is False
        assert list(output.rglob("*.safetensors")) == []
        stable_encoder = forge_out / "canvas" / "qwen-image-2512" / "encoder"
        assert (stable_encoder / "PENDING.txt").is_file()
        _no_weights(stable_encoder)

        edit_fp8 = (
            forge_out
            / "work"
            / "canvas-zimage-tiers-v1"
            / "Qwen--Qwen-Image-Edit-2511-voxprint"
            / "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
        )
        _weight(edit_fp8)
        assert main(argv) == 0
    finally:
        _clear_fakes()

    reused = json.loads((output / ENCODER_REUSE).read_text(encoding="utf-8"))
    assert reused["status"] == "reuse"
    assert reused["copied_weights"] is False
    assert list(output.rglob("*.safetensors")) == []
    assert not (output / "text_encoder").exists()
    linked = forge_out / "canvas" / "qwen-image-2512" / "encoder"
    if linked.is_symlink():
        assert linked.resolve() == edit_fp8.resolve()
    else:
        assert linked.stat().st_ino == edit_fp8.stat().st_ino
    assert "diffusers" not in sys.modules


class _RefuseRemote:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("deferred prepare must not open a remote")


def test_qwen_defer_skips_the_token_prompt(tmp_path: Path) -> None:
    script = (ROOT / "scripts" / "canvas_job_d.sh").read_text(encoding="utf-8")
    assert "FORGE_DEFER_UPLOAD" in script
    assert "--defer-upload" in script
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["FORGE_DEFER_UPLOAD"] = "1"
    env["FORGE_OUT"] = tmp_path.as_posix()
    proc = subprocess.run(
        [*_bash(ROOT / "scripts" / "canvas_job_d.sh"), "--parts", "C,D,E"],
        input=b"",
        capture_output=True,
        env=env,
        timeout=8,
        check=False,
    )
    assert b"non-interactive" not in proc.stderr
    assert proc.returncode != 2


def test_deferred_qwen_parts_then_batch_upload(tmp_path: Path, monkeypatch) -> None:
    """C, D, and E with FORGE_DEFER_UPLOAD, then canvas_upload.sh's two upload lines."""
    from voxprint_forge.diffusion import publish
    from voxprint_forge.diffusion.cli import main as canvas_main
    from voxprint_forge.diffusion.qwen_main import main
    from voxprint_forge.diffusion.qwen_spec import QWEN_ENCODER_FP8

    _install_fakes()
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(publish, "HubRemote", _RefuseRemote)
    forge_out = tmp_path / "forge-out"
    zimage = ROOT / "plans" / "canvas-zimage-tiers-v1.toml"
    try:
        code = main(
            [
                "run",
                str(PLAN),
                "--json",
                "--forge-out",
                str(forge_out),
                "--root",
                str(ROOT),
                "--parts",
                "C,D,E",
                "--force",
                "--defer-upload",
            ]
        )
        assert code == 0
        manifest = next((forge_out / "manifests").glob("*-canvas-qwen-image-2512.json"))
        body = json.loads(manifest.read_text(encoding="utf-8"))
        statuses = {part["id"]: part["status"] for part in body["parts"]}
        assert statuses == {"qwen": "upload-deferred", "edit": "upload-deferred", "encoder": "upload-deferred"}
        qwen_stages = forge_out / "stages" / "canvas-qwen-image-2512"
        edit_stages = forge_out / "stages" / "canvas-zimage-tiers-v1"
        assert (qwen_stages / "qwen-upload-deferred.json").is_file()
        assert (qwen_stages / "encoder-upload-deferred.json").is_file()
        assert (edit_stages / "edit-upload-deferred.json").is_file()
        output = forge_out / "work" / "canvas-qwen-image-2512" / "Qwen--Qwen-Image-2512-voxprint"
        encoder = output / QWEN_ENCODER_FP8
        assert not encoder.exists()
        edit_encoder = (
            forge_out
            / "work"
            / "canvas-zimage-tiers-v1"
            / "Qwen--Qwen-Image-Edit-2511-voxprint"
            / QWEN_ENCODER_FP8
        )
        assert edit_encoder.is_file()
        assert not (forge_out / "work" / "canvas-qwen-image-2512" / "upstream").exists()
        assert (output / "int4" / "DEFERRED.txt").is_file()

        monkeypatch.setattr(publish, "HubRemote", FakeHub)
        assert (
            main(
                [
                    "upload",
                    str(PLAN),
                    "--json",
                    "--forge-out",
                    str(forge_out),
                    "--root",
                    str(ROOT),
                    "--parts",
                    "D,E",
                    "--force",
                ]
            )
            == 0
        )
        assert (
            canvas_main(
                [
                    "upload",
                    str(zimage),
                    "--json",
                    "--forge-out",
                    str(forge_out),
                    "--root",
                    str(ROOT),
                    "--parts",
                    "C",
                    "--force",
                ]
            )
            == 0
        )
        assert (qwen_stages / "qwen-upload.json").is_file()
        assert (qwen_stages / "encoder-upload.json").is_file()
        assert (edit_stages / "edit-upload.json").is_file()
        assert not (qwen_stages / "qwen-upload-deferred.json").exists()
        assert not (qwen_stages / "encoder-upload-deferred.json").exists()
        encoder_marker = json.loads((qwen_stages / "encoder-upload.json").read_text(encoding="utf-8"))
        assert encoder_marker["copied_weights"] is False
        assert not any(item["name"] == QWEN_ENCODER_FP8 for item in encoder_marker["files"])
        assert not encoder.exists()
        assert edit_encoder.is_file()

        monkeypatch.setattr(publish, "HubRemote", _RefuseRemote)
        assert (
            main(
                [
                    "upload",
                    str(PLAN),
                    "--json",
                    "--forge-out",
                    str(forge_out),
                    "--root",
                    str(ROOT),
                    "--parts",
                    "D,E",
                    "--force",
                ]
            )
            == 0
        )
        from voxprint_forge.cli import main as forge_main

        assert (
            forge_main(
                [
                    "--json",
                    "--root",
                    str(ROOT),
                    "--forge-out",
                    str(forge_out),
                    "canvas",
                    "upload",
                    str(zimage),
                    "--parts",
                    "C",
                    "--force",
                ]
            )
            == 0
        )
    finally:
        _clear_fakes()
