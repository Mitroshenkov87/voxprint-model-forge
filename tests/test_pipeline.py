"""Tiny model through every stage, with a fake llama runner and a fake remote."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from voxprint_forge.errors import DiskShortError, StageError
from voxprint_forge.layout import Layout
from voxprint_forge.llama import Completed, Runner
from voxprint_forge.pipeline import Host, _run_quantize, run_all, run_model
from voxprint_forge.plan import load_plan
from voxprint_forge.scrub import scrub
from voxprint_forge.upload import Remote

ROOT = Path(__file__).resolve().parents[1]


def _bash(script: Path) -> list[str]:
    """Git bash on Windows. System32\\bash.exe is a stub and is not used."""
    exe = "bash"
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        exe = str(git_bash) if git_bash.is_file() else "bash"
    return [exe, script.as_posix()]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FakeRunner(Runner):
    def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
        name = Path(argv[0]).name
        if name == "llama-quantize":
            dest = Path(argv[-2])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"Q" * 4096)
        elif name == "llama-imatrix":
            dest = Path(argv[argv.index("-o") + 1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"IMX" * 64)
        elif "--outfile" in argv:
            dest = Path(argv[argv.index("--outfile") + 1])
            dest.write_bytes(b"F16" * 64)
        text = "Final estimate: PPL = 5.50 +/- 0.10\n"
        if name == "llama-cli":
            text = "A short original paragraph about a quiet street.\n"
        return Completed(argv=argv, returncode=0, stdout=text, stderr="")

    def capture_until(self, argv: list[str], *, pattern: str, timeout: float = 180) -> Completed:
        del pattern, timeout
        log = (
            "load_tensors:   CPU_Mapped model buffer size =   521.00 MiB\n"
            "load_tensors:        CUDA0 model buffer size =   100.00 MiB\n"
            "llama_kv_cache:      CUDA0 KV buffer size =    10.00 MiB\n"
        )
        return Completed(argv=argv, returncode=0, stdout=log, stderr="")


class FakeDownloader:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.calls: list[str] = []

    def fetch(self, repo: str, revision: str, filename: str, dest: Path, *, optional: bool = False) -> None:
        del repo, revision
        self.calls.append(filename)
        if filename not in self.files:
            if optional:
                raise FileNotFoundError(filename)
            raise FileNotFoundError(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[filename])


class FakeRemote:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.hf_private: list[str] = []
        self.kaggle_private: list[bool] = []
        self.hf_bytes: dict[str, bytes] = {}
        self.kaggle_bytes: dict[str, bytes] = {}
        self.sleeps: list[float] = []

    def check_hf(self) -> str:
        return "tester"

    def check_kaggle(self) -> str:
        return "mitroshenkov87"

    def hf_private_repo(self, repo_id: str) -> str:
        self.hf_private.append(repo_id)
        return f"https://huggingface.co/{repo_id}"

    def hf_upload(self, repo_id: str, path: Path, name: str) -> None:
        del repo_id
        self.hf_bytes[name] = path.read_bytes()

    def hf_stat(self, repo_id: str, name: str) -> tuple[str | None, int]:
        del repo_id
        if self.failures:
            self.failures -= 1
            return "0" * 64, 1
        payload = self.hf_bytes[name]
        return hashlib.sha256(payload).hexdigest(), len(payload)

    def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
        del owner, slug
        payload = json.loads((folder / "model-metadata.json").read_text(encoding="utf-8"))
        assert payload["isPrivate"] is True
        self.kaggle_private.append(True)
        return "https://www.kaggle.com/models/example"

    def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
        handle = f"{owner}/{slug}/gguf/{variation}"
        for path in folder.iterdir():
            if path.name.endswith(".json"):
                continue
            self.kaggle_bytes[path.name] = path.read_bytes()
        return handle

    def kaggle_stat(self, handle: str, name: str) -> tuple[str | None, int]:
        del handle
        if self.failures:
            self.failures -= 1
            return "0" * 64, 1
        payload = self.kaggle_bytes[name]
        return hashlib.sha256(payload).hexdigest(), len(payload)


def _bins(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("llama-cli", "llama-perplexity", "llama-quantize", "llama-imatrix", "llama-server"):
        path = directory / name
        path.write_text("#!/bin/sh\n", encoding="utf-8")


def _plan(tmp: Path, source: bytes, reuse: bytes, *, optional_id: str | None = None) -> Path:
    source_sha = _sha(source)
    reuse_sha = _sha(reuse)
    optional = ""
    if optional_id:
        optional = f"""
[[models]]
id = "{optional_id}"
enabled = true
optional = true
estimate_min = 80
mode = "requant"
repo = "org/optional"
revision = "optrev"
licence = "apache-2.0"
params_b = 1
base_model = "org/base"
source_format = "gguf"
credits = ["Org"]
[[models.files]]
role = "source"
quant = "F16"
path = "opt.gguf"
sha256 = "{source_sha}"
size = {len(source)}
[[models.tiers]]
name = "16gb"
budget_gb = 16
kind = "quantize"
quants = ["IQ4_XS"]
"""
    text = f"""
[run]
name = "tiny"
output_dir = "{tmp.as_posix()}/work"
wall_clock_budget_min = 30
headroom_gb = 3.0
imatrix_chunks = 1
imatrix_ctx = 512
refusal_limit = 1
ppl_chunks = 1
prose_samples = 1
contexts = [8192, 16384]
llama_cpp_tag = "b11476"
llama_cpp_commit = "988190680d5a89fce97de3c20df2c2813731fd61"
artifact_host = ""
[run.estimates_min]
bootstrap_build = 1
[[models]]
id = "tiny"
enabled = true
optional = false
estimate_min = 5
mode = "requant"
repo = "org/tiny"
revision = "abc123"
licence = "apache-2.0"
params_b = 1
base_model = "org/base"
already_abliterated = true
source_format = "gguf"
credits = ["Org"]
[models.arch]
full_layers = 0
[[models.files]]
role = "source"
quant = "F16"
path = "source.gguf"
sha256 = "{source_sha}"
size = {len(source)}
[[models.files]]
role = "reuse"
quant = "Q5_K_M"
path = "q5.gguf"
sha256 = "{reuse_sha}"
size = {len(reuse)}
[[models.files]]
role = "reuse"
quant = "Q4_K_M"
path = "q4-huge.gguf"
sha256 = "{"ab" * 32}"
size = {50 * 1024 ** 3}
[[models.tiers]]
name = "24gb"
budget_gb = 24
kind = "reuse"
quants = ["Q5_K_M", "Q4_K_M"]
[[models.tiers]]
name = "16gb"
budget_gb = 16
kind = "quantize"
quants = ["IQ4_XS"]
{optional}
"""
    path = tmp / "plan.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _scratch_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    shutil.copytree(ROOT / "templates", repo / "templates")
    shutil.copy(ROOT / "LICENSE", repo / "LICENSE")
    (repo / "manifests").mkdir()
    return repo


def _host(downloader: FakeDownloader, remote: FakeRemote, *, now=None, ablate=None) -> Host:
    return Host(
        runner=FakeRunner(),
        downloader=downloader,
        remote=remote,  # type: ignore[arg-type]
        sleep=remote.sleeps.append,
        now=now or (lambda: 0.0),
        attempts=3,
        stamped="2026-10-10",
        ablate=ablate,
    )


@pytest.fixture()
def llama_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    _bins(folder)
    monkeypatch.setenv("FORGE_LLAMA_BIN", str(folder))
    monkeypatch.setenv("HF_USER", "voxprint-tests")
    monkeypatch.setenv("KAGGLE_USER", "mitroshenkov87")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    # The tiny plan declares a 50 GiB reuse that fit rejects and never writes.
    # ensure_free still requires that size plus the 5 GiB margin (~55 GiB).
    # windows-latest has about 28 GiB free, so the real check would stop the
    # run before the skip. Report a large disk here. test_disk_short covers
    # a short disk, and the production margin is unchanged.
    monkeypatch.setattr("voxprint_forge.disk.free_bytes", _ample_disk)


def test_run_all_uploads_private_and_skips_huge_reuse(tmp_path: Path, llama_bin: None) -> None:
    del llama_bin
    source = b"SRC" * 40
    reuse = b"RE" * 80
    plan = load_plan(_plan(tmp_path, source, reuse))
    files = {"source.gguf": source, "q5.gguf": reuse}
    downloader = FakeDownloader(files)
    remote = FakeRemote(failures=1)
    repo = _scratch_repo(tmp_path)
    layout = Layout(tmp_path / "forge-out")
    result = run_all(plan, _host(downloader, remote), repo, layout)
    assert "q4-huge.gguf" not in downloader.calls
    assert "q5.gguf" in downloader.calls
    assert remote.hf_private == ["voxprint-tests/voxprint-tiny"]
    assert remote.kaggle_private == [True]
    assert remote.sleeps
    upload = layout.stage("tiny", "upload")
    assert upload.is_file()
    payload = json.loads(upload.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload["files"]]
    assert "q5.gguf" in names
    assert any(name.endswith("IQ4_XS.gguf") for name in names)
    for item in payload["files"]:
        if item["kind"] == "gguf":
            assert item["sources"]["hf"].startswith("https://huggingface.co/")
            assert "/gguf/" in item["sources"]["kaggle"]
            assert item["sources"]["torrent"] == "placeholder"
            assert len(item["sha256"]) == 64
    summary = result["summary"]
    assert isinstance(summary, str)
    assert "sha256=" in summary
    manifest = layout.root / "manifests" / "2026-10-10-a100.json"
    assert manifest.is_file()
    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["models"][0]["files"]
    instructions = result["manifest"]["instructions"]
    assert isinstance(instructions, str)
    assert "Copy these files back" in instructions
    assert str(manifest) in instructions
    progress = (layout.root / "progress.log").read_text(encoding="utf-8")
    assert "tiny" in progress and "upload" in progress and "%" in progress


def test_mismatch_does_not_mark_upload(tmp_path: Path, llama_bin: None) -> None:
    del llama_bin
    source = b"SRC" * 40
    reuse = b"RE" * 80
    plan = load_plan(_plan(tmp_path, source, reuse))
    downloader = FakeDownloader({"source.gguf": source, "q5.gguf": reuse})
    remote = FakeRemote(failures=50)
    layout = Layout(tmp_path / "forge-out")
    with pytest.raises(StageError):
        run_model(
            plan,
            plan.models[0],
            _host(downloader, remote),
            _scratch_repo(tmp_path),
            layout,
            stop_after="upload",
        )
    assert not layout.stage("tiny", "upload").is_file()
    quants = list((layout.model_dir("tiny") / "quants").rglob("*.gguf"))
    assert quants
    assert not (layout.model_dir("tiny") / "source").exists()


def test_failed_quantize_removes_the_partial_gguf(tmp_path: Path, llama_bin: None) -> None:
    del llama_bin
    dest = tmp_path / "quants" / "Q4_K_M.gguf"

    class Boom(Runner):
        def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
            del cwd
            out = Path(argv[-2])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"partial-gguf")
            raise StageError("llama-quantize exited 1", code="tool_failed")

    host = _host(FakeDownloader({}), FakeRemote())
    host.runner = Boom()
    with pytest.raises(StageError):
        _run_quantize(host, tmp_path / "src.gguf", tmp_path / "im.gguf", dest, "Q4_K_M")
    assert not dest.exists()


def test_resume_does_not_download_again(tmp_path: Path, llama_bin: None) -> None:
    del llama_bin
    source = b"SRC" * 40
    reuse = b"RE" * 80
    plan = load_plan(_plan(tmp_path, source, reuse))
    downloader = FakeDownloader({"source.gguf": source, "q5.gguf": reuse})
    remote = FakeRemote()
    layout = Layout(tmp_path / "forge-out")
    host = _host(downloader, remote)
    repo = _scratch_repo(tmp_path)
    run_model(plan, plan.models[0], host, repo, layout, stop_after="upload")
    calls = len(downloader.calls)
    run_model(plan, plan.models[0], host, repo, layout, stop_after="upload")
    assert len(downloader.calls) == calls


def test_optional_model_skips_when_time_is_gone(tmp_path: Path, llama_bin: None) -> None:
    del llama_bin
    source = b"SRC" * 20
    reuse = b"RE" * 20
    plan = load_plan(_plan(tmp_path, source, reuse, optional_id="later"))
    downloader = FakeDownloader({"source.gguf": source, "q5.gguf": reuse, "opt.gguf": source})
    remote = FakeRemote()
    ticks = {"n": 0}

    def now() -> float:
        ticks["n"] += 1
        return 0.0 if ticks["n"] == 1 else 10_000.0

    layout = Layout(tmp_path / "forge-out")
    result = run_all(
        plan,
        _host(downloader, remote, now=now),
        _scratch_repo(tmp_path),
        layout,
    )
    skipped = [item for item in result["models"] if isinstance(item, dict) and item.get("skipped")]
    assert skipped and skipped[0]["id"] == "later"
    assert "opt.gguf" not in downloader.calls


def test_ablate_stage_uses_injected_ablator(tmp_path: Path, llama_bin: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del llama_bin
    src = tmp_path / "llama-src"
    src.mkdir()
    (src / "convert_hf_to_gguf.py").write_text("# test\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_LLAMA_SRC", str(src))
    payload = b"SAFE" * 30
    text = f"""
[run]
name = "ablate"
output_dir = "{tmp_path.as_posix()}/work"
wall_clock_budget_min = 30
headroom_gb = 3.0
contexts = [8192]
artifact_host = ""
[[models]]
id = "abl"
enabled = true
mode = "ablate"
repo = "org/abl"
revision = "rev"
licence = "apache-2.0"
source_format = "safetensors"
credits = ["Org"]
base_model = "org/base"
[models.arch]
full_layers = 1
full_kv_heads = 1
full_head_dim = 4
[[models.files]]
role = "source"
quant = "BF16"
path = "model.safetensors"
sha256 = "{_sha(payload)}"
size = {len(payload)}
[[models.tiers]]
name = "16gb"
budget_gb = 16
kind = "quantize"
quants = ["Q4_K_M"]
"""
    path = tmp_path / "plan.toml"
    path.write_text(text, encoding="utf-8")
    plan = load_plan(path)
    seen: list[str] = []

    def ablate(source: Path, dest: Path, weight_bytes: int, free_vram: int | None) -> dict[str, object]:
        del weight_bytes, free_vram
        seen.append(str(source))
        dest.mkdir(parents=True, exist_ok=True)
        return {"layer": 0, "matrices": 1, "mode": "layer", "direction_norm": 1.0}

    downloader = FakeDownloader({"model.safetensors": payload})
    remote = FakeRemote()
    layout = Layout(tmp_path / "forge-out")
    result = run_model(
        plan,
        plan.models[0],
        _host(downloader, remote, ablate=ablate),
        _scratch_repo(tmp_path),
        layout,
        stop_after="upload",
    )
    assert seen
    assert result["id"] == "abl"


def test_disk_short(tmp_path: Path, llama_bin: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del llama_bin
    source = b"SRC" * 10
    reuse = b"RE" * 10
    plan = load_plan(_plan(tmp_path, source, reuse))
    monkeypatch.setattr("voxprint_forge.pipeline.ensure_free", _boom)
    layout = Layout(tmp_path / "forge-out")
    with pytest.raises(DiskShortError):
        run_model(
            plan,
            plan.models[0],
            _host(FakeDownloader({}), FakeRemote()),
            _scratch_repo(tmp_path),
            layout,
            stop_after="fetch",
        )


def _boom(path: Path, need: int, what: str) -> int:
    del path, need, what
    raise DiskShortError("short", code="disk_short")


def _ample_disk(path: Path) -> int:
    del path
    return 200 * 1024**3


def test_token_is_scrubbed_from_upload_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "super-secret-token-value")
    message = scrub("remote said super-secret-token-value")
    assert "super-secret-token-value" not in message
    assert "***" in message


def test_remote_protocol_accepts_fake() -> None:
    remote: Remote = FakeRemote()
    assert remote.check_hf() == "tester"


def test_fetch_checksum(tmp_path: Path) -> None:
    from voxprint_forge.errors import ChecksumError
    from voxprint_forge.fetch import ensure_pinned
    from voxprint_forge.plan import PinnedFile

    pinned = PinnedFile("source", "F16", "a.gguf", "ab" * 32, 4, "org/m", "rev")
    downloader = FakeDownloader({"a.gguf": b"nope"})
    with pytest.raises(ChecksumError):
        ensure_pinned(downloader, pinned, tmp_path / "a.gguf")
    assert not (tmp_path / "a.gguf").exists()


def test_vm_run_invocation_dry_runs_a_tiny_model(
    tmp_path: Path, llama_bin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del llama_bin
    script = (ROOT / "scripts/vm_run.sh").read_text(encoding="utf-8")
    command = '"$VENV/bin/forge" --plan "$ROOT/plans/plan.toml" --root "$ROOT" --forge-out "$FORGE_OUT" run-all'
    assert command in script
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    forge = venv / "forge"
    forge.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FORGE_OUT/argv.txt"\n', encoding="utf-8")
    forge.chmod(0o755)
    captured = tmp_path / "captured"
    captured.mkdir()
    env = os.environ.copy()
    env["FORGE_OUT"] = captured.as_posix()
    env["FORGE_VENV"] = (tmp_path / "venv").as_posix()
    subprocess.run(_bash(ROOT / "scripts/vm_run.sh"), check=True, env=env, cwd=ROOT)
    argv = (captured / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv[-1] == "run-all"
    assert argv.index("--plan") < argv.index("run-all")
    assert argv.index("--root") < argv.index("run-all")
    assert argv.index("--forge-out") < argv.index("run-all")

    source = b"SRC" * 40
    reuse = b"RE" * 80
    plan_path = _plan(tmp_path, source, reuse)
    repo = _scratch_repo(tmp_path)
    host = _host(FakeDownloader({"source.gguf": source, "q5.gguf": reuse}), FakeRemote())
    monkeypatch.setattr("voxprint_forge.cli._live_host", lambda: host)
    from voxprint_forge.cli import main

    rejected = main(
        ["run-all", "--plan", str(plan_path), "--root", str(repo), "--forge-out", str(tmp_path / "bad")]
    )
    assert rejected == 2
    code = main(
        [
            "--plan",
            str(plan_path),
            "--root",
            str(repo),
            "--forge-out",
            str(tmp_path / "good"),
            "run-all",
        ]
    )
    assert code == 0
    assert (tmp_path / "good" / "stages" / "tiny" / "upload.json").is_file()


def test_cli_plan_summary_and_usage() -> None:
    from voxprint_forge.cli import main

    code = main(["--root", str(ROOT), "--json", "plan-summary"])
    assert code == 0
    code = main(["--root", str(ROOT), "fit"])
    assert code == 2
    assert os.environ.get("HF_TOKEN", "") != "super-secret-token-value"
