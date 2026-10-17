"""``forge preflight --vm`` with a mocked machine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxprint_forge import cli, vmcheck
from voxprint_forge.vmcheck import Probe, run_vm_checks

GOOD = {
    "torch": "2.14.1+cu130",
    "torchmetrics": "1.6.3",
    "diffusers": "0.37.1",
    "voxprint-common": "0.1.1",
}


def fake_root(tmp_path: Path, *, llama: str | None = "abc") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "llama.rev").write_text("abc\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["torch>=2.5", "voxprint-common>=0.1.1,<0.2"]\n'
        '[project.optional-dependencies]\ncompare = ["diffusers>=0.37.1,<0.38", "torchmetrics>=1.6"]\n'
    )
    (tmp_path / "vendor").mkdir()
    (tmp_path / vmcheck.WHEEL).write_bytes(b"wheel")
    if llama is not None:
        (tmp_path / "third_party" / "llama.cpp").mkdir(parents=True)
        (tmp_path / "third_party" / "llama.cpp" / "CMakeLists.txt").write_text("")
        (tmp_path / "third_party" / "LLAMA_COMMIT").write_text(llama + "\n")
    return tmp_path


def probe(**overrides) -> Probe:
    versions = {**GOOD, **overrides.pop("versions", {})}
    values = dict(
        which=lambda name: f"/usr/bin/{name}",
        env={"HF_TOKEN": "hf_secret_value", "CUDACXX": "/usr/local/cuda/bin/nvcc"},
        version=versions.get,
        run=lambda argv: "590.48.01\n",
        torch_info=lambda: {"version": "2.14.1+cu130", "cuda": "13.0", "available": True},
        free_gb=lambda path: 500.0,
        python=(3, 14),
        executable=lambda path: True,
    )
    values.update(overrides)
    return Probe(**values)


def by_name(report):
    return {item["name"]: item for item in report["checks"]}


def test_all_pass(tmp_path) -> None:
    report = run_vm_checks(fake_root(tmp_path), tmp_path, probe())
    assert report["ok"], report["text"]
    assert all(line.startswith("PASS ") for line in report["text"].splitlines())
    assert "hf_secret_value" not in report["text"] and "hf_secret_value" not in vmcheck.to_json(report)


def test_old_torchmetrics_fails_with_hint(tmp_path) -> None:
    report = run_vm_checks(fake_root(tmp_path), tmp_path, probe(versions={"torchmetrics": "1.0.3"}))
    line = by_name(report)["requirements[compare]"]
    assert not report["ok"] and "torchmetrics 1.0.3 (needs >=1.6)" in line["detail"]
    assert "FAIL requirements[compare]" in report["text"] and "--extra-index-url" in line["fix"]


def test_torch_against_driver(tmp_path) -> None:
    root = fake_root(tmp_path)
    old = by_name(run_vm_checks(root, tmp_path, probe(run=lambda argv: "570.10\n")))["torch"]
    assert not old["ok"] and "too old" in old["fix"]
    cpu = by_name(
        run_vm_checks(root, tmp_path, probe(torch_info=lambda: {"version": "2.14.1", "cuda": None, "available": False}))
    )["torch"]
    assert not cpu["ok"] and "CPU build" in cpu["fix"]
    gone = by_name(run_vm_checks(root, tmp_path, probe(torch_info=lambda: {"error": "no module"})))["torch"]
    assert not gone["ok"]


def test_missing_tools_fail(tmp_path) -> None:
    report = run_vm_checks(
        fake_root(tmp_path),
        tmp_path,
        probe(which=lambda name: None, env={"HF_TOKEN": "x"}, executable=lambda path: False),
    )
    checks = by_name(report)
    for name in ("cmake", "ninja", "nvcc", "c++"):
        assert not checks[name]["ok"], name
    assert "CUDACXX" in checks["nvcc"]["fix"] and "build-essential" in checks["c++"]["fix"]


def test_llama_wheel_token_disk(tmp_path) -> None:
    report = run_vm_checks(
        fake_root(tmp_path, llama="def"),
        tmp_path,
        probe(env={"CUDACXX": "/x/nvcc"}, versions={"voxprint-common": None}, free_gb=lambda path: 20.0),
    )
    checks = by_name(report)
    assert not checks["llama.cpp"]["ok"] and not checks["voxprint_common"]["ok"]
    assert checks["hf_token"]["detail"] == "missing" and not checks["disk"]["ok"]
    assert not by_name(run_vm_checks(fake_root(tmp_path / "n", llama=None), tmp_path, probe()))["llama.cpp"]["ok"]


def test_python_version(tmp_path) -> None:
    assert not by_name(run_vm_checks(fake_root(tmp_path), tmp_path, probe(python=(3, 11))))["python"]["ok"]


@pytest.mark.parametrize("json_mode", [False, True])
def test_cli_exit_code_and_output(tmp_path, capsys, monkeypatch, json_mode) -> None:
    root = fake_root(tmp_path)
    failing = probe(versions={"torchmetrics": "1.0.3"})
    monkeypatch.setattr(vmcheck, "Probe", lambda: failing)
    argv = (["--json"] if json_mode else []) + ["--root", str(root), "--forge-out", str(tmp_path), "preflight", "--vm"]
    code = cli.main(argv)
    out = capsys.readouterr().out
    assert code == 6
    if json_mode:
        assert json.loads(out)["result"]["ok"] is False
    else:
        assert "FAIL requirements[compare]" in out and out.count("PASS ") >= 5
