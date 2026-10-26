"""MOSS FP8 DiT: torchao weight-only storage, disk cache, and the unpublished mirror."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from voxprint_forge.diffusion import moss
from voxprint_forge.diffusion.moss_fp8 import (
    APACHE_SHA256,
    PINS,
    QUANT_ID,
    TORCHAO_VERSION,
    ZERO_SHA,
    Fp8Pin,
    SourceFile,
    _load_bf16_transformer,
    bind_device_move,
    cover,
    fetch_mirror,
    file_sha256,
    materialize,
    quantize_module,
    status,
    upload_steps,
    write_bundle,
    write_templates,
)
from voxprint_forge.lora.bases import MossImageAdapter
from voxprint_forge.lora.layers import inject
from voxprint_forge.lora.presets import TrainConfig, build_config
from voxprint_forge.mdl import check_document
from voxprint_forge.mdl_adapter import execute_verify, installed_model_ids

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "manifests" / "moss-fp8"


def _linear() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)


def _quantize() -> nn.Linear:
    layer = _linear()
    quantize_module(layer)
    return layer


def _fp8(param: nn.Parameter) -> object:
    if hasattr(param, "qdata"):
        return param
    return param.data


def _produce() -> nn.Linear:
    return _quantize()


def _factory() -> nn.Linear:
    return nn.Linear(8, 8, bias=False)


def test_fp8_pins_match_the_bf16_snapshots() -> None:
    base, sft = PINS
    assert base.source_repo == moss.REPOS["base"].repo
    assert base.source_revision == moss.REPOS["base"].revision
    assert base.source_mirror == moss.REPOS["base"].mirror
    assert sft.source_repo == moss.REPOS["sft"].repo
    assert sft.source_revision == moss.REPOS["sft"].revision
    assert sft.source_mirror == moss.REPOS["sft"].mirror
    assert base.source_revision == "1ceede80badddd6a7812069619edaa5633587192"
    assert sft.source_revision == "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"
    assert moss.LICENSE_SHA256 == APACHE_SHA256
    assert all(not pin.published for pin in PINS)
    assert all(pin.mirror_revision == "unpublished" and pin.weight_sha256 == ZERO_SHA for pin in PINS)


def test_lora_grads_flow_through_the_fp8_linear() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.proj(hidden)

    torch.manual_seed(0)
    model = Tiny().to(dtype=torch.bfloat16)
    quantize_module(model)
    stored = _fp8(model.proj.weight)
    assert model.proj.weight.dtype == torch.bfloat16
    assert stored.qdata.dtype == torch.float8_e4m3fn
    assert not model.proj.weight.requires_grad
    layers = inject(model, ["proj"], 2, 2)
    torch.nn.init.normal_(layers["proj"].lora_B.weight)
    loss = model(torch.randn(2, 8, dtype=torch.bfloat16)).float().sum()
    loss.backward()
    assert layers["proj"].lora_A.weight.grad is not None
    assert layers["proj"].lora_B.weight.grad is not None
    assert float(layers["proj"].lora_A.weight.grad.abs().sum()) > 0
    assert float(layers["proj"].lora_B.weight.grad.abs().sum()) > 0
    assert layers["proj"].base.weight.grad is None
    bind_device_move(model)
    model.to("cpu")
    assert _fp8(layers["proj"].weight).qdata.dtype == torch.float8_e4m3fn


def test_saved_file_reloads_without_quantizing_again(tmp_path: Path) -> None:
    pin = PINS[0]
    made: list[int] = []

    def produce() -> nn.Linear:
        made.append(1)
        return _produce()

    first, action = materialize(tmp_path, pin, factory=_factory, produce=produce)
    assert action == "quantize" and made == [1]
    second, again = materialize(tmp_path, pin, factory=_factory, produce=produce)
    assert again == "cache" and made == [1]
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.allclose(first(hidden), second(hidden))
    assert _fp8(second.weight).qdata.dtype == torch.float8_e4m3fn
    assert status(tmp_path, pin) == "ok"


def test_unpublished_mirror_is_not_downloaded(tmp_path: Path) -> None:
    def download() -> bool:
        raise AssertionError("mirror download")

    _module, action = materialize(tmp_path, PINS[0], factory=_factory, produce=_produce, download=download)
    assert action == "quantize"
    assert fetch_mirror(PINS[0], tmp_path / "mirror") is False


def test_torchao_flag_change_quantizes_again(tmp_path: Path) -> None:
    pin = PINS[1]
    made: list[int] = []

    def produce() -> nn.Linear:
        made.append(1)
        return _produce()

    materialize(tmp_path, pin, factory=_factory, produce=produce)
    manifest = tmp_path / "voxprint-model.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["flags"] = [flag.replace("torchao=0.18.0", "torchao=9.9.9") for flag in document["flags"]]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    assert status(tmp_path, pin) == "stale"
    _module, action = materialize(tmp_path, pin, factory=_factory, produce=produce)
    assert action == "quantize" and made == [1, 1]
    assert status(tmp_path, pin) == "ok"


def test_corrupt_weight_is_repaired_without_a_download(tmp_path: Path) -> None:
    pin = PINS[0]
    made: list[int] = []

    def produce() -> nn.Linear:
        made.append(1)
        return _produce()

    def download() -> bool:
        raise AssertionError("mirror download")

    materialize(tmp_path, pin, factory=_factory, produce=produce)
    weight = tmp_path / "dit-fp8.safetensors"
    blob = bytearray(weight.read_bytes())
    blob[-1] ^= 0xFF
    weight.write_bytes(blob)
    assert status(tmp_path, pin) == "corrupt"
    _module, action = materialize(tmp_path, pin, factory=_factory, produce=produce, download=download)
    assert action == "quantize" and made == [1, 1]
    assert status(tmp_path, pin) == "ok"
    document = json.loads((tmp_path / "voxprint-model.json").read_text(encoding="utf-8"))
    row = next(item for item in document["files"] if item["path"] == "dit-fp8.safetensors")
    assert row["sha256"] == file_sha256(weight)
    assert row["sha256"] != ZERO_SHA


def test_published_mirror_falls_back_when_the_download_fails(tmp_path: Path) -> None:
    pin = replace(PINS[0], mirror_revision="ab" * 20, weight_sha256="cd" * 32, weight_size=4)
    assert pin.published
    calls: list[str] = []

    def download() -> bool:
        calls.append("download")
        raise RuntimeError("mirror not reachable")

    def produce() -> nn.Linear:
        calls.append("quantize")
        return _produce()

    _module, action = materialize(tmp_path, pin, factory=_factory, produce=produce, download=download)
    assert action == "quantize"
    assert calls == ["download", "quantize"]
    assert status(tmp_path, pin) == "ok"
    flags = json.loads((tmp_path / "voxprint-model.json").read_text(encoding="utf-8"))["flags"]
    assert "fallback-used=1" in flags


def test_16_gb_plan_is_under_the_bf16_peak_and_the_14_gb_cap() -> None:
    adapter = MossImageAdapter()
    plan24 = adapter.memory_plan(build_config("balanced", "balanced", 24))
    plan16 = adapter.memory_plan(build_config("balanced", "balanced", 16))
    assert plan24.peak_gb == pytest.approx(11.8, abs=0.05)
    assert plan16.transformer_gb < 5.5
    assert plan16.transformer_gb < plan24.transformer_gb / 2 + 0.3
    assert plan16.peak_gb < 14
    assert plan16.peak_gb < plan24.peak_gb
    assert plan16.encoders_gb > 7
    old = TrainConfig.from_dict(
        {"vram": 16, "rank": 32, "resolution": 1024, "batch": 1, "grad_checkpointing": True, "grad_accum": 2}
    )
    assert old.base_fp8 is False
    assert adapter.memory_plan(old).transformer_gb == pytest.approx(plan16.transformer_gb)


def test_template_records_the_source_pin_and_matches_a_fresh_write(tmp_path: Path) -> None:
    write_templates(tmp_path)
    for pin in PINS:
        for name in ("README.md", "LICENSE", "NOTICE", "voxprint-model.json"):
            fresh = (tmp_path / pin.mdl_id / name).read_bytes()
            assert fresh == (TEMPLATE / pin.mdl_id / name).read_bytes()
        document = json.loads((TEMPLATE / pin.mdl_id / "voxprint-model.json").read_text(encoding="utf-8"))
        check_document(document)
        flags = document["flags"]
        assert f"source-revision={pin.source_revision}" in flags
        assert f"torchao={TORCHAO_VERSION}" in flags
        assert f"quant={QUANT_ID}" in flags
        assert "modified" in flags and "derivative" in flags
        assert "fallback=quantize-bf16" in flags
        assert any(flag.startswith("source-sha=transformer/config.json:") for flag in flags)
        assert document["mirrors"][0]["ref"] == pin.mirror_repo
        assert document["mirrors"][0]["revision"] == "unpublished"
        assert document["mirrors"][0]["ours"] is True
        assert document["license"] == "apache-2.0"
        card = (TEMPLATE / pin.mdl_id / "README.md").read_text(encoding="utf-8")
        assert "modified" in card.lower() or "Modified" in card
        assert pin.source_revision in card
        assert TORCHAO_VERSION in card
        assert "pending" in card
        notice = (TEMPLATE / pin.mdl_id / "NOTICE").read_text(encoding="utf-8")
        assert "modified" in notice.lower()


def test_publish_script_prints_upload_steps_and_does_not_upload(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "publish_moss_fp8.py"), "--out", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Nothing was uploaded" in completed.stdout
    assert "huggingface-cli repo create Mitroshenkov87/voxprint-mirror-moss-image-base-fp8" in completed.stdout
    assert "huggingface-cli upload" in completed.stdout
    assert "fall back" in completed.stdout
    assert (tmp_path / "moss-image-sft-fp8" / "voxprint-model.json").is_file()
    refused = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "publish_moss_fp8.py"), "--out", str(tmp_path), "--upload"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert refused.returncode == 2
    assert "never uploads" in refused.stderr
    source = (ROOT / "scripts" / "publish_moss_fp8.py").read_text(encoding="utf-8")
    assert "HfApi" not in source
    assert "upload_file" not in source
    assert "Nothing was uploaded" in upload_steps()


def test_repair_regenerates_from_the_bf16_source_and_skips_the_downloader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"bf16-source"
    digest = hashlib.sha256(payload).hexdigest()
    pin = replace(PINS[0], sources=(SourceFile("weights.bin", len(payload), digest),))
    root = tmp_path / "models"
    source = root / "model-forge" / pin.source_mdl
    source.mkdir(parents=True)
    (source / "weights.bin").write_bytes(payload)
    built = root / "model-forge" / pin.mdl_id
    write_bundle(built, pin, weight_sha=ZERO_SHA, weight_size=0)
    # model-forge is the app dir, so the walker skips it. A plug-in copy is what
    # would be handed to the upstream downloader; that id has to be filtered out.
    plugin = root / "plugins" / pin.mdl_id
    plugin.mkdir(parents=True)
    (plugin / "voxprint-model.json").write_bytes((built / "voxprint-model.json").read_bytes())
    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.PINS", (pin,))
    produced: list[Path] = []

    def fake_quant(folder: Path, chosen: Fp8Pin, *, verify_source: bool = True) -> nn.Linear:
        assert verify_source is False
        produced.append(folder)
        return _produce()

    downloads: list[str] = []
    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.quantize_snapshot", fake_quant)
    monkeypatch.setattr(
        "voxprint_forge.diffusion.moss_fp8.fetch_mirror",
        lambda *_args, **_kwargs: downloads.append("download") or True,
    )
    manifest = tmp_path / "mdl.json"
    manifest.write_text(json.dumps({"models": []}), encoding="utf-8")
    toy = root / "plugins" / "gemma4-31b"
    toy.mkdir(parents=True)
    (toy / "voxprint-model.json").write_text(
        (ROOT / "tests" / "fixtures" / "descriptors" / "voxprint-gemma4-31b" / "voxprint-model.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    assert pin.mdl_id in installed_model_ids(manifest, root=root)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool = False, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert check is False
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("voxprint_forge.mdl_adapter.mdl_script", lambda: ["voxprint-mdl"])
    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fake_run)
    result = execute_verify(manifest=manifest, repair=True, gui=True, root=root)
    assert calls
    command = calls[0]
    assert command[0] == "voxprint-mdl"
    assert command[1:4] == ["verify", "--app", "model-forge"]
    assert command[-2:] == ["--gui", "--repair"]
    assert pin.mdl_id not in command
    assert downloads == []
    assert produced == [source]
    assert {"id": pin.mdl_id, "status": "repaired"} in result["models"]
    assert result["exit_code"] == 0


def test_repair_skips_a_source_that_does_not_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "voxprint_forge.diffusion.moss_fp8.quantize_snapshot",
        lambda *_args, **_kwargs: pytest.fail("quantized a mismatched source"),
    )
    root = tmp_path / "models" / "model-forge"
    source = root / "moss-image-base" / "transformer"
    source.mkdir(parents=True)
    (source / "config.json").write_bytes(b"not-the-pinned-config")
    (root / "moss-image-base-fp8").mkdir()
    rows = cover(tmp_path / "models", repair=True)
    assert rows == [{"id": "moss-image-base-fp8", "status": "corrupt", "detail": "source mismatch"}]
    absent = tmp_path / "empty"
    (absent / "model-forge" / "moss-image-sft").mkdir(parents=True)
    (absent / "model-forge" / "moss-image-sft-fp8").mkdir()
    missing = cover(absent, repair=True)
    assert missing == [{"id": "moss-image-sft-fp8", "status": "missing", "detail": "source missing"}]
    assert cover(tmp_path / "nowhere", repair=True) == []


def test_a_bf16_snapshot_is_not_a_missing_fp8_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "voxprint_forge.diffusion.moss_fp8.quantize_snapshot",
        lambda *_args, **_kwargs: pytest.fail("quantized a bf16-only install"),
    )
    root = tmp_path / "model-forge" / "moss-image-base"
    root.mkdir(parents=True)
    (root / "transformer").mkdir()
    assert cover(tmp_path, repair=False) == []
    assert cover(tmp_path, repair=True) == []


def test_bf16_load_does_not_leave_the_hub_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    seen: dict[str, str | None] = {}

    class Transformer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            seen["hub"] = os.environ.get("HF_HUB_OFFLINE")
            module = SimpleNamespace()
            module.eval = lambda: module
            return module

    monkeypatch.setattr(
        "voxprint_forge.diffusion.moss.import_vendored",
        lambda: SimpleNamespace(MOSSImageTransformer=Transformer),
    )
    _load_bf16_transformer(tmp_path)
    assert seen["hub"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
