"""MOSS pre-quant build with stand-ins. No downloads and no real weights."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import torch

from tests.lora_fakes import TinyNet
from voxprint_forge.diffusion.moss_prequant import (
    DEFAULT_BUDGET_GB,
    MODEL_ID,
    VRAM_CAP_ENV,
    WEIGHTS_DIR,
    NoQuantFits,
    PinnedFile,
    PromptCase,
    Quality,
    Services,
    SnapshotSet,
    check_budget,
    default_pins,
    quantization_spec,
    run,
    verify_snapshots,
)
from voxprint_forge.diffusion.quant_select import Gpu, known_gpu
from voxprint_forge.errors import ChecksumError, StageError, UsageError
from voxprint_forge.lora.presets import MOSS_VRAM_BUDGET_GB
from voxprint_forge.mdl import check_document, schema_path

CASES = (
    PromptCase("a red cup on a wooden table", 0),
    PromptCase("a stone bridge over a canal", 1),
)
QUALITY = {
    "fp8_e4m3": Quality(0.02, 0.99),
    "int8": Quality(0.05, 0.97),
    "nf4": Quality(0.08, 0.90),
}
SCHEMA = json.loads(schema_path().read_text(encoding="utf-8"))


class _Loader:
    def __init__(self) -> None:
        self.opened = False

    def open(self, snapshots: SnapshotSet) -> None:
        self.opened = True


class _Quantizer:
    def __init__(self, *, corrupt: bool = False) -> None:
        self.built: list[str] = []
        self.corrupt = corrupt
        self.reloaded = False
        self.trained: list[str] = []

    def config(self, mode: str) -> dict[str, object]:
        return quantization_spec(mode)

    def build(self, mode: str, snapshots: SnapshotSet, dest: Path) -> None:
        self.built.append(mode)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "weights.safetensors").write_bytes(mode.encode())
        net = TinyNet()
        loss = net(torch.zeros(1, 16)).sum()
        loss.backward()

    def reload(self, dest: Path) -> None:
        self.reloaded = True
        if self.corrupt:
            (dest / "weights.safetensors").write_bytes(b"corrupt")

    def train_probe(self, dest: Path) -> None:
        self.trained.append(dest.name)
        net = TinyNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        loss = net(torch.zeros(1, 16)).sum()
        loss.backward()
        opt.step()


class _Renderer:
    def __init__(self, *, unstable: bool = False) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.unstable = unstable

    def render_reference(self, snapshots: SnapshotSet, prompt: str, seed: int) -> bytes:
        self.calls.append(("reference", prompt, seed))
        return f"ref|{prompt}|{seed}".encode()

    def render(self, weights: Path, prompt: str, seed: int, *, lora: bool) -> bytes:
        mode = (weights / "weights.safetensors").read_bytes().decode()
        self.calls.append(("candidate", prompt, seed, lora, mode))
        if self.unstable and weights.name == WEIGHTS_DIR:
            return b"different"
        return f"{mode}|{prompt}|{seed}|{int(lora)}".encode()


class _Scorer:
    def score(self, reference: bytes, candidate: bytes, prompt: str) -> Quality:
        del reference, prompt
        mode = candidate.decode().split("|", 1)[0]
        return QUALITY[mode]


class _Probe:
    def __init__(self, gpu: Gpu, peaks: dict[str, tuple[float, float]]) -> None:
        self._gpu = gpu
        self.peaks = peaks
        self.caps: list[tuple[float, str | None]] = []
        self.measured: list[tuple[str, str]] = []

    def gpu(self) -> Gpu:
        return self._gpu

    def cap(self, budget_gb: float) -> None:
        self.caps.append((budget_gb, os.environ.get(VRAM_CAP_ENV)))

    def measure(self, mode: str, kind: str, fn: object) -> float:
        fn()  # type: ignore[operator]
        self.measured.append((mode, kind))
        generate, train = self.peaks[mode]
        return generate if kind == "generate" else train


def _snapshots(tmp: Path) -> tuple[SnapshotSet, dict[str, tuple[PinnedFile, ...]]]:
    pins: dict[str, tuple[PinnedFile, ...]] = {}
    roots: dict[str, Path] = {}
    tmp.mkdir(parents=True, exist_ok=True)
    for role in ("sft", "encoder"):
        folder = tmp / role
        folder.mkdir()
        payload = f"{role}-bytes".encode()
        (folder / "weights.bin").write_bytes(payload)
        pins[role] = (PinnedFile("weights.bin", hashlib.sha256(payload).hexdigest(), len(payload)),)
        roots[role] = folder
    return SnapshotSet(roots["sft"], roots["encoder"]), pins


def _services(
    gpu: Gpu,
    peaks: dict[str, tuple[float, float]],
    *,
    corrupt: bool = False,
    unstable: bool = False,
) -> tuple[Services, _Quantizer, _Renderer, _Probe, _Loader]:
    loader = _Loader()
    quantizer = _Quantizer(corrupt=corrupt)
    renderer = _Renderer(unstable=unstable)
    probe = _Probe(gpu, peaks)
    services = Services(loader, quantizer, renderer, _Scorer(), probe)
    return services, quantizer, renderer, probe, loader


def _fit() -> dict[str, tuple[float, float]]:
    return {"fp8_e4m3": (12.0, 12.4), "int8": (11.0, 11.5), "nf4": (9.0, 9.4)}


def test_default_budget_constant_is_13_5() -> None:
    assert MOSS_VRAM_BUDGET_GB == 13.5
    assert DEFAULT_BUDGET_GB == 13.5
    assert check_budget(13) == 13
    assert check_budget(14) == 14
    with pytest.raises(UsageError) as low:
        check_budget(12.9)
    with pytest.raises(UsageError) as high:
        check_budget(14.01)
    assert low.value.code == high.value.code == "moss_prequant_budget"


def test_picks_the_best_quality_inside_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VRAM_CAP_ENV, "1")
    snapshots, pins = _snapshots(tmp_path / "src")
    services, quantizer, renderer, probe, loader = _services(known_gpu("RTX 4090"), _fit())
    result = run(snapshots, tmp_path / "out", pins=pins, cases=CASES, services=services)
    assert result.chosen == "fp8_e4m3"
    assert result.budget_gb == 13.5
    assert quantizer.built == ["fp8_e4m3", "int8", "nf4"]
    assert quantizer.reloaded is True
    assert loader.opened is True
    assert probe.caps == [(13.5, "13.5")]
    assert os.environ[VRAM_CAP_ENV] == "1"
    refs = [(call[1], call[2]) for call in renderer.calls if call[0] == "reference"]
    assert refs == [(case.prompt, case.seed) for case in CASES]
    for mode in quantizer.built:
        got = [
            (call[1], call[2])
            for call in renderer.calls
            if call[0] == "candidate" and call[4] == mode
        ]
        assert got[: len(CASES)] == refs
    document = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert document["schema"] == "voxprint-model/1"
    assert document["id"] == MODEL_ID == "voxprint-moss-image-sft-compact"
    record = document["prequant"]
    assert record["chosen"] == "fp8_e4m3"
    assert record["budget_gb"] == 13.5
    assert record["quant_modes"] == ["fp8_e4m3", "int8", "nf4"]
    assert record["min_sm"] == {"fp8_e4m3": 89, "int8": 70, "nf4": 70}
    assert record["measured_peak_gb"]["fp8_e4m3"] == {"generate": 12.0, "train": 12.4}
    check_document(document)
    jsonschema.validate(document, SCHEMA)
    card = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    for text in ("Apache-2.0", "OpenMOSS", "Qwen", "FLUX.2", "13.5", "fp8_e4m3"):
        assert text in card
    weight = tmp_path / "out" / WEIGHTS_DIR / "weights.safetensors"
    sidecar = Path(str(weight) + ".sha256")
    assert sidecar.read_text(encoding="utf-8").split()[0] == hashlib.sha256(weight.read_bytes()).hexdigest()


def test_fp8_is_not_built_below_sm89(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    peaks = {"int8": (11.0, 12.0), "nf4": (9.5, 9.8)}
    services, quantizer, _renderer, probe, _loader = _services(known_gpu("RTX 3080 Laptop"), peaks)
    result = run(snapshots, tmp_path / "out", sm=86, pins=pins, cases=CASES, services=services)
    assert quantizer.built == ["int8", "nf4"]
    assert all(mode != "fp8_e4m3" for mode, _kind in probe.measured)
    assert result.chosen == "int8"
    assert result.skipped == ("fp8_e4m3",)
    document = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert "fp8_e4m3" not in document["prequant"]["measured_peak_gb"]
    assert "fp8_e4m3" in document["prequant"]["quant_modes"]
    assert "skipped" in (tmp_path / "out" / "README.md").read_text(encoding="utf-8")


def test_3060_keeps_nf4_when_int8_exceeds_12_gb(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    peaks = {"int8": (12.2, 12.4), "nf4": (9.0, 9.2)}
    services, _quantizer, _renderer, _probe, _loader = _services(known_gpu("RTX 3060"), peaks)
    result = run(snapshots, tmp_path / "out", pins=pins, cases=CASES, services=services)
    assert result.chosen == "nf4"
    assert result.gpu.name == "RTX 3060"


def test_none_fit_raises_and_writes_nothing(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    peaks = {name: (13.8, 13.9) for name in ("fp8_e4m3", "int8", "nf4")}
    services, _quantizer, _renderer, _probe, _loader = _services(known_gpu("RTX 4090"), peaks)
    with pytest.raises(NoQuantFits) as caught:
        run(snapshots, tmp_path / "out", pins=pins, cases=CASES, services=services)
    assert caught.value.code == "moss_prequant_none_fit"
    assert not (tmp_path / "out" / "voxprint-model.json").exists()


def test_sha_mismatch_stops_before_quantizing(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    bad = {
        "sft": (PinnedFile("weights.bin", "0" * 64, pins["sft"][0].size),),
        "encoder": pins["encoder"],
    }
    services, quantizer, _renderer, _probe, loader = _services(known_gpu("RTX 4090"), _fit())
    with pytest.raises(ChecksumError) as caught:
        run(snapshots, tmp_path / "out", pins=bad, cases=CASES, services=services)
    assert caught.value.code == "moss_prequant_sha"
    assert quantizer.built == []
    assert loader.opened is False


def test_missing_snapshot_file_stops(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    (snapshots.encoder / "weights.bin").unlink()
    with pytest.raises(StageError) as caught:
        verify_snapshots(snapshots, pins)
    assert caught.value.code == "moss_prequant_missing"


def test_reload_hash_and_render_are_checked(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    services, _quantizer, _renderer, _probe, _loader = _services(known_gpu("RTX 4080"), _fit(), corrupt=True)
    with pytest.raises(StageError) as caught:
        run(snapshots, tmp_path / "out", pins=pins, cases=CASES, services=services)
    assert caught.value.code == "moss_prequant_reload"
    snapshots, pins = _snapshots(tmp_path / "src2")
    services, _quantizer, _renderer, _probe, _loader = _services(known_gpu("RTX 4080"), _fit(), unstable=True)
    with pytest.raises(StageError) as caught:
        run(snapshots, tmp_path / "out2", pins=pins, cases=CASES, services=services)
    assert caught.value.code == "moss_prequant_reload"


def test_budget_override_is_recorded(tmp_path: Path) -> None:
    snapshots, pins = _snapshots(tmp_path / "src")
    services, _quantizer, _renderer, probe, _loader = _services(known_gpu("RTX 4090"), _fit())
    result = run(snapshots, tmp_path / "out", budget_gb=14, pins=pins, cases=CASES, services=services)
    assert result.budget_gb == 14
    assert probe.caps == [(14.0, "14")]
    document = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert document["prequant"]["budget_gb"] == 14
    assert "Budget: 14 GB" in (tmp_path / "out" / "README.md").read_text(encoding="utf-8")


def test_cli_default_budget_and_the_module_does_not_download() -> None:
    from voxprint_forge.cli import build_parser

    args = build_parser().parse_args(["moss-prequant", "--sft", "s", "--encoder", "e", "--out", "o"])
    assert args.command == "moss-prequant"
    assert args.vram_budget_gb == 13.5
    assert not hasattr(args, "base")
    text = Path(quantization_spec.__code__.co_filename).read_text(encoding="utf-8")
    assert "hf_hub_download" not in text
    assert "snapshot_download" not in text
    assert "urlopen" not in text
    assert "local_files_only=True" in text
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in text
    for mode in ("fp8_e4m3", "int8", "nf4"):
        spec = quantization_spec(mode)
        assert spec["mode"] == mode
        assert spec["backend"] in {"torchao", "bitsandbytes"}
    pins = default_pins()
    assert set(pins) == {"sft", "encoder"}
    assert any(item.path == "model-00001-of-00004.safetensors" for item in pins["encoder"])
    assert any(item.sha256.startswith("e40d3444") for item in pins["sft"])
    assert len(pins["sft"]) == 16
    assert len(pins["encoder"]) == 6


def test_import_does_not_pull_ml_libraries() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import voxprint_forge.diffusion.moss_prequant\n"
            "import voxprint_forge.diffusion.quant_select\n"
            "bad = [name for name in ('torch', 'diffusers', 'transformers', 'bitsandbytes') if name in sys.modules]\n"
            "print(','.join(bad))\n",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
