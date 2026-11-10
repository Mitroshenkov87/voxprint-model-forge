"""MOSS pre-quant build with stand-ins. No downloads and no real weights."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import safetensors
import torch
from PIL import Image

import voxprint_forge.diffusion.moss as moss
import voxprint_forge.diffusion.moss_prequant as prequant
from tests.lora_fakes import TinyNet
from voxprint_forge.diffusion.moss import import_vendored
from voxprint_forge.diffusion.moss_prequant import (
    DEFAULT_BUDGET_GB,
    DEFAULT_HEADROOM_BYTES,
    MODEL_ID,
    VRAM_CAP_ENV,
    WEIGHTS_DIR,
    LiveLoader,
    LiveProbe,
    LiveQuantizer,
    LiveRenderer,
    LiveScorer,
    NoQuantFits,
    PinnedFile,
    PromptCase,
    Quality,
    Services,
    ShardInfo,
    SnapshotSet,
    Step,
    TensorInfo,
    check_budget,
    default_pins,
    live_services,
    plan_stream,
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


def _shard(name: str, tensors: tuple[tuple[str, int], ...]) -> ShardInfo:
    return ShardInfo(name, tuple(TensorInfo(key, size, name) for key, size in tensors))


def _tiny_two_layer() -> list[ShardInfo]:
    return [
        _shard(
            "model-00001-of-00002.safetensors",
            (
                ("model.embed_tokens.weight", 1_000_000),
                ("model.layers.0.self_attn.q_proj.weight", 400_000),
                ("model.layers.0.self_attn.k_proj.weight", 400_000),
                ("model.layers.0.mlp.up_proj.weight", 800_000),
            ),
        ),
        _shard(
            "model-00002-of-00002.safetensors",
            (
                ("model.layers.1.self_attn.q_proj.weight", 400_000),
                ("model.layers.1.mlp.down_proj.weight", 800_000),
                ("model.norm.weight", 8_192),
            ),
        ),
    ]


def _split_layer_across_shards() -> list[ShardInfo]:
    return [
        _shard(
            "a.safetensors",
            (
                ("layers.0.attn.q.weight", 50_000),
                ("layers.0.attn.k.weight", 50_000),
                ("layers.1.attn.q.weight", 50_000),
            ),
        ),
        _shard(
            "b.safetensors",
            (
                ("layers.1.attn.k.weight", 50_000),
                ("layers.1.attn.v.weight", 50_000),
                ("final_linear.weight", 12_000),
            ),
        ),
    ]


def _dit_like_three_shards() -> list[ShardInfo]:
    tensors: list[tuple[str, int]] = [
        ("video_embedder.weight", 786_432),
        ("text_embedder.weight", 3_145_728),
    ]
    layer = 240_000_000
    for index in range(42):
        tensors.append((f"layers.{index}.attn.q.weight", layer // 4))
        tensors.append((f"layers.{index}.attn.k.weight", layer // 4))
        tensors.append((f"layers.{index}.attn.v.weight", layer // 4))
        tensors.append((f"layers.{index}.mlp.weight", layer - 3 * (layer // 4)))
    tensors.append(("final_linear_video.weight", 786_432))
    sizes = (3_400_000_000, 3_400_000_000, 3_400_000_000)
    return _pack(tensors, sizes, "diffusion_pytorch_model")


def _qwen35_9b_encoder_shards() -> list[ShardInfo]:
    """Four shards totalling the pinned Qwen3.5-9B encoder (~15.63 GiB)."""
    sizes = (4_973_910_744, 4_988_364_288, 4_954_202_112, 1_869_001_848)
    embed = 1_244_659_712
    norm = 8_192
    n_layers = 32
    rest = sum(sizes) - embed - norm
    per = rest // n_layers
    extra = rest - per * n_layers
    tensors: list[tuple[str, int]] = [("model.embed_tokens.weight", embed)]
    for index in range(n_layers):
        payload = per + (extra if index == n_layers - 1 else 0)
        qkv = payload // 4
        tensors.extend(
            (
                (f"model.layers.{index}.self_attn.q_proj.weight", qkv),
                (f"model.layers.{index}.self_attn.k_proj.weight", qkv),
                (f"model.layers.{index}.self_attn.v_proj.weight", qkv),
                (f"model.layers.{index}.mlp.up_proj.weight", payload - 3 * qkv),
            )
        )
    tensors.append(("model.norm.weight", norm))
    return _pack(tensors, sizes, "model")


def _pack(tensors: list[tuple[str, int]], sizes: tuple[int, ...], stem: str) -> list[ShardInfo]:
    remaining = list(tensors)
    shards: list[ShardInfo] = []
    count = len(sizes)
    for index, target in enumerate(sizes):
        packed: list[tuple[str, int]] = []
        used = 0
        last_shard = index == count - 1
        while remaining and (last_shard or used + remaining[0][1] <= target or not packed):
            if not last_shard and packed and used + remaining[0][1] > target:
                break
            packed.append(remaining.pop(0))
            used += packed[-1][1]
        name = f"{stem}-{index + 1:05d}-of-{count:05d}.safetensors"
        shards.append(_shard(name, tuple(packed)))
    if remaining:
        last = shards[-1]
        extra = tuple(TensorInfo(name, size, last.name) for name, size in remaining)
        shards[-1] = ShardInfo(last.name, last.tensors + extra)
    return shards


def _assert_plan_fits(steps: list[Step], budget: int) -> None:
    assert steps
    for step in steps:
        assert step.largest_alloc <= budget, step
        assert step.peak <= budget, step
        assert step.source_bytes + step.result_bytes <= budget, step
        assert step.largest_alloc <= step.source_bytes
        assert step.peak >= step.source_bytes


@pytest.mark.parametrize(
    "label,shards,budget,ratio,headroom",
    [
        ("tiny", _tiny_two_layer(), 8_000_000, 0.5, 0),
        ("split-layer", _split_layer_across_shards(), 400_000, 0.5, 0),
        ("dit-like", _dit_like_three_shards(), int(13.5 * 1024**3), 0.5, DEFAULT_HEADROOM_BYTES),
        ("9b-encoder", _qwen35_9b_encoder_shards(), int(13.5 * 1024**3), 0.5, DEFAULT_HEADROOM_BYTES),
        ("9b-encoder-nf4", _qwen35_9b_encoder_shards(), int(13.5 * 1024**3), 0.25, DEFAULT_HEADROOM_BYTES),
    ],
)
def test_plan_stream_never_allocates_above_budget(
    label: str,
    shards: list[ShardInfo],
    budget: int,
    ratio: float,
    headroom: int,
) -> None:
    del label
    steps = plan_stream(shards, budget, quant_ratio=ratio, headroom_bytes=headroom)
    _assert_plan_fits(steps, budget)


def test_plan_stream_groups_respect_layer_boundaries() -> None:
    steps = plan_stream(_split_layer_across_shards(), 400_000, quant_ratio=0.5, headroom_bytes=0)
    names = [step.name for step in steps]
    assert names == ["layers.0", "layers.1", "final_linear"]
    assert set(steps[0].tensors) == {"layers.0.attn.q.weight", "layers.0.attn.k.weight"}
    assert set(steps[1].tensors) == {
        "layers.1.attn.q.weight",
        "layers.1.attn.k.weight",
        "layers.1.attn.v.weight",
    }
    assert steps[1].shards == ("a.safetensors", "b.safetensors")
    assert all("layers.0" not in name or step.name == "layers.0" for step in steps for name in step.tensors)


def test_plan_stream_9b_encoder_fits_13_5_and_stays_under_one_block() -> None:
    shards = _qwen35_9b_encoder_shards()
    assert len(shards) == 4
    total = sum(shard.nbytes for shard in shards)
    assert 15 * 1024**3 < total < 17 * 1024**3
    budget = int(13.5 * 1024**3)
    steps = plan_stream(shards, budget, mode="int8")
    _assert_plan_fits(steps, budget)
    assert any(step.name == "model.embed_tokens" for step in steps)
    assert any(step.name.startswith("model.layers.") for step in steps)
    layer_steps = [step for step in steps if step.name.startswith("model.layers.")]
    assert len(layer_steps) == 32
    for step in layer_steps:
        assert all(name.startswith(step.name + ".") for name in step.tensors)
        assert step.source_bytes < budget
        assert step.largest_alloc < budget
    assert max(step.largest_alloc for step in steps) < 2 * 1024**3
    assert max(step.peak for step in steps) <= budget


def test_plan_stream_oversize_single_tensor_raises() -> None:
    shards = [_shard("one.safetensors", (("huge.weight", 16 * 1024**3),))]
    with pytest.raises(NoQuantFits, match="tensor huge.weight"):
        plan_stream(shards, int(13.5 * 1024**3), headroom_bytes=0, quant_ratio=0.5)


def test_plan_stream_oversize_group_plus_result_raises() -> None:
    shards = [
        _shard(
            "one.safetensors",
            (
                ("model.layers.0.q.weight", 3_000),
                ("model.layers.0.k.weight", 3_000),
                ("model.layers.1.q.weight", 3_000),
            ),
        )
    ]
    with pytest.raises(NoQuantFits, match="group"):
        plan_stream(shards, 5_000, headroom_bytes=0, quant_ratio=1.0)


def test_live_code_does_not_warmup_or_map_the_full_model_to_cuda() -> None:
    text = Path(plan_stream.__code__.co_filename).read_text(encoding="utf-8")
    assert "caching_allocator_warmup" in text
    assert "init_empty_weights" in text
    assert "plan_stream" in text
    assert "safe_open" in text
    assert 'device="cpu"' in text
    assert '"device_map": "cpu"' in text
    assert 'device_map="cuda"' not in text
    assert "device_map='cuda'" not in text
    assert '"": "cuda"' not in text
    assert "memory_allocated" in text
    assert "empty_cache" in text
    assert "from_pretrained" in text


def test_moss_venv_installs_pinned_voxprint_common() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "vm_run_moss_fit.sh").read_text(encoding="utf-8")
    pins = (root / "vendor" / "pins.txt").read_text(encoding="utf-8")
    assert "vendor/voxprint_common-0.7.0-py3-none-any.whl" in script
    assert "vendor/pins.txt" in script
    assert "sha256sum -c" in script
    assert "cryptography>=44" in script
    assert "jeepney>=0.8" in script
    assert "voxprint_common-0.7.0-py3-none-any.whl" in pins
    assert "MOSS_VENV" in script
    assert "moss-venv" in script


def _pinned_bytes(prefix: str) -> int:
    return sum(
        item.size
        for item in default_pins()["sft"]
        if item.path.startswith(prefix) and item.path.endswith(".safetensors")
    )


def _targets_cuda(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value == "cuda" or value.startswith("cuda:")
    if isinstance(value, dict):
        return any(_targets_cuda(item) for item in value.values())
    device_type = getattr(value, "type", None)
    return str(device_type) == "cuda" if device_type is not None else False


def _write_shard(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    header = {"embed.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}
    blob = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 8)
    index = {"weight_map": {"embed.weight": "model.safetensors"}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def _one_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class _Picture:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, buffer: object, format: str = "PNG") -> None:
        assert format == "PNG"
        buffer.write(self._payload)  # type: ignore[attr-defined]


class _Skeleton(torch.nn.Module):
    """Stand-in for an unquantized component. ``.to("cuda")`` on this is a full bf16 load."""

    def __init__(self, role: str, nbytes: int) -> None:
        super().__init__()
        self.embed = torch.nn.Parameter(torch.zeros(2, 2))
        self._placement_role = role
        self._placement_bytes = nbytes
        self._placement_quantized = False
        self._placement_skeleton = True
        self._placement_bf16_te = role == "encoder"

    def save_pretrained(self, dest: Path, safe_serialization: bool = True) -> None:
        del safe_serialization
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(
            json.dumps({"latent_channels": 4, "text_dim": 4}),
            encoding="utf-8",
        )


class _FakeDiT(torch.nn.Module):
    def __init__(self, *, quantized: bool, nbytes: int) -> None:
        super().__init__()
        self.video_embedder = torch.nn.Linear(4, 4)
        self._placement_role = "dit"
        self._placement_bytes = 0 if quantized else nbytes
        self._placement_quantized = quantized
        self._placement_skeleton = False
        self._placement_bf16_te = False

    def forward(self, hidden: torch.Tensor, step: torch.Tensor, text: torch.Tensor, mask: torch.Tensor) -> object:
        del step, text, mask
        layer = self.video_embedder
        if hasattr(layer, "lora_A"):
            value = layer.lora_A.weight.float().sum() + layer.lora_B.weight.float().sum()
        else:
            value = layer.weight.float().sum()
        return SimpleNamespace(sample=hidden.float() * 0 + value)


class _Placement:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record_module(self, module: torch.nn.Module) -> None:
        self.events.append(
            {
                "kind": "module",
                "role": getattr(module, "_placement_role", type(module).__name__),
                "cuda": True,
                "quantized": bool(getattr(module, "_placement_quantized", False)),
                "skeleton": bool(getattr(module, "_placement_skeleton", False)),
                "bf16_te": bool(getattr(module, "_placement_bf16_te", False)),
                "nbytes": int(getattr(module, "_placement_bytes", 0)),
            }
        )

    def record_pretrained(self, role: str, device_map: object, quantized: bool) -> None:
        self.events.append(
            {
                "kind": "pretrained",
                "role": role,
                "cuda": _targets_cuda(device_map),
                "quantized": quantized,
                "skeleton": False,
                "bf16_te": role == "text_encoder" and not quantized,
                "nbytes": 0,
            }
        )


@pytest.fixture
def _restore_sys_modules() -> Iterator[None]:
    """Drop every module a test imports lazily so heavy ML libs never leak between tests."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_capped_prequant_run_keeps_full_bf16_off_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _restore_sys_modules: None,
) -> None:
    """Every live ``run()`` step stays off a full bf16 CUDA text encoder or weight load.

    CUDA is reported as available, so the 24 GB plan would otherwise map the bf16
    text encoder onto the card in one block. The reference DiT is still allowed
    on the card: its pinned weights plus the VAE fit the 13.5 GiB cap, and the
    encoder for that step is not on CUDA with them.

    ``accelerate``/``diffusers``/``transformers`` are imported lazily here;
    ``_restore_sys_modules`` rolls them and every transitive import back out of
    ``sys.modules`` so import-isolation guards in other test files keep passing.
    """
    import accelerate
    import diffusers
    from transformers import AutoModel, AutoTokenizer

    te_bytes = _pinned_bytes("text_encoder/")
    dit_bytes = _pinned_bytes("transformer/diffusion_pytorch_model-")
    vae_bytes = _pinned_bytes("vae/")
    budget = int(DEFAULT_BUDGET_GB * 1024**3)
    assert te_bytes > budget
    assert dit_bytes + vae_bytes <= budget

    placement = _Placement()
    steps: list[str] = []
    png = _one_png()
    monkeypatch.delenv("VOXPRINT_FORGE_TE_QUANT", raising=False)
    monkeypatch.delenv("VOXPRINT_FORGE_TE_DEVICE", raising=False)
    monkeypatch.setenv(VRAM_CAP_ENV, "13.5")
    monkeypatch.setattr("voxprint_forge.lora.presets.detect_vram_gb", lambda: 24)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        name = "NVIDIA GeForce RTX 4090"
        major = 8
        minor = 9
        total_memory = 24 * 1024**3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index=0: _Props())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args, **kwargs: 8 * 1024**3)

    original_module_to = torch.nn.Module.to
    original_tensor_to = torch.Tensor.to

    def module_to(self: torch.nn.Module, *args: object, **kwargs: object) -> torch.nn.Module:
        device = kwargs.get("device", args[0] if args else None)
        if _targets_cuda(device):
            placement.record_module(self)
            return self
        return original_module_to(self, *args, **kwargs)

    def tensor_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        device = kwargs.get("device", args[0] if args else None)
        if _targets_cuda(device):
            placement.events.append(
                {
                    "kind": "tensor",
                    "role": "tensor",
                    "cuda": True,
                    "quantized": False,
                    "skeleton": False,
                    "bf16_te": False,
                    "nbytes": int(self.numel() * self.element_size()),
                }
            )
            return self
        return original_tensor_to(self, *args, **kwargs)

    def factory(fn: object) -> object:
        def wrapped(*args: object, **kwargs: object) -> object:
            if _targets_cuda(kwargs.get("device")):
                kwargs["device"] = "cpu"
            return fn(*args, **kwargs)  # type: ignore[operator]

        return wrapped

    monkeypatch.setattr(torch.nn.Module, "to", module_to)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    for name in ("tensor", "zeros", "ones", "empty", "randn", "rand"):
        monkeypatch.setattr(torch, name, factory(getattr(torch, name)))

    def tokenizer_from_pretrained(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    def automodel_from_pretrained(*args: object, **kwargs: object) -> _Skeleton:
        del args
        quantized = kwargs.get("quantization_config") is not None
        placement.record_pretrained("text_encoder", kwargs.get("device_map"), quantized)
        module = _Skeleton("encoder", te_bytes)
        module._placement_quantized = quantized
        module._placement_bf16_te = not quantized
        module._placement_skeleton = False
        return module

    def dit_from_pretrained(*args: object, **kwargs: object) -> _FakeDiT:
        del args
        quantized = kwargs.get("quantization_config") is not None
        placement.record_pretrained("dit", kwargs.get("device_map"), quantized)
        return _FakeDiT(quantized=quantized, nbytes=dit_bytes)

    def vae_from_pretrained(*args: object, **kwargs: object) -> _Skeleton:
        del args
        placement.record_pretrained("vae", kwargs.get("device_map"), False)
        module = _Skeleton("vae", vae_bytes)
        module._placement_bf16_te = False
        module._placement_skeleton = False
        return module

    class _Pipe:
        def __init__(
            self,
            transformer: torch.nn.Module,
            scheduler: object,
            text_encoder: object,
            tokenizer: object,
            vae: object,
        ) -> None:
            del scheduler, tokenizer
            self.transformer = transformer
            self.text_encoder = text_encoder
            self.vae = vae

        def __call__(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return SimpleNamespace(images=[_Picture(png)])

    def load_pipeline(root: Path, **kwargs: object) -> _Pipe:
        del root
        placement.events.append(
            {
                "kind": "pipeline",
                "role": "pipeline",
                "cuda": False,
                "quantized": False,
                "skeleton": False,
                "bf16_te": False,
                "nbytes": 0,
                "with_text_encoder": kwargs.get("with_text_encoder"),
            }
        )
        dit = _FakeDiT(quantized=False, nbytes=dit_bytes)
        vae = _Skeleton("vae", vae_bytes)
        vae._placement_bf16_te = False
        vae._placement_skeleton = False
        return _Pipe(dit, object(), None, None, vae)

    def encode_on_cpu(pipe: object, prompt: str, max_sequence_length: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        del pipe, prompt, max_sequence_length
        return torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)

    class _Shard:
        def __enter__(self) -> _Shard:
            return self

        def __exit__(self, *args: object) -> bool:
            del args
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            del name
            return torch.zeros(2, 2, dtype=torch.bfloat16)

    vendored = import_vendored()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_from_pretrained)
    monkeypatch.setattr(AutoModel, "from_pretrained", automodel_from_pretrained)
    monkeypatch.setattr(vendored.MOSSImageTransformer, "from_pretrained", dit_from_pretrained)
    monkeypatch.setattr(vendored.MOSSImageScheduler, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(vendored, "MOSSImagePipeline", _Pipe)
    monkeypatch.setattr(diffusers.AutoencoderKLFlux2, "from_pretrained", vae_from_pretrained)
    monkeypatch.setattr(moss, "load_pipeline", load_pipeline)
    monkeypatch.setattr(moss, "encode_on_cpu", encode_on_cpu)
    def materialize(mode: str) -> dict[str, dict[str, str]]:
        return {"dit": {"mode": mode}, "encoder": {"mode": mode}}

    def empty_component(source: object, kind: str) -> _Skeleton:
        del source
        nbytes = te_bytes if kind == "encoder" else dit_bytes
        return _Skeleton(kind, nbytes)

    def score_pair(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        return {"lpips": 0.2, "clip_i": 0.8}

    monkeypatch.setattr(prequant, "materialize_quantization_config", materialize)
    monkeypatch.setattr(prequant, "_empty_component", empty_component)
    monkeypatch.setattr(accelerate, "init_empty_weights", nullcontext)
    monkeypatch.setattr(accelerate.utils, "set_module_tensor_to_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: _Shard())
    monkeypatch.setattr("voxprint_forge.compare.score.make_scorer", lambda: score_pair)

    def watch(name: str, fn: object) -> object:
        def wrapped(*args: object, **kwargs: object) -> object:
            steps.append(name)
            return fn(*args, **kwargs)  # type: ignore[operator]

        return wrapped

    monkeypatch.setattr(LiveProbe, "cap", watch("cap", LiveProbe.cap))
    monkeypatch.setattr(LiveLoader, "open", watch("open", LiveLoader.open))
    monkeypatch.setattr(LiveRenderer, "render_reference", watch("render_reference", LiveRenderer.render_reference))
    monkeypatch.setattr(LiveQuantizer, "build", watch("build", LiveQuantizer.build))
    monkeypatch.setattr(LiveRenderer, "render", watch("render", LiveRenderer.render))
    monkeypatch.setattr(LiveQuantizer, "train_probe", watch("train_probe", LiveQuantizer.train_probe))
    monkeypatch.setattr(LiveQuantizer, "reload", watch("reload", LiveQuantizer.reload))
    monkeypatch.setattr(LiveScorer, "score", watch("score", LiveScorer.score))

    sft = tmp_path / "sft"
    encoder = tmp_path / "encoder"
    sft.mkdir()
    encoder.mkdir()
    (sft / "weights.bin").write_bytes(b"sft-bytes")
    (encoder / "weights.bin").write_bytes(b"enc-bytes")
    _write_shard(sft / "transformer")
    _write_shard(encoder)
    pins: dict[str, tuple[PinnedFile, ...]] = {}
    for role, folder in (("sft", sft), ("encoder", encoder)):
        payload = (folder / "weights.bin").read_bytes()
        pins[role] = (PinnedFile("weights.bin", hashlib.sha256(payload).hexdigest(), len(payload)),)
    snapshots = SnapshotSet(sft, encoder)

    run(
        snapshots,
        tmp_path / "out",
        budget_gb=13.5,
        pins=pins,
        cases=(PromptCase("a red cup", 0),),
        services=live_services(snapshots),
    )

    for name in ("cap", "open", "render_reference", "build", "render", "train_probe", "score", "reload"):
        assert name in steps, steps
    assert steps.index("render_reference") < steps.index("build")
    pipelines = [event for event in placement.events if event.get("kind") == "pipeline"]
    assert pipelines and all(event.get("with_text_encoder") is False for event in pipelines)
    bare_te = [
        event
        for event in placement.events
        if event["kind"] == "pretrained" and event["role"] == "text_encoder" and not event["quantized"]
    ]
    assert bare_te, "the bf16 reference never loaded a text encoder"
    assert all(not event["cuda"] for event in bare_te)
    assert not any(event["kind"] == "pretrained" and event["cuda"] for event in placement.events)
    assert not any(event["kind"] == "module" and event["cuda"] and event["bf16_te"] for event in placement.events)
    assert not any(event["kind"] == "module" and event["cuda"] and event["skeleton"] for event in placement.events)
    assert not any(
        event["kind"] == "module" and event["cuda"] and not event["quantized"] and int(event["nbytes"]) > budget
        for event in placement.events
    )
    huge_tensors = [
        event
        for event in placement.events
        if event["kind"] == "tensor" and event["cuda"] and int(event["nbytes"]) > budget
    ]
    assert not huge_tensors
    reference_dit = [
        event
        for event in placement.events
        if event["kind"] == "module" and event["role"] == "dit" and event["cuda"] and not event["quantized"]
    ]
    assert reference_dit
    assert all(int(event["nbytes"]) <= budget for event in reference_dit)
