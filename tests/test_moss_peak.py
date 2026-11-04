"""Peak measurement for the compact MOSS build. Mocks only: no downloads and no GPU."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from voxprint_forge.cli import build_parser
from voxprint_forge.diffusion.moss_peak import (
    DEFAULT_HEIGHT,
    DEFAULT_PROMPT,
    DEFAULT_RANK,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_WIDTH,
    MemorySample,
    Services,
    locate_build,
    rss_bytes,
    rss_to_bytes,
    run,
)
from voxprint_forge.diffusion.moss_prequant import MODEL_ID, WEIGHTS_DIR
from voxprint_forge.errors import StageError

PHASE_KEYS = {
    "steps",
    "allocated_bytes",
    "allocated_gb",
    "reserved_bytes",
    "reserved_gb",
    "rss_bytes",
}
REPORT_KEYS = {
    "mode",
    "build_id",
    "quant_mode",
    "gpu",
    "timestamp",
    "rank",
    "seed",
    "steps",
    "train_steps",
    "resolution",
    "prompt",
}


class _Clock:
    def timestamp(self) -> str:
        return "2026-11-04T00:00:00+00:00"


class _Probe:
    def __init__(self, samples: list[MemorySample]) -> None:
        self.events: list[str] = []
        self._samples = samples
        self._index = 0

    def reset(self) -> None:
        self.events.append("reset")

    def sample(self) -> MemorySample:
        self.events.append("sample")
        item = self._samples[self._index]
        self._index += 1
        return item

    def gpu_name(self) -> str:
        return "Mock GPU"


class _Session:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def load(self, weights: Path, *, quant_mode: str) -> None:
        self.events.append(("load", Path(weights), quant_mode))

    def generate(
        self,
        *,
        steps: int,
        seed: int,
        rank: int,
        prompt: str,
        width: int,
        height: int,
    ) -> None:
        self.events.append(
            (
                "generate",
                {
                    "steps": steps,
                    "seed": seed,
                    "rank": rank,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                },
            )
        )

    def train(self, *, steps: int, seed: int, rank: int, prompt: str) -> None:
        self.events.append(
            ("train", {"steps": steps, "seed": seed, "rank": rank, "prompt": prompt})
        )


def _services(samples: list[MemorySample] | None = None) -> tuple[Services, _Session, _Probe]:
    session = _Session()
    probe = _Probe(samples or [MemorySample(1024**3, 2 * 1024**3, 4096)])
    return Services(session, probe, _Clock()), session, probe


def _seal(folder: Path) -> None:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "voxprint-model.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def _fake_build(root: Path, *, quant: str = "fp8_e4m3") -> Path:
    weights = root / WEIGHTS_DIR
    (weights / "sft").mkdir(parents=True)
    (weights / "encoder").mkdir()
    (weights / "sft" / "config.json").write_text("{}\n", encoding="utf-8")
    (weights / "sft" / "model.safetensors").write_bytes(b"sft")
    (weights / "encoder" / "config.json").write_text("{}\n", encoding="utf-8")
    (weights / "encoder" / "model.safetensors").write_bytes(b"enc")
    (weights / "mode.json").write_text(json.dumps({"mode": quant}) + "\n", encoding="utf-8")
    _seal(weights)
    manifest = {"id": MODEL_ID, "prequant": {"chosen": quant}}
    (root / "voxprint-model.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def _listing(root: Path) -> list[tuple[str, int]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), path.stat().st_size))
    return rows


def _files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def test_parser_accepts_gen_train_both_and_rejects_other() -> None:
    parser = build_parser()
    for mode in ("gen", "train", "both"):
        args = parser.parse_args(["moss-peak", "--mode", mode, "--build", "b", "--out", "o.json"])
        assert args.command == "moss-peak"
        assert args.mode == mode
        assert args.lora_rank == DEFAULT_RANK == 16
        assert args.steps == DEFAULT_STEPS == 4
        assert args.train_steps == DEFAULT_TRAIN_STEPS == 5
        assert args.seed == DEFAULT_SEED == 0
        assert args.width == DEFAULT_WIDTH == 1024
        assert args.height == DEFAULT_HEIGHT == 1024
        assert args.prompt == DEFAULT_PROMPT
    with pytest.raises(SystemExit) as rejected:
        parser.parse_args(["moss-peak", "--mode", "eval", "--build", "b", "--out", "o.json"])
    assert rejected.value.code == 2
    with pytest.raises(SystemExit) as missing:
        parser.parse_args(["moss-peak", "--build", "b", "--out", "o.json"])
    assert missing.value.code == 2


def test_gen_report_uses_defaults_and_leaves_the_build_untouched(tmp_path: Path) -> None:
    build = _fake_build(tmp_path / "build")
    before = _listing(build)
    out = build / ".." / "peak.json"
    services, session, probe = _services()
    result = run(build, out, mode="gen", services=services)
    payload = json.loads(out.resolve().read_text(encoding="utf-8"))
    assert payload == result
    assert set(payload) == REPORT_KEYS | {"gen"}
    assert payload["mode"] == "gen"
    assert payload["build_id"] == MODEL_ID
    assert payload["quant_mode"] == "fp8_e4m3"
    assert payload["gpu"] == "Mock GPU"
    assert payload["timestamp"] == "2026-11-04T00:00:00+00:00"
    assert payload["rank"] == 16
    assert payload["seed"] == 0
    assert payload["steps"] == 4
    assert payload["train_steps"] == 5
    assert payload["resolution"] == {"width": 1024, "height": 1024}
    assert payload["prompt"] == DEFAULT_PROMPT
    assert set(payload["gen"]) == PHASE_KEYS
    assert payload["gen"]["steps"] == 4
    assert payload["gen"]["allocated_bytes"] == 1024**3
    assert payload["gen"]["allocated_gb"] == 1.0
    assert payload["gen"]["reserved_bytes"] == 2 * 1024**3
    assert payload["gen"]["reserved_gb"] == 2.0
    assert payload["gen"]["rss_bytes"] == 4096
    assert "train" not in payload
    assert probe.events == ["reset", "sample"]
    assert session.events[0] == ("load", (build / WEIGHTS_DIR).resolve(), "fp8_e4m3")
    assert session.events[1][0] == "generate"
    assert session.events[1][1] == {
        "steps": 4,
        "seed": 0,
        "rank": 16,
        "prompt": DEFAULT_PROMPT,
        "width": 1024,
        "height": 1024,
    }
    assert _listing(build) == before
    assert out.resolve() == (tmp_path / "peak.json").resolve()


def test_refuses_out_inside_the_build_and_writes_nothing(tmp_path: Path) -> None:
    build = _fake_build(tmp_path / "build")
    manifest = (build / "voxprint-model.json").read_bytes()
    before = _listing(build)
    services, session, probe = _services()
    blocked = [
        build / "peak.json",
        build / WEIGHTS_DIR / "peak.json",
        build / WEIGHTS_DIR / "sft" / "peak.json",
        build / "nope" / ".." / "peak.json",
        build / "voxprint-model.json",
    ]
    for out in blocked:
        with pytest.raises(StageError) as caught:
            run(build, out, mode="gen", services=services)
        assert caught.value.code == "moss_peak_out"
        assert not (out.resolve().is_file() and out.resolve() != build / "voxprint-model.json")
    weights = build / WEIGHTS_DIR
    with pytest.raises(StageError) as from_weights:
        run(weights, build / "inside.json", mode="train", services=services)
    assert from_weights.value.code == "moss_peak_out"
    assert session.events == []
    assert probe.events == []
    assert _listing(build) == before
    assert (build / "voxprint-model.json").read_bytes() == manifest
    assert not (build / "inside.json").exists()


def test_train_and_both_write_only_the_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    build = _fake_build(tmp_path / "build", quant="nf4")
    reports = tmp_path / "reports"
    reports.mkdir()
    build_before = _listing(build)
    before = _files(tmp_path)

    train_services, train_session, train_probe = _services([MemorySample(5 * 1024**3, 6 * 1024**3, 2048)])
    train_out = reports / "train.json"
    train_result = run(build, train_out, mode="train", services=train_services)
    after_train = _files(tmp_path)
    assert after_train - before == {"reports/train.json"}
    assert _files(work) == set()
    assert _listing(build) == build_before
    train_payload = json.loads(train_out.read_text(encoding="utf-8"))
    assert train_payload == train_result
    assert train_payload["mode"] == "train"
    assert train_payload["quant_mode"] == "nf4"
    assert train_payload["train_steps"] == 5
    assert train_payload["steps"] == 4
    assert train_payload["rank"] == 16
    assert set(train_payload["train"]) == PHASE_KEYS
    assert train_payload["train"]["steps"] == 5
    assert train_payload["train"]["allocated_gb"] == 5.0
    assert "gen" not in train_payload
    assert train_probe.events == ["reset", "sample"]
    assert [item[0] for item in train_session.events] == ["load", "train"]
    assert train_session.events[1][1]["steps"] == 5
    assert train_session.events[1][1]["rank"] == 16

    both_services, both_session, both_probe = _services(
        [
            MemorySample(1024**3, 2 * 1024**3, 4096),
            MemorySample(3 * 1024**3, 4 * 1024**3, 8192),
        ]
    )
    both_out = reports / "both.json"
    both_result = run(
        build / WEIGHTS_DIR,
        both_out,
        mode="both",
        rank=8,
        steps=7,
        train_steps=2,
        seed=3,
        prompt="a red cube on a white table",
        width=640,
        height=768,
        services=both_services,
    )
    after_both = _files(tmp_path)
    assert after_both - before == {"reports/train.json", "reports/both.json"}
    assert _files(work) == set()
    assert _listing(build) == build_before
    both_payload = json.loads(both_out.read_text(encoding="utf-8"))
    assert both_payload == both_result
    assert set(both_payload) == REPORT_KEYS | {"gen", "train"}
    assert list(both_payload)[-2:] == ["gen", "train"]
    assert both_payload["mode"] == "both"
    assert both_payload["build_id"] == MODEL_ID
    assert both_payload["quant_mode"] == "nf4"
    assert both_payload["rank"] == 8
    assert both_payload["seed"] == 3
    assert both_payload["steps"] == 7
    assert both_payload["train_steps"] == 2
    assert both_payload["resolution"] == {"width": 640, "height": 768}
    assert both_payload["prompt"] == "a red cube on a white table"
    assert both_payload["gen"]["steps"] == 7
    assert both_payload["gen"]["allocated_bytes"] == 1024**3
    assert both_payload["train"]["steps"] == 2
    assert both_payload["train"]["allocated_bytes"] == 3 * 1024**3
    assert both_payload["train"]["reserved_gb"] == 4.0
    assert both_payload["train"]["rss_bytes"] == 8192
    assert both_probe.events == ["reset", "sample", "reset", "sample"]
    assert [item[0] for item in both_session.events] == ["load", "generate", "train"]
    assert both_session.events[0] == ("load", (build / WEIGHTS_DIR).resolve(), "nf4")
    assert both_session.events[1][1] == {
        "steps": 7,
        "seed": 3,
        "rank": 8,
        "prompt": "a red cube on a white table",
        "width": 640,
        "height": 768,
    }
    assert both_session.events[2][1] == {
        "steps": 2,
        "seed": 3,
        "rank": 8,
        "prompt": "a red cube on a white table",
    }
    written = {path.suffix for path in reports.iterdir()}
    assert written == {".json"}


def test_weights_directory_resolves_the_parent_manifest(tmp_path: Path) -> None:
    build = _fake_build(tmp_path / "build", quant="int8")
    located = locate_build(build)
    from_weights = locate_build(build / WEIGHTS_DIR)
    assert located.weights == from_weights.weights == (build / WEIGHTS_DIR).resolve()
    assert located.root == from_weights.root == build.resolve()
    assert located.build_id == from_weights.build_id == MODEL_ID
    assert located.quant_mode == from_weights.quant_mode == "int8"


def test_bad_sidecar_writes_nothing_and_does_not_load(tmp_path: Path) -> None:
    build = _fake_build(tmp_path / "build")
    weight = build / WEIGHTS_DIR / "sft" / "model.safetensors"
    weight.write_bytes(b"tampered")
    before = _listing(build)
    out = tmp_path / "peak.json"
    services, session, _probe = _services()
    with pytest.raises(StageError) as caught:
        run(build, out, mode="gen", services=services)
    assert caught.value.code == "moss_prequant_reload"
    assert session.events == []
    assert not out.exists()
    assert _listing(build) == before


def test_wrong_build_id_and_missing_quant_write_nothing(tmp_path: Path) -> None:
    build = _fake_build(tmp_path / "build")
    manifest = build / "voxprint-model.json"
    out = tmp_path / "peak.json"
    manifest.write_text(json.dumps({"id": "other-model", "prequant": {"chosen": "nf4"}}) + "\n", encoding="utf-8")
    before = _listing(build)
    services, session, _probe = _services()
    with pytest.raises(StageError) as wrong_id:
        run(build, out, mode="gen", services=services)
    assert wrong_id.value.code == "moss_peak_manifest"
    assert session.events == []
    assert not out.exists()
    assert _listing(build) == before

    manifest.write_text(json.dumps({"id": MODEL_ID}) + "\n", encoding="utf-8")
    before = _listing(build)
    with pytest.raises(StageError) as missing:
        run(build, out, mode="train", services=services)
    assert missing.value.code == "moss_peak_quant"
    assert session.events == []
    assert not out.exists()
    assert _listing(build) == before


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss kilobytes are Linux-only")
def test_linux_rss_kilobytes_are_bytes() -> None:
    assert rss_to_bytes(10, "linux") == 10240
    assert rss_to_bytes(10, "darwin") == 10
    assert rss_bytes() > 0


def test_import_does_not_pull_ml_libraries() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import voxprint_forge.diffusion.moss_peak\n"
            "bad = [name for name in ('torch', 'diffusers', 'transformers', 'bitsandbytes', 'peft') "
            "if name in sys.modules]\n"
            "print(','.join(bad))\n",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


def test_live_path_does_not_save_or_download() -> None:
    text = Path(run.__code__.co_filename).read_text(encoding="utf-8")
    assert text.count(".write_text(") == 1
    assert "write_bytes" not in text
    assert "save_pretrained" not in text
    assert "torch.save" not in text
    assert "save_image" not in text
    assert "hf_hub_download" not in text
    assert "snapshot_download" not in text
    assert "urlopen" not in text
    assert "local_files_only=True" in text
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in text
    assert "reset_peak_memory_stats" in text
    assert "max_memory_allocated" in text
    assert "max_memory_reserved" in text
    assert "ru_maxrss" in text
    assert "torch.Generator" in text
    assert "AdamW" in text
    assert "mse_loss" in text
    assert "_assert_sidecars" in text
    assert "ATTENTION_TARGETS" in text
