"""Live progress files: cadence, stage boundaries, flush and fsync."""

from __future__ import annotations

import json
from pathlib import Path

from voxprint_forge.lora.progress_log import TRAIN_FILE, VRAM_FILE, ProgressWriter, read_progress


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _note(writer: ProgressWriter, *, stage: str, step: int = 1, loss: float | None = 0.25) -> bool:
    return writer.note(step=step, total=10, loss=loss, eta=40, stage=stage)


def test_progress_writes_every_60s_and_on_a_stage_change(tmp_path: Path, monkeypatch) -> None:
    synced: list[int] = []
    monkeypatch.setattr("voxprint_forge.lora.progress_log.os.fsync", lambda fd: synced.append(fd))
    clock = _Clock()
    writer = ProgressWriter(tmp_path, clock=clock, vram=lambda: (1.25, 2.5))
    assert _note(writer, stage="load", step=0, loss=None)
    clock.now = 30
    assert not _note(writer, stage="load", step=0, loss=None)
    clock.now = 60
    assert _note(writer, stage="load", step=0, loss=None)
    clock.now = 61
    assert _note(writer, stage="step", step=3, loss=0.5)
    clock.now = 90
    assert not _note(writer, stage="step", step=4, loss=0.4)
    row = read_progress(tmp_path)
    assert row is not None
    assert row["step"] == 3
    assert row["total"] == 10
    assert row["loss"] == 0.5
    assert row["elapsed_seconds"] == 61
    assert row["eta_seconds"] == 40
    assert row["stage"] == "step"
    log = (tmp_path / "train.log").read_text(encoding="utf-8").splitlines()
    assert len(log) == 3
    assert log[-1].startswith("stage=step step=3/10 loss=0.5000 elapsed=61s eta=40")
    vram = (tmp_path / VRAM_FILE).read_text(encoding="utf-8").splitlines()
    assert vram[0] == "elapsed_seconds,allocated_gb,reserved_gb"
    assert vram[1:] == ["0.000,1.250,2.500", "60.000,1.250,2.500", "61.000,1.250,2.500"]
    # json + log + csv, once per write that landed
    assert len(synced) == 9
    text = (tmp_path / TRAIN_FILE).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["stage"] == "step"


def test_round_one_extra_is_stored_and_logged(tmp_path: Path) -> None:
    clock = _Clock()
    writer = ProgressWriter(tmp_path, clock=clock, vram=lambda: (None, None))
    assert writer.note(
        step=40, total=200, loss=0.2, eta=400, stage="step",
        extra={"round": 1, "first_pictures_seconds": 480, "seconds_per_sample": 18, "sample_seconds_estimated": False},
    )
    row = read_progress(tmp_path)
    assert row is not None
    assert row["first_pictures_seconds"] == 480
    assert row["seconds_per_sample"] == 18
    assert "first_pictures=480" in (tmp_path / "train.log").read_text(encoding="utf-8")


def test_a_non_finite_loss_is_stored_as_unknown(tmp_path: Path) -> None:
    clock = _Clock()
    writer = ProgressWriter(tmp_path, clock=clock, vram=lambda: (None, None))
    assert _note(writer, stage="step", loss=float("nan"))
    assert read_progress(tmp_path)["loss"] is None
    line = (tmp_path / VRAM_FILE).read_text(encoding="utf-8").splitlines()[1]
    assert line == "0.000,,"
