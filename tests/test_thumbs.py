"""Thumbnails are scaled off the UI thread. The photos step does not decode a full image."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtGui  # noqa: E402

from voxprint_forge.app import thumbs as thumbs_mod  # noqa: E402
from voxprint_forge.app.thumbs import ThumbService, decode_thumbnail  # noqa: E402


def test_decode_thumbnail_scales_from_the_header(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    QtGui.QImage(1800, 900, QtGui.QImage.Format.Format_RGB32).save(str(path))
    image = decode_thumbnail(path, 72)
    assert not image.isNull()
    assert max(image.width(), image.height()) <= 72


def test_request_returns_before_the_worker_decodes(qtbot, tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "wide.png"
    QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB32).save(str(path))
    ui = threading.get_ident()
    started = threading.Event()
    release = threading.Event()
    seen: list[int] = []

    def fake(file: Path, size: int) -> QtGui.QImage:
        seen.append(threading.get_ident())
        started.set()
        assert release.wait(2)
        image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("#112233"))
        return image

    monkeypatch.setattr(thumbs_mod, "decode_thumbnail", fake)
    service = ThumbService()
    qtbot.addWidget(service)
    holder: list[QtGui.QPixmap] = []
    pixmap = service.request(path, 48, holder.append)
    assert not pixmap.isNull()
    assert holder == []
    assert started.wait(2)
    assert seen[0] != ui
    release.set()
    service.wait()
    qtbot.waitUntil(lambda: bool(holder), timeout=2000)
    assert holder[0].width() == 48


def test_the_window_does_not_decode_a_full_pixmap() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "voxprint_forge" / "app" / "window.py").read_text(encoding="utf-8")
    assert "QPixmap(str(" not in text
    assert "QImageReader" not in text
