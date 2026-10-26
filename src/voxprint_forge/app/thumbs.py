"""Thumbnails decoded off the UI thread, never at full resolution.

``QPixmap`` is built on the GUI thread from a ``QImage`` the pool already scaled.
The label shows a drawn placeholder until that image arrives.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtGui


def decode_thumbnail(path: Path, size: int) -> QtGui.QImage:
    """Read ``path`` scaled to fit in ``size`` px. Header size only, then a scaled decode.

    Call this from a worker. A full-resolution decode is not performed.
    """
    reader = QtGui.QImageReader(str(path))
    reader.setAutoTransform(True)
    header = reader.size()
    width, height = header.width(), header.height()
    if width <= 0 or height <= 0 or size <= 0:
        return QtGui.QImage()
    scale = min(size / width, size / height, 1.0)
    reader.setScaledSize(QtCore.QSize(max(1, round(width * scale)), max(1, round(height * scale))))
    image = reader.read()
    if image.isNull():
        return QtGui.QImage()
    return image


class ThumbService(QtCore.QObject):
    """Cache of scaled images. :meth:`request` returns immediately on the GUI thread."""

    _ready = QtCore.Signal(str, int, int, object)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._cache: dict[tuple[str, int, int], QtGui.QImage] = {}
        self._waiters: dict[tuple[str, int, int], list[Callable[[QtGui.QPixmap], None]]] = {}
        self._queued: set[tuple[str, int, int]] = set()
        self._placeholders: dict[int, QtGui.QPixmap] = {}
        self._ready.connect(self._deliver)

    def request(
        self,
        path: Path,
        size: int,
        on_ready: Callable[[QtGui.QPixmap], None] | None = None,
    ) -> QtGui.QPixmap:
        """A cached thumbnail, or a drawn placeholder while a worker decodes one."""
        mtime = _mtime(path)
        key = (str(path), size, mtime)
        image = self._cache.get(key)
        if image is not None and not image.isNull():
            return QtGui.QPixmap.fromImage(image)
        if on_ready is not None:
            self._waiters.setdefault(key, []).append(on_ready)
        if key not in self._queued:
            self._queued.add(key)
            self._pool.start(_ThumbTask(path, size, mtime, self))
        return self.placeholder(size)

    def placeholder(self, size: int) -> QtGui.QPixmap:
        """A drawn stand-in. It does not read a file."""
        found = self._placeholders.get(size)
        if found is not None:
            return found
        image = QtGui.QImage(max(size, 1), max(size, 1), QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("#2a2a33"))
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QPen(QtGui.QColor("#6e6e80")))
        painter.drawRect(1, 1, max(size - 3, 1), max(size - 3, 1))
        painter.end()
        pixmap = QtGui.QPixmap.fromImage(image)
        self._placeholders[size] = pixmap
        return pixmap

    def wait(self, msec: int = 5000) -> None:
        """Block until queued decodes finish. Tests use this."""
        self._pool.waitForDone(msec)

    def _deliver(self, path: str, size: int, mtime: int, image: object) -> None:
        key = (path, size, mtime)
        self._queued.discard(key)
        pixmap: QtGui.QPixmap | None = None
        if isinstance(image, QtGui.QImage) and not image.isNull():
            self._cache[key] = image
            pixmap = QtGui.QPixmap.fromImage(image)
        if pixmap is None:
            pixmap = self.placeholder(size)
        for callback in self._waiters.pop(key, ()):
            callback(pixmap)


class _ThumbTask(QtCore.QRunnable):
    """One scaled decode. The result is delivered on the GUI thread."""

    def __init__(self, path: Path, size: int, mtime: int, service: ThumbService) -> None:
        super().__init__()
        self._path = path
        self._size = size
        self._mtime = mtime
        self._service = service

    def run(self) -> None:
        image = decode_thumbnail(self._path, self._size)
        self._service._ready.emit(str(self._path), self._size, self._mtime, image)


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0
