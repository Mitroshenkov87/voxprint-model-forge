"""Windows 11 Acrylic. Any other platform, or any error, stays on the plain dark theme."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

MIN_BUILD = 26100
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_TRANSIENTWINDOW = 3
DWMWCP_ROUND = 2


class _Margins(ctypes.Structure):
    _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]


def windows_build() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        return int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
    except Exception:
        return None


def apply_backdrop(hwnd: int) -> str:
    """Return 'acrylic' on Windows 11 build 26100+, otherwise 'plain'."""
    if sys.platform != "win32":
        return "plain"
    build = windows_build()
    if build is not None and build < MIN_BUILD:
        return "plain"
    try:
        dwm = ctypes.windll.dwmapi
        handle = wintypes.HWND(hwnd)

        def _set(attr: int, value: int) -> int:
            packed = ctypes.c_int(value)
            return int(
                dwm.DwmSetWindowAttribute(handle, ctypes.c_uint(attr), ctypes.byref(packed), ctypes.sizeof(packed))
            )

        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
        _set(DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
        margins = _Margins(-1, -1, -1, -1)
        dwm.DwmExtendFrameIntoClientArea(handle, ctypes.byref(margins))
        hr = _set(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW)
    except Exception:
        return "plain"
    return "acrylic" if hr == 0 else "plain"
