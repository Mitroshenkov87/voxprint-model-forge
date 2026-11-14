"""The Linux installer script must install the [gui] app plus the pinned MiVOLO gate, and add a menu entry."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "installer" / "linux" / "install.sh"

MIVOLO_COMMIT = "37475e3f8818b5f22448003feec3e64b01bfb188"
COMMON_WHEEL = "voxprint_common-0.7.0-py3-none-any.whl"


def test_linux_installer_installs_gui_extra_and_pinned_mivolo() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    # Window app: the vendored wheel that exists in vendor/, then this checkout with [gui].
    assert COMMON_WHEEL in text
    assert "[gui]" in text
    # Gate estimator: pinned MiVOLO commit, no deps, no build isolation, setuptools<81 first.
    assert MIVOLO_COMMIT in text
    assert "--no-deps" in text
    assert "--no-build-isolation" in text
    assert "setuptools<81" in text
    # MiVOLO's own pins are not needed for the classifier and must not be pulled.
    for pkg in ("ultralytics", "yt-dlp", "lapx"):
        assert pkg not in text
    # GNOME/freedesktop menu entry with a copied icon.
    assert ".local/share/applications/voxprint-model-forge.desktop" in text
    assert "Icon=" in text


def test_cli_docs_document_the_same_mivolo_install() -> None:
    text = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    assert MIVOLO_COMMIT in text
    assert "--no-deps --no-build-isolation" in text
    assert '"setuptools<81" wheel' in text
    assert '.[gui]' in text
