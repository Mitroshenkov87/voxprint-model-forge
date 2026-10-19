"""Release identity. The build number is read by ``voxprint_common.buildinfo``.

``VOXPRINT_BUILD`` wins. Otherwise the number is ``GITHUB_RUN_NUMBER`` plus the
``offset`` in ``BUILD.json``. That sum lives in the shared loader. This module
does not read ``GITHUB_RUN_NUMBER``. A missing recorded build stays null and
:func:`version_line` renders it as ``dev``.

A Windows pack writes ``build_info.json`` beside the catalogue, with the resolved
``build`` and ``rc``. The loader uses that recorded build when both variables
are unset, which is the installed app on a laptop.
"""

from __future__ import annotations

from pathlib import Path

from voxprint_common.buildinfo import BuildInfo, load

from voxprint_forge import __version__

__all__ = ["version_line"]


def version_line() -> str:
    """``X.Y.Z build N "Codename"``, or ``dev`` when no build number is recorded."""
    info = load(package_dir=Path(__file__).resolve().parent, app="model-forge")
    if info.version:
        return info.label()
    filled = BuildInfo(
        app=info.app or "model-forge",
        version=__version__,
        rc=info.rc,
        build=info.build,
        codename=info.codename or "Charash",
        show_rc=info.show_rc,
    )
    return filled.label()
