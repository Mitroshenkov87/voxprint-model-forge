"""Offscreen Qt for the GUI tests. Set before PySide6 is imported."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REAL_MODELS = os.environ.get("VOXPRINT_REAL_MODELS") == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """``real_models`` tests need downloads; skip them unless VOXPRINT_REAL_MODELS=1."""
    if REAL_MODELS:
        return
    skip = pytest.mark.skip(reason="downloads real models (set VOXPRINT_REAL_MODELS=1 to run)")
    for item in items:
        if item.get_closest_marker("real_models") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _suite_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model components out of the real suite directory."""
    root = tmp_path_factory.mktemp("suite-home")
    monkeypatch.setenv("VOXPRINT_HOME", str(root))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE", raising=False)


@pytest.fixture(autouse=True)
def _no_real_model_downloads(monkeypatch: pytest.MonkeyPatch):
    """Fail a test loudly when it tries to download models or reach beyond localhost.

    Worker children spawned by tests are covered separately: they must run the
    stand-in handlers from :mod:`tests.image_worker_fake` (see
    :func:`tests.forge_app_stub.fake_image_checks`), never the real loader.
    """
    if REAL_MODELS:
        yield
        return
    import socket
    import subprocess

    def _bail(*args: object, **kwargs: object) -> None:
        raise AssertionError("test attempted to download a model (VOXPRINT_REAL_MODELS is not 1)")

    # The worker child inherits the environment, so the hub stays offline there too.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _bail)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _bail)
    except ImportError:
        pass

    real_popen = subprocess.Popen
    real_run = subprocess.run

    def _is_mdl_download(args: object) -> bool:
        if isinstance(args, (str, bytes)):
            return False  # shell=True is not used with voxprint-mdl
        try:
            argv = [str(a) for a in args]  # type: ignore[iterable]
        except TypeError:
            return False
        exe = argv[0].removesuffix(".exe") if argv else ""
        named_mdl = exe.endswith("voxprint-mdl") or (
            "-m" in argv and any(a.endswith("voxprint_common.mdl") or a == "voxprint_common.mdl" for a in argv)
        )
        return named_mdl and "download" in argv[1:]

    def popen(args: object, *a: object, **kw: object):
        if _is_mdl_download(args):
            raise AssertionError("test launched `voxprint-mdl download` (VOXPRINT_REAL_MODELS is not 1)")
        return real_popen(args, *a, **kw)

    def run(args: object, *a: object, **kw: object):
        if _is_mdl_download(args):
            raise AssertionError("test launched `voxprint-mdl download` (VOXPRINT_REAL_MODELS is not 1)")
        return real_run(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", run)

    real_connect = socket.socket.connect

    def connect(sock: socket.socket, address: object) -> None:
        # AF_UNIX (and other non-INET) addresses are a path string/bytes, not (host, port).
        if not isinstance(address, tuple):
            return real_connect(sock, address)
        host = address[0]
        if str(host) not in ("localhost", "127.0.0.1", "::1"):
            raise AssertionError(f"test opened a network connection to {host!r} (VOXPRINT_REAL_MODELS is not 1)")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield


@pytest.fixture(autouse=True)
def _join_check_workers():
    """Join check-gate threads so none outlives its test (Qt aborts on a running QThread)."""
    yield
    try:
        from voxprint_forge.ui.check_gate import join_workers
    except ImportError:
        return
    assert join_workers(), "a check-gate worker did not finish"


@pytest.fixture(autouse=True)
def _reset_model_park():
    """Drop the process-wide ModelPark so a recycled object id can't leak between tests."""
    yield
    try:
        from voxprint_common.model_park import reset_process_park
    except ImportError:
        return
    reset_process_park()
