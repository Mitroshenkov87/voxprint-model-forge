"""The third-party floods print once per process; other warnings and log lines are untouched."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from voxprint_forge.quiet_warnings import quiet_third_party_warnings, set_cpp_log_level_default

CAST = "MatMul8bitLt: inputs will be cast from %s to float16 during quantization"
KWS = "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`"

_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None

# Import voxprint_forge.cli, then torch (as a command's lazy import would), recording the
# env var at every torch import: the default must already be ERROR by then.
_SUBPROCESS_SNIPPET = """
    import builtins
    import os

    real_import = builtins.__import__
    seen = []

    def tracked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            seen.append(os.environ.get("TORCH_CPP_LOG_LEVEL"))
        return real_import(name, *args, **kwargs)

    builtins.__import__ = tracked
    try:
        import voxprint_forge.cli  # noqa: F401
        import torch  # noqa: F401
    finally:
        builtins.__import__ = real_import

    assert os.environ.get("TORCH_CPP_LOG_LEVEL") == "ERROR", os.environ.get("TORCH_CPP_LOG_LEVEL")
    assert seen, "torch was not imported in the subprocess"
    assert all(value == "ERROR" for value in seen), seen
"""


def test_matmul_cast_log_shows_once_and_other_records_stay(caplog) -> None:
    quiet_third_party_warnings()
    quiet_third_party_warnings()
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    with caplog.at_level(logging.WARNING, logger="bitsandbytes.autograd._functions"):
        for _ in range(50):
            logger.warning(CAST, "torch.bfloat16")
        logger.warning("something else")
        logger.warning("something else")
    texts = [record.getMessage() for record in caplog.records]
    assert texts.count(CAST % "torch.bfloat16") == 1
    assert texts.count("something else") == 2


def test_deprecated_state_warning_shows_once_and_other_warnings_stay() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_third_party_warnings()
        for _ in range(20):
            warnings.warn("MatmulLtState.CxBt is deprecated and will be removed", FutureWarning, stacklevel=1)
        warnings.warn("something else", UserWarning, stacklevel=1)
        warnings.warn("something else", UserWarning, stacklevel=1)
    texts = [str(item.message) for item in caught]
    assert sum(text.startswith("MatmulLtState.CxBt") for text in texts) == 1
    assert texts.count("something else") == 2


def test_processor_kwargs_log_shows_once_per_text_and_other_records_stay(caplog) -> None:
    quiet_third_party_warnings()
    quiet_third_party_warnings()
    logger = logging.getLogger("transformers.processing_utils")
    other = "Token indices sequence length is longer than the specified maximum"
    with caplog.at_level(logging.WARNING, logger="transformers.processing_utils"):
        for _ in range(50):
            logger.warning(KWS)
        logger.warning(other)
        logger.warning(other)
    texts = [record.getMessage() for record in caplog.records]
    assert texts.count(KWS) == 1
    assert texts.count(other) == 2
    # A distinct message text is a separate message: it gets its own single showing.
    distinct = KWS + " (extra detail)"
    with caplog.at_level(logging.WARNING, logger="transformers.processing_utils"):
        logger.warning(distinct)
        logger.warning(distinct)
    texts = [record.getMessage() for record in caplog.records]
    assert texts.count(distinct) == 1


def test_cpp_log_level_default_set_when_absent(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_CPP_LOG_LEVEL", raising=False)
    set_cpp_log_level_default()
    assert os.environ["TORCH_CPP_LOG_LEVEL"] == "ERROR"


def test_cpp_log_level_default_keeps_user_value(monkeypatch) -> None:
    monkeypatch.setenv("TORCH_CPP_LOG_LEVEL", "INFO")
    set_cpp_log_level_default()
    assert os.environ["TORCH_CPP_LOG_LEVEL"] == "INFO"


@pytest.mark.skipif(not _TORCH_INSTALLED, reason="torch is not installed")
def test_cli_import_sets_cpp_log_level_before_torch() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env.pop("TORCH_CPP_LOG_LEVEL", None)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not pythonpath else str(src) + os.pathsep + pythonpath
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SUBPROCESS_SNIPPET)],
        check=True,
        env=env,
        timeout=180,
    )


def test_cli_main_installs_the_filter(monkeypatch) -> None:
    import voxprint_forge.cli as cli

    called: list[bool] = []
    monkeypatch.setattr(cli, "quiet_third_party_warnings", lambda: called.append(True))
    cli.main(["--version"])
    assert called == [True]
