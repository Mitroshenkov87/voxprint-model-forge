"""Keep known third-party warning floods off the terminal.

bitsandbytes logs a warning on every int8 matmul (logger ``bitsandbytes.autograd._functions``:
``MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization``), so
``lora prepare`` printed tens of thousands of identical lines while the int8 vision model
checked and described photos. Reading the ``MatmulLtState`` fields that
:mod:`voxprint_forge.model_park` moves between devices also raises a deprecation
``FutureWarning`` per field and module. ``transformers.processing_utils`` warns on every
processor call that still passes processing kwargs through ``**kwargs`` ("Kwargs passed to
`processor.__call__` have to be in `processor_kwargs`"), which the vision pipeline
triggers per photo. Each of these messages is shown once per process; every other warning
and log record is left as it is.

Intentionally not filtered: the native CUDA caching allocator line
``memory allocation failed with OOM on device ... while trying to allocate ...``.
It is a C++ ``LOG(WARNING)`` written straight to stderr by libtorch (see
``c10/cuda/CUDACachingAllocator.cpp``), not a Python warning and not a Python log record,
so neither ``warnings.filterwarnings`` nor a ``logging.Filter`` can see it. Upstream
already downgraded it to opt-in ``LOG(INFO)`` after pytorch/pytorch#193195, but torch
2.14 still logs it on every failed ``cudaMalloc``, including attempts the
release-and-retry path then recovers. :func:`set_cpp_log_level_default` instead defaults
``TORCH_CPP_LOG_LEVEL`` to ``ERROR`` before torch is first imported. That hides C++
``LOG(WARNING)`` lines only: C++ ``LOG(ERROR)`` and fatal messages still print, Python
exceptions (including ``torch.OutOfMemoryError``) are raised as before, and ``TORCH_WARN``
from ops called from Python still reach the ``warnings`` module, where this module's
filters govern them. The allocator retries and the real OOM exception are unaffected.
The default uses ``os.environ.setdefault`` so a user-provided ``TORCH_CPP_LOG_LEVEL``
wins, and it must be set before libc10 initializes (the level is read at import time):
``voxprint_forge.cli`` calls it at module import, and nothing reachable from the CLI
imports torch at import time (every forge torch import is lazy inside a function), so
the default is always in place before torch loads.
"""

from __future__ import annotations

import logging
import os
import re
import warnings

# (message regex, category) for ``warnings.warn`` floods. Matched against the start of the message.
ONCE_PATTERNS: tuple[tuple[str, type[Warning]], ...] = ((r"MatmulLtState\.\w+ is deprecated", FutureWarning),)
# (logger name, message regex) for ``logging`` floods.
ONCE_LOG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bitsandbytes.autograd._functions", r"MatMul8bitLt: inputs will be cast"),
    ("transformers.processing_utils", r"Kwargs passed to `processor\.__call__` have to be in `processor_kwargs`"),
)


class OnceFilter(logging.Filter):
    """Let a record whose message matches ``pattern`` through once per distinct text."""

    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = re.compile(pattern)
        self.seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if not self.pattern.match(text):
            return True
        if text in self.seen:
            return False
        self.seen.add(text)
        return True


def set_cpp_log_level_default() -> None:
    """Default ``TORCH_CPP_LOG_LEVEL`` to ``ERROR`` so libtorch C++ ``LOG(WARNING)`` lines stay quiet.

    Must run before torch (libc10) is first imported in the process; the level is read at
    import time. ``setdefault`` keeps a user-provided ``TORCH_CPP_LOG_LEVEL``. C++
    ``LOG(ERROR)``/fatal messages, Python exceptions (including ``torch.OutOfMemoryError``)
    and ``TORCH_WARN`` raised as Python warnings are not affected.
    """
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")


def quiet_third_party_warnings() -> None:
    """Show each known flood message at most once in this process. Safe to call twice."""
    for pattern, category in ONCE_PATTERNS:
        warnings.filterwarnings("once", message=pattern, category=category)
    for name, pattern in ONCE_LOG_PATTERNS:
        logger = logging.getLogger(name)
        if not any(isinstance(item, OnceFilter) and item.pattern.pattern == pattern for item in logger.filters):
            logger.addFilter(OnceFilter(pattern))
