"""Exit codes for ``forge compare``: 0 pass, 1 fail, 2 setup, 3 no GPU."""

from __future__ import annotations

from voxprint_forge.errors import ForgeError

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SETUP = 2
EXIT_NO_GPU = 3


class CompareSetup(ForgeError):
    """The job cannot start. Exit code 2."""

    exit_code = EXIT_SETUP


class CompareNoGpu(ForgeError):
    """This compare job needs a CUDA device. Exit code 3."""

    exit_code = EXIT_NO_GPU
