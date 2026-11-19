"""Stable process exit codes for the forge CLI."""

from __future__ import annotations


class ForgeError(Exception):
    """Failure the CLI turns into a documented exit code."""

    exit_code = 1

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or self.__class__.__name__


class UsageError(ForgeError):
    """Bad arguments. Exit code 2."""
    exit_code = 2


class LicenceError(ForgeError):
    """A weight licence blocked the run. Exit code 3."""
    exit_code = 3


class ChecksumError(ForgeError):
    """A download hash did not match. Exit code 4."""
    exit_code = 4


class DownloadError(ForgeError):
    """A download failed. Exit code 5."""
    exit_code = 5


class StageError(ForgeError):
    """A pipeline stage failed. Exit code 6."""
    exit_code = 6

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        report: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code)
        self.report = report


class DiskShortError(ForgeError):
    """Free space is below the next stage. Stop without deleting finished quants."""

    exit_code = 7


class PlanError(ForgeError):
    """The plan file is missing or invalid. Exit code 8."""
    exit_code = 8


class ToolMissingError(ForgeError):
    """A required binary or file is missing. Exit code 9."""
    exit_code = 9


# 0 success, including a clean skip of optional models that did not fit the clock.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_LICENCE = 3
EXIT_CHECKSUM = 4
EXIT_DOWNLOAD = 5
EXIT_STAGE = 6
EXIT_DISK = 7
EXIT_PLAN = 8
EXIT_TOOL = 9
