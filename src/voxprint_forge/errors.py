"""Stable process exit codes for the forge CLI."""

from __future__ import annotations


class ForgeError(Exception):
    """Failure the CLI turns into a documented exit code."""

    exit_code = 1

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or self.__class__.__name__


class UsageError(ForgeError):
    exit_code = 2


class LicenceError(ForgeError):
    exit_code = 3


class ChecksumError(ForgeError):
    exit_code = 4


class DownloadError(ForgeError):
    exit_code = 5


class StageError(ForgeError):
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
    exit_code = 8


class ToolMissingError(ForgeError):
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
