"""LoRA errors. Each carries a ``lora.err.*`` key and its parameters."""

from __future__ import annotations

from voxprint_forge.errors import ForgeError, StageError, UsageError
from voxprint_forge.lora.i18n import text


class LoraUsageError(UsageError):
    """Missing or invalid input, such as consent or the photo count. Exit code 2."""

    def __init__(self, key: str, **params: object) -> None:
        super().__init__(text(key, "en", **params), code=key)
        self.key = key
        self.params = params


class LoraStageError(StageError):
    """A run that could not finish, such as no GPU or a diverged run. Exit code 6."""

    def __init__(self, key: str, **params: object) -> None:
        super().__init__(text(key, "en", **params), code=key)
        self.key = key
        self.params = params


class LoraBlocked(ForgeError):
    """The safety gate blocked the run. Exit code 6. There is no override."""

    exit_code = 6

    def __init__(self, key: str = "lora.err.blocked", **params: object) -> None:
        super().__init__(text(key, "en", **params), code=key)
        self.key = key
        self.params = params
