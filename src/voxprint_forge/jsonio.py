"""One JSON object on stdout when --json is set. Logs stay on stderr."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from voxprint_forge.scrub import active_secrets, scrub


def emit(
    command: str,
    result: object,
    *,
    ok: bool,
    exit_code: int,
    json_mode: bool,
    error: str | None = None,
    error_code: str | None = None,
    stream: TextIO | None = None,
) -> None:
    out = stream or sys.stdout
    if error is not None:
        error = scrub(error, active_secrets())
    if json_mode:
        payload: dict[str, object] = {
            "ok": ok,
            "command": command,
            "exit_code": exit_code,
            "result": result,
        }
        if error is not None:
            payload["error"] = {"code": error_code or "error", "message": error}
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
        return
    if error:
        print(f"error: {error}", file=sys.stderr)
    if isinstance(result, str):
        print(result, file=out)
    elif result is not None:
        print(json.dumps(result, indent=2, sort_keys=True), file=out)
