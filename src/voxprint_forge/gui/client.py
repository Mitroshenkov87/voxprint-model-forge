"""Call `forge --json` locally, or `forge status --json` over SSH. No pipeline logic."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable

from voxprint_forge.errors import UsageError

SshRun = Callable[[list[str]], str]
ForgeRun = Callable[[list[str], dict[str, str]], dict[str, object]]


def forge_args(command: str, *, plan: str, root: str, forge_out: str, model: str = "") -> list[str]:
    """Global flags precede the subcommand, matching the forge CLI."""
    args: list[str] = []
    if plan:
        args.extend(["--plan", plan])
    if root:
        args.extend(["--root", root])
    if forge_out:
        args.extend(["--forge-out", forge_out])
    args.append(command)
    if model:
        args.extend(["--model", model])
    return args


class SubprocessForge:
    def __call__(self, args: list[str], env: dict[str, str]) -> dict[str, object]:
        merged = os.environ.copy()
        merged.update(env)
        done = subprocess.run(
            [sys.executable, "-m", "voxprint_forge", "--json", *args],
            check=False,
            capture_output=True,
            text=True,
            env=merged,
        )
        text = done.stdout.strip()
        if not text:
            return {
                "ok": False,
                "command": args[-1] if args else "",
                "exit_code": done.returncode,
                "result": None,
                "error": {"code": "empty", "message": (done.stderr or "")[-500:]},
            }
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise UsageError("forge --json did not print an object", code="usage")
        return loaded


def result_of(payload: dict[str, object]) -> dict[str, object]:
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return {}


def format_checks(payload: dict[str, object]) -> str:
    result = result_of(payload)
    checks = result.get("checks")
    lines: list[str] = []
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", "check"))
            detail = str(check.get("detail", ""))
            if check.get("ok") is True:
                lines.append(f"{name}: ok — {detail}")
            else:
                lines.append(f"{name}: FAILED — {detail}\nFix: {check.get('fix', '')}")
    if lines:
        return "\n".join(lines)
    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return "preflight returned no checks"


def status_items(payload: dict[str, object]) -> list[dict[str, object]]:
    items = result_of(payload).get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _ssh_run(argv: list[str]) -> str:
    done = subprocess.run(argv, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise UsageError((done.stderr or done.stdout or "ssh failed")[-500:], code="ssh")
    return done.stdout


def ssh_status(
    host: str,
    user: str,
    key: str,
    forge_out: str = "",
    *,
    run: SshRun | None = None,
) -> dict[str, object]:
    """Poll `forge --json status` on a VM. Host, user, and key are chosen by the caller."""
    if not host.strip() or not user.strip() or not key.strip():
        raise UsageError("host, user, and key are required", code="usage")
    remote = "forge --json status"
    if forge_out.strip():
        remote = f"forge --json --forge-out {shlex.quote(forge_out.strip())} status"
    argv = [
        "ssh",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{user}@{host}",
        remote,
    ]
    stdout = (run or _ssh_run)(argv)
    loaded = json.loads(stdout)
    if not isinstance(loaded, dict):
        raise UsageError("forge status --json did not print an object", code="usage")
    return loaded


def state_badge(state: str, error: str = "") -> tuple[str, bool]:
    """Map a forge state to the label shown in the table. A failure is FAILED, never done."""
    text = state.strip()
    upper = text.upper()
    failed_words = {"FAILED", "FAIL", "ERROR", "DONE", "COMPLETE", "FINISHED"}
    if upper.startswith("STAGEFAIL") or upper in failed_words or (not text and bool(error.strip())):
        return "FAILED", True
    if not text:
        return "—", False
    return text, False
