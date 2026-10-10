"""`python -m voxprint_forge.diffusion.qwen_main run PLAN`.

This entry is separate from `python -m voxprint_forge.diffusion`. A running
canvas A–C job does not import this module. diffusers loads only inside the
quality stage, after the guards.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voxprint_forge.diffusion.deadline import resolve_deadline
from voxprint_forge.diffusion.defer import defer_requested
from voxprint_forge.diffusion.guard import llm_job_running
from voxprint_forge.diffusion.qwen_guard import canvas_ac_running
from voxprint_forge.diffusion.qwen_plan import load_qwen_plan
from voxprint_forge.diffusion.qwen_select import parse_parts, refuse_retired
from voxprint_forge.diffusion.qwen_spec import PLAN_NAME
from voxprint_forge.errors import EXIT_STAGE, EXIT_UNEXPECTED, EXIT_USAGE, ForgeError, UsageError
from voxprint_forge.jsonio import emit


def execute(
    *,
    repo: Path,
    plan_path: Path,
    forge_out: Path,
    force: bool,
    deadline_text: str | None,
    parts_text: str | None,
    defer_upload: bool = False,
) -> dict[str, object]:
    """Run a Qwen-Image plan from the CLI arguments."""
    if not force and llm_job_running(forge_out):
        raise UsageError(
            "LLM bootstrap is still running (bootstrap.lock or scripts/bootstrap.sh). "
            "Pass --force to start anyway.",
            code="llm_running",
        )
    if not force and canvas_ac_running():
        raise UsageError(
            "scripts/canvas_job.sh is still running. Wait until it exits, or pass --force.",
            code="canvas_running",
        )
    plan = load_qwen_plan(plan_path)
    parts = parse_parts(parts_text)
    refuse_retired(parts, plan.retired, force=force)
    deadline = resolve_deadline(deadline_text)
    from voxprint_forge.diffusion.qwen_job import run_selected

    token = os.environ.get("HF_TOKEN", "").strip() or None
    return run_selected(
        repo=repo,
        plan=plan,
        forge_out=forge_out,
        parts=parts,
        deadline=deadline,
        token=token,
        defer_upload=defer_requested(defer_upload),
    )


def _repo(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("FORGE_ROOT")
    if env:
        return Path(env)
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "plans" / PLAN_NAME).is_file():
            return candidate
    packaged = Path(__file__).resolve().parents[3]
    if (packaged / "plans" / PLAN_NAME).is_file():
        return packaged
    raise UsageError("cannot find the forge repo; pass --root", code="root")


def main(argv: list[str] | None = None) -> int:
    """Run the Qwen-Image CLI and return the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m voxprint_forge.diffusion.qwen_main")
    parser.add_argument("command", choices=["run", "upload"])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--defer-upload", action="store_true")
    parser.add_argument("--deadline", default=None)
    parser.add_argument("--parts", default=None)
    parser.add_argument("--forge-out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE
    forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
    try:
        if args.command == "upload":
            from voxprint_forge.diffusion.qwen_job import release_deferred
            from voxprint_forge.diffusion.qwen_plan import load_qwen_plan

            if not args.force and llm_job_running(forge_out):
                raise UsageError(
                    "LLM bootstrap is still running (bootstrap.lock or scripts/bootstrap.sh). "
                    "Pass --force to start anyway.",
                    code="llm_running",
                )
            if not args.force and canvas_ac_running():
                raise UsageError(
                    "scripts/canvas_job.sh is still running. Wait until it exits, or pass --force.",
                    code="canvas_running",
                )
            result = release_deferred(
                repo=_repo(args.root),
                plan=load_qwen_plan(args.plan),
                forge_out=forge_out,
                token=os.environ.get("HF_TOKEN", "").strip() or None,
                parts_text=args.parts,
                force=bool(args.force),
            )
        else:
            result = execute(
                repo=_repo(args.root),
                plan_path=args.plan,
                forge_out=forge_out,
                force=bool(args.force),
                deadline_text=args.deadline,
                parts_text=args.parts,
                defer_upload=bool(args.defer_upload),
            )
        code = EXIT_STAGE if isinstance(result, dict) and result.get("ok") is False else 0
        emit("qwen", result, ok=code == 0, exit_code=code, json_mode=args.json)
        return code
    except ForgeError as exc:
        emit(
            "qwen",
            None,
            ok=False,
            exit_code=exc.exit_code,
            json_mode=args.json,
            error=str(exc),
            error_code=exc.code,
        )
        return exc.exit_code
    except Exception as exc:
        emit(
            "qwen",
            None,
            ok=False,
            exit_code=EXIT_UNEXPECTED,
            json_mode=args.json,
            error=str(exc),
            error_code="unexpected",
        )
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
