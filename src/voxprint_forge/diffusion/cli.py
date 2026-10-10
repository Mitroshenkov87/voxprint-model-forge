"""`python -m voxprint_forge.diffusion run PLAN`.

diffusers is imported with the job, after the LLM-running guard, so a refused
start and the LLM commands do not load it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voxprint_forge.diffusion.deadline import resolve_deadline
from voxprint_forge.diffusion.defer import defer_requested
from voxprint_forge.diffusion.guard import llm_job_running
from voxprint_forge.diffusion.planload import load_canvas_plan
from voxprint_forge.errors import EXIT_STAGE, EXIT_UNEXPECTED, EXIT_USAGE, ForgeError, UsageError
from voxprint_forge.jsonio import emit


def execute(
    *,
    repo: Path,
    plan_path: Path,
    forge_out: Path,
    force: bool,
    deadline_text: str | None,
    defer_upload: bool = False,
) -> dict[str, object]:
    if not force and llm_job_running(forge_out):
        raise UsageError(
            "LLM bootstrap is still running (bootstrap.lock or scripts/bootstrap.sh). "
            "Pass --force to start anyway.",
            code="llm_running",
        )
    deadline = resolve_deadline(deadline_text)
    plan = load_canvas_plan(plan_path)
    from voxprint_forge.diffusion.job import run_job

    token = os.environ.get("HF_TOKEN", "").strip() or None
    return run_job(
        repo=repo,
        plan=plan,
        forge_out=forge_out,
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
        if (candidate / "plans" / "canvas-zimage-tiers-v1.toml").is_file():
            return candidate
    packaged = Path(__file__).resolve().parents[3]
    if (packaged / "plans" / "canvas-zimage-tiers-v1.toml").is_file():
        return packaged
    raise UsageError("cannot find the forge repo; pass --root", code="root")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxprint_forge.diffusion")
    parser.add_argument("command", choices=["run", "upload"])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--defer-upload", action="store_true")
    parser.add_argument("--parts", default=None)
    parser.add_argument("--deadline", default=None)
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
            from voxprint_forge.diffusion.release import release_canvas_plan

            if not args.force and llm_job_running(forge_out):
                raise UsageError(
                    "LLM bootstrap is still running (bootstrap.lock or scripts/bootstrap.sh). "
                    "Pass --force to start anyway.",
                    code="llm_running",
                )
            result = release_canvas_plan(
                repo=_repo(args.root),
                plan_path=args.plan,
                forge_out=forge_out,
                token=os.environ.get("HF_TOKEN", "").strip() or None,
                parts_text=args.parts,
            )
        else:
            result = execute(
                repo=_repo(args.root),
                plan_path=args.plan,
                forge_out=forge_out,
                force=bool(args.force),
                deadline_text=args.deadline,
                defer_upload=bool(args.defer_upload),
            )
        code = EXIT_STAGE if isinstance(result, dict) and result.get("ok") is False else 0
        emit("canvas", result, ok=code == 0, exit_code=code, json_mode=args.json)
        return code
    except ForgeError as exc:
        emit(
            "canvas",
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
            "canvas",
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
