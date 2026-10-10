"""forge command line. `--json` prints one object; logs stay on stderr."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voxprint_forge.accounts import execute_accounts
from voxprint_forge.diffusion.planload import load_canvas_plan
from voxprint_forge.diffusion.retain import consumers_for, release_consumer
from voxprint_forge.errors import (
    EXIT_STAGE,
    EXIT_UNEXPECTED,
    EXIT_USAGE,
    ForgeError,
    PlanError,
    StageError,
    UsageError,
)
from voxprint_forge.fetch import HfDownloader
from voxprint_forge.jsonio import emit
from voxprint_forge.layout import Layout
from voxprint_forge.llama import Runner
from voxprint_forge.pipeline import ORDER, Host, run_all, run_model
from voxprint_forge.plan import format_estimates, load_plan
from voxprint_forge.preflight import run_preflight
from voxprint_forge.statusfile import inspect_status, status_path
from voxprint_forge.upload import HubRemote, verify_logins
from voxprint_forge.version import version_line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forge")
    parser.add_argument("--version", action="version", version=version_line())
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--forge-out", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan-summary")
    sub.add_parser("verify-logins")
    sub.add_parser("preflight")
    sub.add_parser("status")
    accounts = sub.add_parser("accounts")
    account_sub = accounts.add_subparsers(dest="accounts_command", required=True)
    account_add = account_sub.add_parser("add")
    account_add.add_argument("--target", required=True, choices=("hf", "kaggle", "drive"))
    account_add.add_argument("--token", default="")
    account_add.add_argument("--login", choices=("paste", "browser"), default="paste")
    account_sub.add_parser("list")
    account_test = account_sub.add_parser("test")
    account_test.add_argument("--target", required=True, choices=("hf", "kaggle", "drive"))
    account_remove = account_sub.add_parser("remove")
    account_remove.add_argument("--target", required=True, choices=("hf", "kaggle", "drive"))
    sources = sub.add_parser("sources")
    source_sub = sources.add_subparsers(dest="sources_command", required=True)
    source_release = source_sub.add_parser("release")
    source_release.add_argument("--source", required=True)
    source_release.add_argument("--consumer", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--jobs", type=int, default=None)
    for name in ("fetch", "abliterate", "quantize", "eval", "card", "publish-prep", "upload"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--model", required=True)
    fit.add_argument("--budget-gb", type=float, action="append", default=None)
    canvas = sub.add_parser("canvas")
    canvas_sub = canvas.add_subparsers(dest="canvas_command", required=True)
    canvas_run = canvas_sub.add_parser("run")
    canvas_run.add_argument("canvas_plan", type=Path, metavar="PLAN")
    canvas_run.add_argument("--force", action="store_true")
    canvas_run.add_argument("--defer-upload", action="store_true")
    canvas_run.add_argument("--deadline", default=None)
    canvas_upload = canvas_sub.add_parser("upload")
    canvas_upload.add_argument("canvas_plan", type=Path, metavar="PLAN")
    canvas_upload.add_argument("--parts", default=None)
    canvas_upload.add_argument("--force", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE
    try:
        result = _dispatch(args)
        code = _exit_code(args.command, result)
        emit(args.command, result, ok=code == 0, exit_code=code, json_mode=args.json)
        return code
    except ForgeError as exc:
        emit(
            args.command,
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
            args.command,
            None,
            ok=False,
            exit_code=EXIT_UNEXPECTED,
            json_mode=args.json,
            error=str(exc),
            error_code="unexpected",
        )
        return EXIT_UNEXPECTED


def _canvas_upload(root: Path, plan_path: Path, forge_out: Path, parts: str | None, force: bool) -> dict[str, object]:
    from voxprint_forge.diffusion.guard import llm_job_running
    from voxprint_forge.errors import PlanError

    if not force and llm_job_running(forge_out):
        raise UsageError(
            "LLM bootstrap is still running (bootstrap.lock or scripts/bootstrap.sh). "
            "Pass --force to start anyway.",
            code="llm_running",
        )
    token = os.environ.get("HF_TOKEN", "").strip() or None
    try:
        from voxprint_forge.diffusion.release import release_canvas_plan

        return release_canvas_plan(
            repo=root,
            plan_path=plan_path,
            forge_out=forge_out,
            token=token,
            parts_text=parts,
        )
    except PlanError:
        from voxprint_forge.diffusion.qwen_job import release_deferred
        from voxprint_forge.diffusion.qwen_plan import load_qwen_plan

        return release_deferred(
            repo=root,
            plan=load_qwen_plan(plan_path),
            forge_out=forge_out,
            token=token,
            parts_text=parts,
        )


def _exit_code(command: str, result: object) -> int:
    if not isinstance(result, dict):
        return 0
    if command == "status":
        return int(result.get("exit_code", 4))
    if result.get("ok") is False:
        return EXIT_STAGE
    return 0


def _jobs(explicit: int | None) -> int:
    if explicit is None:
        text = os.environ.get("FORGE_JOBS", "1").strip() or "1"
        try:
            explicit = int(text)
        except ValueError as exc:
            raise UsageError(f"FORGE_JOBS {text!r} is not an integer", code="usage") from exc
    if explicit < 1:
        raise UsageError("jobs must be at least 1", code="usage")
    return explicit


def _release_source(root: Path, plan_path: Path, forge_out: Path, source_id: str, consumer: str) -> dict[str, object]:
    plan = load_canvas_plan(plan_path)
    names = consumers_for(plan, source_id)
    if consumer not in names:
        raise UsageError(f"{consumer} is not a consumer of {source_id}", code="usage")
    removed = release_consumer(forge_out, plan.id, source_id, consumer, names)
    return {"source": source_id, "consumer": consumer, "removed": removed, "consumers": list(names)}


def _dispatch(args: argparse.Namespace) -> object:
    root = _repo_root(args.root)
    if args.command == "accounts":
        return execute_accounts(args)
    if args.command == "sources":
        if args.sources_command != "release":
            raise UsageError(f"unknown sources command {args.sources_command}", code="usage")
        plan_path = args.plan or (root / "plans" / "canvas-zimage-tiers-v1.toml")
        forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
        return _release_source(root, plan_path, forge_out, str(args.source), str(args.consumer))
    if args.command == "status":
        forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
        if not status_path(forge_out).is_file():
            return {
                "items": [],
                "stalled": False,
                "failed": True,
                "exit_code": 4,
                "error": "status.json is missing",
            }
        return inspect_status(forge_out)
    if args.command == "canvas":
        # diffusers is an optional extra and must not load for LLM commands.
        forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
        if args.canvas_command == "upload":
            return _canvas_upload(root, args.canvas_plan, forge_out, args.parts, bool(args.force))
        if args.canvas_command == "run":
            from voxprint_forge.diffusion.cli import execute

            return execute(
                repo=root,
                plan_path=args.canvas_plan,
                forge_out=forge_out,
                force=bool(args.force),
                deadline_text=args.deadline,
                defer_upload=bool(args.defer_upload),
            )
        raise UsageError(f"unknown canvas command {args.canvas_command}", code="usage")
    plan_path = args.plan or (root / "plans" / "plan.toml")
    if args.command == "verify-logins":
        return verify_logins()
    if not plan_path.is_file():
        raise PlanError(f"missing plan {plan_path}", code="plan_io")
    plan = load_plan(plan_path)
    if args.command == "plan-summary":
        return format_estimates(plan)
    forge_out = args.forge_out or plan.output_dir.parent
    layout = Layout(forge_out)
    host = _live_host()
    if args.command in {"preflight", "run-all"} or args.command in ORDER:
        try:
            report = run_preflight(plan, root, forge_out)
        except StageError as exc:
            if args.command == "preflight" and isinstance(exc.report, dict):
                return exc.report
            raise
        if args.command == "preflight":
            return report
    if args.command == "run-all":
        return run_all(plan, host, root, layout, jobs=_jobs(args.jobs))
    model_id = str(args.model)
    model = next((item for item in plan.models if item.id == model_id), None)
    if model is None:
        raise UsageError(f"unknown model {model_id}", code="usage")
    if not model.enabled:
        raise UsageError(f"{model_id} is disabled in the plan", code="usage")
    budgets = None
    if args.command == "fit" and args.budget_gb:
        budgets = {float(item) for item in args.budget_gb}
    stop = args.command
    if stop not in ORDER:
        raise UsageError(f"unknown stage {stop}", code="usage")
    return run_model(plan, model, host, root, layout, stop_after=stop, budgets=budgets)


def _live_host() -> Host:
    hf_token = os.environ.get("HF_TOKEN")
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    return Host(
        runner=Runner(),
        downloader=HfDownloader(hf_token),
        remote=HubRemote(hf_token, kaggle_token),
    )


def _repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("FORGE_ROOT")
    if env:
        return Path(env)
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "plans" / "plan.toml").is_file():
            return candidate
    packaged = Path(__file__).resolve().parents[2]
    if (packaged / "plans" / "plan.toml").is_file():
        return packaged
    raise UsageError("cannot find the forge repo; pass --root", code="root")


if __name__ == "__main__":
    sys.exit(main())
