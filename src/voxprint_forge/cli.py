"""forge command line. `--json` prints one object; logs stay on stderr."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

from voxprint_forge.accounts import execute_accounts
from voxprint_forge.compare.cli import add_command as add_compare_command
from voxprint_forge.compare.cli import execute as execute_compare
from voxprint_forge.diffusion.int4.cli import (
    add_commands as add_int4_commands,
)
from voxprint_forge.diffusion.int4.cli import (
    perform as perform_int4,
)
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
from voxprint_forge.kl.cli import add_command as add_kl_command
from voxprint_forge.kl.cli import execute as execute_kl
from voxprint_forge.layout import Layout
from voxprint_forge.llama import Runner
from voxprint_forge.lora.cli import add_command as add_lora_command
from voxprint_forge.lora.cli import execute as execute_lora
from voxprint_forge.mdl import host_descriptors, push_index, revision_map, run_backfill, run_build
from voxprint_forge.pipeline import ORDER, Host, run_all, run_model
from voxprint_forge.plan import format_estimates, load_plan
from voxprint_forge.preflight import run_preflight
from voxprint_forge.statusfile import inspect_status, status_path
from voxprint_forge.upload import HubRemote, verify_logins
from voxprint_forge.version import version_line


def build_parser() -> argparse.ArgumentParser:
    """The forge argument parser. The CLI reference is generated from this."""
    # Python 3.14 colors help when stdout is a terminal. The reference is plain text.
    kwargs: dict[str, bool] = {}
    if "color" in inspect.signature(argparse.ArgumentParser).parameters:
        kwargs["color"] = False
    parser = argparse.ArgumentParser(prog="forge", formatter_class=_Help, **kwargs)
    parser.add_argument("--version", action="version", version=version_line())
    parser.add_argument("--json", action="store_true", help="Print one JSON object on stdout.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root. Default: search from the working directory.",
    )
    parser.add_argument("--plan", type=Path, default=None, help="Plan TOML. Default: plans/plan.toml.")
    parser.add_argument("--forge-out", type=Path, default=None, help="Output root. Default: the plan output directory.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan-summary", help="Print the plan's time estimate.")
    sub.add_parser("verify-logins", help="Check Hugging Face and Kaggle logins.")
    preflight = sub.add_parser("preflight", help="Run the seconds-long checks and print the report.")
    preflight.add_argument(
        "--vm",
        action="store_true",
        help="Check a fresh VM image before a paid GPU job: Python, torch/CUDA, [compare] versions, build tools, "
        "llama.cpp, the convert_hf_to_gguf.py imports, the voxprint_common wheel, HF_TOKEN and disk. "
        "Exit 6 on any FAIL.",
    )
    preflight.add_argument(
        "--for",
        dest="vm_scope",
        choices=("all", "lora"),
        default="all",
        help="With --vm: what the VM is for. 'lora' skips cmake, ninja, nvcc, c++, llama.cpp and the converter "
        "imports. Default: all.",
    )
    preflight.add_argument("--min-free-gb", type=float, default=150.0, help="Disk needed for --vm. Default: 150.")
    preflight.add_argument(
        "--refset",
        action="store_true",
        help="Check this computer for Build a set from references: diffusers img2img, Z-Image-Turbo, the photo "
        "preparation models and env, DINOv2, CUDA 16 GB+, 32 GB RAM and disk. Exit 6 on a missing requirement.",
    )
    sub.add_parser("status", help="Read status.json and report stalled or failed work.")
    accounts = sub.add_parser("accounts", help="Store Hugging Face, Kaggle, and Drive credentials in the OS keyring.")
    account_sub = accounts.add_subparsers(dest="accounts_command", required=True)
    account_add = account_sub.add_parser("add", help="Save a token for one target.")
    account_add.add_argument("--target", required=True, choices=("hf", "kaggle", "drive"))
    account_add.add_argument("--token", default="")
    account_add.add_argument("--login", choices=("paste", "browser"), default="paste")
    account_sub.add_parser("list", help="Show connected account names.")
    account_test = account_sub.add_parser("test", help="Check one stored connection.")
    account_test.add_argument("--target", required=True, choices=("hf", "kaggle", "drive"))
    account_remove = account_sub.add_parser("remove", help="Delete one stored credential.")
    account_remove.add_argument("--target", required=True, choices=("hf", "kaggle", "drive"))
    sources = sub.add_parser("sources", help="Record that a consumer has finished with a shared download.")
    source_sub = sources.add_subparsers(dest="sources_command", required=True)
    source_release = source_sub.add_parser("release", help="Release one source for one consumer.")
    source_release.add_argument("--source", required=True)
    source_release.add_argument("--consumer", required=True)
    run = sub.add_parser("run-all", help="Run every enabled model through upload.")
    run.add_argument("--jobs", type=int, default=None, help="How many models run at once. Default: FORGE_JOBS or 1.")
    stage_help = {
        "fetch": "Download pinned files for one model.",
        "abliterate": "Run the refusal-direction ablation for one model.",
        "quantize": "Build the importance matrix and GGUF quants.",
        "eval": "Record perplexity, refusal rate, and prose samples.",
        "card": "Render the model card.",
        "publish-prep": "Write the registry draft.",
        "upload": "Upload files and verify sha256 and size. Hub repos are public. Kaggle stays private.",
    }
    for name in ("fetch", "abliterate", "quantize", "eval", "card", "publish-prep", "upload"):
        command = sub.add_parser(name, help=stage_help[name])
        command.add_argument("--model", required=True, help="Model id from the plan.")
    fit = sub.add_parser("fit", help="Measure quants and record the VRAM/RAM split for each tier.")
    fit.add_argument("--model", required=True, help="Model id from the plan.")
    fit.add_argument(
        "--budget-gb",
        type=float,
        action="append",
        default=None,
        help="Keep tiers of this size. Repeatable.",
    )
    canvas = sub.add_parser("canvas", help="Prepare canvas image tiers.")
    canvas_sub = canvas.add_subparsers(dest="canvas_command", required=True)
    canvas_run = canvas_sub.add_parser("run", help="Run a canvas plan.")
    canvas_run.add_argument("canvas_plan", type=Path, metavar="PLAN")
    canvas_run.add_argument("--force", action="store_true", help="Start even if an LLM job is running.")
    canvas_run.add_argument("--defer-upload", action="store_true", help="Prepare files and skip the upload.")
    canvas_run.add_argument("--deadline", default=None, help="ISO time. Skip parts that would miss it.")
    canvas_upload = canvas_sub.add_parser("upload", help="Upload parts that were deferred.")
    canvas_upload.add_argument("canvas_plan", type=Path, metavar="PLAN")
    canvas_upload.add_argument("--parts", default=None, help="Comma-separated part ids. Default: the whole plan.")
    canvas_upload.add_argument("--force", action="store_true", help="Start even if an LLM job is running.")
    int4 = sub.add_parser("diffusion-int4", help="Wan 2.2 GGUF and FP8.")
    add_int4_commands(int4.add_subparsers(dest="int4_command", required=True))
    add_compare_command(sub)
    add_kl_command(sub)
    add_lora_command(sub)
    index = sub.add_parser("index", help="Build or push the model index, or backfill voxprint-model.json.")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    index_build = index_sub.add_parser(
        "build",
        help="Write index.json from local voxprint-model.json files. Does not upload.",
    )
    index_build.add_argument("--from", dest="manifest_from", type=Path, required=True, help="Directory to scan.")
    index_build.add_argument("--out", type=Path, required=True, help="Where to write index.json.")
    index_build.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="REPO=SHA",
        help="Commit of voxprint-model.json in REPO, 40 hex digits. Repeat for each repo.",
    )
    index_build.add_argument(
        "--serial",
        type=int,
        default=None,
        help="Index serial. Default: the published serial plus one, at least 2.",
    )
    index_push = index_sub.add_parser(
        "push",
        help="Sign index.json and upload it with index.json.sig. Does nothing unless --push is set.",
    )
    index_push.add_argument(
        "--push",
        action="store_true",
        help="Upload to Mitroshenkov87/voxprint-model-index. Required; omitted runs do not upload.",
    )
    index_push.add_argument("--index", type=Path, required=True, help="index.json to upload.")
    index_push.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Ed25519 private key PEM. Default: VOXPRINT_MDL_INDEX_KEY_FILE. Never printed.",
    )
    index_host = index_sub.add_parser(
        "host",
        help="Upload official-source voxprint-model.json files into the index dataset. Needs --push.",
    )
    index_host.add_argument("--from", dest="manifest_from", type=Path, required=True, help="Directory to scan.")
    index_host.add_argument(
        "--push",
        action="store_true",
        help="Upload to Mitroshenkov87/voxprint-model-index under descriptors/<id>/. Required.",
    )
    index_backfill = index_sub.add_parser(
        "backfill",
        help="Write voxprint-model.json for published voxprint-* repos. Does not upload.",
    )
    index_backfill.add_argument("--out", type=Path, required=True, help="Directory for the generated files.")
    index_backfill.add_argument("--user", default="Mitroshenkov87", help="Hugging Face user to list.")
    return parser


class _Help(argparse.HelpFormatter):
    """Help text at 88 columns. The usage line stays unwrapped.

    Python 3.12 splits a subparser's ``{cmd} ...`` and may wrap the ellipsis.
    Python 3.14 keeps that token together. A wide usage line is the same on both.
    """

    def __init__(self, prog: str) -> None:
        super().__init__(prog, width=88)

    def _format_usage(
        self,
        usage: str | None,
        actions: list[argparse.Action],
        groups: list[argparse._ArgumentGroup],
        prefix: str | None,
    ) -> str:
        saved = self._width
        self._width = 10_000
        try:
            return super()._format_usage(usage, actions, groups, prefix)
        finally:
            self._width = saved


def main(argv: list[str] | None = None) -> int:
    """Run the forge CLI and return the process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE
    try:
        result = _dispatch(args)
        if (
            not args.json
            and args.command == "diffusion-int4"
            and getattr(args, "int4_command", "") == "estimates"
            and isinstance(result, dict)
            and isinstance(result.get("text"), str)
        ):
            print(result["text"])
            return 0
        if not args.json and args.command == "preflight" and getattr(args, "vm", False) and isinstance(result, dict):
            print(result["text"])
            return _exit_code(args.command, result)
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

    if not force and llm_job_running(forge_out):
        raise UsageError(
            "LLM bootstrap is still running (bootstrap.lock or scripts/bootstrap.sh). "
            "Pass --force to start anyway.",
            code="llm_running",
        )
    token = os.environ.get("HF_TOKEN", "").strip() or None
    from voxprint_forge.diffusion.release import release_canvas_plan

    return release_canvas_plan(
        repo=root,
        plan_path=plan_path,
        forge_out=forge_out,
        token=token,
        parts_text=parts,
    )


def _index(args: argparse.Namespace) -> dict[str, object]:
    if args.index_command == "build":
        return run_build(args.manifest_from, args.out, revisions=revision_map(args.revision), serial=args.serial)
    if args.index_command == "push":
        if not args.push:
            raise UsageError("index push does not upload unless --push is set", code="usage")
        return push_index(args.index, key_file=args.key_file)
    if args.index_command == "host":
        if not args.push:
            raise UsageError("index host does not upload unless --push is set", code="usage")
        return host_descriptors(args.manifest_from)
    if args.index_command == "backfill":
        return run_backfill(args.out, str(args.user))
    raise UsageError(f"unknown index command {args.index_command}", code="usage")


def _exit_code(command: str, result: object) -> int:
    if not isinstance(result, dict):
        return 0
    if command == "compare":
        return int(result.get("exit_code", 2))
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
    if args.command == "preflight" and getattr(args, "refset", False):
        from voxprint_forge.preflight import refset_preflight

        return refset_preflight()
    root = _repo_root(args.root)
    if args.command == "accounts":
        return execute_accounts(args)
    if args.command == "preflight" and args.vm:
        from voxprint_forge.vmcheck import run_vm_checks

        forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
        return run_vm_checks(root, forge_out, min_free_gb=args.min_free_gb, scope=args.vm_scope)
    if args.command == "index":
        return _index(args)
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
    if args.command == "diffusion-int4":
        return perform_int4(args, root)
    if args.command == "lora":
        return execute_lora(args)
    if args.command == "compare":
        forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
        return execute_compare(
            root,
            forge_out,
            args.job,
            args.plan,
            make_fp8=bool(args.make_fp8),
            out=args.out,
            reclaim_cache=bool(args.reclaim_cache),
        )
    if args.command == "kl":
        forge_out = args.forge_out or Path(os.environ.get("FORGE_OUT", Path.home() / "forge-out"))
        return execute_kl(
            root,
            forge_out,
            job_id=args.job,
            plan_path=args.plan,
            bf16=args.bf16,
            quants=args.quant,
            chunks=args.chunks,
            ctx=args.ctx,
            render=args.render,
        )
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
