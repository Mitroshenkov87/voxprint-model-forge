"""``python -m voxprint_forge.diffusion.int4`` and the forge diffusion-int4 command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.disk import guard_disk, reclaim_verified
from voxprint_forge.diffusion.int4.fetch import fetch_weights
from voxprint_forge.diffusion.int4.gate import OwnEditDecision, decide_own_edit, load_phase0_baseline, parse_own_edit
from voxprint_forge.diffusion.int4.gpu import planning_gpu, step_minutes, total_minutes
from voxprint_forge.diffusion.int4.markers import done_dir
from voxprint_forge.diffusion.int4.pins import DEFAULT_SVD_RANK, DISK_FLOOR, PHASE0_RESULT, SCHEMA, planned_steps
from voxprint_forge.diffusion.int4.stage import format_estimates, run
from voxprint_forge.diffusion.int4.watch import watch_path
from voxprint_forge.errors import EXIT_UNEXPECTED, ForgeError, UsageError
from voxprint_forge.jsonio import emit


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register run, estimates, and disk-floor on an existing subparser group."""
    run_parser = sub.add_parser("run", help="Calibrate, quantize, evaluate, card, and upload.")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU smoke with tiny files. No GPU and no downloads.",
    )
    run_parser.add_argument("--svd-rank", type=int, default=DEFAULT_SVD_RANK, help="SVD rank. Default 256.")
    run_parser.add_argument("--qwen-image", action="store_true", help="Also quantize Qwen-Image INT4 r256.")
    run_parser.add_argument(
        "--done-dir",
        type=Path,
        default=None,
        help="Marker directory. Default: ~/.done/diffusion-int4.",
    )
    run_parser.add_argument("--watch", type=Path, default=None, help="Watch file. Default: ~/watch.json.")
    run_parser.add_argument("--only", default=None, help="Run one step. Earlier steps must already have markers.")
    _add_edit_flags(run_parser)
    estimates = sub.add_parser("estimates", help="Print the planning minutes for each step.")
    estimates.add_argument("--qwen-image", action="store_true", help="Include the optional Qwen-Image steps.")
    _add_edit_flags(estimates)
    disk = sub.add_parser("disk-floor", help="Exit 2 when free space stays under 15 percent after reclaim.")
    disk.add_argument("--path", type=Path, default=None, help="Filesystem to check. Default: forge-out.")
    disk.add_argument(
        "--done-dir",
        type=Path,
        default=None,
        help="Marker directory whose verified sources can be deleted.",
    )
    fetched = sub.add_parser("fetch", help="Download QuantFunc, the edit base, Wan experts, and the FP8 reference.")
    fetched.add_argument("--dest", type=Path, required=True, help="Directory that receives the snapshots.")


def _add_edit_flags(parser: argparse.ArgumentParser) -> None:
    """``--own-edit`` defaults to on. Phase 0 is an optional comparison baseline."""
    parser.add_argument(
        "--own-edit",
        choices=("auto", "on", "off"),
        default="on",
        help="on (default) always runs the own INT4 r256 quant. auto may skip it. off mirrors QuantFunc.",
    )
    parser.add_argument(
        "--phase0-result",
        type=Path,
        default=Path(PHASE0_RESULT),
        help="Phase 0 result.json. Missing is ignored. Present is an eval and card baseline, not the winner.",
    )


def _edit_plan(args: argparse.Namespace) -> tuple[OwnEditDecision, dict[str, object] | None, bool]:
    """Resolve the own-edit decision and the optional Phase 0 baseline."""
    path = args.phase0_result if isinstance(args.phase0_result, Path) else None
    decision = decide_own_edit(parse_own_edit(str(args.own_edit)), path)
    baseline = load_phase0_baseline(path)
    return decision, baseline, decision.action == "mirror"


def _estimate_text(
    decision: OwnEditDecision,
    baseline: dict[str, object] | None,
    *,
    qwen_image: bool,
    mirror: bool,
) -> str:
    """Decision line plus the A100 and H100 tables."""
    if baseline is None:
        note = " Phase 0 result is absent, so the eval has no Phase 0 baseline."
    else:
        note = " Phase 0 result is an eval and card baseline and does not choose the winner."
    table = format_estimates(qwen_image=qwen_image, mirror=mirror)
    return decision.reason + note + "\n\n" + table


def build_parser() -> argparse.ArgumentParser:
    """Parser for ``python -m voxprint_forge.diffusion.int4``."""
    parser = argparse.ArgumentParser(prog="python -m voxprint_forge.diffusion.int4")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--forge-out", type=Path, default=None)
    add_commands(parser.add_subparsers(dest="command", required=True))
    return parser


def perform(args: argparse.Namespace, repo: Path) -> dict[str, object]:
    """Run one diffusion-int4 command and return the result object."""
    command = str(getattr(args, "command", None) or getattr(args, "int4_command", ""))
    forge_out = args.forge_out or Path.home() / "forge-out"
    if command == "estimates":
        decision, baseline, mirror = _edit_plan(args)
        steps = planned_steps(qwen_image=bool(args.qwen_image), mirror=mirror)
        gpu = planning_gpu()
        return {
            "schema": SCHEMA,
            "gpu": gpu,
            "own_edit": decision.as_dict(),
            "phase0_baseline": baseline,
            "expected_min": total_minutes(steps, gpu),
            "gpus": {
                name: {
                    "expected_min": total_minutes(steps, name),
                    "steps": [
                        {
                            "id": step.step_id,
                            "expected_min": step_minutes(step, name),
                            "note": step.note,
                        }
                        for step in steps
                    ],
                }
                for name in ("a100", "h100")
            },
            "steps": [
                {"id": step.step_id, "expected_min": step_minutes(step, gpu), "note": step.note} for step in steps
            ],
            "text": _estimate_text(decision, baseline, qwen_image=bool(args.qwen_image), mirror=mirror),
        }
    if command == "fetch":
        token = os.environ.get("HF_TOKEN")
        paths = fetch_weights(args.dest, token, forge_out=forge_out)
        return {"schema": SCHEMA, "paths": paths}
    if command == "disk-floor":
        markers = done_dir(getattr(args, "done_dir", None))
        ratio = guard_disk(
            args.path or forge_out,
            floor=DISK_FLOOR,
            reclaim=lambda: reclaim_verified(markers),
        )
        return {"free_ratio": ratio, "floor": DISK_FLOOR}
    if command != "run":
        raise UsageError(f"unknown diffusion-int4 command {command}", code="usage")
    if args.svd_rank < 1:
        raise UsageError("svd rank must be at least 1", code="usage")
    decision, _baseline, _mirror = _edit_plan(args)
    ctx = StageContext(
        repo=repo,
        forge_out=forge_out,
        dry_run=bool(args.dry_run),
        svd_rank=int(args.svd_rank),
    )
    phase0 = args.phase0_result if isinstance(args.phase0_result, Path) else None
    return run(
        ctx,
        done=done_dir(args.done_dir),
        watch=watch_path(args.watch),
        qwen_image=bool(args.qwen_image),
        only=args.only,
        own_edit=decision,
        phase0_result=phase0,
    )


def execute(args: argparse.Namespace, *, repo: Path, json_mode: bool) -> int:
    """Run one diffusion-int4 command and return the process exit code."""
    command = str(getattr(args, "command", None) or getattr(args, "int4_command", ""))
    try:
        result = perform(args, repo)
        if command == "estimates" and not json_mode and isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                print(text)
                return 0
        emit(command, result, ok=True, exit_code=0, json_mode=json_mode)
        return 0
    except ForgeError as exc:
        emit(
            command or "diffusion-int4",
            None,
            ok=False,
            exit_code=exc.exit_code,
            json_mode=json_mode,
            error=str(exc),
            error_code=exc.code,
        )
        return exc.exit_code
    except Exception as exc:
        emit(
            command or "diffusion-int4",
            None,
            ok=False,
            exit_code=EXIT_UNEXPECTED,
            json_mode=json_mode,
            error=str(exc),
            error_code="unexpected",
        )
        return EXIT_UNEXPECTED


def main(argv: list[str] | None = None) -> int:
    """Module entry point."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    repo = _repo(args.root)
    return execute(args, repo=repo, json_mode=bool(args.json))


def _repo(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "data" / "diffusion_calib" / "prompts.json").is_file():
            return candidate
    packaged = Path(__file__).resolve().parents[4]
    if (packaged / "data" / "diffusion_calib" / "prompts.json").is_file():
        return packaged
    raise UsageError("cannot find the forge repo; pass --root", code="root")


if __name__ == "__main__":
    sys.exit(main())
