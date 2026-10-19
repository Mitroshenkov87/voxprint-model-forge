"""``forge compare`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from voxprint_forge.compare.cache import reclaim_all
from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.plan import load_compare_plan
from voxprint_forge.compare.privacy import private_root_from_env


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the compare subcommand."""
    command = sub.add_parser(
        "compare",
        help="Score original and compressed arms for one synth4 job.",
    )
    command.add_argument("--job", default=None, help="Job id from the compare plan, for example D1.")
    command.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Compare plan. Default: plans/compare-synth4.toml.",
    )
    command.add_argument(
        "--make-fp8",
        action="store_true",
        help="Write local FP8 files for D1 or D4 from the reference snapshot and exit.",
    )
    command.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for --make-fp8. Files stay local and are not uploaded.",
    )
    command.add_argument(
        "--reclaim-cache",
        action="store_true",
        help="Delete every snapshot under $FORGE_OUT/compare-cache and exit.",
    )


def execute(
    root: Path,
    forge_out: Path,
    job_id: str | None,
    plan_path: Path | None,
    *,
    make_fp8: bool = False,
    out: Path | None = None,
    reclaim_cache: bool = False,
) -> dict[str, object]:
    """Load the plan and run one job. Exit codes are 0, 1, 2, and 3.

    ``--make-fp8`` writes the local FP8 files and does not need a GPU.
    ``--reclaim-cache`` deletes the shared snapshot directory and does not run a job.
    """
    from voxprint_forge.compare.fp8_make import export_fp8
    from voxprint_forge.compare.run import run_compare

    if reclaim_cache:
        removed = reclaim_all(forge_out / "compare-cache")
        return {"exit_code": 0, "removed": removed, "uploaded": False}
    if not job_id:
        raise CompareSetup("compare needs --job", code="compare_plan")
    path = plan_path or (root / "plans" / "compare-synth4.toml")
    plan = load_compare_plan(path)
    job = plan.job(job_id)
    if make_fp8:
        if out is None:
            raise CompareSetup("compare --make-fp8 needs --out", code="compare_plan")
        return export_fp8(job, out, forge_out)
    results = forge_out / "results" / job.id
    return run_compare(
        job,
        results,
        private_root=private_root_from_env(),
        cache=forge_out / "compare-cache",
        plan=plan,
    )
