"""``forge compare`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from voxprint_forge.compare.plan import load_compare_plan
from voxprint_forge.compare.privacy import private_root_from_env
from voxprint_forge.compare.run import run_compare


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the compare subcommand."""
    command = sub.add_parser(
        "compare",
        help="Score original and compressed arms for one synth4 job.",
    )
    command.add_argument("--job", required=True, help="Job id from the compare plan, for example D1.")
    command.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Compare plan. Default: plans/compare-synth4.toml.",
    )


def execute(root: Path, forge_out: Path, job_id: str, plan_path: Path | None) -> dict[str, object]:
    """Load the plan and run one job. Exit codes are 0, 1, 2, and 3."""
    path = plan_path or (root / "plans" / "compare-synth4.toml")
    plan = load_compare_plan(path)
    job = plan.job(job_id)
    results = forge_out / "results" / job.id
    return run_compare(job, results, private_root=private_root_from_env())
