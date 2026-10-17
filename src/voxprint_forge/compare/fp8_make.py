"""Write the local FP8 files a compare arm reads. Nothing here is uploaded."""

from __future__ import annotations

from pathlib import Path
from typing import Never

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.plan import Job
from voxprint_forge.diffusion.fp8 import FORMAT, Profile, read_format
from voxprint_forge.diffusion.names import QWEN3_FP8_NAME, TURBO_FP8_NAME, ZIMAGE_FP8_NAME
from voxprint_forge.diffusion.weights import write_fp8_file
from voxprint_forge.scrub import active_secrets, scrub


def fp8_targets(job_id: str) -> tuple[tuple[str, Profile, str], ...]:
    """Directories and file names ``--make-fp8`` writes for one job."""
    if job_id == "D1":
        return (
            ("transformer", "diffusion", TURBO_FP8_NAME),
            ("text_encoder", "qwen3_encoder", QWEN3_FP8_NAME),
        )
    if job_id == "D4":
        return (("transformer", "diffusion", ZIMAGE_FP8_NAME),)
    return _unexpected_job(job_id)


def export_fp8(job: Job, out: Path, forge_out: Path) -> dict[str, object]:
    """Download the reference snapshot once and write the job's FP8 files."""
    snapshot = forge_out / "fp8-source" / job.id
    download_reference(job, snapshot)
    return produce_fp8(job, out, snapshot)


def download_reference(job: Job, dest: Path) -> Path:
    """Fetch the reference repo folders ``--make-fp8`` quantizes. No token is printed."""
    from huggingface_hub import snapshot_download

    arm = job.reference_arm()
    if not arm.repo:
        raise CompareSetup(f"{job.id} reference has no repo", code="compare_weights")
    folders = [folder for folder, _profile, _name in fp8_targets(job.id)]
    patterns = ["model_index.json", *[f"{folder}/**" for folder in folders]]
    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {
        "repo_id": arm.repo,
        "local_dir": str(dest),
        "allow_patterns": patterns,
    }
    if arm.revision:
        kwargs["revision"] = arm.revision
    try:
        snapshot_download(**kwargs)
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        raise CompareSetup(f"download failed for {arm.repo}: {message}", code="compare_download") from exc
    return dest


def produce_fp8(job: Job, out: Path, snapshot: Path) -> dict[str, object]:
    """Quantize the reference snapshot. An existing ``voxprint-fp8-v1`` file is left in place."""
    written: list[str] = []
    skipped: list[str] = []
    out.mkdir(parents=True, exist_ok=True)
    for folder, profile, name in fp8_targets(job.id):
        source = snapshot / folder
        if not source.is_dir():
            raise CompareSetup(f"{job.id} snapshot has no {folder}", code="compare_weights")
        dest = out / name
        if dest.is_file() and read_format(dest) == FORMAT:
            skipped.append(name)
            continue
        write_fp8_file(source, dest, profile)
        written.append(name)
    return {
        "job": job.id,
        "out": str(out),
        "written": written,
        "skipped": skipped,
        "exit_code": 0,
        "uploaded": False,
    }


def _unexpected_job(job_id: str) -> Never:
    raise CompareSetup(f"{job_id} has no FP8 export (use D1 or D4)", code="compare_plan")
