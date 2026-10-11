"""Card, sha256, public Hugging Face, and the Drive fallback.

Sources are deleted by the reclaim step, and only after ``verified`` is set.
A refused licence does not fall through to Drive.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.diffusion.cardrender import CardContext, render_canvas_card, today_stamp
from voxprint_forge.diffusion.int4.pins import DRIVE_FOLDER_ID, HF_USER_DEFAULT
from voxprint_forge.drive import upload_drive
from voxprint_forge.errors import LicenceError, StageError
from voxprint_forge.hashing import sha256_file, write_sha256
from voxprint_forge.upload import Artifact, HubRemote, assert_public_upload, upload_verified

INT4_METHOD = (
    "SVDQuant INT4 (W4A4, group 64) with a high-precision low-rank branch, "
    "run by MIT HAN Lab deepcompressor at the pinned commit. The fast_int4 "
    "overlay uses the upstream fast calibration budget and the SVD rank on "
    "this card. The checkpoint is exported to the Nunchaku svdquant safetensors "
    "layout. Qwen-Image-Edit-2511 keeps zero_cond_t from the base transformer "
    "config. Inference for that edit model uses the pinned tonera vitoom-nunchaku "
    "wheel, because upstream Nunchaku does not apply zero_cond_t."
)
INT4_CALIBRATION = (
    "Calibration prompts and image refs live in data/diffusion_calib. "
    "The prompts are original Russian and English edit instructions covering "
    "people, text, and hands. The image refs are original synthetic swatches, "
    "not photographs of real people. Both are Apache-2.0 and written for this "
    "repository. This is not the COCO or MJHQ calibration set."
)
INT4_LIMITS = (
    "A 16 GB fit means the sampled nvidia-smi peak is at most 15.0 GiB. "
    "A 24 GB fit means that peak is at most 23.0 GiB. Seconds per image are "
    "measured after one warm-up. This card is not a safety filter. "
    "Dry-run numbers are synthetic and are not a quality claim."
)
INT4_NOTICE = "Modified by Voxprint: SVDQuant INT4 calibration and Nunchaku export."
WAN_METHOD = (
    "Wan2.2-T2V-A14B high-noise and low-noise experts are converted with "
    "city96 ComfyUI-GGUF at the pinned commit and quantized with llama.cpp "
    "llama-quantize at tag b3962 (the commit that patch targets) to Q4_K_M "
    "and Q5_K_M. The 5D tensors are restored with fix_5d_tensors.py. "
    "A separate FP8 copy uses voxprint-fp8-v1 (float8_e4m3fn, per-tensor BF16 scale)."
)
WAN_CALIBRATION = "No imatrix. GGUF types are K-quants without an importance matrix. FP8 is a dtype conversion."
WAN_LIMITS = (
    "The smoke test is 5 frames at 832*480 and 4 sampling steps with offload. "
    "It checks that the pinned Wan generate.py accepts the checkpoint. "
    "It is not a quality score. This card is not a safety filter."
)
WAN_NOTICE = "Modified by Voxprint: GGUF Q4_K_M and Q5_K_M quantization and FP8 conversion."


@dataclass(frozen=True)
class Published:
    """One verified upload, or a recorded failure that did not verify."""

    verified: bool
    records: tuple[dict[str, object], ...]
    hf: str
    drive: str


def render_int4_card(
    repo: Path,
    *,
    title: str,
    base_models: tuple[str, ...],
    pipeline_tag: str,
    quant_lines: tuple[str, ...],
    eval_lines: tuple[str, ...],
    credits: tuple[str, ...],
    kind: str,
) -> str:
    """Render the canvas card template. Method text is Forge-authored."""
    if kind == "wan":
        method, calibration, limits, notice = WAN_METHOD, WAN_CALIBRATION, WAN_LIMITS, WAN_NOTICE
    else:
        method, calibration, limits, notice = INT4_METHOD, INT4_CALIBRATION, INT4_LIMITS, INT4_NOTICE
    card = CardContext(
        title=title,
        base_models=base_models,
        base_lines=tuple(f"Quantized from: `{item}`" for item in base_models),
        pipeline_tag=pipeline_tag,
        tags=("voxprint", "quantized", kind),
        language=("ru", "en"),
        quant_lines=quant_lines,
        eval_lines=eval_lines,
        credits=credits,
        stamped=today_stamp(),
    )
    return render_canvas_card(
        repo,
        card,
        method=method,
        calibration=calibration,
        limits=limits,
        modification_notice=notice,
    )


def write_publish_tree(output: Path, files: list[Path], card_text: str) -> list[Path]:
    """Copy deliverables, write the card and sha256 sidecars, and return the upload set."""
    output.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    readme = output / "README.md"
    readme.write_text(card_text, encoding="utf-8")
    published.append(readme)
    used = {"README.md"}
    for src in files:
        name = src.name
        if name in used:
            name = f"{src.parent.name}__{src.name}"
        used.add(name)
        dest = output / name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        published.append(dest)
    for path in list(published):
        published.append(write_sha256(path))
    return published


def _artifacts(paths: list[Path]) -> list[Artifact]:
    found: list[Artifact] = []
    for path in paths:
        if path.name.endswith(".sha256"):
            continue
        digest = sha256_file(path)
        found.append(Artifact(path, path.name, digest, path.stat().st_size, "shared", "", "file"))
    return found


def _hub_repo(slug: str) -> str:
    user = os.environ.get("HF_USER", "").strip() or HF_USER_DEFAULT
    return f"{user}/{slug}"


class LocalMirror:
    """Copy files into a directory and report their sha256. Used by --dry-run."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def upload_hf(self, repo_id: str, files: list[Artifact]) -> list[dict[str, object]]:
        """Copy to the local Hub stand-in and check the hash."""
        folder = self.root / "hf" / repo_id
        return self._copy(folder, files, target="hf", url=f"file://{folder.as_posix()}")

    def upload_drive(self, folder_id: str, files: list[Artifact]) -> list[dict[str, object]]:
        """Copy to the local Drive stand-in and check the hash."""
        folder = self.root / "drive" / folder_id
        return self._copy(folder, files, target="drive", url=f"file://{folder.as_posix()}")

    def _copy(
        self,
        folder: Path,
        files: list[Artifact],
        *,
        target: str,
        url: str,
    ) -> list[dict[str, object]]:
        folder.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for artifact in files:
            dest = folder / artifact.name
            shutil.copy2(artifact.path, dest)
            digest = sha256_file(dest)
            if digest != artifact.sha256 or dest.stat().st_size != artifact.size:
                raise StageError(f"{target} sha256 mismatch for {artifact.name}", code="upload_mismatch")
            records.append(
                {
                    "name": artifact.name,
                    "sha256": digest,
                    "size": artifact.size,
                    "target": target,
                    "url": url,
                }
            )
        return records


def publish_verified(
    output: Path,
    *,
    slug: str,
    card_text: str,
    licence: str,
    hub: Callable[[str, list[Artifact]], list[dict[str, object]]] | None = None,
    drive: Callable[[str, list[Artifact]], list[dict[str, object]]] | None = None,
    folder_id: str = DRIVE_FOLDER_ID,
) -> Published:
    """Upload after the licence gate. Drive runs when Hub fails for another reason.

    A licence or card refusal is not sent to Drive. One verified mirror is enough
    to set ``verified``.
    """
    files = _artifacts([path for path in output.iterdir() if path.is_file()])
    repo_id = _hub_repo(slug)
    assert_public_upload(licence=licence, card_text=card_text, repo=repo_id)
    records: list[dict[str, object]] = []
    hf_state = "skipped"
    drive_state = "skipped"
    hf_error: StageError | None = None
    if hub is not None:
        try:
            records.extend(hub(repo_id, files))
            hf_state = "uploaded"
        except LicenceError:
            raise
        except StageError as exc:
            hf_error = exc
            hf_state = "failed"
    if hf_state != "uploaded" and drive is not None:
        try:
            records.extend(drive(folder_id, files))
            drive_state = "uploaded"
        except StageError:
            if hf_error is not None:
                raise
            drive_state = "failed"
    elif hf_state == "uploaded" and drive is not None:
        try:
            records.extend(drive(folder_id, files))
            drive_state = "uploaded"
        except StageError:
            drive_state = "failed"
    verified = hf_state == "uploaded" or drive_state == "uploaded"
    if not verified:
        if hf_error is not None:
            raise hf_error
        raise StageError(f"{slug} upload was not verified", code="upload_mismatch")
    return Published(verified=True, records=tuple(records), hf=hf_state, drive=drive_state)


def live_hub(repo_id: str, files: list[Artifact]) -> list[dict[str, object]]:
    """Create a public Hub repo and upload each file once the remote hash matches."""
    remote = HubRemote(os.environ.get("HF_TOKEN"), os.environ.get("KAGGLE_API_TOKEN"))
    url = remote.hf_private_repo(repo_id)
    records: list[dict[str, object]] = []
    for artifact in files:
        upload_verified(
            lambda artifact=artifact: remote.hf_upload(repo_id, artifact.path, artifact.name),
            lambda artifact=artifact: remote.hf_stat(repo_id, artifact.name),
            local_sha=artifact.sha256,
            local_size=artifact.size,
            attempts=4,
            sleep=time.sleep,
        )
        records.append(
            {
                "name": artifact.name,
                "sha256": artifact.sha256,
                "size": artifact.size,
                "target": "hf",
                "url": url,
            }
        )
    return records


def live_drive(
    folder_id: str,
    files: list[Artifact],
    *,
    model_id: str = "diffusion-int4",
) -> list[dict[str, object]]:
    """Upload to the Voxprint Drive folder. The folder id is set for this call only."""
    previous = os.environ.get("FORGE_DRIVE_ROOT_FOLDER_ID")
    os.environ["FORGE_DRIVE_ROOT_FOLDER_ID"] = folder_id
    try:
        return upload_drive(files, model_id=model_id, date=today_stamp())
    finally:
        if previous is None:
            os.environ.pop("FORGE_DRIVE_ROOT_FOLDER_ID", None)
        else:
            os.environ["FORGE_DRIVE_ROOT_FOLDER_ID"] = previous
