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
from voxprint_forge.mdl import (
    MANIFEST_NAME,
    ManifestInput,
    build_document,
    card_url_for,
    drive_mirror,
    hub_mirrors,
    infer_manifest,
    manifest_artifact,
    refresh_manifest,
    retarget_mirrors,
    rows_from_artifacts,
    write_manifest,
)
from voxprint_forge.upload import Artifact, HubRemote, assert_public_upload, upload_verified

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
    method_text: str | None = None,
    calibration_text: str | None = None,
    notice: str | None = None,
) -> str:
    """Render the canvas card template. Method text is Forge-authored."""
    if kind != "wan":
        raise StageError(f"unknown diffusion-int4 card {kind}", code="card")
    method, calibration, limits, notice_text = WAN_METHOD, WAN_CALIBRATION, WAN_LIMITS, WAN_NOTICE
    if method_text is not None:
        method = method_text
    if calibration_text is not None:
        calibration = calibration_text
    if notice is not None:
        notice_text = notice
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
        modification_notice=notice_text,
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


def _attach_manifest(
    output: Path,
    paths: list[Path],
    spec: ManifestInput,
    *,
    slug: str,
    repo_id: str,
    stamp: str,
    hub: Callable[[str, list[Artifact]], list[dict[str, object]]] | None,
    drive: Callable[[str, list[Artifact]], list[dict[str, object]]] | None,
) -> list[Artifact]:
    files = _artifacts(paths)
    mirrors: list[dict[str, str]] = []
    if hub is not None:
        mirrors.extend(hub_mirrors(hf_repo=repo_id))
    if drive is not None:
        mirrors.append(drive_mirror(slug, stamp))
    if not mirrors:
        raise StageError(f"{slug} has no mirror", code="mdl")
    document = build_document(
        model_id=spec.model_id,
        display_name=spec.display_name,
        app_ids=spec.app_ids,
        task=spec.task,
        license=spec.license,
        files=rows_from_artifacts(files),
        mirrors=mirrors,
        min_vram_gb=spec.min_vram_gb,
        recommended_vram_gb=spec.recommended_vram_gb,
        gpu_min_cc=spec.gpu_min_cc,
        engine=spec.engine,
        version_req=spec.version_req,
        base_model=spec.base_model,
        card_url=card_url_for(mirrors),
        flags=spec.flags,
        on_demand=spec.on_demand,
        tag=spec.tag,
        date=spec.date or stamp,
    )
    path = write_manifest(output, document)
    files.append(manifest_artifact(path))
    return files


def publish_verified(
    output: Path,
    *,
    slug: str,
    card_text: str,
    licence: str,
    hub: Callable[[str, list[Artifact]], list[dict[str, object]]] | None = None,
    drive: Callable[[str, list[Artifact]], list[dict[str, object]]] | None = None,
    folder_id: str = DRIVE_FOLDER_ID,
    manifest: ManifestInput | None = None,
    date: str = "",
) -> Published:
    """Upload after the licence gate. Drive runs when Hub fails for another reason.

    A licence or card refusal is not sent to Drive. One verified mirror is enough
    to set ``verified``. ``voxprint-model.json`` is written beside the card and
    uploaded with the other files.
    """
    repo_id = _hub_repo(slug)
    assert_public_upload(licence=licence, card_text=card_text, repo=repo_id)
    stamp = date or today_stamp()
    spec = manifest if manifest is not None else infer_manifest(slug, licence)
    paths = [path for path in output.iterdir() if path.is_file() and path.name != MANIFEST_NAME]
    files = _attach_manifest(output, paths, spec, slug=slug, repo_id=repo_id, stamp=stamp, hub=hub, drive=drive)
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
            manifest_path = output / MANIFEST_NAME
            retarget_mirrors(manifest_path, [drive_mirror(slug, stamp)])
            files = refresh_manifest(files, manifest_path)
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
