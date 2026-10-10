"""Pinned imatrix corpus. The bundled llama-imatrix rejects a file shorter than 2×ctx."""

from __future__ import annotations

from voxprint_forge.errors import ChecksumError, PlanError
from voxprint_forge.evaluate import data_dir
from voxprint_forge.hashing import sha256_file
from voxprint_forge.plan import Plan

# bartowski calibration_datav3. The gist states no licence.
CALIBRATION_URL = "https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8"
CALIBRATION_SHA256 = "200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026"
# GPT-2 content tokens of that file, with no added BOS. llama.cpp adds BOS on top.
CALIBRATION_TOKENS = 74048
LICENCE_STATUS = "licence unverified, calibration-only, not redistributed in model files"


def tokens_required(ctx: int, chunks: int) -> int:
    """Tokens required before llama-imatrix will honour ctx and chunks.

    tools/imatrix/imatrix.cpp at 98819068 errors when the tokenized file is
    shorter than 2*n_ctx. It also caps the chunk count at len(tokens)//n_ctx,
    so a request of `chunks` is real only when the file holds at least
    ctx*chunks tokens. The preflight requires both.
    """
    return max(2 * ctx, ctx * chunks)


def assert_calibration(plan: Plan) -> int:
    """Return the pinned token count, or raise when the file or the plan is short."""
    path = data_dir() / "imatrix.txt"
    digest = sha256_file(path)
    if digest != CALIBRATION_SHA256:
        raise ChecksumError(
            f"imatrix.txt sha256 {digest} is not the pinned calibration_datav3 {CALIBRATION_SHA256}",
            code="checksum",
        )
    need = tokens_required(plan.imatrix_ctx, plan.imatrix_chunks)
    if CALIBRATION_TOKENS < need:
        raise PlanError(
            f"calibration file has {CALIBRATION_TOKENS} tokens; need {need} "
            f"(at least 2*ctx={2 * plan.imatrix_ctx} and ctx*chunks={plan.imatrix_ctx * plan.imatrix_chunks})",
            code="plan_shape",
        )
    return CALIBRATION_TOKENS
