"""Load QuantFunc Qwen-Image-Edit INT4 on the pinned vitoom wheel.

QuantFunc packs ``img_mod.1`` and ``txt_mod.1`` as TinyChat W4A16: int32
``qweight``, ``wscales``, ``wzeros``, and no ``smooth_factor``. That matches
official Nunchaku ``AWQW4A16Linear``. The pinned Linux wheel
``nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311`` and the Windows
``dev20260629`` wheel share one ``transformer_qwenimage.py`` (CRLF aside) and
both replace those linears with ``SVDQW8A16Linear``. ``patch_scale_key`` runs
before ``load_state_dict`` and raises when ``smooth_factor`` is missing. The
same source would fail if it were rebuilt. Z-Image does not use this module,
which is why that INT4 file loads on the same wheel.

The shim fills identity smooth (ones) and unpacks the W4 matrix to a float
``.weight``, dropping ``qweight``, ``wscales``, and ``wzeros``. The wheel
already skips missing ``qweight`` / ``wscales`` when ``.weight`` is present,
then ``_quantize_from_fp_weight_bias`` requantizes on CUDA. The unpacked
matrix stays in the adanorm order the wheel's ``0::6`` split expects.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TypedDict

import torch

from voxprint_forge.diffusion.int4.pins import (
    QUANTFUNC_R128,
    QUANTFUNC_REPO,
    QUANTFUNC_REVISION,
)
from voxprint_forge.diffusion.int4.qwen_pack import GROUP_SIZE, dequant_tinychat_w4
from voxprint_forge.errors import StageError

try:
    import nunchaku.models.transformers.transformer_qwenimage as _qwen_transformer
    import nunchaku.models.transformers.utils as _nunchaku_utils
except ImportError:
    _qwen_transformer = None
    _nunchaku_utils = None

_MOD_TAILS = (".img_mod.1", ".txt_mod.1")
_PACKED_INT32 = frozenset({"I32", "INT32"})
_HEADER_LIMIT = 8_000_000


class TensorHeader(TypedDict):
    """One safetensors tensor record from the JSON header."""

    dtype: str
    shape: list[int]


def quantfunc_header_url(filename: str = QUANTFUNC_R128) -> str:
    """Hugging Face resolve URL for one QuantFunc safetensors file."""
    return f"https://huggingface.co/{QUANTFUNC_REPO}/resolve/{QUANTFUNC_REVISION}/{filename}"


def fetch_safetensors_header(url: str) -> dict[str, TensorHeader]:
    """Read a safetensors JSON header with HTTP range requests.

    The tensor bytes stay on the server. The first request is 8 bytes (the
    header length). The second request is that header and nothing after it.
    """
    length_bytes = _read_range(url, 0, 7)
    if len(length_bytes) != 8:
        raise StageError(f"safetensors length prefix from {url} was {len(length_bytes)} bytes", code="quantfunc_header")
    length = int.from_bytes(length_bytes, "little")
    if length < 2 or length > _HEADER_LIMIT:
        raise StageError(f"safetensors header length {length} from {url} is not usable", code="quantfunc_header")
    raw = _read_range(url, 8, 8 + length - 1)
    if len(raw) != length:
        raise StageError(
            f"safetensors header from {url} was {len(raw)} bytes, expected {length}",
            code="quantfunc_header",
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageError(f"safetensors header from {url} is not JSON", code="quantfunc_header") from exc
    if not isinstance(payload, dict):
        raise StageError(f"safetensors header from {url} is not an object", code="quantfunc_header")
    tensors: dict[str, TensorHeader] = {}
    for key, info in payload.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        dtype = info.get("dtype")
        shape = info.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            continue
        tensors[key] = {"dtype": dtype, "shape": [int(item) for item in shape]}
    if not tensors:
        raise StageError(f"safetensors header from {url} has no tensors", code="quantfunc_header")
    return tensors


def quantfunc_smooth_plan(tensors: dict[str, TensorHeader]) -> dict[str, list[int]]:
    """Identity smooth shapes for W4A16 mods that omit ``smooth_factor``.

    ``in_features`` is ``qweight.shape[1] * 2`` because TinyChat stores int16
    ``[oc / 4, ic]`` and the file views it as int32 ``[oc / 4, ic / 2]``.
    """
    plan: dict[str, list[int]] = {}
    for key, info in tensors.items():
        prefix = _mod_prefix(key, info["dtype"])
        if prefix is None:
            continue
        if f"{prefix}.smooth_factor" in tensors:
            continue
        shape = info["shape"]
        if len(shape) != 2:
            continue
        features = int(shape[1]) * 2
        plan[f"{prefix}.smooth_factor"] = [features]
        plan[f"{prefix}.smooth_factor_orig"] = [features]
    return plan


def adapt_quantfunc_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inject identity smooth and replace TinyChat W4A16 mods with float weights.

    Attention W4A4 tensors (int8 ``qweight``, no ``wzeros``) are left as they
    are. Only ``img_mod.1`` and ``txt_mod.1`` change.
    """
    prefixes = []
    for key, value in state.items():
        prefix = _mod_prefix(key, "I32" if value.dtype == torch.int32 else "")
        if prefix is None or prefix in prefixes:
            continue
        prefixes.append(prefix)
    for prefix in prefixes:
        qweight = state.get(f"{prefix}.qweight")
        wscales = state.get(f"{prefix}.wscales")
        wzeros = state.get(f"{prefix}.wzeros")
        if qweight is None or wscales is None or wzeros is None:
            continue
        if qweight.dtype != torch.int32 or qweight.ndim != 2 or wscales.ndim != 2:
            continue
        features = int(qweight.shape[1]) * 2
        ones = torch.ones((features,), dtype=torch.bfloat16)
        state.setdefault(f"{prefix}.smooth_factor", ones)
        state.setdefault(f"{prefix}.smooth_factor_orig", ones.clone())
        state[f"{prefix}.weight"] = dequant_tinychat_w4(qweight, wscales, wzeros, group_size=GROUP_SIZE).to(
            dtype=torch.bfloat16
        )
        state.pop(f"{prefix}.qweight", None)
        state.pop(f"{prefix}.wscales", None)
        state.pop(f"{prefix}.wzeros", None)
    return state


def preflight_missing_keys(
    module_keys: list[str],
    checkpoint: dict[str, torch.Tensor],
    w8_prefixes: set[str],
) -> list[str]:
    """Keys that vitoom ``patch_scale_key`` would still raise on.

    This is the wheel's filter: ``.qweight``, ``.wscales``, and ``.bias`` on an
    ``SVDQW8A16Linear`` are skipped when ``.weight`` is already in the
    checkpoint. ``smooth_factor`` is not skipped.
    """
    missing = [key for key in module_keys if key not in checkpoint]
    filtered: list[str] = []
    for key in missing:
        if not key.endswith((".qweight", ".wscales", ".bias")):
            filtered.append(key)
            continue
        module_name = key.rsplit(".", 1)[0]
        if (
            key.endswith(".bias")
            and module_name in w8_prefixes
            and f"{module_name}.qweight" in checkpoint
            and f"{module_name}.wscales" in checkpoint
        ):
            continue
        if module_name in w8_prefixes and f"{module_name}.weight" in checkpoint:
            continue
        filtered.append(key)
    return [key for key in filtered if ".wcscales" not in key]


def install_quantfunc_shim() -> bool:
    """Wrap the wheel's ``patch_scale_key`` so QuantFunc mods load. Idempotent.

    Returns false when the vitoom Qwen transformer module is not importable.
    The Qwen module binds the name at import, so both that binding and
    ``utils.patch_scale_key`` are replaced.
    """
    if _nunchaku_utils is None or _qwen_transformer is None:
        return False
    current = _nunchaku_utils.patch_scale_key
    if getattr(current, "_voxprint_quantfunc", False):
        _qwen_transformer.patch_scale_key = current
        return True

    def wrapped(transformer: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
        adapt_quantfunc_state(state_dict)
        current(transformer, state_dict)

    wrapped._voxprint_quantfunc = True  # type: ignore[attr-defined]
    _nunchaku_utils.patch_scale_key = wrapped
    _qwen_transformer.patch_scale_key = wrapped
    return True


def _mod_prefix(key: str, dtype: str) -> str | None:
    if not key.endswith(".qweight") or dtype not in _PACKED_INT32:
        return None
    prefix = key[: -len(".qweight")]
    if prefix.endswith(_MOD_TAILS):
        return prefix
    return None


def _read_range(url: str, start: int, end: int) -> bytes:
    want = end - start + 1
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "voxprint-forge"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise StageError(
                    f"{url} did not honor Range bytes={start}-{end}",
                    code="quantfunc_header",
                )
            return response.read(want)
    except urllib.error.URLError as exc:
        raise StageError(f"header range request failed for {url}: {exc}", code="quantfunc_header") from exc
