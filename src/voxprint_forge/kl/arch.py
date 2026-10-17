"""Architecture and MTP-head checks.

Safetensors jobs compare ``config.json`` with the converted F16 GGUF.
A published F16 GGUF is its own reference: the MTP head is ``nextn_predict_layers``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import gguf

from voxprint_forge.errors import StageError
from voxprint_forge.fitmath import Arch

_LAYER = re.compile(r"(?:^|\.)mtp\.layers\.(\d+)\.")


def read_model_arch(path: Path) -> tuple[Arch, dict[str, object]]:
    """KV shape from one GGUF, plus the raw fields the summary records.

    ``block_count`` includes MTP layers when the converter appended them.
    A sliding-window pattern uses true for SWA and false for full attention.
    ``full_attention_interval`` splits the stack when the pattern is absent:
    one full layer per interval, and the rest sliding.
    """
    reader = gguf.GGUFReader(str(path))
    name = _text(reader, gguf.Keys.General.ARCHITECTURE)
    if not name:
        raise StageError(f"{path.name} has no architecture", code="kl_arch")
    blocks = _optional_int(reader, gguf.Keys.LLM.BLOCK_COUNT.format(arch=name))
    kv_heads = _optional_int(reader, gguf.Keys.Attention.HEAD_COUNT_KV.format(arch=name))
    if blocks is None or kv_heads is None or blocks < 1 or kv_heads < 1:
        raise StageError(f"{path.name} is missing block_count or head_count_kv", code="kl_arch")
    key_length = _optional_int(reader, gguf.Keys.Attention.KEY_LENGTH.format(arch=name))
    embed = _optional_int(reader, gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=name))
    heads = _optional_int(reader, gguf.Keys.Attention.HEAD_COUNT.format(arch=name))
    head_dim = key_length or (embed // heads if embed and heads else None)
    if head_dim is None or head_dim < 1:
        raise StageError(f"{path.name} is missing attention.key_length", code="kl_arch")
    window = _optional_int(reader, gguf.Keys.Attention.SLIDING_WINDOW.format(arch=name)) or 0
    interval = _optional_int(reader, gguf.Keys.LLM.FULL_ATTENTION_INTERVAL.format(arch=name)) or 0
    pattern = _pattern(reader, gguf.Keys.Attention.SLIDING_WINDOW_PATTERN.format(arch=name))
    swa_dim = _optional_int(reader, gguf.Keys.Attention.KEY_LENGTH_SWA.format(arch=name)) or head_dim
    nextn = _optional_int(reader, gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.format(arch=name)) or 0
    full, swa = _split(blocks, window, interval, pattern)
    arch = Arch(
        full_layers=full,
        full_kv_heads=kv_heads,
        full_head_dim=head_dim,
        swa_layers=swa,
        swa_kv_heads=kv_heads if swa else 0,
        swa_head_dim=swa_dim if swa else 0,
        swa_window=window if swa else 0,
    )
    meta: dict[str, object] = {
        "architecture": name,
        "block_count": blocks,
        "head_count_kv": kv_heads,
        "head_dim": head_dim,
        "sliding_window": window,
        "full_attention_interval": interval,
        "nextn_predict_layers": nextn,
        "full_layers": full,
        "swa_layers": swa,
    }
    return arch, meta


def source_mtp_layers(folder: Path) -> tuple[int, int]:
    """``(num_hidden_layers, mtp_layers)`` from ``config.json`` and the index.

    ``text_config`` wins when it is present. A missing ``mtp_num_hidden_layers``
    falls back to the distinct ``mtp.layers.N`` names in the safetensors index.
    """
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise StageError("checkpoint has no config.json", code="kl_mtp")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError("checkpoint config.json is not JSON", code="kl_mtp") from exc
    if not isinstance(payload, dict):
        raise StageError("checkpoint config.json is not an object", code="kl_mtp")
    text = payload.get("text_config")
    merged = dict(payload)
    if isinstance(text, dict):
        merged.update(text)
    layers = _positive(merged.get("num_hidden_layers"))
    declared = _positive(merged.get("mtp_num_hidden_layers"))
    indexed = _index_mtp(folder / "model.safetensors.index.json")
    return layers, declared or indexed


def gguf_reference_mtp(meta: dict[str, object]) -> dict[str, object]:
    """MTP status when the reference is already an F16 GGUF.

    The published file is the quantisation source, so there is no
    ``config.json`` to compare. ``nextn_predict_layers`` is the MTP head.
    A missing head, or a head larger than ``block_count``, fails the job.
    """
    block_count = int(meta["block_count"])
    nextn = int(meta["nextn_predict_layers"])
    if nextn < 1:
        raise StageError("F16 GGUF reference has no MTP head", code="kl_mtp")
    if nextn > block_count:
        raise StageError(
            f"F16 GGUF nextn_predict_layers {nextn} exceeds block_count {block_count}",
            code="kl_mtp",
        )
    return mtp_status(block_count - nextn, nextn, meta)


def mtp_status(source_layers: int, source_mtp: int, meta: dict[str, object]) -> dict[str, object]:
    """Whether the F16 GGUF kept the MTP head the checkpoint declared."""
    block_count = int(meta["block_count"])
    nextn = int(meta["nextn_predict_layers"])
    if source_mtp <= 0:
        kept = True
    else:
        kept = nextn == source_mtp and block_count >= source_layers + source_mtp
    return {
        "source_layers": source_layers,
        "source_mtp_layers": source_mtp,
        "gguf_block_count": block_count,
        "gguf_nextn_predict_layers": nextn,
        "mtp_kept": kept,
    }


def require_mtp(status: dict[str, object]) -> None:
    """Fail the job when a declared MTP head did not survive conversion."""
    if status["mtp_kept"]:
        return
    raise StageError(
        "llama.cpp b11476 did not keep the Qwen3.6 MTP head "
        f"(source layers {status['source_mtp_layers']}, "
        f"gguf nextn {status['gguf_nextn_predict_layers']})",
        code="kl_mtp",
    )


def _split(blocks: int, window: int, interval: int, pattern: list[bool] | None) -> tuple[int, int]:
    if pattern:
        swa = sum(1 for item in pattern if item)
        return blocks - swa, swa
    if window > 0 and interval > 1:
        full = blocks // interval
        return full, blocks - full
    return blocks, 0


def _pattern(reader: gguf.GGUFReader, key: str) -> list[bool] | None:
    field = reader.get_field(key)
    if field is None:
        return None
    value: Any = field.contents()
    if isinstance(value, list) and value and isinstance(value[0], (bool, int)):
        return [bool(item) for item in value]
    return None


def _text(reader: gguf.GGUFReader, key: str) -> str:
    field = reader.get_field(key)
    if field is None:
        return ""
    value = field.contents()
    return value if isinstance(value, str) else ""


def _optional_int(reader: gguf.GGUFReader, key: str) -> int | None:
    field = reader.get_field(key)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, bool) or isinstance(value, list):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _positive(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if value > 0 else 0


def _index_mtp(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return 0
    found: set[int] = set()
    for name in weight_map:
        if not isinstance(name, str):
            continue
        match = _LAYER.search(name)
        if match:
            found.add(int(match.group(1)))
    return len(found)
