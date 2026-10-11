"""Short real Qwen calibration and a 60-block extrapolation.

The deepcompressor syh checkout is tried on CUDA first. When that cannot use
CUDA, including on Windows, the same first blocks are calibrated and quantized
on CPU with the syh INT4 pack. The H100 estimate scales the laptop time by the
geometric mean of dense FP16 tensor TFLOP/s and memory bandwidth.

torch, diffusers, and safetensors are imported inside the timing functions.
Importing this module must not load that stack: the laptop entry imports it
only after the CPU-only contract tests have already imported the package.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from voxprint_forge.diffusion.int4.calib import calibration_prompts, image_root, write_prompt_yaml
from voxprint_forge.diffusion.int4.config import collect_argv, quant_argv, write_configs
from voxprint_forge.diffusion.int4.patch import apply_checkout
from voxprint_forge.diffusion.int4.pins import DEFAULT_SVD_RANK

FULL_BLOCKS = 60
SMOKE_SAMPLES = 2
# RTX 4090 Laptop (AD103, 9728 CUDA): 39.7 FP32 TFLOP/s. Dense FP16 tensor is 2x.
# Memory is 16 GB GDDR6, 256-bit, 18 GT/s, 576 GB/s.
LAPTOP_FP16_TFLOPS = 79.4
LAPTOP_BW_GBPS = 576.0
# H100 SXM5 dense FP16 tensor, without 2:4 sparsity. HBM3 bandwidth is 3350 GB/s.
H100_FP16_TFLOPS = 989.0
H100_BW_GBPS = 3350.0
CPU_FALLBACK = "cpu fallback: deepcompressor could not use CUDA"
SCALE_ASSUMPTION = (
    "H100 seconds = laptop seconds / sqrt((H100 FP16 tensor TFLOP/s / laptop FP16 tensor TFLOP/s) "
    "* (H100 GB/s / laptop GB/s)). Laptop is the RTX 4090 Laptop: 79.4 dense FP16 tensor TFLOP/s "
    "(2x its 39.7 FP32) and 576 GB/s. H100 SXM5 is 989 dense FP16 tensor TFLOP/s, without sparsity, "
    "and 3350 GB/s. Calibration time is kept as measured. Block SVD and quant scale with the 60 blocks."
)


def extrapolate_quant(
    *,
    sample_seconds: list[float],
    block_seconds: list[float],
    full_blocks: int = FULL_BLOCKS,
) -> dict[str, object]:
    """Extrapolate measured sample and block times to 60 blocks and to an H100."""
    if not sample_seconds or not block_seconds:
        raise ValueError("extrapolation needs at least one sample and one block")
    if any(item < 0 for item in sample_seconds + block_seconds):
        raise ValueError("timings must be non-negative")
    per_sample = sum(sample_seconds) / len(sample_seconds)
    per_block = sum(block_seconds) / len(block_seconds)
    laptop_seconds = sum(sample_seconds) + per_block * full_blocks
    compute_ratio = H100_FP16_TFLOPS / LAPTOP_FP16_TFLOPS
    bandwidth_ratio = H100_BW_GBPS / LAPTOP_BW_GBPS
    scale = math.sqrt(compute_ratio * bandwidth_ratio)
    return {
        "full_blocks": full_blocks,
        "blocks_measured": len(block_seconds),
        "samples_measured": len(sample_seconds),
        "seconds_per_calibration_sample": per_sample,
        "seconds_per_block": per_block,
        "laptop_seconds": laptop_seconds,
        "h100_seconds": laptop_seconds / scale,
        "scale": scale,
        "laptop_fp16_tflops": LAPTOP_FP16_TFLOPS,
        "laptop_bandwidth_gbps": LAPTOP_BW_GBPS,
        "h100_fp16_tflops": H100_FP16_TFLOPS,
        "h100_bandwidth_gbps": H100_BW_GBPS,
        "assumption": SCALE_ASSUMPTION,
    }


def reuse_step(previous: dict[str, object] | None, step_id: str, *, force: bool) -> dict[str, object] | None:
    """Return a passed step from an earlier result. ``force`` runs it again."""
    if force or not previous:
        return None
    steps = previous.get("steps")
    if not isinstance(steps, list):
        return None
    for item in steps:
        if not isinstance(item, dict):
            continue
        if item.get("id") == step_id and item.get("status") == "pass":
            notes = [str(note) for note in item.get("notes", [])] if isinstance(item.get("notes"), list) else []
            if "reused" not in notes:
                notes.append("reused")
            carried = dict(item)
            carried["notes"] = notes
            return carried
    return None


def time_real_blocks(repo: Path, snapshot: Path, *, blocks: int, rank: int) -> dict[str, object]:
    """Calibrate and INT4-quantize the first ``blocks`` of the real transformer on CPU."""
    if blocks < 1:
        raise ValueError("blocks must be at least 1")
    import torch

    from voxprint_forge.diffusion.int4.calib import load_prompts
    from voxprint_forge.diffusion.int4.qwen_tiny import TINY_RANK, _edit_batch, _quantize_block, _sample_images

    model = _load_truncated(snapshot, blocks)
    prompts = load_prompts(repo)
    images = _sample_images(prompts, repo)
    sample_seconds: list[float] = []
    captured: dict[int, dict[str, list[torch.Tensor]]] = {index: {} for index in range(blocks)}
    hooks = []
    for index in range(blocks):
        for name, module in model.transformer_blocks[index].named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_pre_hook(_capture(index, name, captured)))
    try:
        for index, path in enumerate(images):
            prompt = str(prompts[index].get("prompt", "edit"))
            batch = _edit_batch(model, path, prompt)
            started = time.perf_counter()
            with torch.no_grad():
                model(**batch)
            sample_seconds.append(time.perf_counter() - started)
    finally:
        for hook in hooks:
            hook.remove()
    chosen_rank = rank if rank > 0 else TINY_RANK
    block_seconds: list[float] = []
    for index in range(blocks):
        started = time.perf_counter()
        _quantize_block(model, captured[index], rank=chosen_rank, block_index=index)
        block_seconds.append(time.perf_counter() - started)
    report = extrapolate_quant(sample_seconds=sample_seconds, block_seconds=block_seconds)
    report["device"] = "cpu"
    report["rank"] = chosen_rank
    return {
        "sample_seconds": sample_seconds,
        "block_seconds": block_seconds,
        "extrapolation": report,
    }


def try_deepcompressor(
    cache: Path,
    snapshot: Path,
    work: Path,
    *,
    repo: Path,
    blocks: int,
    timeout: float,
) -> dict[str, object]:
    """Run syh collect and fast INT4 on a truncated pipeline. Raises on failure.

    Collect and quant both use ``SMOKE_SAMPLES`` and the H100 SVD rank. Config
    files are written under the deepcompressor checkout, and the process cwd
    is ``examples/diffusion``, which is the only tree omniconfig accepts.
    """
    checkout = _ensure_checkout(cache / "deepcompressor")
    apply_checkout(checkout)
    pipeline = _linked_pipeline(snapshot, work / "pipeline", blocks)
    diffusion_cwd = checkout / "examples" / "diffusion"
    saved = {
        "FORGE_DEEPCOMPRESSOR": os.environ.get("FORGE_DEEPCOMPRESSOR"),
        "FORGE_QWEN_EDIT": os.environ.get("FORGE_QWEN_EDIT"),
    }
    os.environ["FORGE_DEEPCOMPRESSOR"] = str(checkout)
    os.environ["FORGE_QWEN_EDIT"] = str(pipeline)
    try:
        configs = write_configs(
            work,
            "qwen-image-edit-2511",
            rank=DEFAULT_SVD_RANK,
            num_samples=SMOKE_SAMPLES,
        )
        prompts = calibration_prompts(repo)[:SMOKE_SAMPLES]
        write_prompt_yaml(prompts, configs["prompts"], image_root=image_root(repo))
        env = os.environ.copy()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = work / "deepcompressor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        started = time.perf_counter()
        _run(collect_argv(configs), env, log, timeout, cwd=diffusion_cwd)
        calib_seconds = time.perf_counter() - started
        remaining = max(1.0, timeout - calib_seconds)
        started = time.perf_counter()
        argv = quant_argv(checkout, configs, work / "quant")
        _run(argv, env, log, remaining, cwd=diffusion_cwd)
        quant_seconds = time.perf_counter() - started
    per_block = quant_seconds / blocks
    sample_seconds = [calib_seconds / SMOKE_SAMPLES] * SMOKE_SAMPLES
    block_seconds = [per_block] * blocks
    report = extrapolate_quant(sample_seconds=sample_seconds, block_seconds=block_seconds)
    report["device"] = "cuda"
    report["path"] = "deepcompressor-syh"
    report["rank"] = DEFAULT_SVD_RANK
    report["samples"] = SMOKE_SAMPLES
    return {
        "sample_seconds": sample_seconds,
        "block_seconds": block_seconds,
        "extrapolation": report,
        "notes": ["deepcompressor syh", "cuda", f"samples {SMOKE_SAMPLES}", f"rank {DEFAULT_SVD_RANK}"],
    }


def _run(argv: list[str], env: dict[str, str], log: object, timeout: float, *, cwd: Path) -> None:
    proc = subprocess.run(
        argv,
        env=env,
        cwd=str(cwd),
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"deepcompressor exited {proc.returncode}")


def _ensure_checkout(dest: Path) -> Path:
    marker = dest / "deepcompressor" / "app" / "diffusion" / "ptq.py"
    if marker.is_file():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    from voxprint_forge.diffusion.int4.pins import DEEPCOMPRESSOR_COMMIT, DEEPCOMPRESSOR_REPO

    clone = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", DEEPCOMPRESSOR_REPO, str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if clone.returncode != 0:
        raise RuntimeError(clone.stderr.strip() or "git clone failed")
    fetch = subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", DEEPCOMPRESSOR_COMMIT],
        capture_output=True,
        text=True,
        check=False,
    )
    if fetch.returncode != 0:
        raise RuntimeError(fetch.stderr.strip() or "git fetch failed")
    checkout = subprocess.run(
        ["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if checkout.returncode != 0:
        raise RuntimeError(checkout.stderr.strip() or "git checkout failed")
    return dest


def _linked_pipeline(snapshot: Path, dest: Path, blocks: int) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for child in snapshot.iterdir():
        target = dest / child.name
        if child.name == "transformer":
            _write_truncated_transformer(child, target, blocks)
        else:
            _link_or_copy(child, target)
    return dest


def _link_or_copy(src: Path, dest: Path) -> None:
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _link_or_copy(child, dest / child.name)
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _write_truncated_transformer(src: Path, dest: Path, blocks: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    config_path = src / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_layers"] = blocks
    (dest / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for child in src.iterdir():
        if child.name == "config.json" or child.suffix == ".safetensors" or child.name.endswith(".index.json"):
            continue
        _link_or_copy(child, dest / child.name)
    kept: dict[str, object] = {}
    from safetensors.torch import save_file

    tensors: dict[str, object] = {}
    for shard in _transformer_shards(src):
        from safetensors import safe_open

        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if _skip_block(key, blocks):
                    continue
                tensors[key] = handle.get_tensor(key)
                kept[key] = "diffusion_pytorch_model.safetensors"
    if not tensors:
        raise RuntimeError(f"no transformer tensors kept from {src}")
    save_file(tensors, dest / "diffusion_pytorch_model.safetensors")  # type: ignore[arg-type]
    index = {
        "metadata": {"voxprint_blocks": blocks},
        "weight_map": kept,
    }
    (dest / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def _load_truncated(snapshot: Path, blocks: int):
    import torch
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    config_path = snapshot / "transformer" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_layers"] = blocks
    model = QwenImageTransformer2DModel.from_config(config)
    state: dict[str, torch.Tensor] = {}
    for shard in _transformer_shards(snapshot / "transformer"):
        from safetensors import safe_open

        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if _skip_block(key, blocks):
                    continue
                name = key[len("transformer.") :] if key.startswith("transformer.") else key
                state[name] = handle.get_tensor(key)
    missing, _unexpected = model.load_state_dict(state, strict=False)
    real_missing = [key for key in missing if not _skip_block(key, blocks)]
    if real_missing:
        raise RuntimeError(f"truncated transformer is missing {real_missing[:5]}")
    model.eval()
    return model


def _transformer_shards(root: Path) -> list[Path]:
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        names = sorted({str(name) for name in weight_map.values()})
        return [root / name for name in names]
    single = root / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [single]
    return sorted(root.glob("*.safetensors"))


def _skip_block(key: str, blocks: int) -> bool:
    marker = "transformer_blocks."
    start = key.find(marker)
    if start < 0:
        return False
    rest = key[start + len(marker) :]
    number = rest.split(".", 1)[0]
    if not number.isdigit():
        return False
    return int(number) >= blocks


def _capture(index: int, name: str, sink: dict[int, dict[str, list]]):
    import torch

    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if not args:
            return
        sink[index].setdefault(name, []).append(args[0].detach())

    return hook
