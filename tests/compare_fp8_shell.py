"""Load a tiny Z-Image pipeline whose transformer weights were not linked.

Run as a child process so the pytest session does not import diffusers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2Model

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.engine import _native_text, _text_pipe
from voxprint_forge.compare.plan import load_compare_plan
from voxprint_forge.diffusion.fp8 import quantize_weight, save_quantized
from voxprint_forge.diffusion.fp8_native import Fp8Linear

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "compare-synth4.toml"


def main(root: Path, mode: str) -> None:
    """Build the fixture, then require a complete FP8 overlay on the config shell."""
    if mode == "complete":
        _assert_complete(root)
        return
    if mode in {"missing", "mismatch"}:
        _assert_gap(root, mode)
        return
    raise SystemExit("usage: compare_fp8_shell.py DIR {complete,missing,mismatch}")


def _assert_complete(root: Path) -> None:
    folder = _tiny_fp8_arm(root)
    base = folder / "base"
    if any((base / "transformer").glob("*.safetensors")):
        raise SystemExit("transformer weights were not removed")
    try:
        ZImagePipeline.from_pretrained(base, torch_dtype=torch.bfloat16)
    except OSError:
        pass
    else:
        raise SystemExit("from_pretrained succeeded without transformer weights")
    job = load_compare_plan(PLAN).job("D1")
    native = _native_text(ZImagePipeline, job.arm("c"), folder, torch.bfloat16)
    note = native.forge_fp8
    if note["mode"] != "fp8-native" or int(note["layers_swapped"]) < 1:
        raise SystemExit(f"native overlay did not swap layers: {note}")
    swapped = [module for module in native.transformer.modules() if isinstance(module, Fp8Linear)]
    if not swapped or swapped[0].weight.dtype != torch.float8_e4m3fn:
        raise SystemExit("native transformer has no float8 linear")
    dequant = _text_pipe(ZImagePipeline, job.arm("c-dq"), folder, torch.bfloat16)
    if dequant.forge_fp8 != {"arm": "c-dq", "mode": "bf16-dequant"}:
        raise SystemExit(f"dequant stamp was {dequant.forge_fp8}")
    embedder = dequant.transformer.all_x_embedder["2-1"]
    if not isinstance(embedder, nn.Linear) or embedder.weight.dtype != torch.bfloat16:
        raise SystemExit("dequant embedder is not a BF16 linear")
    if any((base / "transformer").glob("*.safetensors")):
        raise SystemExit("load linked transformer weights back into the base")
    print(f"layers_swapped {note['layers_swapped']}")


def _assert_gap(root: Path, mode: str) -> None:
    folder = _tiny_fp8_arm(root)
    key = _mutate_overlay(folder / "weights" / "transformer-fp8.safetensors", mode)
    for arm_id in ("c", "c-dq"):
        _expect_coverage_error(folder, arm_id, key)
    print(f"{mode} {key}")


def _expect_coverage_error(folder: Path, arm_id: str, needle: str) -> None:
    job = load_compare_plan(PLAN).job("D1")
    arm = job.arm(arm_id)
    try:
        if arm.compute == "fp8-native":
            _native_text(ZImagePipeline, arm, folder, torch.bfloat16)
        else:
            _text_pipe(ZImagePipeline, arm, folder, torch.bfloat16)
    except CompareSetup as exc:
        text = str(exc)
        if exc.code != "compare_weights" or needle not in text or "fp8 shell did not cover" not in text:
            raise SystemExit(f"{arm_id} bad error {exc.code}: {text}") from exc
        return
    raise SystemExit(f"{arm_id} accepted an incomplete fp8 shell")


def _mutate_overlay(path: Path, mode: str) -> str:
    tensors = load_file(path)
    key = _coverage_key(tensors)
    if mode == "missing":
        del tensors[key]
        tensors.pop(f"{key}.scale_weight", None)
    elif mode == "mismatch":
        value = tensors[key]
        shape = (value.shape[0] + 1,) if value.ndim == 1 else (*value.shape[:-1], value.shape[-1] + 1)
        tensors[key] = torch.zeros(shape, dtype=value.dtype)
    else:
        raise SystemExit(f"unknown mutation {mode}")
    save_quantized(tensors, path)
    return key


def _coverage_key(tensors: dict[str, torch.Tensor]) -> str:
    for key in sorted(tensors):
        if key.endswith(".scale_weight"):
            continue
        if "norm" in key and key.endswith(".weight"):
            return key
    raise SystemExit("fixture has no norm weight")


def _tiny_fp8_arm(root: Path) -> Path:
    """Write a Z-Image pipeline whose transformer directory has config.json and no weights."""
    transformer = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=128,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=1,
        n_kv_heads=1,
        cap_feat_dim=64,
        axes_dims=[32, 48, 48],
        axes_lens=[32, 32, 32],
    )
    text = Qwen2Model(
        Qwen2Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=32,
        )
    )
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "a": 2}, unk_token="<unk>"))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        latent_channels=4,
        layers_per_block=1,
    )
    pipe = ZImagePipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=text,
        tokenizer=fast,
        transformer=transformer,
    )
    base = root / "base"
    pipe.save_pretrained(base)
    for path in (base / "transformer").glob("*.safetensors"):
        path.unlink()
    weights = root / "weights"
    weights.mkdir()
    save_quantized(_quantize_transformer(transformer), weights / "transformer-fp8.safetensors")
    return root


def _quantize_transformer(module: nn.Module) -> dict[str, torch.Tensor]:
    linear_weights = {f"{name}.weight" for name, child in module.named_modules() if isinstance(child, nn.Linear)}
    tensors: dict[str, torch.Tensor] = {}
    for key, value in module.state_dict().items():
        if key in linear_weights:
            quantized, scale = quantize_weight(value)
            tensors[key] = quantized
            tensors[f"{key}.scale_weight"] = scale
            continue
        if value.is_floating_point() and value.dtype != torch.bfloat16:
            tensors[key] = value.detach().to(dtype=torch.bfloat16)
        else:
            tensors[key] = value.detach()
    return tensors


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_fp8_shell.py DIR {complete,missing,mismatch}")
    main(Path(sys.argv[1]), sys.argv[2])
