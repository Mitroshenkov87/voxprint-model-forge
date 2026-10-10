"""Quality check for Qwen-Image-2512. Imported only when the quality stage runs.

FP8 is dequantized to BF16 before the matmul. The A100 has no FP8 tensor cores.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from diffusers import QwenImagePipeline

from voxprint_forge.diffusion.generate import (
    ClipScorer,
    _apply,
    _cells,
    _component,
    _cpu_state,
    _cuda,
    _dequant,
    _lpips_net,
    _pair_stats,
    _quiet,
    _tick,
    save_avif,
)
from voxprint_forge.diffusion.prompts import BASE_PROMPTS, BASE_SEEDS
from voxprint_forge.diffusion.qwen_policy import decide_qwen_fp8
from voxprint_forge.diffusion.qwen_spec import QWEN_SIZE, QWEN_STEPS, QWEN_TRUE_CFG
from voxprint_forge.diffusion.scenes import contact_sheet

Progress = Callable[[str, int, int], None]


def _one_image(pipe: QwenImagePipeline, device: torch.device, prompt: str, seed: int) -> object:
    return pipe(
        prompt=prompt,
        negative_prompt=" ",
        true_cfg_scale=QWEN_TRUE_CFG,
        num_inference_steps=QWEN_STEPS,
        height=QWEN_SIZE,
        width=QWEN_SIZE,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="pil",
    ).images[0]


def evaluate_qwen(
    *,
    model_dir: Path,
    transformer_fp8: list[Path],
    grid_path: Path,
    progress: Progress | None = None,
) -> dict[str, object]:
    """32 prompts x 2 seeds at 1328. One BF16 pass, then one dequantized FP8 pass."""
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    pipe = QwenImagePipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe.to(device)
    _quiet(pipe)
    prepared = [(prompt, seed) for prompt in BASE_PROMPTS for seed in BASE_SEEDS]
    total = len(prepared)
    refs = []
    for done, (prompt, seed) in enumerate(prepared, start=1):
        refs.append(_one_image(pipe, device, prompt, seed))
        _tick(progress, "qwen-bf16", done, total)
    prompts = [prompt for prompt, _seed in prepared]
    transformer_state = _cpu_state(pipe.transformer)
    _apply(pipe.transformer, _dequant(transformer_fp8))
    candidates = [_one_image(pipe, device, prompt, seed) for prompt, seed in prepared]
    summary = _pair_stats(refs, candidates, prompts, metric, clip, device, image_score=False)
    _apply(pipe.transformer, transformer_state)
    del pipe
    torch.cuda.empty_cache()
    save_avif(contact_sheet(_cells(list(zip(refs, candidates, strict=True))), 2), grid_path)
    decision = decide_qwen_fp8(float(summary["lpips_mean"]), float(summary["clip_drop"]))
    return {"components": {"qwen-transformer": _component(summary, decision)}}
