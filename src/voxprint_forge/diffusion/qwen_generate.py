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
    _component,
    _cpu_state,
    _cuda,
    _dequant,
    _lpips_net,
    _next_chunk,
    _pair_stats,
    _quiet,
    _score_series,
    _text_pair_decided,
    _tick,
    _write_review,
)
from voxprint_forge.diffusion.prompts import LEAN_BASE_PROMPTS, LEAN_BASE_SEEDS, LEAN_MINIMUM
from voxprint_forge.diffusion.qwen_policy import decide_qwen_fp8
from voxprint_forge.diffusion.qwen_spec import QWEN_SIZE, QWEN_STEPS, QWEN_TRUE_CFG

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
    """A few hard prompts at the Qwen size. Stop when the LPIPS interval decides the gate."""
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    pipe = QwenImagePipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe.to(device)
    _quiet(pipe)
    prepared = [(prompt, seed) for prompt in LEAN_BASE_PROMPTS for seed in LEAN_BASE_SEEDS]
    total = len(prepared)
    transformer_state = _cpu_state(pipe.transformer)
    fp8_state = _dequant(transformer_fp8)
    refs = []
    candidates = []
    prompts: list[str] = []
    seeds: list[int] = []
    lpips_scores: list[float] = []
    drops: list[float] = []
    while True:
        chunk = _next_chunk(
            prepared,
            len(prompts),
            decided=_text_pair_decided(lpips_scores, drops),
            minimum=LEAN_MINIMUM,
        )
        if not chunk:
            break
        _apply(pipe.transformer, transformer_state)
        for prompt, seed in chunk:
            refs.append(_one_image(pipe, device, prompt, seed))
            prompts.append(prompt)
            seeds.append(seed)
            _tick(progress, "qwen-bf16", len(prompts), total)
        _apply(pipe.transformer, fp8_state)
        fresh = []
        for prompt, seed in chunk:
            fresh.append(_one_image(pipe, device, prompt, seed))
            _tick(progress, "qwen-fp8", len(candidates) + len(fresh), total)
        candidates.extend(fresh)
        start = len(prompts) - len(chunk)
        scores, chunk_drops = _score_series(
            refs[start:], candidates[start:], prompts[start:], metric, clip, device, image_score=False
        )
        lpips_scores.extend(scores)
        drops.extend(chunk_drops)
    summary = _pair_stats(refs, candidates, prompts, metric, clip, device, image_score=False)
    del pipe
    torch.cuda.empty_cache()
    _write_review(grid_path, refs, candidates, prompts, seeds)
    decision = decide_qwen_fp8(float(summary["lpips_mean"]), float(summary["clip_drop"]))
    return {"components": {"qwen-transformer": _component(summary, decision)}}
