"""Quality checks. FP8 is dequantized to BF16 because the A100 has no FP8 tensor cores."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import lpips
import numpy as np
import pillow_avif  # noqa: F401
import torch
from diffusers import (
    QwenImageEditPlusPipeline,
    ZImageControlNetInpaintPipeline,
    ZImageControlNetModel,
    ZImageControlNetPipeline,
    ZImagePipeline,
)
from PIL import Image
from safetensors import safe_open
from transformers import CLIPModel, CLIPProcessor

from voxprint_forge.diffusion.alias import alias_key as _alias_key
from voxprint_forge.diffusion.controlnet_link import swap_controlnet
from voxprint_forge.diffusion.controlnet_union import places_for_count, union_config
from voxprint_forge.diffusion.fp8 import dequant_state
from voxprint_forge.diffusion.policy import (
    CLIP_DROP_MAX,
    CLIP_I_MIN,
    LPIPS_EDIT_MAX,
    LPIPS_MAX,
    clip_drop,
    decide_base,
    decide_control,
    decide_edit,
)
from voxprint_forge.diffusion.prompts import (
    CONTROL_PROMPTS,
    LEAN_BASE_PROMPTS,
    LEAN_BASE_SEEDS,
    LEAN_CONTROL,
    LEAN_CONTROL_MINIMUM,
    LEAN_EDIT,
    LEAN_EDIT_MINIMUM,
    LEAN_INPAINT,
    LEAN_INPAINT_MINIMUM,
    LEAN_MINIMUM,
    edit_budget,
)
from voxprint_forge.diffusion.scenes import canny_map, depth_map, inpaint_mask, pose_map, render_scene
from voxprint_forge.diffusion.spec import (
    CONTROL_GUIDANCE,
    CONTROL_SCALE,
    CONTROL_STEPS,
    CREATIVE_GUIDANCE,
    CREATIVE_STEPS,
    EDIT_STEPS,
    EDIT_TRUE_CFG,
    IMAGE_SIZE,
    TURBO_GUIDANCE,
    TURBO_STEPS,
)
from voxprint_forge.diffusion.stage_free import free_eval_stage
from voxprint_forge.diffusion.weights import load_tensors, materialise_bf16
from voxprint_forge.errors import StageError, ToolMissingError
from voxprint_forge.lean import REVIEW_LIMIT, metrics_decided, review_document, wave_size

Progress = Callable[[str, int, int], None]
_Case = TypeVar("_Case")


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ToolMissingError("torch.cuda.is_available() is false", code="cuda")
    return torch.device("cuda")


def _quiet(pipe: object) -> None:
    setter = getattr(pipe, "set_progress_bar_config", None)
    if setter is not None:
        setter(disable=True)


def _pil(rgb: torch.Tensor) -> Image.Image:
    array = rgb.detach().clamp(0, 1).mul(255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _from_pil(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float() / 255.0


def _as_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    embeds = getattr(output, "image_embeds", None)
    if isinstance(embeds, torch.Tensor):
        return embeds
    raise StageError("CLIP features were not a tensor", code="clip")


def save_avif(image: torch.Tensor, path: Path) -> None:
    """Write an RGB image as AVIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).mul(255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path, format="AVIF")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_n = torch.nn.functional.normalize(left.float().flatten(), dim=0)
    right_n = torch.nn.functional.normalize(right.float().flatten(), dim=0)
    return float((left_n * right_n).sum())


class ClipScorer:
    """CLIP similarity of a text prompt or another image."""
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def text_image(self, image: Image.Image, text: str) -> float:
        """Cosine similarity of this image and a text prompt."""
        image_inputs = self.processor(images=image, return_tensors="pt")
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        pixels = image_inputs["pixel_values"].to(self.device)
        input_ids = text_inputs["input_ids"].to(self.device)
        mask = text_inputs.get("attention_mask")
        attention = mask.to(self.device) if mask is not None else None
        with torch.no_grad():
            image_features = _as_tensor(self.model.get_image_features(pixel_values=pixels))
            text_features = _as_tensor(self.model.get_text_features(input_ids=input_ids, attention_mask=attention))
        return _cosine(image_features[0], text_features[0])

    def image_image(self, left: Image.Image, right: Image.Image) -> float:
        """Cosine similarity of two images."""
        inputs = self.processor(images=[left, right], return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = _as_tensor(self.model.get_image_features(pixel_values=pixels))
        return _cosine(features[0], features[1])


def _lpips_net(device: torch.device) -> torch.nn.Module:
    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    return metric


def _lpips_value(metric: torch.nn.Module, left: Image.Image, right: Image.Image, device: torch.device) -> float:
    def prep(image: Image.Image) -> torch.Tensor:
        return _from_pil(image).unsqueeze(0).to(device) * 2 - 1

    with torch.no_grad():
        return float(metric(prep(left), prep(right)).mean())


def _cn_load(path: object) -> ZImageControlNetModel:
    """Load a Union 2.1 single file with the VideoX-Fun layer map.

    diffusers reads a 3-layer lite file as the v1 net (control_in_dim 16) and then
    rejects control_all_x_embedder. Lite places are [0, 10, 20], from
    config/z_image/z_image_control_2.1_lite.yaml.
    """
    with safe_open(str(path), framework="pt") as handle:
        layers = sorted({int(key.split(".")[1]) for key in handle.keys() if key.startswith("control_layers.")})
    if not layers:
        raise StageError(f"{path} has no control_layers", code="controlnet")
    folder = tempfile.mkdtemp(prefix="forge-cn21-")
    (Path(folder) / "config.json").write_text(
        json.dumps(union_config(places_for_count(len(layers)))),
        encoding="utf-8",
    )
    return ZImageControlNetModel.from_single_file(str(path), config=folder, torch_dtype=torch.bfloat16)


def _apply(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    device = next(module.parameters()).device
    mapped: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = _alias_key(raw_key, current)
        if key is None:
            continue
        tensor = value.detach()
        target = current[key]
        if tensor.shape != target.shape:
            continue
        if tensor.is_floating_point() and tensor.dtype != target.dtype:
            tensor = tensor.to(dtype=target.dtype)
        mapped[key] = tensor.to(device=device)
    weight_keys = [key for key in current if key.endswith(".weight")]
    if len(mapped) < max(1, len(weight_keys) // 2):
        raise StageError(
            f"converted weights match {len(mapped)} of {len(current)} module tensors",
            code="state_dict",
        )
    module.load_state_dict(mapped, strict=False)


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _dequant(paths: list[Path]) -> dict[str, torch.Tensor]:
    if not paths:
        raise StageError("no FP8 files to dequantize", code="fp8_scale")
    return dequant_state(load_tensors(paths))


def _tick(progress: Progress | None, stage: str, done: int, total: int) -> None:
    if progress is not None:
        progress(stage, done, total)


def _zimage(
    pipe: object,
    prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    device: torch.device,
    **extra: object,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pil",
        **extra,
    )
    return result.images[0]


def _mean(values: list[float]) -> float:
    if not values:
        raise StageError("quality check produced no samples", code="quality_empty")
    return sum(values) / len(values)


def _pair_stats(
    refs: list[Image.Image],
    cands: list[Image.Image],
    prompts: list[str],
    metric: torch.nn.Module,
    clip: ClipScorer,
    device: torch.device,
    *,
    image_score: bool,
) -> dict[str, float]:
    lpips_scores: list[float] = []
    clip_refs: list[float] = []
    clip_cands: list[float] = []
    clip_i: list[float] = []
    for ref, cand, prompt in zip(refs, cands, prompts, strict=True):
        lpips_scores.append(_lpips_value(metric, ref, cand, device))
        if image_score:
            clip_i.append(clip.image_image(ref, cand))
        else:
            clip_refs.append(clip.text_image(ref, prompt))
            clip_cands.append(clip.text_image(cand, prompt))
    summary: dict[str, float] = {"lpips_mean": _mean(lpips_scores), "samples": float(len(lpips_scores))}
    if image_score:
        summary["clip_i"] = _mean(clip_i)
    else:
        reference = _mean(clip_refs)
        candidate = _mean(clip_cands)
        summary["clip_reference"] = reference
        summary["clip_candidate"] = candidate
        summary["clip_drop"] = (reference - candidate) / abs(reference) if reference else 0.0
    return summary


def _component(summary: dict[str, float], decision: str) -> dict[str, object]:
    body: dict[str, object] = {"decision": decision, "samples": int(summary["samples"])}
    for key, value in summary.items():
        if key != "samples":
            body[key] = value
    return body


def _score_series(
    refs: list[Image.Image],
    cands: list[Image.Image],
    prompts: list[str],
    metric: torch.nn.Module,
    clip: ClipScorer,
    device: torch.device,
    *,
    image_score: bool,
) -> tuple[list[float], list[float]]:
    """Per-sample LPIPS and either CLIP-I or per-sample CLIP drop."""
    lpips_scores: list[float] = []
    secondary: list[float] = []
    for ref, cand, prompt in zip(refs, cands, prompts, strict=True):
        lpips_scores.append(_lpips_value(metric, ref, cand, device))
        if image_score:
            secondary.append(clip.image_image(ref, cand))
        else:
            secondary.append(clip_drop(clip.text_image(ref, prompt), clip.text_image(cand, prompt)))
    return lpips_scores, secondary


def _write_review(
    grid_path: Path,
    refs: list[Image.Image],
    cands: list[Image.Image],
    prompts: list[str],
    seeds: list[int],
) -> None:
    """A few full-resolution labelled frames beside the old grid path."""
    directory = grid_path.parent / f"review-{grid_path.stem}"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, int, str]] = []
    kept = min(REVIEW_LIMIT, len(refs), len(cands))
    for index in range(kept):
        ref_name = f"{index:02d}-reference.png"
        cand_name = f"{index:02d}-candidate.png"
        refs[index].save(directory / ref_name)
        cands[index].save(directory / cand_name)
        rows.append((ref_name, prompts[index], seeds[index], "reference"))
        rows.append((cand_name, prompts[index], seeds[index], "candidate"))
    (directory / "labels.md").write_text(review_document(rows), encoding="utf-8")


def _next_chunk(cases: list[_Case], done: int, *, decided: bool, minimum: int) -> list[_Case]:
    count = wave_size(done, len(cases), decided=decided, minimum=minimum)
    return cases[done : done + count]


def _text_pair_decided(lpips_scores: list[float], drops: list[float]) -> bool:
    return metrics_decided(
        [(lpips_scores, LPIPS_MAX, True), (drops, CLIP_DROP_MAX, True)],
        minimum=LEAN_MINIMUM,
    )


def evaluate_base(
    *,
    turbo_dir: Path,
    base_dir: Path,
    turbo_fp8: Path,
    base_fp8: Path,
    encoder_fp8: Path,
    grid_path: Path,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Score a few hard prompts and write a full-resolution review pack."""
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    cases = [(prompt, seed) for prompt in LEAN_BASE_PROMPTS for seed in LEAN_BASE_SEEDS]
    total = len(cases)

    turbo = ZImagePipeline.from_pretrained(str(turbo_dir), torch_dtype=torch.bfloat16)
    turbo.to(device)
    _quiet(turbo)
    transformer_state = _cpu_state(turbo.transformer)
    encoder_state = _cpu_state(turbo.text_encoder)
    turbo_fp8_state = _dequant([turbo_fp8])
    encoder_fp8_state = _dequant([encoder_fp8])
    refs: list[Image.Image] = []
    turbo_cands: list[Image.Image] = []
    encoder_cands: list[Image.Image] = []
    prompts: list[str] = []
    seeds: list[int] = []
    turbo_lpips: list[float] = []
    turbo_drop: list[float] = []
    encoder_lpips: list[float] = []
    encoder_drop: list[float] = []
    while True:
        decided = _text_pair_decided(turbo_lpips, turbo_drop) and _text_pair_decided(encoder_lpips, encoder_drop)
        chunk = _next_chunk(cases, len(prompts), decided=decided, minimum=LEAN_MINIMUM)
        if not chunk:
            break
        _apply(turbo.transformer, transformer_state)
        _apply(turbo.text_encoder, encoder_state)
        for prompt, seed in chunk:
            refs.append(_zimage(turbo, prompt, seed, TURBO_STEPS, TURBO_GUIDANCE, device))
            prompts.append(prompt)
            seeds.append(seed)
            _tick(progress, "turbo-bf16", len(prompts), total)
        _apply(turbo.transformer, turbo_fp8_state)
        fresh: list[Image.Image] = []
        for prompt, seed in chunk:
            fresh.append(_zimage(turbo, prompt, seed, TURBO_STEPS, TURBO_GUIDANCE, device))
            _tick(progress, "turbo-fp8", len(turbo_cands) + len(fresh), total)
        turbo_cands.extend(fresh)
        _apply(turbo.transformer, transformer_state)
        _apply(turbo.text_encoder, encoder_fp8_state)
        enc_fresh: list[Image.Image] = []
        for prompt, seed in chunk:
            enc_fresh.append(_zimage(turbo, prompt, seed, TURBO_STEPS, TURBO_GUIDANCE, device))
            _tick(progress, "encoder-fp8", len(encoder_cands) + len(enc_fresh), total)
        encoder_cands.extend(enc_fresh)
        _apply(turbo.text_encoder, encoder_state)
        start = len(prompts) - len(chunk)
        chunk_prompts = prompts[start:]
        scores, drops = _score_series(
            refs[start:], turbo_cands[start:], chunk_prompts, metric, clip, device, image_score=False
        )
        turbo_lpips.extend(scores)
        turbo_drop.extend(drops)
        scores, drops = _score_series(
            refs[start:], encoder_cands[start:], chunk_prompts, metric, clip, device, image_score=False
        )
        encoder_lpips.extend(scores)
        encoder_drop.extend(drops)
    turbo_summary = _pair_stats(refs, turbo_cands, prompts, metric, clip, device, image_score=False)
    encoder_summary = _pair_stats(refs, encoder_cands, prompts, metric, clip, device, image_score=False)
    _write_review(grid_path, refs, turbo_cands, prompts, seeds)
    del turbo
    torch.cuda.empty_cache()

    base = ZImagePipeline.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16)
    base.to(device)
    _quiet(base)
    base_state = _cpu_state(base.transformer)
    base_fp8_state = _dequant([base_fp8])
    base_refs: list[Image.Image] = []
    base_cands: list[Image.Image] = []
    base_prompts: list[str] = []
    base_seeds: list[int] = []
    base_lpips: list[float] = []
    base_drop: list[float] = []
    while True:
        chunk = _next_chunk(
            cases,
            len(base_prompts),
            decided=_text_pair_decided(base_lpips, base_drop),
            minimum=LEAN_MINIMUM,
        )
        if not chunk:
            break
        _apply(base.transformer, base_state)
        for prompt, seed in chunk:
            base_refs.append(_zimage(base, prompt, seed, CREATIVE_STEPS, CREATIVE_GUIDANCE, device))
            base_prompts.append(prompt)
            base_seeds.append(seed)
            _tick(progress, "base-bf16", len(base_prompts), total)
        _apply(base.transformer, base_fp8_state)
        fresh = []
        for prompt, seed in chunk:
            fresh.append(_zimage(base, prompt, seed, CREATIVE_STEPS, CREATIVE_GUIDANCE, device))
            _tick(progress, "base-fp8", len(base_cands) + len(fresh), total)
        base_cands.extend(fresh)
        start = len(base_prompts) - len(chunk)
        scores, drops = _score_series(
            base_refs[start:], base_cands[start:], base_prompts[start:], metric, clip, device, image_score=False
        )
        base_lpips.extend(scores)
        base_drop.extend(drops)
    base_summary = _pair_stats(base_refs, base_cands, base_prompts, metric, clip, device, image_score=False)
    del base
    torch.cuda.empty_cache()
    return {
        "components": {
            "turbo-transformer": _component(
                turbo_summary,
                decide_base(turbo_summary["lpips_mean"], turbo_summary["clip_drop"]),
            ),
            "zimage-transformer": _component(
                base_summary,
                decide_base(base_summary["lpips_mean"], base_summary["clip_drop"]),
            ),
            "qwen3-4b-encoder": _component(
                encoder_summary,
                decide_base(encoder_summary["lpips_mean"], encoder_summary["clip_drop"]),
            ),
        }
    }


def _control_image(kind: str, index: int) -> torch.Tensor:
    scene = render_scene(index, IMAGE_SIZE)
    if kind == "canny":
        return canny_map(scene)
    if kind == "depth":
        return depth_map(index, IMAGE_SIZE)
    if kind == "pose":
        return pose_map(index, IMAGE_SIZE)
    raise StageError(f"unknown control {kind}", code="control")


def _lpips_decided(scores: list[float], minimum: int) -> bool:
    return metrics_decided([(scores, LPIPS_MAX, True)], minimum=minimum)


def _control_samples() -> list[tuple[str, int, str, torch.Tensor]]:
    samples: list[tuple[str, int, str, torch.Tensor]] = []
    for index, kind in LEAN_CONTROL:
        samples.append((CONTROL_PROMPTS[index], 0, kind, _control_image(kind, index)))
    return samples


def evaluate_control(
    *,
    turbo_dir: Path,
    lite_bf16: Path,
    lite_fp8: Path,
    grid_path: Path,
    inpaint_grid: Path,
    work_dir: Path,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Score a few hard control cases and write full-resolution review packs."""
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    work_dir.mkdir(parents=True, exist_ok=True)
    dequant_path = work_dir / "lite-dequant-bf16.safetensors"
    materialise_bf16([lite_fp8], dequant_path)
    samples = _control_samples()
    bf16_net = _cn_load(lite_bf16)
    fp8_net = _cn_load(dequant_path)
    pipe = ZImageControlNetPipeline.from_pretrained(
        str(turbo_dir),
        controlnet=bf16_net,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    _quiet(pipe)
    refs: list[Image.Image] = []
    cands: list[Image.Image] = []
    prompts: list[str] = []
    seeds: list[int] = []
    lpips_scores: list[float] = []
    while True:
        chunk = _next_chunk(
            samples,
            len(prompts),
            decided=_lpips_decided(lpips_scores, LEAN_CONTROL_MINIMUM),
            minimum=LEAN_CONTROL_MINIMUM,
        )
        if not chunk:
            break
        pipe.controlnet = bf16_net.to(device)
        for prompt, seed, _kind, control in chunk:
            refs.append(
                _zimage(
                    pipe,
                    prompt,
                    seed,
                    CONTROL_STEPS,
                    CONTROL_GUIDANCE,
                    device,
                    control_image=_pil(control),
                    controlnet_conditioning_scale=CONTROL_SCALE,
                )
            )
            prompts.append(prompt)
            seeds.append(seed)
            _tick(progress, "control-bf16", len(prompts), len(samples))
        # Swapped nets must share the transformer embedders like the pipeline constructor does.
        swap_controlnet(pipe, fp8_net, device, ZImageControlNetModel.from_transformer)
        fresh: list[Image.Image] = []
        for prompt, seed, _kind, control in chunk:
            fresh.append(
                _zimage(
                    pipe,
                    prompt,
                    seed,
                    CONTROL_STEPS,
                    CONTROL_GUIDANCE,
                    device,
                    control_image=_pil(control),
                    controlnet_conditioning_scale=CONTROL_SCALE,
                )
            )
            _tick(progress, "control-fp8", len(cands) + len(fresh), len(samples))
        cands.extend(fresh)
        start = len(prompts) - len(chunk)
        scores, _drops = _score_series(
            refs[start:], cands[start:], prompts[start:], metric, clip, device, image_score=False
        )
        lpips_scores.extend(scores)
    summary = _pair_stats(refs, cands, prompts, metric, clip, device, image_score=False)
    _write_review(grid_path, refs, cands, prompts, seeds)
    del pipe
    torch.cuda.empty_cache()

    inpaint_refs, inpaint_cands, inpaint_prompts, inpaint_seeds = _inpaint(
        turbo_dir, lite_bf16, dequant_path, device, metric, clip, progress
    )
    inpaint_summary = _pair_stats(inpaint_refs, inpaint_cands, inpaint_prompts, metric, clip, device, image_score=False)
    _write_review(inpaint_grid, inpaint_refs, inpaint_cands, inpaint_prompts, inpaint_seeds)
    # The published gate is the sample-weighted mean of the control set and the inpaint set.
    control_mass = summary["lpips_mean"] * summary["samples"]
    inpaint_mass = inpaint_summary["lpips_mean"] * inpaint_summary["samples"]
    combined = (control_mass + inpaint_mass) / (summary["samples"] + inpaint_summary["samples"])
    unweighted = _mean([summary["lpips_mean"], inpaint_summary["lpips_mean"]])
    return {
        "components": {
            "controlnet-lite": _component(
                {"lpips_mean": combined, "samples": summary["samples"] + inpaint_summary["samples"]},
                decide_control(combined),
            )
        },
        "control": summary,
        "inpaint": inpaint_summary,
        "lpips_unweighted_mean": unweighted,
    }


def _inpaint_one(
    pipe: ZImageControlNetInpaintPipeline,
    prompt: str,
    seed: int,
    scene: Image.Image,
    mask: Image.Image,
    control: Image.Image,
    device: torch.device,
) -> Image.Image:
    return pipe(
        prompt,
        image=scene,
        mask_image=mask,
        control_image=control,
        controlnet_conditioning_scale=CONTROL_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=CONTROL_STEPS,
        guidance_scale=CONTROL_GUIDANCE,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="pil",
    ).images[0]


def _inpaint(
    turbo_dir: Path,
    lite_bf16: Path,
    dequant_path: Path,
    device: torch.device,
    metric: torch.nn.Module,
    clip: ClipScorer,
    progress: Progress | None,
) -> tuple[list[Image.Image], list[Image.Image], list[str], list[int]]:
    prepared: list[tuple[str, int, Image.Image, Image.Image, Image.Image]] = []
    for index, prompt in enumerate(LEAN_INPAINT):
        scene = _pil(render_scene(1000 + index, IMAGE_SIZE))
        mask = _pil(inpaint_mask(IMAGE_SIZE))
        control = _pil(canny_map(render_scene(1000 + index, IMAGE_SIZE)))
        prepared.append((prompt, index, scene, mask, control))
    bf16_net = _cn_load(lite_bf16)
    fp8_net = _cn_load(dequant_path)
    pipe = ZImageControlNetInpaintPipeline.from_pretrained(
        str(turbo_dir),
        controlnet=bf16_net,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    _quiet(pipe)
    refs: list[Image.Image] = []
    cands: list[Image.Image] = []
    prompts: list[str] = []
    seeds: list[int] = []
    scores: list[float] = []
    while True:
        chunk = _next_chunk(
            prepared,
            len(prompts),
            decided=_lpips_decided(scores, LEAN_INPAINT_MINIMUM),
            minimum=LEAN_INPAINT_MINIMUM,
        )
        if not chunk:
            break
        pipe.controlnet = bf16_net.to(device)
        for prompt, seed, scene, mask, control in chunk:
            refs.append(_inpaint_one(pipe, prompt, seed, scene, mask, control, device))
            prompts.append(prompt)
            seeds.append(seed)
            _tick(progress, "inpaint-bf16", len(prompts), len(prepared))
        # Swapped nets must share the transformer embedders like the pipeline constructor does.
        swap_controlnet(pipe, fp8_net, device, ZImageControlNetModel.from_transformer)
        fresh: list[Image.Image] = []
        for prompt, seed, scene, mask, control in chunk:
            fresh.append(_inpaint_one(pipe, prompt, seed, scene, mask, control, device))
            _tick(progress, "inpaint-fp8", len(cands) + len(fresh), len(prepared))
        cands.extend(fresh)
        start = len(prompts) - len(chunk)
        chunk_scores, _drops = _score_series(
            refs[start:], cands[start:], prompts[start:], metric, clip, device, image_score=False
        )
        scores.extend(chunk_scores)
    del pipe
    torch.cuda.empty_cache()
    return refs, cands, prompts, seeds


def evaluate_edit(
    *,
    edit_dir: Path,
    transformer_fp8: list[Path],
    encoder_fp8: Path,
    grid_path: Path,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Score a few hard edit pairs and write a full-resolution review pack."""
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(edit_dir), torch_dtype=torch.bfloat16)
    pipe.to(device)
    _quiet(pipe)
    allowed, budget_seeds = edit_budget()
    allowed_set = set(allowed)
    seed = budget_seeds[0] if budget_seeds else 0
    prepared = [
        (instruction, seed, _pil(render_scene(scene_seed, IMAGE_SIZE)))
        for instruction, scene_seed in LEAN_EDIT
        if (instruction, scene_seed) in allowed_set
    ]
    total = len(prepared)

    def _edit(instruction: str, edit_seed: int, image: Image.Image) -> Image.Image:
        return pipe(
            image=image,
            prompt=instruction,
            negative_prompt=" ",
            true_cfg_scale=EDIT_TRUE_CFG,
            num_inference_steps=EDIT_STEPS,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=torch.Generator(device=device).manual_seed(edit_seed),
            output_type="pil",
        ).images[0]

    def _edit_decided(lpips_scores: list[float], clip_i: list[float]) -> bool:
        return metrics_decided(
            [(lpips_scores, LPIPS_EDIT_MAX, True), (clip_i, CLIP_I_MIN, False)],
            minimum=LEAN_EDIT_MINIMUM,
        )

    transformer_state = _cpu_state(pipe.transformer)
    encoder_state = _cpu_state(pipe.text_encoder)
    transformer_fp8_state = _dequant(transformer_fp8)
    encoder_fp8_state = _dequant([encoder_fp8])
    refs: list[Image.Image] = []
    transformer_cands: list[Image.Image] = []
    encoder_cands: list[Image.Image] = []
    prompts: list[str] = []
    seeds: list[int] = []
    transformer_lpips: list[float] = []
    transformer_clip: list[float] = []
    encoder_lpips: list[float] = []
    encoder_clip: list[float] = []
    while True:
        decided = _edit_decided(transformer_lpips, transformer_clip) and _edit_decided(encoder_lpips, encoder_clip)
        chunk = _next_chunk(prepared, len(prompts), decided=decided, minimum=LEAN_EDIT_MINIMUM)
        if not chunk:
            break
        _apply(pipe.transformer, transformer_state)
        _apply(pipe.text_encoder, encoder_state)
        for instruction, edit_seed, image in chunk:
            refs.append(_edit(instruction, edit_seed, image))
            prompts.append(instruction)
            seeds.append(edit_seed)
            _tick(progress, "edit-bf16", len(prompts), total)
        free_eval_stage()
        _apply(pipe.transformer, transformer_fp8_state)
        fresh: list[Image.Image] = []
        for instruction, edit_seed, image in chunk:
            fresh.append(_edit(instruction, edit_seed, image))
            _tick(progress, "edit-transformer-fp8", len(transformer_cands) + len(fresh), total)
        transformer_cands.extend(fresh)
        free_eval_stage()
        _apply(pipe.transformer, transformer_state)
        _apply(pipe.text_encoder, encoder_fp8_state)
        enc_fresh: list[Image.Image] = []
        for instruction, edit_seed, image in chunk:
            enc_fresh.append(_edit(instruction, edit_seed, image))
            _tick(progress, "edit-encoder-fp8", len(encoder_cands) + len(enc_fresh), total)
        encoder_cands.extend(enc_fresh)
        start = len(prompts) - len(chunk)
        scores, clip_i = _score_series(
            refs[start:], transformer_cands[start:], prompts[start:], metric, clip, device, image_score=True
        )
        transformer_lpips.extend(scores)
        transformer_clip.extend(clip_i)
        scores, clip_i = _score_series(
            refs[start:], encoder_cands[start:], prompts[start:], metric, clip, device, image_score=True
        )
        encoder_lpips.extend(scores)
        encoder_clip.extend(clip_i)
    transformer_summary = _pair_stats(refs, transformer_cands, prompts, metric, clip, device, image_score=True)
    encoder_summary = _pair_stats(refs, encoder_cands, prompts, metric, clip, device, image_score=True)
    pipe = free_eval_stage(pipe)
    if pipe is not None:
        raise StageError("edit pipeline was not released", code="quality")
    _write_review(grid_path, refs, transformer_cands, prompts, seeds)
    return {
        "components": {
            "edit-transformer": _component(
                transformer_summary,
                decide_edit(transformer_summary["lpips_mean"], transformer_summary["clip_i"]),
            ),
            "qwen2.5-vl-encoder": _component(
                encoder_summary,
                decide_edit(encoder_summary["lpips_mean"], encoder_summary["clip_i"]),
            ),
        }
    }
