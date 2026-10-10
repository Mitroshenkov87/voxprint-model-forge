"""Quality checks. FP8 is dequantized to BF16 because the A100 has no FP8 tensor cores."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
from transformers import CLIPModel, CLIPProcessor

from voxprint_forge.diffusion.alias import alias_key as _alias_key
from voxprint_forge.diffusion.fp8 import dequant_state
from voxprint_forge.diffusion.policy import decide_base, decide_control, decide_edit
from voxprint_forge.diffusion.prompts import (
    BASE_PROMPTS,
    BASE_SEEDS,
    CONTROL_PROMPTS,
    CONTROL_SEEDS,
    CONTROL_TYPES,
    EDIT_PAIRS,
    EDIT_SEEDS,
    INPAINT_PROMPTS,
)
from voxprint_forge.diffusion.scenes import (
    canny_map,
    contact_sheet,
    depth_map,
    downscale,
    inpaint_mask,
    pose_map,
    render_scene,
)
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
from voxprint_forge.diffusion.weights import load_tensors, materialise_bf16
from voxprint_forge.errors import StageError, ToolMissingError

Progress = Callable[[str, int, int], None]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).mul(255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path, format="AVIF")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_n = torch.nn.functional.normalize(left.float().flatten(), dim=0)
    right_n = torch.nn.functional.normalize(right.float().flatten(), dim=0)
    return float((left_n * right_n).sum())


class ClipScorer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def text_image(self, image: Image.Image, text: str) -> float:
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


def _cells(pairs: list[tuple[Image.Image, Image.Image]]) -> list[torch.Tensor]:
    cells: list[torch.Tensor] = []
    for ref, cand in pairs:
        cells.append(downscale(_from_pil(ref)))
        cells.append(downscale(_from_pil(cand)))
    return cells


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
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    prompts = list(BASE_PROMPTS)
    total = len(prompts) * len(BASE_SEEDS)

    turbo = ZImagePipeline.from_pretrained(str(turbo_dir), torch_dtype=torch.bfloat16)
    turbo.to(device)
    _quiet(turbo)
    refs: list[Image.Image] = []
    done = 0
    for prompt in prompts:
        for seed in BASE_SEEDS:
            refs.append(_zimage(turbo, prompt, seed, TURBO_STEPS, TURBO_GUIDANCE, device))
            done += 1
            _tick(progress, "turbo-bf16", done, total)
    repeated = [prompt for prompt in prompts for _seed in BASE_SEEDS]

    transformer_state = _cpu_state(turbo.transformer)
    _apply(turbo.transformer, _dequant([turbo_fp8]))
    turbo_cands = [
        _zimage(turbo, prompt, seed, TURBO_STEPS, TURBO_GUIDANCE, device)
        for prompt in prompts
        for seed in BASE_SEEDS
    ]
    turbo_summary = _pair_stats(refs, turbo_cands, repeated, metric, clip, device, image_score=False)
    turbo_pairs = list(zip(refs, turbo_cands, strict=True))
    _apply(turbo.transformer, transformer_state)

    _apply(turbo.text_encoder, _dequant([encoder_fp8]))
    encoder_cands = [
        _zimage(turbo, prompt, seed, TURBO_STEPS, TURBO_GUIDANCE, device)
        for prompt in prompts
        for seed in BASE_SEEDS
    ]
    encoder_summary = _pair_stats(refs, encoder_cands, repeated, metric, clip, device, image_score=False)
    del turbo
    torch.cuda.empty_cache()

    base = ZImagePipeline.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16)
    base.to(device)
    _quiet(base)
    base_refs = [
        _zimage(base, prompt, seed, CREATIVE_STEPS, CREATIVE_GUIDANCE, device)
        for prompt in prompts
        for seed in BASE_SEEDS
    ]
    _apply(base.transformer, _dequant([base_fp8]))
    base_cands = [
        _zimage(base, prompt, seed, CREATIVE_STEPS, CREATIVE_GUIDANCE, device)
        for prompt in prompts
        for seed in BASE_SEEDS
    ]
    base_summary = _pair_stats(base_refs, base_cands, repeated, metric, clip, device, image_score=False)
    del base
    torch.cuda.empty_cache()

    cells = _cells(turbo_pairs)
    cells.extend(_cells(list(zip(base_refs, base_cands, strict=True))))
    save_avif(contact_sheet(cells, 2), grid_path)
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


def _control_samples() -> list[tuple[str, int, str, torch.Tensor]]:
    samples: list[tuple[str, int, str, torch.Tensor]] = []
    for index, prompt in enumerate(CONTROL_PROMPTS):
        for kind in CONTROL_TYPES:
            control = _control_image(kind, index)
            for seed in CONTROL_SEEDS:
                samples.append((prompt, seed, kind, control))
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
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    work_dir.mkdir(parents=True, exist_ok=True)
    dequant_path = work_dir / "lite-dequant-bf16.safetensors"
    materialise_bf16([lite_fp8], dequant_path)
    samples = _control_samples()
    bf16_net = ZImageControlNetModel.from_single_file(str(lite_bf16), torch_dtype=torch.bfloat16)
    pipe = ZImageControlNetPipeline.from_pretrained(
        str(turbo_dir),
        controlnet=bf16_net,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    _quiet(pipe)
    refs: list[Image.Image] = []
    for done, (prompt, seed, _kind, control) in enumerate(samples, start=1):
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
        _tick(progress, "control-bf16", done, len(samples))
    pipe.controlnet = ZImageControlNetModel.from_single_file(str(dequant_path), torch_dtype=torch.bfloat16).to(device)
    cands: list[Image.Image] = []
    for done, (prompt, seed, _kind, control) in enumerate(samples, start=1):
        cands.append(
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
        _tick(progress, "control-fp8", done, len(samples))
    prompts = [prompt for prompt, _seed, _kind, _control in samples]
    summary = _pair_stats(refs, cands, prompts, metric, clip, device, image_score=False)
    save_avif(contact_sheet(_cells(list(zip(refs, cands, strict=True))), 2), grid_path)
    del pipe
    torch.cuda.empty_cache()

    inpaint_refs, inpaint_cands = _inpaint(turbo_dir, lite_bf16, dequant_path, device, metric, clip, progress)
    inpaint_prompts = list(INPAINT_PROMPTS)
    inpaint_summary = _pair_stats(inpaint_refs, inpaint_cands, inpaint_prompts, metric, clip, device, image_score=False)
    save_avif(contact_sheet(_cells(list(zip(inpaint_refs, inpaint_cands, strict=True))), 2), inpaint_grid)
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


def _inpaint(
    turbo_dir: Path,
    lite_bf16: Path,
    dequant_path: Path,
    device: torch.device,
    metric: torch.nn.Module,
    clip: ClipScorer,
    progress: Progress | None,
) -> tuple[list[Image.Image], list[Image.Image]]:
    del metric, clip
    net = ZImageControlNetModel.from_single_file(str(lite_bf16), torch_dtype=torch.bfloat16)
    pipe = ZImageControlNetInpaintPipeline.from_pretrained(
        str(turbo_dir),
        controlnet=net,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    _quiet(pipe)
    refs: list[Image.Image] = []
    prepared: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for index, prompt in enumerate(INPAINT_PROMPTS):
        scene = _pil(render_scene(1000 + index, IMAGE_SIZE))
        mask = _pil(inpaint_mask(IMAGE_SIZE))
        control = _pil(canny_map(render_scene(1000 + index, IMAGE_SIZE)))
        prepared.append((prompt, scene, mask, control))
        refs.append(
            pipe(
                prompt,
                image=scene,
                mask_image=mask,
                control_image=control,
                controlnet_conditioning_scale=CONTROL_SCALE,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=CONTROL_STEPS,
                guidance_scale=CONTROL_GUIDANCE,
                generator=torch.Generator(device=device).manual_seed(index),
                output_type="pil",
            ).images[0]
        )
        _tick(progress, "inpaint-bf16", index + 1, len(INPAINT_PROMPTS))
    pipe.controlnet = ZImageControlNetModel.from_single_file(str(dequant_path), torch_dtype=torch.bfloat16).to(device)
    cands: list[Image.Image] = []
    for index, (prompt, scene, mask, control) in enumerate(prepared):
        cands.append(
            pipe(
                prompt,
                image=scene,
                mask_image=mask,
                control_image=control,
                controlnet_conditioning_scale=CONTROL_SCALE,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=CONTROL_STEPS,
                guidance_scale=CONTROL_GUIDANCE,
                generator=torch.Generator(device=device).manual_seed(index),
                output_type="pil",
            ).images[0]
        )
        _tick(progress, "inpaint-fp8", index + 1, len(INPAINT_PROMPTS))
    del pipe
    torch.cuda.empty_cache()
    return refs, cands


def evaluate_edit(
    *,
    edit_dir: Path,
    transformer_fp8: list[Path],
    encoder_fp8: Path,
    grid_path: Path,
    progress: Progress | None = None,
) -> dict[str, object]:
    device = _cuda()
    metric = _lpips_net(device)
    clip = ClipScorer(device)
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(edit_dir), torch_dtype=torch.bfloat16)
    pipe.to(device)
    _quiet(pipe)
    prepared = [
        (instruction, seed, _pil(render_scene(scene_seed, IMAGE_SIZE)))
        for instruction, scene_seed in EDIT_PAIRS
        for seed in EDIT_SEEDS
    ]
    total = len(prepared)
    editor = pipe

    def _edit(instruction: str, seed: int, image: Image.Image) -> Image.Image:
        return editor(
            image=image,
            prompt=instruction,
            negative_prompt=" ",
            true_cfg_scale=EDIT_TRUE_CFG,
            num_inference_steps=EDIT_STEPS,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="pil",
        ).images[0]

    refs: list[Image.Image] = []
    for done, (instruction, seed, image) in enumerate(prepared, start=1):
        refs.append(_edit(instruction, seed, image))
        _tick(progress, "edit-bf16", done, total)
    prompts = [instruction for instruction, _seed, _image in prepared]

    transformer_state = _cpu_state(pipe.transformer)
    _apply(pipe.transformer, _dequant(transformer_fp8))
    transformer_cands = [_edit(instruction, seed, image) for instruction, seed, image in prepared]
    transformer_summary = _pair_stats(refs, transformer_cands, prompts, metric, clip, device, image_score=True)
    _apply(pipe.transformer, transformer_state)

    _apply(pipe.text_encoder, _dequant([encoder_fp8]))
    encoder_cands = [_edit(instruction, seed, image) for instruction, seed, image in prepared]
    encoder_summary = _pair_stats(refs, encoder_cands, prompts, metric, clip, device, image_score=True)
    del pipe
    torch.cuda.empty_cache()

    cells = _cells(list(zip(refs, transformer_cands, strict=True)))
    save_avif(contact_sheet(cells, 2), grid_path)
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
