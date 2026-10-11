"""Idempotent Qwen loader patch for the pinned deepcompressor tree.

Pipeline and dataset anchors are the text at
``69f3473f5e1c1504bae35cc50c7858ef900a9b17``. The pinned ``syh`` commit
``019cbdea`` still has those pipeline anchors, a different collect call, and
the issue 116 scale assert. A checkout that matches neither collect line is
refused.
"""

from __future__ import annotations

from pathlib import Path

from voxprint_forge.errors import LicenceError, StageError

_TASK_ANCHOR = (
    '        elif self.name == "flux.1-fill-dev":\n'
    '            self.task = "inpainting"\n'
)
_TASK_INSERT = (
    '        elif self.name == "flux.1-fill-dev":\n'
    '            self.task = "inpainting"\n'
    '        elif self.name.startswith("qwen-image-edit"):\n'
    "            # Voxprint: Qwen-Image-Edit-2511 takes an image. zero_cond_t is in the transformer config.\n"
    '            self.task = "image-edit"\n'
)
_BUILD_ANCHOR = (
    "        else:\n"
    "            pipeline = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype)\n"
)
_BUILD_INSERT = (
    '        elif name.startswith("qwen-image"):\n'
    "            # Voxprint: Qwen-Image and Qwen-Image-Edit load the class named in model_index.\n"
    "            from diffusers import DiffusionPipeline as _QwenPipeline\n"
    "\n"
    "            pipeline = _QwenPipeline.from_pretrained(path, torch_dtype=dtype)\n"
    '            if name.startswith("qwen-image-edit") and hasattr(pipeline, "transformer"):\n'
    "                pipeline.transformer.register_to_config(zero_cond_t=True)\n"
    "        else:\n"
    "            pipeline = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype)\n"
)
_YAML_ANCHOR = (
    '    ret = {"filename": [], "prompt": [], "meta_path": []}\n'
    "    idx = 0\n"
    "    for name in names:\n"
    "        prompt = meta[name]\n"
    "        for j in range(repeat):\n"
    '            ret["filename"].append(f"{name}-{j}")\n'
    '            ret["prompt"].append(prompt)\n'
    '            ret["meta_path"].append(meta_path)\n'
    "            idx += 1\n"
    "    return ret\n"
)
_YAML_INSERT = (
    '    ret = {"filename": [], "prompt": [], "meta_path": [], "image": []}\n'
    "    idx = 0\n"
    "    for name in names:\n"
    "        raw = meta[name]\n"
    "        if isinstance(raw, dict):\n"
    '            prompt = raw["prompt"]\n'
    '            image = str(raw.get("image", ""))\n'
    "        else:\n"
    "            prompt = raw\n"
    '            image = ""\n'
    "        for j in range(repeat):\n"
    '            ret["filename"].append(f"{name}-{j}")\n'
    '            ret["prompt"].append(prompt)\n'
    '            ret["meta_path"].append(meta_path)\n'
    '            ret["image"].append(image)\n'
    "            idx += 1\n"
    "    return ret\n"
)
_FEATURES_ANCHOR = (
    "                    \"filename\": datasets.Value(\"string\"),\n"
    "                    \"prompt\": datasets.Value(\"string\"),\n"
    "                    \"meta_path\": datasets.Value(\"string\"),\n"
)
_FEATURES_INSERT = (
    "                    \"filename\": datasets.Value(\"string\"),\n"
    "                    \"prompt\": datasets.Value(\"string\"),\n"
    "                    \"meta_path\": datasets.Value(\"string\"),\n"
    "                    \"image\": datasets.Value(\"string\"),\n"
)
_COLLECT_ANCHOR = "        result_images = pipeline(prompts, generator=generators, **pipeline_kwargs).images\n"
_COLLECT_INSERT = (
    "        if task == \"image-edit\":\n"
    "            from PIL import Image\n"
    "\n"
    "            pipeline_kwargs[\"image\"] = [Image.open(path).convert(\"RGB\") for path in batch[\"image\"]]\n"
    "            scale = pipeline_kwargs.pop(\"guidance_scale\", None)\n"
    "            if scale is not None:\n"
    "                pipeline_kwargs[\"true_cfg_scale\"] = scale\n"
    "            pipeline_kwargs.setdefault(\"negative_prompt\", \" \")\n"
    "        result_images = pipeline(prompts, generator=generators, **pipeline_kwargs).images\n"
)
_COLLECT_SYH_ANCHOR = (
    "        result_images = pipeline(prompt_embeds=prompt_embeds, "
    "prompt_embeds_mask=prompt_embeds_mask, negative_prompt_embeds=negative_prompt_embeds, "
    "negative_prompt_embeds_mask=negative_prompt_embeds_mask,generator=generators, "
    "target_seq_len=target_seq_len, **pipeline_kwargs).images\n"
)
_COLLECT_SYH_INSERT = (
    "        if task == \"image-edit\":\n"
    "            from PIL import Image\n"
    "\n"
    "            pipeline_kwargs[\"image\"] = [Image.open(path).convert(\"RGB\") for path in batch[\"image\"]]\n"
    "            scale = pipeline_kwargs.pop(\"guidance_scale\", None)\n"
    "            if scale is not None:\n"
    "                pipeline_kwargs[\"true_cfg_scale\"] = scale\n"
    "            pipeline_kwargs.setdefault(\"negative_prompt\", \" \")\n"
    "            result_images = pipeline(\n"
    "                prompt_embeds=prompt_embeds,\n"
    "                prompt_embeds_mask=prompt_embeds_mask,\n"
    "                negative_prompt_embeds=negative_prompt_embeds,\n"
    "                negative_prompt_embeds_mask=negative_prompt_embeds_mask,\n"
    "                generator=generators,\n"
    "                **pipeline_kwargs,\n"
    "            ).images\n"
    "        else:\n"
    "            "
    + _COLLECT_SYH_ANCHOR.lstrip(" ")
)
_W4_SCALE_ANCHOR = '    assert scale.ndim == 4, "scale tensor should be 4D."\n    assert scale.shape[0] == oc\n'
_W4_SCALE_INSERT = (
    '    assert scale.ndim == 4, "scale tensor should be 4D."\n'
    "    # Voxprint: nunchaku-ai/deepcompressor issue 116. img_mod scales can be [1, 1, 1, 1].\n"
    "    if int(scale.shape[0]) == 1 and oc != 1:\n"
    "        scale = scale.expand(oc, int(scale.shape[1]), int(scale.shape[2]), int(scale.shape[3])).contiguous()\n"
    "    assert scale.shape[0] == oc\n"
)
_W4_SCALE_PRESENT = "issue 116"


def assert_apache(path: Path, *, what: str) -> None:
    """Refuse a checkout whose licence file is not Apache-2.0."""
    if not path.is_file():
        raise LicenceError(f"{what} has no licence file at {path}", code="untagged")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in text or "Version 2.0" not in text:
        raise LicenceError(f"{what} licence at {path} is not Apache-2.0", code="non_permissive")


def _replace_once(path: Path, old: str, new: str, *, present: str) -> str:
    text = path.read_text(encoding="utf-8")
    if present in text:
        return "present"
    if old not in text:
        raise StageError(f"{path} does not match the pinned deepcompressor text", code="patch_anchor")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "applied"


def apply_pipeline(path: Path) -> str:
    """Register the Qwen pipeline name and the image-edit task. Idempotent."""
    task = _replace_once(path, _TASK_ANCHOR, _TASK_INSERT, present='self.task = "image-edit"')
    build = _replace_once(path, _BUILD_ANCHOR, _BUILD_INSERT, present='name.startswith("qwen-image")')
    if task == "present" and build == "present":
        return "present"
    return "applied"


def apply_dataset(path: Path) -> str:
    """Accept a prompt entry that names an image. Idempotent."""
    features = _replace_once(path, _FEATURES_ANCHOR, _FEATURES_INSERT, present='"image": datasets.Value("string")')
    body = _replace_once(path, _YAML_ANCHOR, _YAML_INSERT, present='ret["image"].append(image)')
    if features == "present" and body == "present":
        return "present"
    return "applied"


def apply_collect(path: Path) -> str:
    """Pass the calibration image and ``true_cfg_scale`` for image-edit. Idempotent."""
    text = path.read_text(encoding="utf-8")
    if 'task == "image-edit"' in text:
        return "present"
    if _COLLECT_ANCHOR in text:
        path.write_text(text.replace(_COLLECT_ANCHOR, _COLLECT_INSERT, 1), encoding="utf-8")
        return "applied"
    if _COLLECT_SYH_ANCHOR in text:
        path.write_text(text.replace(_COLLECT_SYH_ANCHOR, _COLLECT_SYH_INSERT, 1), encoding="utf-8")
        return "applied"
    raise StageError(f"{path} does not match the pinned deepcompressor text", code="patch_anchor")


def apply_w4_scale(path: Path) -> str:
    """Broadcast a per-tensor W4 scale before the ``oc`` assert. Idempotent.

    The ``syh`` checkout has this assert. A tree without ``utils.py`` is skipped
    so the older anchor tests still apply.
    """
    if not path.is_file():
        return "skipped"
    return _replace_once(path, _W4_SCALE_ANCHOR, _W4_SCALE_INSERT, present=_W4_SCALE_PRESENT)


def apply_checkout(root: Path) -> dict[str, str]:
    """Patch one deepcompressor checkout and check its Apache-2.0 file."""
    assert_apache(root / "LICENSE", what="deepcompressor")
    pipeline = root / "deepcompressor" / "app" / "diffusion" / "pipeline" / "config.py"
    dataset = root / "deepcompressor" / "app" / "diffusion" / "dataset" / "data" / "__init__.py"
    collect = root / "deepcompressor" / "app" / "diffusion" / "dataset" / "collect" / "calib.py"
    utils = root / "deepcompressor" / "backend" / "nunchaku" / "utils.py"
    return {
        "pipeline": apply_pipeline(pipeline),
        "dataset": apply_dataset(dataset),
        "collect": apply_collect(collect),
        "convert": apply_w4_scale(utils),
    }
