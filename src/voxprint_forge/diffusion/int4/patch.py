"""Idempotent Qwen loader patch for the pinned deepcompressor tree.

The anchors are the text at commit ``69f3473f5e1c1504bae35cc50c7858ef900a9b17``.
A checkout that does not contain them is refused, so a moved pin cannot
silently quantize the wrong code.
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
    return _replace_once(path, _COLLECT_ANCHOR, _COLLECT_INSERT, present='task == "image-edit"')


def apply_checkout(root: Path) -> dict[str, str]:
    """Patch one deepcompressor checkout and check its Apache-2.0 file."""
    assert_apache(root / "LICENSE", what="deepcompressor")
    pipeline = root / "deepcompressor" / "app" / "diffusion" / "pipeline" / "config.py"
    dataset = root / "deepcompressor" / "app" / "diffusion" / "dataset" / "data" / "__init__.py"
    collect = root / "deepcompressor" / "app" / "diffusion" / "dataset" / "collect" / "calib.py"
    return {
        "pipeline": apply_pipeline(pipeline),
        "dataset": apply_dataset(dataset),
        "collect": apply_collect(collect),
    }
