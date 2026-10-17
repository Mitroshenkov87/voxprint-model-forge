"""GPU generation for a compare arm.

``diffusers`` is imported inside the load functions. It is an optional extra
and must not load for the LLM commands.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Never, Protocol

from PIL import Image

from voxprint_forge.compare.cases import Case
from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.int4 import load_zimage_int4
from voxprint_forge.compare.plan import Arm, Job, Kind
from voxprint_forge.diffusion.scenes import canny_map, depth_map, pose_map, render_scene
from voxprint_forge.scrub import active_secrets, scrub

GeneratorFactory = Callable[[str], "Seedable"]


class Seedable(Protocol):
    """Object returned by ``torch.Generator``."""

    def manual_seed(self, seed: int) -> Seedable:
        """Set the seed and return this generator."""
        ...


def seeded_generator(seed: int, factory: GeneratorFactory | None = None) -> Seedable:
    """``torch.Generator("cuda").manual_seed(seed)`` for every arm and case."""
    if factory is None:
        import torch

        return torch.Generator("cuda").manual_seed(seed)
    return factory("cuda").manual_seed(seed)


def fetch_arm(arm: Arm, dest: Path) -> None:
    """Download one public arm. No token is printed. Private data is not fetched.

    ``local`` arms copy named files from ``FORGE_FP8_DIR`` and do not publish them.
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    if arm.base_repo:
        _snapshot(snapshot_download, arm.base_repo, arm.base_revision, dest / "base", None)
    if arm.local:
        _copy_local(arm, dest / "weights")
    elif arm.repo:
        patterns = list(arm.files) or None
        _snapshot(snapshot_download, arm.repo, arm.revision, dest / "weights", patterns)
    elif not arm.base_repo:
        raise CompareSetup(f"arm {arm.id} has no repo", code="compare_weights")


def make_generator(job: Job, weights: Path) -> _Resident:
    """Return a generator that keeps a single pipeline resident.

    Call ``load_arm`` before timing images. ``release`` drops that pipeline
    before the next arm loads, so the peak is one arm and not the whole job.
    """
    del job
    return _Resident(weights)


class _Resident:
    """One compare pipeline. Loading another arm frees the previous one."""

    def __init__(self, weights: Path) -> None:
        self.weights = weights
        self.pipe: object | None = None
        self.arm_id: str | None = None
        self.int4_notes: list[dict[str, str]] = []

    def load_arm(self, arm: Arm, job: Job) -> None:
        """Load ``arm`` after freeing the pipeline that was resident."""
        if self.pipe is not None and self.arm_id == arm.id:
            return
        self.release()
        pipe = load_pipe(job, arm, self.weights / arm.id)
        info = getattr(pipe, "forge_int4", None)
        if isinstance(info, dict):
            self.int4_notes.append({str(key): str(value) for key, value in info.items()})
        self.pipe = pipe
        self.arm_id = arm.id

    def release(self) -> None:
        """Drop the pipeline, collect garbage, and empty the CUDA cache."""
        pipe = self.pipe
        self.pipe = None
        self.arm_id = None
        _free_pipe(pipe)

    def __call__(self, arm: Arm, case: Case, job: Job) -> Image.Image:
        """Render one case. Loads the arm when the caller has not."""
        if self.pipe is None or self.arm_id != arm.id:
            self.load_arm(arm, job)
        if self.pipe is None:
            raise CompareSetup(f"arm {arm.id} pipeline was released", code="compare_generate")
        return render(self.pipe, job, case)


def load_pipe(job: Job, arm: Arm, folder: Path) -> object:
    """Load one arm in BF16 compute. FP8 weights are dequantised first."""
    import torch
    from diffusers import ZImageControlNetPipeline, ZImagePipeline

    dtype = torch.bfloat16
    kind: Kind = job.kind
    if kind == "control":
        base = _require_dir(folder / "base", arm.id)
        control_file = _control_file(arm, folder)
        net = load_controlnet(control_file)
        pipe = ZImageControlNetPipeline.from_pretrained(str(base), controlnet=net, torch_dtype=dtype)
    elif kind == "turbo" or kind == "creative":
        if arm.quant == "int4" or arm.compute == "int4":
            pipe = _base_pipe(ZImagePipeline, arm, folder, dtype)
            weight = _int4_file(arm, folder)
            model, info = load_zimage_int4(weight)
            pipe.transformer = model
            pipe.forge_int4 = {"arm": arm.id, **info}
        else:
            pipe = _text_pipe(ZImagePipeline, arm, folder, dtype)
    else:
        return _unexpected_kind(kind)
    _place(pipe, arm)
    return pipe


def render(pipe: object, job: Job, case: Case) -> Image.Image:
    """Generate one image. The caller must not log ``case.prompt``."""
    kwargs: dict[str, object] = {
        "prompt": case.prompt,
        "height": case.height,
        "width": case.width,
        "num_inference_steps": job.steps,
        "guidance_scale": job.guidance,
        "generator": seeded_generator(case.seed),
        "output_type": "pil",
    }
    if case.negative:
        kwargs["negative_prompt"] = case.negative
    if case.control:
        kwargs["control_image"] = control_image(case.control, case.seed, case.width)
        kwargs["controlnet_conditioning_scale"] = job.control_scale
    result = pipe(**kwargs)
    images = getattr(result, "images", None)
    if not images:
        raise CompareSetup(f"arm produced no image for prompt id {case.prompt_id}", code="compare_generate")
    image = images[0]
    if not isinstance(image, Image.Image):
        raise CompareSetup(f"arm image was not a picture for prompt id {case.prompt_id}", code="compare_generate")
    return image


def control_image(kind: str, seed: int, size: int) -> Image.Image:
    """Synthetic canny, depth, or pose map. These are drawn in-job, not photos."""
    if kind == "canny":
        tensor = canny_map(render_scene(seed, size))
    elif kind == "depth":
        tensor = depth_map(seed, size)
    elif kind == "pose":
        tensor = pose_map(seed, size)
    else:
        raise CompareSetup(f"unknown control {kind}", code="compare_control")
    array = tensor.detach().clamp(0, 1).mul(255).round().to(dtype=tensor.dtype)
    import torch

    pixels = array.to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels, mode="RGB")


def load_controlnet(path: Path) -> object:
    """Load a Union 2.1 single file, including a 3-layer lite checkpoint."""
    import torch
    from diffusers import ZImageControlNetModel
    from safetensors import safe_open

    from voxprint_forge.diffusion.controlnet_union import places_for_count, union_config

    with safe_open(str(path), framework="pt") as handle:
        layers = sorted({int(key.split(".")[1]) for key in handle.keys() if key.startswith("control_layers.")})
    if not layers:
        raise CompareSetup(f"{path.name} has no control layers", code="compare_control")
    folder = Path(tempfile.mkdtemp(prefix="forge-compare-cn-"))
    (folder / "config.json").write_text(
        json.dumps(union_config(places_for_count(len(layers)))),
        encoding="utf-8",
    )
    return ZImageControlNetModel.from_single_file(str(path), config=str(folder), torch_dtype=torch.bfloat16)


def _base_pipe(pipeline_cls: object, arm: Arm, folder: Path, dtype: object) -> object:
    base = folder / "base"
    if not base.is_dir():
        weights = folder / "weights"
        if (weights / "model_index.json").is_file():
            return pipeline_cls.from_pretrained(str(weights), torch_dtype=dtype)
        raise CompareSetup(f"arm {arm.id} is missing a base snapshot", code="compare_weights")
    return pipeline_cls.from_pretrained(str(base), torch_dtype=dtype)


def _int4_file(arm: Arm, folder: Path) -> Path:
    weights = folder / "weights"
    files = sorted(weights.rglob("*.safetensors")) if weights.is_dir() else []
    if len(files) != 1:
        raise CompareSetup(f"arm {arm.id} expected one INT4 file", code="compare_weights")
    return files[0]


def _copy_local(arm: Arm, dest: Path) -> None:
    raw = os.environ.get("FORGE_FP8_DIR", "").strip()
    if not raw or not Path(raw).is_dir():
        raise CompareSetup("FORGE_FP8_DIR is not a directory", code="compare_weights")
    root = Path(raw)
    dest.mkdir(parents=True, exist_ok=True)
    if not arm.files:
        raise CompareSetup(f"arm {arm.id} has no local files", code="compare_weights")
    for pattern in arm.files:
        matches = [path for path in sorted(root.glob(pattern)) if path.is_file()]
        if len(matches) != 1:
            raise CompareSetup(f"arm {arm.id} expected one local file named {pattern}", code="compare_weights")
        shutil.copyfile(matches[0], dest / matches[0].name)


def _text_pipe(pipeline_cls: object, arm: Arm, folder: Path, dtype: object) -> object:
    weights = folder / "weights"
    if (weights / "model_index.json").is_file():
        return pipeline_cls.from_pretrained(str(weights), torch_dtype=dtype)
    if (folder / "model_index.json").is_file():
        return pipeline_cls.from_pretrained(str(folder), torch_dtype=dtype)
    base = _require_dir(folder / "base", arm.id)
    pipe = pipeline_cls.from_pretrained(str(base), torch_dtype=dtype)
    overlay_root = weights if weights.is_dir() else folder
    _overlay(pipe, overlay_root)
    return pipe


def _control_file(arm: Arm, folder: Path) -> Path:
    weights = folder / "weights"
    files = sorted(weights.rglob("*.safetensors")) if weights.is_dir() else []
    if not files:
        files = sorted(path for path in folder.rglob("*.safetensors") if "base" not in path.parts)
    if not files:
        raise CompareSetup(f"arm {arm.id} has no controlnet file", code="compare_weights")
    chosen = files[0]
    if arm.quant != "fp8" and arm.compute != "bf16-dequant":
        return chosen
    from voxprint_forge.diffusion.weights import materialise_bf16

    dest = folder / "dequant-bf16.safetensors"
    return materialise_bf16([chosen], dest)


def _overlay(pipe: object, folder: Path) -> None:
    """Merge every shard of a module, then apply that module once.

    A Hugging Face ``*.safetensors.index.json`` names the files when it is
    present. Coverage is checked on the merged module, not on each shard.
    """
    files = [path for path in sorted(folder.rglob("*.safetensors")) if "base" not in path.parts]
    if not files:
        raise CompareSetup("mirror folder has no safetensors to overlay", code="compare_weights")
    loaded = {path: _load_state(path) for path in files}
    for label, states in _shard_groups(pipe, folder, loaded):
        merged: dict[str, object] = {}
        for state in states:
            merged.update(state)
        module = _best_module(pipe, merged)
        if module is None:
            raise CompareSetup(f"{label} matched no pipeline module", code="compare_weights")
        _apply(module, merged, label)


def _shard_groups(
    pipe: object,
    folder: Path,
    loaded: dict[Path, dict[str, object]],
) -> list[tuple[str, list[dict[str, object]]]]:
    claimed: set[Path] = set()
    groups: list[tuple[str, list[dict[str, object]], object | None]] = []
    for index in _index_files(folder):
        members = _index_members(index, loaded, claimed)
        if not members:
            continue
        states = [loaded[path] for path in members]
        merged: dict[str, object] = {}
        for state in states:
            merged.update(state)
        groups.append((index.name, states, _best_module(pipe, merged)))
    buckets: dict[str, tuple[object, list[dict[str, object]]]] = {}
    for path, state in loaded.items():
        if path in claimed:
            continue
        module = _best_module(pipe, state)
        if module is None:
            raise CompareSetup(f"{path.name} matched no pipeline module", code="compare_weights")
        attached = False
        for _label, states, grouped in groups:
            if grouped is module:
                states.append(state)
                attached = True
                break
        if attached:
            continue
        name = _module_attr(pipe, module) or path.name
        bucket = buckets.get(name)
        if bucket is None:
            buckets[name] = (module, [state])
        else:
            bucket[1].append(state)
    groups.extend((name, states, module) for name, (module, states) in buckets.items())
    return [(label, states) for label, states, _module in groups]


def _index_files(folder: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder.rglob("*.safetensors.index.json"))
        if "base" not in path.parts
    ]


def _index_members(index: Path, loaded: dict[Path, dict[str, object]], claimed: set[Path]) -> list[Path]:
    members: list[Path] = []
    for filename in dict.fromkeys(_weight_map(index).values()):
        match = _match_shard(index, filename, loaded)
        if match is None or match in claimed:
            continue
        claimed.add(match)
        members.append(match)
    return members


def _weight_map(index: Path) -> dict[str, str]:
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompareSetup(f"{index.name} is not JSON", code="compare_weights") from exc
    raw = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {}
    mapping: dict[str, str] = {}
    for key, filename in raw.items():
        if isinstance(key, str) and isinstance(filename, str):
            mapping[key] = filename
    return mapping


def _match_shard(index: Path, filename: str, loaded: dict[Path, dict[str, object]]) -> Path | None:
    wanted = (index.parent / filename).resolve()
    named = [path for path in loaded if path.name == Path(filename).name]
    for path in loaded:
        if path.resolve() == wanted:
            return path
    if len(named) == 1:
        return named[0]
    return None


def _module_attr(pipe: object, module: object) -> str | None:
    for name in ("transformer", "text_encoder", "vae", "controlnet"):
        if getattr(pipe, name, None) is module:
            return name
    return None


def _free_pipe(pipe: object | None) -> None:
    if pipe is None:
        return
    for name in ("transformer", "text_encoder", "vae", "controlnet", "tokenizer", "scheduler"):
        if hasattr(pipe, name):
            setattr(pipe, name, None)
    del pipe
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_state(path: Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file

    from voxprint_forge.diffusion.fp8 import load_dequant_bf16, read_format

    if read_format(path) == "voxprint-fp8-v1":
        return load_dequant_bf16(path)
    tensors = load_file(str(path))
    if any(getattr(value, "dtype", None) == torch.float8_e4m3fn for value in tensors.values()):
        return load_dequant_bf16(path)
    return tensors


def _best_module(pipe: object, state: dict[str, object]) -> object | None:
    best: object | None = None
    hits = 0
    for name in ("transformer", "text_encoder", "vae", "controlnet"):
        module = getattr(pipe, name, None)
        if module is None or not hasattr(module, "state_dict"):
            continue
        current = module.state_dict()
        matched = sum(1 for key in state if _alias(key, current) is not None)
        if matched > hits:
            hits = matched
            best = module
    if hits < 1:
        return None
    return best


def _apply(module: object, state: dict[str, object], label: str) -> None:
    import torch

    from voxprint_forge.diffusion.alias import alias_key

    current = module.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or raw_key.endswith(".scale_weight"):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        key = alias_key(raw_key, current)
        if key is None or key not in current:
            continue
        target = current[key]
        tensor = value.detach()
        if tensor.shape != target.shape:
            continue
        if tensor.is_floating_point() and tensor.dtype != target.dtype:
            tensor = tensor.to(dtype=target.dtype)
        mapped[key] = tensor
    weight_keys = [key for key in current if key.endswith(".weight")]
    if len(mapped) < max(1, len(weight_keys) // 2):
        raise CompareSetup(f"{label} matched {len(mapped)} tensors", code="compare_weights")
    module.load_state_dict(mapped, strict=False)


def _alias(key: str, current: dict[str, object]) -> str | None:
    from voxprint_forge.diffusion.alias import alias_key

    if key.endswith(".scale_weight"):
        return None
    return alias_key(key, current)


def _place(pipe: object, arm: Arm) -> None:
    if arm.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    setter = getattr(pipe, "set_progress_bar_config", None)
    if setter is not None:
        setter(disable=True)


def _require_dir(path: Path, arm_id: str) -> Path:
    if not path.is_dir():
        raise CompareSetup(f"arm {arm_id} is missing a base snapshot", code="compare_weights")
    return path


def _snapshot(download: object, repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {"repo_id": repo, "local_dir": str(dest)}
    if revision:
        kwargs["revision"] = revision
    if patterns:
        kwargs["allow_patterns"] = patterns
    try:
        download(**kwargs)
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        raise CompareSetup(f"download failed for {repo}: {message}", code="compare_download") from exc


def _unexpected_kind(kind: Never) -> Never:
    raise CompareSetup(f"unhandled compare kind {kind}", code="compare_plan")
