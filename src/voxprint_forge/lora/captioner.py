"""Captioners: Qwen3-VL-4B-Instruct (default) and Florence-2-large (fallback), via MDL.

The captioner is loaded, captions every photo, and is unloaded before training;
it never stays in memory next to the base model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

DEFAULT = "qwen3-vl-4b-instruct"
FALLBACK = "florence2-large"
LICENCES = {DEFAULT: "Apache-2.0", FALLBACK: "MIT"}
LONG = "Describe this photo for training an image model in two or three plain sentences. Do not guess names."
SHORT = "Describe this photo in one short phrase."


class Captioner(Protocol):
    """Writes a long and a short description for each photo."""

    name: str

    def caption(self, image: Any) -> dict[str, str]:
        """``{"long": ..., "short": ...}``."""
        ...

    def unload(self) -> None:
        """Free memory."""
        ...


class QwenCaptioner:
    """Qwen3-VL-4B-Instruct (Apache-2.0)."""

    name = DEFAULT

    def __init__(self, folder: Path, device: str = "cuda") -> None:
        from voxprint_forge.lora.backends import VlmJudge

        self.vlm = VlmJudge(folder, device)

    def caption(self, image: Any) -> dict[str, str]:
        """Two descriptions."""
        system = "You write factual image descriptions."
        return {
            "long": self.vlm.ask(image, system, LONG, 160).strip(),
            "short": self.vlm.ask(image, system, SHORT, 32).strip(),
        }

    def unload(self) -> None:
        """Free graphics memory."""
        self.vlm.unload()


class FlorenceCaptioner:
    """Florence-2-large (MIT)."""

    name = FALLBACK

    def __init__(self, folder: Path, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(folder), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(folder), trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)

    def _run(self, image: Any, task: str) -> str:
        import torch

        inputs = self.processor(text=task, images=image.convert("RGB"), return_tensors="pt").to(
            self.model.device, self.model.dtype
        )
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=160, num_beams=3
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.replace(task, "").strip()

    def caption(self, image: Any) -> dict[str, str]:
        """Two descriptions."""
        return {"long": self._run(image, "<MORE_DETAILED_CAPTION>"), "short": self._run(image, "<CAPTION>")}

    def unload(self) -> None:
        """Free graphics memory."""
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_LOADERS: dict[str, Callable[[Path], Captioner]] = {DEFAULT: QwenCaptioner, FALLBACK: FlorenceCaptioner}


def load_captioner(choice: str = DEFAULT) -> Captioner | None:
    """The chosen captioner, else the other one, else ``None`` (``lora.err.captioner_missing``)."""
    from voxprint_forge.lora.backends import model_dir

    order = [choice] + [name for name in (DEFAULT, FALLBACK) if name != choice]
    for name in order:
        folder = model_dir(name)
        if not folder.is_dir():
            continue
        try:
            return _LOADERS[name](folder)
        except Exception:  # noqa: BLE001 - try the fallback
            continue
    return None
