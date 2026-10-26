"""Captioners: Qwen3-VL-4B-Instruct (default) and Florence-2-large (fallback), via MDL.

The captioner is loaded, captions every photo, and is unloaded before training;
it never stays in memory next to the base model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from voxprint_forge.lora.backends import VlmJudge, installed_dir
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.presets import detect_vram_gb
from voxprint_forge.lora.vram_stage import TIER_GB, require_quantized

DEFAULT = "qwen3-vl-4b-instruct"
FALLBACK = "florence2-large"
LICENCES = {DEFAULT: "Apache-2.0", FALLBACK: "MIT"}
LONG = "Describe this photo for training an image model in two or three plain sentences. Do not guess names."
SHORT = "Describe this photo in one short phrase."
_UNCLOTHED_ASK = "Answer yes or no: is the subject unclothed?"
_UNCLOTHED_WORD = re.compile(r"\b(?:nude|naked|unclothed|undressed)\b", re.IGNORECASE)


def text_marks_unclothed(*parts: str) -> bool:
    """True when caption text contains an unclothed word."""
    return any(_UNCLOTHED_WORD.search(part) for part in parts)


def answer_yes(text: str) -> bool:
    """True when a yes/no answer starts with yes."""
    return text.strip().lower().startswith("yes")


class Captioner(Protocol):
    """Writes a long and a short description for each photo."""

    name: str

    def caption(self, image: Any) -> dict[str, str | bool]:
        """``{"long": ..., "short": ..., "unclothed": bool}``."""
        ...

    def unload(self) -> None:
        """Free memory."""
        ...


class QwenCaptioner:
    """Qwen3-VL-4B-Instruct (Apache-2.0)."""

    name = DEFAULT

    def __init__(self, folder: Path, device: str = "cuda") -> None:
        self.vlm = VlmJudge(folder, device)

    def caption(self, image: Any) -> dict[str, str | bool]:
        """Two descriptions and whether the subject is unclothed."""
        system = "You write factual image descriptions."
        return {
            "long": self.vlm.ask(image, system, LONG, 160).strip(),
            "short": self.vlm.ask(image, system, SHORT, 32).strip(),
            "unclothed": answer_yes(self.vlm.ask(image, system, _UNCLOTHED_ASK, 8)),
        }

    def unload(self) -> None:
        """Free graphics memory."""
        self.vlm.unload()


class FlorenceCaptioner:
    """Florence-2-large (MIT)."""

    name = FALLBACK

    def __init__(self, folder: Path, device: str = "cuda") -> None:
        if detect_vram_gb() == TIER_GB:
            require_quantized(TIER_GB, False, FALLBACK)
        # Imported here so ``voxprint-forge --version`` does not pull torch or transformers.
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(folder), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(folder), trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)

    def _run(self, image: Any, task: str) -> str:
        # Imported here so ``voxprint-forge --version`` does not pull torch.
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

    def caption(self, image: Any) -> dict[str, str | bool]:
        """Two descriptions and whether the subject is unclothed."""
        long = self._run(image, "<MORE_DETAILED_CAPTION>")
        short = self._run(image, "<CAPTION>")
        return {"long": long, "short": short, "unclothed": text_marks_unclothed(long, short)}

    def unload(self) -> None:
        """Free graphics memory."""
        # Imported here so ``voxprint-forge --version`` does not pull torch.
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_LOADERS: dict[str, Callable[[Path], Captioner]] = {DEFAULT: QwenCaptioner, FALLBACK: FlorenceCaptioner}


def load_captioner(choice: str = DEFAULT) -> Captioner | None:
    """The chosen captioner, else the other one, else ``None`` (``lora.err.captioner_missing``).

    A 16 GB card does not substitute Florence for Qwen, and a stage error from the
    quantizer is not swallowed.
    """
    tight = detect_vram_gb() == TIER_GB
    order = [choice]
    if not tight:
        order += [name for name in (DEFAULT, FALLBACK) if name != choice]
    seen: set[str] = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        folder = installed_dir(name)
        if folder is None:
            continue
        try:
            return _LOADERS[name](folder)
        except LoraStageError:
            raise
        except Exception:
            if tight:
                raise
            continue
    return None
