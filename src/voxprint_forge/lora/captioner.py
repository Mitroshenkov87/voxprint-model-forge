"""Captioners: Qwen3-VL-4B-Instruct (default) and Florence-2-large (fallback), via MDL.

The captioner is loaded, captions every photo, and is unloaded before training;
it never stays in memory next to the base model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from voxprint_forge.lora.backends import VlmJudge, installed_dir
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.presets import detect_vram_gb
from voxprint_forge.lora.vram_stage import TIER_GB, require_quantized

DEFAULT = "qwen3-vl-4b-instruct"
FALLBACK = "florence2-large"
LICENCES = {DEFAULT: "Apache-2.0", FALLBACK: "MIT"}
# The description prompts ask for the attire as it is. Without this the model
# fell back on its own wording and named clothing the subject was not wearing.
ATTIRE = (
    "If a person is shown, say what they are wearing exactly as it appears. If they are unclothed or partly "
    "unclothed, say so plainly with a neutral word such as unclothed, nude or topless. Do not name clothing "
    "that is not in the photo."
)
LONG = (
    "Describe this photo for training an image model in two or three plain sentences. Do not guess names. "
    + ATTIRE
)
SHORT = "Describe this photo in one short phrase. " + ATTIRE
SYSTEM = "You write factual, neutral image descriptions. Describe what is in the photo as it is."
# Added to every Qwen prompt when ``release.txt`` says ``Adult (18+): yes`` and ``No minors
# in photos: yes``. It only shapes the wording; the visual age check has already run.
ADULT_HINT = "The person shown is an adult, 18 or older; no minors are present. Describe them as an adult."
_UNCLOTHED_ASK = "Answer yes or no: is the subject unclothed?"
_UNCLOTHED_WORD = re.compile(r"\b(?:nude|naked|unclothed|undressed)\b", re.IGNORECASE)


def text_marks_unclothed(*parts: str) -> bool:
    """True when caption text contains an unclothed word."""
    return any(_UNCLOTHED_WORD.search(part) for part in parts)


def unclothed_flag(flag: object, *parts: str) -> bool | None:
    """The stored ``unclothed`` flag: true when the captioner said so or its text does.

    The text is kept as written; the flag only follows it. A missing flag with
    clothed text stays ``None``.
    """
    if text_marks_unclothed(*parts):
        return True
    return flag if isinstance(flag, bool) else None


def answer_yes(text: str) -> bool:
    """True when a yes/no answer starts with yes."""
    return text.strip().lower().startswith("yes")


def _prompts(adult_hint: bool) -> tuple[str, str, str]:
    """The Qwen system, long and short prompts, each with :data:`ADULT_HINT` when ``adult_hint``."""
    if not adult_hint:
        return SYSTEM, LONG, SHORT
    return tuple(f"{text} {ADULT_HINT}" for text in (SYSTEM, LONG, SHORT))  # type: ignore[return-value]


class Captioner(Protocol):
    """Writes a long and a short description for each photo.

    A captioner may also have ``caption_batch(images, adult_hint=False) -> list[dict]``: one
    ``caption`` result per image, in input order. It is optional; ``lora prepare`` uses it when
    present and falls back to ``caption`` per image when it is missing, raises or miscounts.
    """

    name: str

    def caption(self, image: Any, adult_hint: bool = False) -> dict[str, str | bool]:
        """``{"long": ..., "short": ..., "unclothed": bool}``.

        ``adult_hint`` tells the model the subject is an adult and no minors are present.
        """
        ...

    def unload(self) -> None:
        """Free memory."""
        ...


class QwenCaptioner:
    """Qwen3-VL-4B-Instruct (Apache-2.0)."""

    name = DEFAULT

    def __init__(self, folder: Path, device: str = "cuda") -> None:
        self.vlm = VlmJudge(folder, device)

    def caption(self, image: Any, adult_hint: bool = False) -> dict[str, str | bool]:
        """Two descriptions and whether the subject is unclothed."""
        system, long_ask, short_ask = _prompts(adult_hint)
        long = self.vlm.ask(image, system, long_ask, 160).strip()
        short = self.vlm.ask(image, system, short_ask, 48).strip()
        asked = answer_yes(self.vlm.ask(image, system, _UNCLOTHED_ASK, 8))
        return {"long": long, "short": short, "unclothed": unclothed_flag(asked, long, short)}

    def caption_batch(self, images: Sequence[Any], adult_hint: bool = False) -> list[dict[str, str | bool]]:
        """:meth:`caption` for several images: the same three asks, each one batched call over ``images``."""
        images = list(images)
        system, long_ask, short_ask = _prompts(adult_hint)
        longs = [text.strip() for text in self.vlm.ask_batch(images, system, long_ask, 160)]
        shorts = [text.strip() for text in self.vlm.ask_batch(images, system, short_ask, 48)]
        asked = [answer_yes(text) for text in self.vlm.ask_batch(images, system, _UNCLOTHED_ASK, 8)]
        if not len(longs) == len(shorts) == len(asked) == len(images):
            raise RuntimeError(f"batch caption gave {len(longs)}/{len(shorts)}/{len(asked)} answers for {len(images)}")
        return [
            {"long": long, "short": short, "unclothed": unclothed_flag(flag, long, short)}
            for long, short, flag in zip(longs, shorts, asked, strict=True)
        ]

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

    def caption(self, image: Any, adult_hint: bool = False) -> dict[str, str | bool]:
        """Two descriptions and whether the subject is unclothed. Florence takes no prompt; ``adult_hint`` is unused."""
        del adult_hint
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
