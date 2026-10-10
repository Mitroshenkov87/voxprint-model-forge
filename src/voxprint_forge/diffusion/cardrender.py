"""Render templates/canvas_model_card.md.jinja. The LLM card template is not used."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from voxprint_forge.errors import ToolMissingError

MODIFICATION_NOTICE = "Modified by Voxprint: dtype conversion."
METHOD = (
    "Dtype conversion only. Linear weights are stored as float8_e4m3fn with a "
    "per-tensor BF16 scale (amax/448) under the safetensors metadata format "
    "voxprint-fp8-v1. Embeddings, norms, and the layers named in the job stay "
    "BF16. This is not an ablation and it is not a fine-tune. Quality checks "
    "dequantize FP8 back to BF16, because the A100 has no FP8 tensor cores."
)
CALIBRATION = "None. No calibration set and no training."
LIMITS = (
    "The FP8 files are a storage format. On an A100 they are dequantized to "
    "BF16 before the matmul. A component that misses its LPIPS or CLIP gate "
    "is published as BF16. This card is not a safety filter. TensorRT engines "
    "are not part of this job."
)


@dataclass(frozen=True)
class CardContext:
    """Fields the canvas model card template renders."""
    title: str
    base_models: tuple[str, ...]
    base_lines: tuple[str, ...]
    pipeline_tag: str
    tags: tuple[str, ...]
    language: tuple[str, ...]
    quant_lines: tuple[str, ...]
    eval_lines: tuple[str, ...]
    credits: tuple[str, ...]
    stamped: str


def template_path(repo: Path) -> Path:
    """Path of the canvas model card template in the repo."""
    path = repo / "templates" / "canvas_model_card.md.jinja"
    if not path.is_file():
        raise ToolMissingError(f"missing {path}", code="template_missing")
    return path


def licence_text(repo: Path) -> str:
    """One-line Apache-2.0 pointer. The card links the licence instead of pasting it.

    ``repo`` is the checkout the card is rendered from. The line does not
    read ``LICENSE`` from that tree.
    """
    del repo
    return (
        "Apache-2.0, same as the base model. "
        "See https://www.apache.org/licenses/LICENSE-2.0"
    )


def render_canvas_card(repo: Path, card: CardContext) -> str:
    """Render a canvas model card from ``card``."""
    env = Environment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(template_path(repo).read_text(encoding="utf-8"))
    return template.render(
        title=card.title,
        base_models=card.base_models,
        base_lines=card.base_lines,
        pipeline_tag=card.pipeline_tag,
        tags=card.tags,
        language=card.language,
        method=METHOD,
        calibration=CALIBRATION,
        quant_lines=card.quant_lines,
        eval_lines=card.eval_lines,
        limits=LIMITS,
        credits=card.credits,
        modification_notice=MODIFICATION_NOTICE,
        apache_text=licence_text(repo),
        stamped=card.stamped,
    )


def today_stamp() -> str:
    """Today's date as an ISO stamp."""
    return date.today().isoformat()
