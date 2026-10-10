"""Render the model card from templates/model_card.md.jinja."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Never

from jinja2 import Environment, StrictUndefined

from voxprint_forge.errors import StageError, ToolMissingError
from voxprint_forge.plan import ModelPlan

MODIFICATION_NOTICE = (
    "Modified from the base model named above by Voxprint AI Model Forge. "
    "Changes are requantization and, where the plan says so, an in-house "
    "difference-of-means refusal-direction ablation. Upstream weights keep "
    "the licence recorded in this card. Credits for the base model are listed above."
)

LIMITS = (
    "Refusals are reduced. This is not a production safety filter and it is "
    "not for safety-critical use. Russian prose is a short sample, not a "
    "fluency benchmark; read it before shipping a tier. Quality moves with "
    "the quant and, on ablated models, with the direction removed."
)


def template_path(repo: Path) -> Path:
    """Path of ``templates/model_card.md.jinja`` in the repo."""
    path = repo / "templates" / "model_card.md.jinja"
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


def _unexpected_mode(mode: Never) -> Never:
    raise StageError(f"unhandled mode {mode}", code="card_mode")


def card_base_model(model: ModelPlan) -> str:
    """Hub id this GGUF was quantized from.

    Requants use the abliterated upstream. Ablations use the source repo
    (the T-Bank repo for T-lite and T-pro).
    """
    mode = model.mode
    if mode == "requant":
        return model.repo
    if mode == "ablate":
        return model.repo
    _unexpected_mode(mode)


def calibration_note(model: ModelPlan) -> str:
    """Card paragraph for requant or ablation calibration data."""
    mode = model.mode
    if mode == "requant":
        return (
            "This requant does not repeat the upstream ablation and does not include "
            "that author's calibration set. The importance matrix and the eval prompts "
            "are original text in src/voxprint_forge/data, under the Apache-2.0 licence "
            "recorded in src/voxprint_forge/data/LICENCES.md."
        )
    if mode == "ablate":
        return (
            "The refusal direction is calibrated on original prompts in "
            "src/voxprint_forge/data/harmful.txt and harmless.txt. Those files, the "
            "importance-matrix text, and the eval prompts are original Apache-2.0 text. "
            "See src/voxprint_forge/data/LICENCES.md. No third-party corpus is included."
        )
    _unexpected_mode(mode)


def what_changed(model: ModelPlan, ablation: dict[str, object] | None) -> str:
    """Card paragraph describing the requant or the refusal-direction ablation."""
    if model.mode == "requant" or model.already_abliterated:
        return (
            f"Requant of upstream abliterated {model.repo} at revision {model.revision}. "
            "Forge does not repeat that ablation. The 24 GB tier reuses a pinned "
            "upstream GGUF when one fits; the 16 GB tier is quantized here."
        )
    if model.mode == "ablate":
        layer = "the layer with the largest gap" if not ablation else f"layer {ablation.get('layer')}"
        mode = "" if not ablation else f" Compute mode: {ablation.get('mode')}."
        return (
            "Own difference-of-means refusal-direction ablation. The direction is the "
            "normalised gap between the mean last-token residual on harmful prompts and "
            f"the mean on harmless prompts, taken at {layer}. Writing matrices whose "
            "output size matches the hidden size (o_proj, out_proj, down_proj, wo, "
            "dense_4h_to_h) are orthogonalised: W := W - r r^T W. "
            "Calibration prompts are original Apache-2.0 text shipped with Forge "
            "(src/voxprint_forge/data/LICENCES.md)."
            f"{mode}"
        )
    raise StageError(f"unhandled mode {model.mode}", code="card_mode")


def render_card(
    repo: Path,
    model: ModelPlan,
    tiers: list[dict[str, object]],
    *,
    ablation: dict[str, object] | None,
    imatrix_note: str,
    stamped: str | None = None,
) -> str:
    """Render the model card markdown for these tiers."""
    template = template_path(repo).read_text(encoding="utf-8")
    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    try:
        html = env.from_string(template)
        return html.render(
            model_id=model.id,
            title=f"Voxprint {model.id}",
            card_base=card_base_model(model),
            foundation=model.base_model,
            revision=model.revision,
            licence=model.licence,
            what_changed=what_changed(model, ablation),
            calibration=calibration_note(model),
            quant_lines=[
                (
                    f"{item.get('name')}: {item.get('quant')} "
                    f"(min_vram_gb={item.get('min_vram_gb')}, split={item.get('split', 'gpu')})"
                )
                for item in tiers
            ],
            tiers=tiers,
            imatrix_note=imatrix_note,
            limits=LIMITS,
            credits=list(model.credits),
            apache_text=licence_text(repo),
            modification_notice=MODIFICATION_NOTICE,
            stamped=stamped or date.today().isoformat(),
            params_b=model.params_b,
        )
    except Exception as exc:
        raise StageError(f"model card render failed: {exc}", code="card") from exc
