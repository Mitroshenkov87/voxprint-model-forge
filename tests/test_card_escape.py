"""Model cards escape upstream text and leave Forge's own paragraphs raw."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from voxprint_forge.cards import render_card
from voxprint_forge.diffusion.cardrender import CardContext, render_canvas_card
from voxprint_forge.plan import load_plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = '<script>alert("xss")</script>'
ESCAPED = "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;"

TIERS = [
    {
        "name": "24gb",
        "quant": "Q5_K_M",
        "min_vram_gb": 24,
        "budget_gb": 24,
        "perplexity": 5.5,
        "refusal_en": 0.0,
        "refusal_ru": 0.0,
        "measurements": [
            {"ctx": 8192, "weights_bytes": 1000, "kv_bytes": 10, "fits": True, "nvidia_used_bytes": 0}
        ],
        "prose": ["A short original paragraph."],
    }
]


def test_script_in_a_model_name_is_escaped() -> None:
    plan = load_plan(ROOT / "plans" / "plan.toml")
    model = next(item for item in plan.models if item.id == "huihui-qwen36-27b")
    poisoned = replace(model, id=SCRIPT, repo=f"org/{SCRIPT}", base_model=SCRIPT)
    text = render_card(
        ROOT,
        poisoned,
        TIERS,
        ablation=None,
        imatrix_note="imatrix.gguf sha256 abc",
        stamped="2026-10-10",
    )
    assert SCRIPT not in text
    assert ESCAPED in text
    assert "author's calibration set" in text


def test_script_in_a_canvas_title_is_escaped() -> None:
    text = render_canvas_card(
        ROOT,
        CardContext(
            title=SCRIPT,
            base_models=(SCRIPT,),
            base_lines=(f"Quantized from: `{SCRIPT}`",),
            pipeline_tag="text-to-image",
            tags=("voxprint",),
            language=("en",),
            quant_lines=(SCRIPT,),
            eval_lines=(SCRIPT,),
            credits=(SCRIPT,),
            stamped="2026-10-10",
        ),
    )
    assert SCRIPT not in text
    assert text.count(ESCAPED) >= 5
    assert "Dtype conversion only." in text
