"""Rendered model cards carry Hub front matter."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import ModelCard

from voxprint_forge.cards import render_card
from voxprint_forge.plan import load_plan

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BASE = {
    "huihui-qwen36-27b": "huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF",
    "gemma4-31b": "paperscarecrow/Gemma-4-31B-it-abliterated",
    "t-lite-it-2.1": "t-tech/T-lite-it-2.1",
    "t-pro-it-2.1": "t-tech/T-pro-it-2.1",
}

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
            {
                "ctx": 8192,
                "weights_bytes": 1000,
                "kv_bytes": 10,
                "fits": True,
                "nvidia_used_bytes": 0,
            }
        ],
        "prose": ["A short original paragraph."],
    },
    {
        "name": "16gb",
        "quant": "IQ4_XS",
        "min_vram_gb": 16,
        "budget_gb": 16,
        "perplexity": 6.1,
        "refusal_en": 0.0,
        "refusal_ru": 0.25,
        "measurements": [
            {
                "ctx": 8192,
                "weights_bytes": 800,
                "kv_bytes": 10,
                "fits": True,
                "nvidia_used_bytes": 0,
            }
        ],
        "prose": ["Ещё одна короткая фраза."],
    },
]


def test_model_card_front_matter_matches_hub_card() -> None:
    plan = load_plan(ROOT / "plans" / "plan.toml")
    for model_id, expected in EXPECTED_BASE.items():
        model = next(item for item in plan.models if item.id == model_id)
        ablation = {"layer": 3, "mode": "resident"} if model.mode == "ablate" else None
        text = render_card(
            ROOT,
            model,
            TIERS,
            ablation=ablation,
            imatrix_note="imatrix.gguf sha256 abc",
            stamped="2026-10-10",
        )
        assert text.startswith("---\n")
        body = text.split("---", 2)[2].lstrip()
        assert body.startswith(f"# VOXPrint {model_id}\n")
        card = ModelCard(text)
        data = card.data.to_dict()
        assert card.data.license == "apache-2.0"
        assert card.data.base_model == expected
        assert data["base_model_relation"] == "quantized"
        assert card.data.pipeline_tag == "text-generation"
        assert list(card.data.language) == ["ru", "en"]
        assert list(card.data.tags) == [
            "voxprint",
            "gguf",
            "abliterated",
            "llama.cpp",
            "imatrix",
            "russian",
        ]
        front = text.split("---", 2)[1]
        assert "finetune" not in front
        assert "This repository is private" not in text
        assert "## License" in text
        assert "Apache-2.0, same as the base model. See https://www.apache.org/licenses/LICENSE-2.0" in text
        licence = text.split("## License", 1)[1].split("\n## ", 1)[0]
        assert "```" not in licence
        assert "## Credits" in text
        assert "## Modification notice" in text
        assert "Modified from the base model named above" in text
        assert "## Made for Voxprint AI Media Suite" in text
        assert "https://github.com/Mitroshenkov87/voxprint-plotweaver" in text
        assert "min_vram_gb" in text
        assert expected in text
        made = text.split("## Made for Voxprint AI Media Suite", 1)[1]
        assert "## Base model" not in made
