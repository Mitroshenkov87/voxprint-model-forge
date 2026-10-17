"""Expert values and the preset document for the LoRA trainer (plan §2, §3).

Setting ids are the plan's settings keys (``lora.*``). Each Expert field is
tagged with the wizard step that shows it, so the shared Expert panel lists
only that step's values. Presets come from the engine
(:mod:`voxprint_forge.lora.presets`) so the GUI and the CLI never disagree.
Safety and age values are not settings: ``SettingSpec`` refuses such keys.
"""

from __future__ import annotations

from voxprint_common.ui.presets import PresetFile, parse_presets
from voxprint_common.ui.settings import Choice, Settings, SettingSpec, SettingType

from voxprint_forge.lora import captioner

SECTION = "train"
WIZARD = "lora"
PRESET_GROUP = "lora.preset"
# Settings key -> TrainConfig field.
CONFIG_FIELD = {
    "lora.rank": "rank",
    "lora.alpha": "alpha",
    "lora.steps": "steps",
    "lora.lr": "lr",
    "lora.scheduler": "scheduler",
    "lora.warmup": "warmup",
    "lora.batch": "batch",
    "lora.grad_accum": "grad_accum",
    "lora.resolution": "resolution",
    "lora.precision": "precision",
    "lora.grad_checkpointing": "grad_checkpointing",
    "lora.train_text_encoder": "train_text_encoder",
    "lora.caption_mode": "caption_mode",
    "lora.caption_dropout": "caption_dropout",
    "lora.seed": "seed",
    "lora.save_every": "save_every",
    "lora.preview_every": "preview_every",
    "lora.mix_old": "replay",
}
CAPTION_MODELS = (captioner.DEFAULT, captioner.FALLBACK)
# The v0.2.5 Expert panel shows a FLOAT with one decimal, so small numbers are lists.
FLOAT_CHOICES = {
    "lora.lr": (1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 2e-4, 3e-4, 5e-4),
    "lora.caption_dropout": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
    "lora.mix_old": (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
}


def number_text(value: float) -> str:
    """The stored text of a listed number: ``1e-4`` -> ``"0.0001"``."""
    return f"{float(value):g}"


def _numbers(key: str, extra: float) -> tuple[Choice, ...]:
    values = sorted({number_text(item) for item in (*FLOAT_CHOICES[key], extra)}, key=float)
    return tuple(Choice(value, f"forge.app.number.{value}") for value in values)


def _spec(key: str, kind: SettingType, default: object, step: str, **extra: object) -> SettingSpec:
    name = key.split(".", 1)[1]
    tip = extra.pop("tip", f"lora.expert.{name}.tip")
    return SettingSpec(
        key,
        f"forge.app.expert.{name}",
        str(tip),
        kind,
        default,  # type: ignore[arg-type]
        expert=True,
        group="lora",
        section=SECTION,
        step=step,
        **extra,  # type: ignore[arg-type]
    )


def _choices(name: str, values: tuple[str, ...]) -> tuple[Choice, ...]:
    return tuple(Choice(value, f"forge.app.expert.{name}.{value}") for value in values)


def lora_settings() -> Settings:
    """Every Expert value of the trainer, with the Balanced / 24 GB defaults."""
    from voxprint_forge.lora.presets import CAPTION_MODES, PRECISIONS, SCHEDULERS, build_config

    base = build_config("balanced", "balanced", 24)
    integer = SettingType.INT
    specs = [
        _spec("lora.caption_model", SettingType.ENUM, CAPTION_MODELS[0], "prepare", choices=_choices(
            "caption_model", CAPTION_MODELS)),
        _spec("lora.rank", integer, base.rank, "train", minimum=1, maximum=256),
        _spec("lora.alpha", integer, base.alpha, "train", minimum=1, maximum=256),
        _spec("lora.steps", integer, base.steps, "train", minimum=1, maximum=20000),
        _spec("lora.lr", SettingType.ENUM, number_text(base.lr), "train", choices=_numbers("lora.lr", base.lr)),
        _spec("lora.scheduler", SettingType.ENUM, base.scheduler, "train", choices=_choices("scheduler", SCHEDULERS)),
        _spec("lora.warmup", integer, base.warmup, "train", minimum=0, maximum=5000,
              tip="forge.app.expert.warmup.tip"),
        _spec("lora.batch", integer, base.batch, "train", minimum=1, maximum=16),
        _spec("lora.grad_accum", integer, base.grad_accum, "train", minimum=1, maximum=16,
              tip="forge.app.expert.grad_accum.tip"),
        _spec("lora.resolution", integer, base.resolution, "train", minimum=256, maximum=2048),
        _spec("lora.precision", SettingType.ENUM, base.precision, "train", choices=tuple(
            Choice(value, f"forge.app.expert.precision.{value}") for value in PRECISIONS)),
        _spec("lora.grad_checkpointing", SettingType.BOOL, base.grad_checkpointing, "train",
              tip="lora.expert.checkpointing.tip"),
        _spec("lora.train_text_encoder", SettingType.BOOL, base.train_text_encoder, "train",
              tip="forge.app.expert.train_text_encoder.tip"),
        _spec("lora.caption_mode", SettingType.ENUM, base.caption_mode, "train", choices=_choices(
            "caption_mode", CAPTION_MODES), tip="lora.expert.caption.tip"),
        _spec("lora.caption_dropout", SettingType.ENUM, number_text(base.caption_dropout), "train",
              choices=_numbers("lora.caption_dropout", base.caption_dropout)),
        _spec("lora.seed", integer, base.seed, "train", minimum=0, maximum=2**31 - 1),
        _spec("lora.save_every", integer, base.save_every, "train", minimum=1, maximum=20000),
        _spec("lora.preview_every", integer, base.preview_every, "train", minimum=1, maximum=20000),
        _spec("lora.mix_old", SettingType.ENUM, number_text(base.replay), "train",
              choices=_numbers("lora.mix_old", base.replay)),
    ]
    return Settings(specs)


def config_values(preset: str, likeness: str, vram: str) -> dict[str, object]:
    """The live Expert values a Simple choice writes (engine ``build_config``)."""
    from voxprint_forge.lora.presets import build_config

    config = build_config(preset, likeness, vram if vram == "auto" else int(vram)).to_dict()
    values = {key: config[field] for key, field in CONFIG_FIELD.items() if field in config}
    for key in FLOAT_CHOICES:
        if key in values:
            values[key] = number_text(values[key])  # type: ignore[arg-type]
    return values


def preset_file(settings: Settings) -> PresetFile:
    """The ``voxprint-presets/1`` document: Quick test, Balanced, Best quality."""
    from voxprint_forge.lora.presets import PRESETS

    options = []
    for preset in PRESETS:
        values = config_values(preset, "balanced", "24")
        options.append({"id": preset, "label_key": f"lora.preset.{preset}", "values": values})
    document = {
        "schema": "voxprint-presets/1",
        "groups": [{"id": PRESET_GROUP, "label_key": "forge.app.preset.label", "default": "balanced",
                    "options": options}],
    }
    return parse_presets(document, settings)
