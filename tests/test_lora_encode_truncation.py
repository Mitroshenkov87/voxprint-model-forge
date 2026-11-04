"""Token caps apply at encode time. Cache keys include the base.

The stored caption stays the full string. Z-Image and MOSS declare their own
caps and do not share a truncated embedding.
"""

from __future__ import annotations

from PIL import Image

from tests.lora_fakes import TinyAdapter, gradient, make_project
from voxprint_forge.lora import bases
from voxprint_forge.lora.bases import BaseInfo, MossImageAdapter, ZImageAdapter
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.project import Item


class _Words:
    """One token per whitespace-separated word."""

    def encode(self, text: str):
        return text.split()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(part) for part in ids)


class _CapMoss(TinyAdapter):
    info = BaseInfo("moss-image-base", "example/moss", "0" * 40, "Apache-2.0", "MOSS stand-in")
    max_tokens = 6


class _CapZ(TinyAdapter):
    info = BaseInfo("z-image", "example/z", "0" * 40, "Apache-2.0", "Z stand-in")
    max_tokens = 3


def test_real_adapters_declare_their_token_caps() -> None:
    assert ZImageAdapter.max_tokens == 512
    assert MossImageAdapter.max_tokens == 4096


def test_hiding_z_image_leaves_moss(monkeypatch) -> None:
    monkeypatch.delenv("VOXPRINT_FORGE_MOSS_LORA", raising=False)
    monkeypatch.delenv("VOXPRINT_FORGE_ZIMAGE_LORA", raising=False)
    monkeypatch.setenv("VOXPRINT_FORGE_ZIMAGE_LORA", "0")
    assert bases.hidden() == frozenset({"z-image"})
    assert [info.mdl_id for info in bases.trainable()] == ["moss-image-base"]
    assert bases.get_adapter("z-image").info.mdl_id == "z-image"


def test_encode_truncates_per_base_and_cache_keys_do_not_collide(tmp_path) -> None:
    saved = {mdl: bases._REGISTRY[mdl] for mdl in ("moss-image-base", "z-image")}
    bases.register(_CapMoss)
    bases.register(_CapZ)
    try:
        project = make_project(tmp_path, count=1)
        item = project.items()[0]
        long = " ".join(f"word{index}" for index in range(12))
        project.write_items([Item(item.id, item.file, long, "short one", 0)])
        image = Image.open(project.resolve(item.file))
        config = build_config("quick", "balanced", 24, steps=1, resolution=64, previews=False)
        captions = [{"long": long, "short": "short one"}]
        moss = bases.get_adapter("moss-image-base")
        zed = bases.get_adapter("z-image")
        moss.tokenizer = _Words()
        zed.tokenizer = _Words()
        moss.encode([image], captions, config)
        zed.encode([image], captions, config)
    finally:
        bases._REGISTRY.update(saved)

    def bodies(adapter, kind: str) -> set[str]:
        prefix = f"{adapter.info.mdl_id}\x1f{kind}\x1f"
        return {key[len(prefix):] for key in adapter._embed_cache if key.startswith(prefix)}

    assert bodies(moss, "embedding") >= {" ".join(f"word{index}" for index in range(6)), "short one"}
    assert bodies(zed, "embedding") >= {" ".join(f"word{index}" for index in range(3)), "short one"}
    assert long not in bodies(moss, "embedding") and long not in bodies(zed, "embedding")
    assert bodies(moss, "latent") == {"0"} and bodies(zed, "latent") == {"0"}
    assert set(moss._embed_cache).isdisjoint(zed._embed_cache)
    assert project.items()[0].long == long
    # The stand-in above is registered under the real ids only inside this test.
    # A missing tokenizer on the plain TinyAdapter leaves the caption whole.
    plain = TinyAdapter()
    plain.info = BaseInfo("tiny-test", "example/tiny", "0" * 40, "Apache-2.0", "Tiny")
    sample = plain.encode([gradient(1)], [{"long": long}], config)[0]
    assert "long" in sample.captions
