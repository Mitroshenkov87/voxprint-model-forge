"""Picture formats and opt-in metadata. Synthetic pixels only; no models and no downloads."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, features
from voxprint_common.ui.settings import SettingType

from tests.lora_fakes import gradient
from voxprint_forge.app import engine
from voxprint_forge.app.draft import Draft
from voxprint_forge.app.settings import lora_settings
from voxprint_forge.cli import build_parser
from voxprint_forge.diffusion.moss import generation_parser
from voxprint_forge.imageio_save import (
    LOSSY_QUALITY,
    ImageSaveError,
    embed_metadata_enabled,
    embed_metadata_scope,
    format_of,
    read_metadata,
    save_image,
)
from voxprint_forge.lora.train import PreviewStamp, read_preview_text, save_preview

_SECRET = "SECRETPROMPTTOKEN9F3C"
_PROMPT = 'say "hi" & bye\nnext line'


def _noise(size: int = 48, seed: int = 1) -> Image.Image:
    """An RGB picture whose pixels are not a flat color, so lossy encodes change them."""
    rnd = random.Random(seed)
    payload = bytes(rnd.randrange(256) for _ in range(size * size * 3))
    return Image.frombytes("RGB", (size, size), payload)


def _pixels(path: Path) -> bytes:
    with Image.open(path) as image:
        return image.convert("RGB").tobytes()


def _details() -> dict[str, object]:
    digest = "ab" * 32
    return {
        "prompt": _PROMPT,
        "seed": 1234,
        "steps": 20,
        "base_model": "z-image",
        "strength": 0.75,
        "sha256": digest,
        "lora": 'face "a"',
        "loras": [
            {"name": 'face "a"', "strength": 0.75, "sha256": digest, "base": "z-image"},
            {"name": "style & co", "strength": 0.4, "sha256": "cd" * 32, "base": "moss-image-base"},
        ],
    }


def test_default_is_webp_and_lossy_quality_is_90(tmp_path: Path) -> None:
    """A regular picture with no format is WebP, and the lossy quality is 90."""
    source = _noise()
    path = save_image(source, tmp_path / "shot.png", "regular")
    assert path == tmp_path / "shot.webp"
    assert path.is_file()
    assert LOSSY_QUALITY == 90
    assert _pixels(path) != source.tobytes()


@pytest.mark.parametrize("kind", ["master", "series", "handoff", "hand-off"])
def test_lossless_kinds_keep_every_pixel(tmp_path: Path, kind: str) -> None:
    """Masters, series and hand-offs are lossless WebP, including the hyphenated name."""
    source = _noise(seed=3)
    path = save_image(source, tmp_path / kind, kind)
    assert path.suffix == ".webp"
    assert _pixels(path) == source.tobytes()


def test_png_is_exact_and_jpeg_uses_the_named_suffix(tmp_path: Path) -> None:
    """PNG stays lossless. JPEG is ``.jpg``, and an existing ``.jpeg`` suffix is kept."""
    source = _noise(seed=5)
    png = save_image(source, tmp_path / "frame", "series", fmt="png")
    assert png.suffix == ".png"
    assert _pixels(png) == source.tobytes()
    jpg = save_image(source, tmp_path / "frame", "master", fmt="JPEG")
    assert jpg == tmp_path / "frame.jpg"
    assert _pixels(jpg) != source.tobytes()
    kept = save_image(source, tmp_path / "scan.jpeg", "regular", fmt="jpg")
    assert kept == tmp_path / "scan.jpeg"


def test_avif_writes_when_pillow_can_encode_it(tmp_path: Path) -> None:
    """AVIF is written when Pillow can encode it. A series is fuller than a regular picture."""
    if not features.check("avif"):
        pytest.skip("this Pillow cannot encode AVIF")
    source = _noise(seed=7)
    path = save_image(source, tmp_path / "series", "series", fmt="avif")
    regular = save_image(source, tmp_path / "regular", "regular", fmt="avif")
    assert path.suffix == ".avif"
    with Image.open(path) as image:
        assert image.size == source.size
    assert path.stat().st_size > regular.stat().st_size


def test_avif_without_support_names_the_missing_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Pillow build without AVIF gets a clear error, and nothing is written."""
    real = features.check

    def check(name: str, *args: object, **kwargs: object) -> bool:
        if name == "avif":
            return False
        return bool(real(name, *args, **kwargs))

    monkeypatch.setattr(features, "check", check)
    with pytest.raises(ImageSaveError, match="AVIF needs a Pillow build with AVIF support"):
        save_image(_noise(), tmp_path / "shot", "series", fmt="avif")
    assert not (tmp_path / "shot.avif").exists()


@pytest.mark.parametrize(("kind", "fmt"), [("thumb", None), ("regular", "gif")])
def test_unknown_kind_or_format_is_rejected(tmp_path: Path, kind: str, fmt: str | None) -> None:
    """A bad kind or format raises ImageSaveError and names the value."""
    shown = kind if fmt is None else fmt
    with pytest.raises(ImageSaveError, match=shown) as caught:
        save_image(_noise(), tmp_path / "shot", kind, fmt=fmt)
    assert isinstance(caught.value, ValueError)


def test_metadata_stays_out_until_opted_in(tmp_path: Path) -> None:
    """A metadata mapping is ignored unless the call or the current scope asks for it."""
    source = _noise(seed=9)
    source.info["exif"] = _SECRET.encode()
    path = save_image(source, tmp_path / "quiet", "regular", metadata={"prompt": _SECRET, "seed": 7})
    assert read_metadata(path)["prompt"] is None
    assert read_metadata(path)["seed"] is None
    assert _SECRET.encode() not in path.read_bytes()
    assert embed_metadata_enabled() is False

    opted = save_image(source, tmp_path / "opted", "regular", metadata=_details(), embed_metadata=True)
    _assert_details(read_metadata(opted))
    with embed_metadata_scope(True):
        assert embed_metadata_enabled() is True
        scoped = save_image(source, tmp_path / "scoped", "series", fmt="png", metadata=_details())
        with embed_metadata_scope(False):
            assert embed_metadata_enabled() is False
            inner = save_image(source, tmp_path / "inner", "regular", metadata={"prompt": _SECRET})
        assert embed_metadata_enabled() is True
        still = save_image(
            source, tmp_path / "still", "regular", metadata=_details(), embed_metadata=False,
        )
    assert embed_metadata_enabled() is False
    _assert_details(read_metadata(scoped))
    assert read_metadata(inner)["prompt"] is None
    _assert_details(read_metadata(still))


def test_metadata_round_trips_for_each_container(tmp_path: Path) -> None:
    """WebP, PNG and JPEG carry the same document. AVIF does too when Pillow can write it."""
    formats = ["webp", "png", "jpeg"]
    if features.check("avif"):
        formats.append("avif")
    for fmt in formats:
        path = save_image(
            _noise(seed=11), tmp_path / fmt, "handoff", fmt=fmt, metadata=_details(), embed_metadata=True,
        )
        _assert_details(read_metadata(path))


def test_format_of_reads_the_suffix() -> None:
    """``.jpg`` and ``.jpeg`` are JPEG. An unknown suffix names no format."""
    assert format_of("out.WEBP") == "webp"
    assert format_of(Path("out.PNG")) == "png"
    assert format_of("out.jpg") == "jpeg"
    assert format_of("out.JPEG") == "jpeg"
    assert format_of("out.avif") == "avif"
    assert format_of("out.bin") is None


def test_preview_png_keeps_voxprint_text_and_embeds_only_when_opted_in(tmp_path: Path) -> None:
    """Likeness text is always on the preview. The shared document follows the opt-in scope."""
    stamp = PreviewStamp("z-image", "ab" * 32, 500)
    meta = {"prompt": _SECRET, "seed": 1234, "base": "z-image", "strength": 1.0, "sha256": "ab" * 32}
    plain = tmp_path / "step-000500-0.png"
    save_preview(gradient(1, 32), plain, stamp.text(1234), metadata=meta)
    assert read_preview_text(plain) == stamp.text(1234)
    assert read_metadata(plain)["prompt"] is None
    assert _SECRET.encode() not in plain.read_bytes()

    opted = tmp_path / "step-000500-1.png"
    with embed_metadata_scope(True):
        save_preview(gradient(2, 32), opted, stamp.text(99), metadata=meta)
    assert read_preview_text(opted) == stamp.text(99)
    found = read_metadata(opted)
    assert found["prompt"] == _SECRET
    assert found["seed"] == 1234
    assert found["base"] == "z-image"
    assert found["sha256"] == "ab" * 32


def test_app_setting_defaults_off_and_the_flag_is_opt_in(tmp_path: Path) -> None:
    """The Expert setting is off. Train args and the parsers send the flag only when asked."""
    spec = lora_settings().get("image.embed_metadata")
    assert spec.type is SettingType.BOOL
    assert spec.default is False
    assert spec.expert is True
    assert spec.label_key == "forge.app.expert.embed_metadata"
    assert spec.help_key == "forge.app.expert.embed_metadata.tip"
    assert "--embed-metadata" not in engine.train_args(tmp_path, {}, expert=False)
    simple = engine.train_args(tmp_path, {"image.embed_metadata": True}, expert=False)
    assert "--embed-metadata" in simple
    expert = engine.train_args(tmp_path, {"image.embed_metadata": True, "lora.rank": 8}, expert=True)
    assert "--embed-metadata" in expert and expert[expert.index("--rank") + 1] == "8"
    draft = Draft(kind="object")
    quiet = engine.refset_round_args(draft, tmp_path, count=4, strength=0.4, steps=4)
    asked = engine.refset_round_args(draft, tmp_path, count=4, strength=0.4, steps=4, embed_metadata=True)
    assert "--embed-metadata" not in quiet
    assert "--embed-metadata" in asked
    assert engine.embed_metadata_args(False) == []
    assert engine.embed_metadata_args(True) == ["--embed-metadata"]

    parser = build_parser()
    train = parser.parse_args(["lora", "train", "--project", "p"])
    assert train.embed_metadata is False
    asked_train = parser.parse_args(["lora", "train", "--project", "p", "--embed-metadata"])
    assert asked_train.embed_metadata is True
    compare = parser.parse_args(["compare"])
    assert compare.embed_metadata is False
    asked_compare = parser.parse_args(["compare", "--embed-metadata"])
    assert asked_compare.embed_metadata is True
    moss = generation_parser().parse_args(["--snapshot", "snap", "--prompt", "p", "--out", "out.png"])
    assert moss.embed_metadata is False
    asked_moss = generation_parser().parse_args(
        ["--snapshot", "snap", "--prompt", "p", "--out", "out.png", "--embed-metadata"],
    )
    assert asked_moss.embed_metadata is True


def _assert_details(found: dict[str, object]) -> None:
    assert found["prompt"] == _PROMPT
    assert found["seed"] == 1234
    assert found["steps"] == 20
    assert found["base"] == "z-image"
    assert found["base_model"] == "z-image"
    assert found["lora_strength"] == pytest.approx(0.75)
    assert found["sha256"] == "ab" * 32
    loras = found["loras"]
    assert isinstance(loras, list) and len(loras) == 2
    assert loras[0]["name"] == 'face "a"'
    assert loras[0]["strength"] == pytest.approx(0.75)
    assert loras[0]["sha256"] == "ab" * 32
    assert loras[0]["base"] == "z-image"
    assert loras[1]["name"] == "style & co"
    assert loras[1]["base"] == "moss-image-base"
