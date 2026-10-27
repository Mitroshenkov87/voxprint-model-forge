"""One plural helper for every count string (GUI and CLI) and one source for the set-size range."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from voxprint_common.i18n import plural_category

from voxprint_forge.lora import limits
from voxprint_forge.lora.i18n import catalog, text
from voxprint_forge.plural import category, pick, ru_plural

ROOT = Path(__file__).resolve().parents[1]
APP_LOCALES = ROOT / "src" / "voxprint_forge" / "app" / "locales"
NUMBERS = (1, 2, 5, 11, 21, 22, 24, 25)
EXPECTED = {1: "one", 2: "few", 5: "many", 11: "many", 21: "one", 22: "few", 24: "few", 25: "many"}


def _app(lang: str) -> dict:
    return json.loads((APP_LOCALES / f"{lang}.json").read_text(encoding="utf-8"))


def _all(lang: str) -> dict:
    return {**_app(lang), **catalog(lang)}


@pytest.mark.parametrize("count", NUMBERS)
def test_ru_plural_picks_one_few_many(count) -> None:
    assert ru_plural(count, "one", "few", "many") == EXPECTED[count]
    assert ru_plural(count, "картинка", "картинки", "картинок") == {
        "one": "картинка", "few": "картинки", "many": "картинок"}[EXPECTED[count]]


def test_ru_plural_agrees_with_the_shell_rule() -> None:
    for count in range(0, 250):
        assert category(count, "ru") == plural_category(count, "ru"), count
        assert category(count, "en") == plural_category(count, "en"), count


def test_every_count_string_has_full_forms() -> None:
    en = _all("en")
    plural_keys = {k for k, v in en.items() if isinstance(v, dict)}
    assert "lora.progress.refset_done" in plural_keys and "forge.app.refset.simple" in plural_keys
    for key in plural_keys:
        assert set(en[key]) - {"by"} == {"one", "other"}, key


@pytest.mark.parametrize("count", NUMBERS)
def test_gui_and_cli_render_the_same_form(count) -> None:
    pytest.importorskip("PySide6")
    from voxprint_common.i18n import set_language

    from voxprint_forge.app.text import tr
    from voxprint_forge.app.window import register_catalogs

    register_catalogs()
    set_language("en")
    for key, value in _all("en").items():
        if not isinstance(value, dict):
            continue
        by = value.get("by", "count")
        params = {"count": 0, "round": 1, "low": 1, "high": 0, "minutes": 3, "need": 0, "extra": 0, "shown": 0,
                  "steps": 0, "done": 0, "planned": 0, "total": 9, "kept": 0, "dropped": 0, by: count}
        want = value[category(count, "en")].format(**{**limits.text_params(), **params})
        assert tr(key, **params) == want, key
        if key.startswith("lora."):
            assert text(key, "en", **params) == want, key


def test_english_singular() -> None:
    assert text("lora.progress.refset_done", "en", round=2, shown=1) == "Round 2 is ready: 1 picture to review"
    assert text("lora.err.too_few", "en", need=1) == "Add at least 1 more photo to continue."
    assert text("lora.err.too_many", "en", extra=1, high=60) == "Remove 1 photo; 60 is the limit."
    assert pick({"one": "a", "other": "b"}, {}, "en") == "b"


RANGE = re.compile(r"(?:от\s+)?(\d+)\s*(?:to|–|-|до)\s*(\d+)\s+(?:[\w«»*]+\s+){0,1}?(photos|pictures|фото|картин\w*)")


def _ranges(value: str) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b, _noun in RANGE.findall(value)}


def test_catalogs_take_the_set_range_from_the_constant() -> None:
    """No catalog string spells the recommended or Expert range itself; they use {rec_low}… placeholders."""
    literal = {limits.RECOMMENDED_RANGE, limits.EXPERT_RANGE}
    for lang in ("en",):
        for key, value in _all(lang).items():
            for form in (value.values() if isinstance(value, dict) else (value,)):
                assert not (_ranges(form) & literal), (lang, key, form)
                for pair in literal:
                    assert "{}–{}".format(*pair) not in form and "{}-{}".format(*pair) not in form, (lang, key)
    assert text("lora.step.photos.desc", "en").startswith("Add {} to {} photos".format(*limits.RECOMMENDED_RANGE))


def test_the_manual_uses_the_constant_ranges() -> None:
    allowed = {limits.RECOMMENDED_RANGE, limits.EXPERT_RANGE, (1, 10), (15, 20)}  # (15, 20): the "usually enough" tip
    for lang, recommended, expert in (("en", "Recommended: {} to {}", "Allowed in Expert: {} to {}"),):
        manual = "\n".join(page.read_text(encoding="utf-8")
                           for page in sorted((ROOT / "docs" / "manual" / lang).glob("*.md")))
        assert manual
        assert recommended.format(*limits.RECOMMENDED_RANGE) in manual
        if limits.expert_range_differs():
            assert expert.format(*limits.EXPERT_RANGE) in manual
        assert _ranges(manual) <= allowed, (lang, _ranges(manual) - allowed)


def test_expert_range_is_named_only_when_it_differs(monkeypatch) -> None:
    assert limits.expert_range_differs()
    monkeypatch.setattr(limits, "EXPERT_RANGE", limits.RECOMMENDED_RANGE)
    assert not limits.expert_range_differs()
