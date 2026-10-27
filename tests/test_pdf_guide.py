"""The PDF user guide: both languages, more than 10 pages, screenshots, extractable Cyrillic, neutral wording.

The installer workflow builds the PDFs and runs this with VOXPRINT_PDF_GUIDE_DIR set. Elsewhere it builds them
into a temporary folder when WeasyPrint and the DejaVu fonts are available, and skips otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_pdf_guide  # noqa: E402

FORBIDDEN = re.compile(
    r"\bminors?\b|\bchild|\bkids?\b|\bunder[- ]?(18|20)\b(?! ?GB)|age (check|estimat|gate|filter)"
    r"|safety check|nsfw|\bnud(e|ity)\b"
    r"|несовершеннолет|ребён|ребен|детск|подрост|возраст|младше 18|проверк[аи] безопасност",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    given = os.environ.get("VOXPRINT_PDF_GUIDE_DIR")
    if given:
        folder = Path(given)
    else:
        fonts = build_pdf_guide.FONT_DIR / "DejaVuSans.ttf"
        if importlib.util.find_spec("weasyprint") is None or not fonts.is_file():
            pytest.skip("WeasyPrint or the DejaVu fonts are not installed")
        folder = tmp_path_factory.mktemp("pdf")
        build_pdf_guide.build(folder)
    return {lang: folder / build_pdf_guide.NAME.format(lang=lang) for lang in ("en",)}


def _text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def test_it_exists_with_more_than_ten_pages(pdfs: dict[str, Path]) -> None:
    from pypdf import PdfReader

    for lang, path in pdfs.items():
        assert path.is_file(), lang
        assert len(PdfReader(path).pages) > 10, lang


@pytest.mark.skipif(shutil.which("pdfimages") is None, reason="needs poppler-utils (pdfimages)")
def test_screenshots_are_embedded(pdfs: dict[str, Path]) -> None:
    for lang, path in pdfs.items():
        listing = subprocess.run(["pdfimages", "-list", str(path)], capture_output=True, text=True, check=True)
        assert len(listing.stdout.strip().splitlines()) - 2 > 0, lang


def test_cover_toc_and_sections(pdfs: dict[str, Path]) -> None:
    from pypdf import PdfReader

    version = build_pdf_guide.version()
    for lang, words in (("en", ("Install", "Training", "Presets", "Contents", "User guide")),):
        first = PdfReader(pdfs[lang]).pages[0].extract_text()
        assert version in first, lang
        text = _text(pdfs[lang])
        for word in words:
            assert word in text, (lang, word)


def test_nothing_explains_the_filter(pdfs: dict[str, Path]) -> None:
    for lang, path in pdfs.items():
        hits = sorted({m.group(0) for m in FORBIDDEN.finditer(_text(path))})
        assert not hits, (lang, hits)
