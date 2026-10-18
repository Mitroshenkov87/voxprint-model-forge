"""Render the user manual (docs/manual/{en,ru}) into one PDF per language with WeasyPrint.

    python scripts/build_pdf_guide.py OUT_DIR [--font-dir DIR]

Writes OUT_DIR/VOXPrint-Model-Forge-Guide-en.pdf and -ru.pdf: a cover page with the version, a table of
contents with page numbers, then the manual pages in nav order with their screenshots and numbered pages.
The fonts (DejaVu Sans and Sans Mono, which cover Cyrillic) are embedded from --font-dir; the oblique
faces are used when present there.
Pinned dependencies: installer/requirements-pdf.txt.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_guide import LANGS, MANUAL, SITE, manual_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONTS = {("normal", "normal"): "DejaVuSans.ttf", ("bold", "normal"): "DejaVuSans-Bold.ttf",
         ("normal", "italic"): "DejaVuSans-Oblique.ttf", ("bold", "italic"): "DejaVuSans-BoldOblique.ttf"}
MONO = "DejaVuSansMono.ttf"
NAME = "VOXPrint-Model-Forge-Guide-{lang}.pdf"
TEXT = {
    "en": {"title": "VOXPrint AI Model Forge", "subtitle": "User guide", "version": "Version",
           "contents": "Contents", "lang": "English"},
    "ru": {"title": "VOXPrint AI Model Forge", "subtitle": "Руководство пользователя", "version": "Версия",
           "contents": "Содержание", "lang": "Русский"},
}


def css(font_dir: Path) -> str:
    """Print styles: A4, embedded DejaVu, page numbers, TOC leaders."""
    faces = "".join(
        f"@font-face{{font-family:'Guide Sans';font-weight:{w};font-style:{s};src:url('{(font_dir / f).as_uri()}')}}"
        for (w, s), f in FONTS.items() if (font_dir / f).is_file()
    )
    faces += f"@font-face{{font-family:'Guide Mono';src:url('{(font_dir / MONO).as_uri()}')}}"
    return faces + """
@page{size:A4;margin:20mm 18mm 22mm 18mm;@bottom-center{content:counter(page);font:9pt 'Guide Sans';color:#555}}
@page cover{@bottom-center{content:none}}
html{font-family:'Guide Sans';font-size:10.5pt;line-height:1.45;color:#1d1d1f}
code,pre{font-family:'Guide Mono';font-size:9pt}
pre{background:#f3f3f5;padding:6px 8px;white-space:pre-wrap;border-radius:4px}
code{background:#f3f3f5;padding:0 2px}
a{color:#1d4ed8;text-decoration:none}
h1{font-size:20pt;margin:0 0 10pt;break-before:page;bookmark-level:1}
h2{font-size:14pt;margin:14pt 0 6pt;bookmark-level:2;break-after:avoid}
h3{font-size:11.5pt;margin:10pt 0 4pt;bookmark-level:3;break-after:avoid}
img{max-width:100%;max-height:120mm;display:block;margin:6pt auto;border:1px solid #ccc;border-radius:3px}
table{border-collapse:collapse;width:100%;margin:6pt 0;font-size:9.5pt}
td,th{border:1px solid #bbb;padding:3pt 5pt;vertical-align:top;text-align:left}
th{background:#f0f0f2}
tr,img{break-inside:avoid}
section.cover{page:cover;text-align:center;padding-top:70mm}
section.cover .title{font-size:28pt;font-weight:bold}
section.cover .subtitle{font-size:18pt;margin-top:8mm}
section.cover .meta{font-size:11pt;margin-top:20mm;color:#444}
nav.toc{break-before:page}
nav.toc h1{break-before:avoid;bookmark-level:none}
nav.toc ul{list-style:none;padding:0;margin:0}
nav.toc li{margin:3pt 0}
nav.toc li.l2{margin-left:8mm;font-size:9.5pt}
nav.toc a{color:#1d1d1f}
nav.toc a::after{content:leader('.') target-counter(attr(href),page)}
"""


def version() -> str:
    """The BUILD.json version, with the rc suffix when one is set."""
    data = json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))
    return f"{data['version']}-rc{data['rc']}" if data.get("rc") else str(data["version"])


def document(lang: str) -> str:
    """Cover, contents and the manual pages of one language as one HTML document."""
    words = TEXT[lang]
    body = markdown.markdown(manual_text(lang), extensions=["tables", "sane_lists", "fenced_code", "toc"])
    body = re.sub(r'href="(en|ru)\.html"', lambda m: f'href="{SITE}manual/{m.group(1)}/"', body)
    entries = []
    for level, ident, title in re.findall(r'<h([12]) id="([^"]+)">(.*?)</h\1>', body):
        label = re.sub(r"<[^>]+>", "", title)
        entries.append(f'<li class="l{level}"><a href="#{ident}">{label}</a></li>')
    cover = (
        f'<section class="cover"><div class="title">{html.escape(words["title"])}</div>'
        f'<div class="subtitle">{html.escape(words["subtitle"])}</div>'
        f'<div class="meta">{html.escape(words["version"])} {html.escape(version())} · '
        f'{html.escape(words["lang"])}</div>'
        "</section>"
    )
    toc = f'<nav class="toc"><h1>{html.escape(words["contents"])}</h1><ul>{"".join(entries)}</ul></nav>'
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{html.escape(words["title"])}'
            f" — {html.escape(words['subtitle'])}</title></head><body>{cover}{toc}{body}</body></html>")


def build(out: Path, font_dir: Path = FONT_DIR) -> list[Path]:
    """Write both PDFs into ``out`` and return their paths."""
    from weasyprint import CSS, HTML
    from weasyprint.text.fonts import FontConfiguration

    needed = (FONTS[("normal", "normal")], FONTS[("bold", "normal")], MONO)
    missing = [f for f in needed if not (font_dir / f).is_file()]
    if missing:
        raise SystemExit(f"fonts missing in {font_dir}: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for lang in LANGS:
        path = out / NAME.format(lang=lang)
        fonts = FontConfiguration()
        HTML(string=document(lang), base_url=str(MANUAL) + "/").write_pdf(
            path, stylesheets=[CSS(string=css(font_dir), font_config=fonts)], font_config=fonts)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry: build the PDFs and print their paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--font-dir", type=Path, default=FONT_DIR)
    args = parser.parse_args(argv)
    for path in build(args.out, args.font_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
