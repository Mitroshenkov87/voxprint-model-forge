"""Render docs/manual/{en,ru}.md into the offline guide shipped with the app.

Output: src/voxprint_forge/app/guide/{index,en,ru}.html plus img/. The installer's
wheel carries it (package-data), so More -> Open guide works without a network.

    python scripts/build_guide.py          # write
    python scripts/build_guide.py --check  # fail if the committed guide is stale
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"
GUIDE = ROOT / "src" / "voxprint_forge" / "app" / "guide"
LANGS = ("en", "ru")
STYLE = (
    "body{font-family:system-ui,'Segoe UI',sans-serif;max-width:860px;margin:2em auto;padding:0 1em;"
    "line-height:1.5;color:#e8e6e3;background:#1e1e24}a{color:#f6d365}img{max-width:100%;border-radius:6px}"
    "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 8px}code{background:#2a2a33;padding:0 3px}"
)


def page(title: str, body: str, lang: str) -> str:
    """A self-contained HTML page: no scripts, no remote assets."""
    return (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def render() -> dict[str, bytes]:
    """Every guide file by relative path."""
    files: dict[str, bytes] = {}
    links = []
    for lang in LANGS:
        text = (MANUAL / f"{lang}.md").read_text(encoding="utf-8")
        text = re.sub(r"\]\((en|ru)\.md\)", r"](\1.html)", text)
        title = text.splitlines()[0].lstrip("# ").strip()
        body = markdown.markdown(text, extensions=["tables", "sane_lists"])
        files[f"{lang}.html"] = page(title, body, lang).encode("utf-8")
        links.append(f'<p><a href="{lang}.html">{html.escape(title)}</a></p>')
    files["index.html"] = page("VOXPrint AI Model Forge", "\n".join(links), "en").encode("utf-8")
    for image in sorted((MANUAL / "img").glob("*.png")):
        files[f"img/{image.name}"] = image.read_bytes()
    return files


def stale() -> list[str]:
    """Files that differ from a fresh render, or should not be there."""
    want = render()
    have = {p.relative_to(GUIDE).as_posix() for p in GUIDE.rglob("*") if p.is_file()} if GUIDE.is_dir() else set()
    wrong = [name for name, data in want.items() if not (GUIDE / name).is_file() or (GUIDE / name).read_bytes() != data]
    return sorted(wrong + sorted(have - set(want)))


def main() -> int:
    """Write the guide, or with ``--check`` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        wrong = stale()
        if wrong:
            print("guide is stale, run scripts/build_guide.py:", ", ".join(wrong))
            return 1
        return 0
    for name, data in render().items():
        target = GUIDE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
