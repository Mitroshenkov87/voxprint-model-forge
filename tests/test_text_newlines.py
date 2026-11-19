"""Text writes under src must pin newline to LF."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def text_write_problems(root: Path) -> list[str]:
    """Return ``path:line`` rows whose text write does not pass ``newline="\\n"``."""
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    found: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "write_text":
                if not _newline_is_lf(node):
                    found.append(f"{path}:{node.lineno}: write_text")
                continue
            if name not in {"open", "fdopen"}:
                continue
            mode = _mode_text(node)
            if mode is not None and _text_write_mode(mode) and not _newline_is_lf(node):
                found.append(f"{path}:{node.lineno}: open")
    return found


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _newline_is_lf(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "newline":
            value = keyword.value
            return isinstance(value, ast.Constant) and value.value == "\n"
    return False


def _mode_text(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return _constant(keyword.value)
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open" and len(node.args) > 1:
        return _constant(node.args[1])
    if isinstance(func, ast.Attribute) and func.attr == "fdopen" and len(node.args) > 1:
        return _constant(node.args[1])
    if isinstance(func, ast.Attribute) and func.attr == "open" and node.args:
        first = _constant(node.args[0])
        if first is not None and _looks_like_mode(first):
            return first
        if len(node.args) > 1:
            second = _constant(node.args[1])
            if second is not None and _looks_like_mode(second):
                return second
    return None


def _constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _looks_like_mode(text: str) -> bool:
    return bool(text) and set(text) <= set("rwatb+xUt")


def _text_write_mode(mode: str) -> bool:
    return bool(mode) and set(mode) <= set("w+t") and "w" in mode


def test_src_text_writes_pin_lf_newlines() -> None:
    assert text_write_problems(SRC) == []


def test_scanner_flags_text_writes_and_ignores_binary_and_append(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "def f(path, mode):",
                "    path.write_text('a\\n', encoding='utf-8')",
                "    open(path, 'w', encoding='utf-8')",
                "    open(path, 'wb')",
                "    path.open(mode)",
                "    path.open('a', encoding='utf-8')",
                "    path.write_text('a\\n', encoding='utf-8', newline=None)",
                "    path.write_text('a\\n', encoding='utf-8', newline='\\n')",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    found = text_write_problems(sample)
    assert [item.rsplit(": ", 1)[-1] for item in found] == ["write_text", "open", "write_text"]
