"""MODELS.md rows for one canvas part. The file stays under ~/forge-out."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    """One row of the canvas MODELS table."""
    repo: str
    revision: str
    file: str
    size: int
    sha256: str
    licence: str
    tier: str
    vram_gb: int | None


def render_models_md(rows: list[Row]) -> str:
    """Render MODELS markdown from these rows."""
    lines = [
        "| repo | revision | file | bytes | sha256 | licence | tier | vram_gb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        vram = "" if row.vram_gb is None else str(row.vram_gb)
        lines.append(
            f"| {row.repo} | {row.revision} | {row.file} | {row.size} | {row.sha256} | "
            f"{row.licence} | {row.tier} | {vram} |"
        )
    return "\n".join(lines) + "\n"
