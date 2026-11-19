"""Sha256 manifest and MODELS.md snippets. No weight hashes are invented here."""

from __future__ import annotations

import json
from pathlib import Path

from voxprint_forge.layout import Layout


def models_markdown(payload: dict[str, object]) -> str:
    """MODELS.md snippet for the files in a run payload."""
    lines = [
        "# MODELS registry snippet",
        "",
        "Sources for app failover. `torrent` is a placeholder until a torrent exists.",
        "Hugging Face repos in this file are public. Kaggle models stay private.",
        "",
    ]
    models = payload.get("models")
    if not isinstance(models, list):
        return "\n".join(lines)
    for model in models:
        if not isinstance(model, dict):
            continue
        lines.append(f"## {model.get('id')}")
        lines.append("")
        lines.append(f"- base: `{model.get('base_model')}`")
        lines.append(f"- revision: `{model.get('revision')}`")
        lines.append(f"- licence: `{model.get('licence')}`")
        files = model.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                sources = item.get("sources")
                if not isinstance(sources, dict):
                    sources = {}
                lines.append(
                    f"- tier `{item.get('tier')}` quant `{item.get('quant')}` "
                    f"file `{item.get('name')}`"
                )
                lines.append(f"  - sha256: `{item.get('sha256')}`")
                lines.append(f"  - size: {item.get('size')}")
                lines.append(f"  - min_vram_gb: {item.get('min_vram_gb')}")
                lines.append(f"  - split: `{item.get('split', 'gpu')}`")
                lines.append(f"  - ram_bytes: {item.get('ram_bytes', 0)}")
                lines.append(f"  - hf: {sources.get('hf', '')}")
                lines.append(f"  - kaggle: {sources.get('kaggle', '')}")
                lines.append(f"  - torrent: {sources.get('torrent', 'placeholder')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def copy_back_instructions(date: str, folder: Path) -> str:
    """Tell the operator which manifest files to copy back."""
    manifest = folder / f"{date}-a100.json"
    snippet = folder / f"{date}-a100.MODELS.md"
    return (
        "Copy these files back to the forge repo. This machine has no GitHub remote.\n"
        f"  {manifest}\n"
        f"  {snippet}\n"
    )


def write_run_manifest(
    layout: Layout,
    payload: dict[str, object],
    *,
    date: str,
) -> dict[str, object]:
    """Write the JSON manifest and the MODELS snippet for this date."""
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    snippet = models_markdown(payload)
    name = f"{date}-a100.json"
    snippet_name = f"{date}-a100.MODELS.md"
    folder = layout.manifest_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(body, encoding="utf-8", newline="\n")
    (folder / snippet_name).write_text(snippet, encoding="utf-8", newline="\n")
    instructions = copy_back_instructions(date, folder)
    return {
        "manifests": [str(folder / name)],
        "instructions": instructions,
        "markdown": snippet,
    }


def summary_lines(payload: dict[str, object]) -> list[str]:
    """Short per-file lines for the end-of-run summary."""
    lines = ["Forge summary"]
    models = payload.get("models")
    if not isinstance(models, list):
        return lines
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("skipped"):
            lines.append(f"{model.get('id')}: skipped ({model.get('reason')})")
            continue
        lines.append(str(model.get("id")))
        files = model.get("files")
        if not isinstance(files, list):
            continue
        for item in files:
            if not isinstance(item, dict):
                continue
            sources = item.get("sources")
            if not isinstance(sources, dict):
                sources = {}
            lines.append(
                f"  {item.get('name')} sha256={item.get('sha256')} "
                f"hf={sources.get('hf', '')} kaggle={sources.get('kaggle', '')}"
            )
    return lines
