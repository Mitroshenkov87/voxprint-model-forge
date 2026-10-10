"""Sha256 manifest and MODELS.md snippets. No weight hashes are invented here."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from voxprint_forge.errors import StageError
from voxprint_forge.layout import Layout
from voxprint_forge.scrub import scrub

GIT_NAME = "Mitroshenkov87"
GIT_EMAIL = "223849179+Mitroshenkov87@users.noreply.github.com"


def models_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# MODELS registry snippet",
        "",
        "Sources for app failover. `torrent` is a placeholder until a torrent exists.",
        "Every Hugging Face repo and Kaggle model in this file is private.",
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
                lines.append(f"  - hf: {sources.get('hf', '')}")
                lines.append(f"  - kaggle: {sources.get('kaggle', '')}")
                lines.append(f"  - torrent: {sources.get('torrent', 'placeholder')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def git_instructions(date: str) -> str:
    manifest = f"manifests/{date}-a100.json"
    snippet = f"manifests/{date}-a100.MODELS.md"
    return (
        f"git add {manifest} {snippet}\n"
        f'git commit -m "Record A100 sha256 manifest {date}"\n'
        "git push\n"
    )


def write_run_manifest(
    repo: Path,
    layout: Layout,
    payload: dict[str, object],
    *,
    date: str,
    push: bool,
) -> dict[str, object]:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    snippet = models_markdown(payload)
    name = f"{date}-a100.json"
    snippet_name = f"{date}-a100.MODELS.md"
    destinations = [layout.manifest_dir(), repo / "manifests"]
    written: list[str] = []
    for folder in destinations:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(body, encoding="utf-8")
        (folder / snippet_name).write_text(snippet, encoding="utf-8")
        written.append(str(folder / name))
    instructions = git_instructions(date)
    pushed = False
    if push and os.environ.get("GH_TOKEN"):
        pushed = _push_manifest(repo, date)
    return {
        "manifests": written,
        "instructions": instructions,
        "pushed": pushed,
        "markdown": snippet,
    }


def summary_lines(payload: dict[str, object]) -> list[str]:
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


def _push_manifest(repo: Path, date: str) -> bool:
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        return False
    manifest = f"manifests/{date}-a100.json"
    snippet = f"manifests/{date}-a100.MODELS.md"
    header = f"Authorization: Bearer {token}"
    add = subprocess.run(
        ["git", "-C", str(repo), "add", "--", manifest, snippet],
        check=False,
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise StageError(scrub(add.stderr or add.stdout), code="manifest_git")
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            f"user.name={GIT_NAME}",
            "-c",
            f"user.email={GIT_EMAIL}",
            "commit",
            "-m",
            f"Record A100 sha256 manifest {date}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
        raise StageError(scrub(commit.stderr or commit.stdout), code="manifest_git")
    push = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "http.extraHeader=" + header,
            "push",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        raise StageError(scrub(push.stderr or push.stdout), code="manifest_git")
    return True
