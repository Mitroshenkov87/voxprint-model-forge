"""One download per ``repo@revision`` for a compare run.

Arm folders hardlink the files they load. They do not copy the snapshot.
A local FP8 arm does not link ``transformer/*.safetensors``; the base config
is enough because the FP8 file replaces those weights.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.scrub import active_secrets, scrub

Skip = Callable[[Path], bool]


def ensure_cached(
    download: object,
    root: Path,
    repo: str,
    revision: str,
    patterns: tuple[str, ...] | None,
) -> Path:
    """Download ``repo@revision`` once. A later call with the same key returns the directory."""
    dest = root / cache_name(repo, revision, patterns)
    if (dest / ".complete").is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "source.txt").write_text(_source_text(repo, revision, patterns), encoding="utf-8", newline="\n")
    kwargs: dict[str, object] = {"repo_id": repo, "local_dir": str(dest)}
    if revision:
        kwargs["revision"] = revision
    if patterns:
        kwargs["allow_patterns"] = list(patterns)
    try:
        download(**kwargs)
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        raise CompareSetup(f"download failed for {repo}: {message}", code="compare_download") from exc
    (dest / ".complete").write_text("ok\n", encoding="utf-8", newline="\n")
    return dest


def bind_tree(source: Path, dest: Path, *, skip: Skip | None = None) -> int:
    """Hardlink the files this arm loads. A cross-filesystem link is a setup error."""
    dest.mkdir(parents=True, exist_ok=True)
    linked = 0
    for src in sorted(source.rglob("*")):
        if not src.is_file() or src.name in {".complete", "source.txt"}:
            continue
        rel = src.relative_to(source)
        if skip is not None and skip(rel):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(src, target)
            except OSError as exc:
                raise CompareSetup(
                    f"could not hardlink {rel.as_posix()} (cache and job folder must be on one filesystem)",
                    code="compare_weights",
                ) from exc
        linked += 1
    if linked < 1:
        raise CompareSetup("cached snapshot has no files for this arm", code="compare_weights")
    return linked


def skip_transformer_weights(rel: Path) -> bool:
    """True for ``transformer/*.safetensors``, including shards under that directory."""
    return "transformer" in rel.parts and rel.suffix == ".safetensors"


def cache_name(repo: str, revision: str, patterns: tuple[str, ...] | None) -> str:
    """Stable directory name for one repo, revision, and pattern list."""
    pattern_text = "*" if not patterns else "\n".join(patterns)
    digest = hashlib.sha256(f"{repo}\n{revision}\n{pattern_text}".encode()).hexdigest()[:20]
    return digest


def entries(root: Path) -> list[tuple[Path, str, str]]:
    """Cache directories with the repo and revision recorded in ``source.txt``."""
    if not root.is_dir():
        return []
    found: list[tuple[Path, str, str]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        repo, revision = _read_source(path / "source.txt")
        if repo:
            found.append((path, repo, revision))
    return found


def remove_entry(path: Path) -> None:
    """Delete one cache directory."""
    shutil.rmtree(path)


def reclaim_all(root: Path) -> list[str]:
    """Delete every cached snapshot. Review images are not in this directory."""
    removed: list[str] = []
    if not root.is_dir():
        return removed
    for path in sorted(root.iterdir()):
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def _source_text(repo: str, revision: str, patterns: tuple[str, ...] | None) -> str:
    pattern_text = "*" if not patterns else "\n".join(patterns)
    return f"{repo}\n{revision}\n{pattern_text}\n"


def _read_source(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return "", ""
    return lines[0].strip(), lines[1].strip()
