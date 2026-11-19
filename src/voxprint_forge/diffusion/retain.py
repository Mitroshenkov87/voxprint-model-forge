"""Keep a downloaded source until every plan consumer has finished.

Scratch files under ``upstream/<part>`` are removed when that part finishes.
Shared weights live in ``work/<job>/sources/<id>`` and are removed only after
each name in the plan ``consumers`` list has a release marker.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from voxprint_forge.diffusion.planload import CanvasPlan, load_canvas_plan
from voxprint_forge.scrub import log


def source_store(forge_out: Path, job_id: str) -> Path:
    return forge_out / "work" / job_id / "sources"


def _markers(forge_out: Path, job_id: str, source_id: str) -> Path:
    return forge_out / "work" / job_id / "source-releases" / source_id


def consumers_for(plan: CanvasPlan, source_id: str) -> tuple[str, ...]:
    """Union of ``consumers`` on every part that lists this source. Default is the part id."""
    names: list[str] = []
    for part in plan.parts:
        for source in part.sources:
            if source.id != source_id:
                continue
            declared = source.consumers or (part.id,)
            for name in declared:
                if name not in names:
                    names.append(name)
    return tuple(names)


def release_consumer(
    forge_out: Path,
    job_id: str,
    source_id: str,
    consumer: str,
    consumers: tuple[str, ...],
) -> bool:
    """Record ``consumer`` as finished. Delete the shared tree when the list is complete."""
    folder = _markers(forge_out, job_id, source_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / consumer).write_text("released\n", encoding="utf-8", newline="\n")
    if consumers:
        (folder / "_consumers").write_text("\n".join(consumers) + "\n", encoding="utf-8", newline="\n")
    if not consumers or any(not (folder / name).is_file() for name in consumers):
        return False
    tree = source_store(forge_out, job_id) / source_id
    _drop_tree(tree)
    return True


def _drop_tree(tree: Path) -> None:
    if not tree.exists():
        return
    size = 0
    for item in tree.rglob("*"):
        if item.is_symlink() or item.is_file():
            try:
                size += item.lstat().st_size
            except OSError:
                continue
    shutil.rmtree(tree)
    log(f"reclaim {tree} {size} bytes")


def release_consumed(forge_out: Path, repo: Path, plan_rel: str, source_ids: tuple[str, ...], consumer: str) -> None:
    """Release ``consumer`` on sources declared by another plan, then delete if it was last."""
    plan = load_canvas_plan(repo / plan_rel)
    for source_id in source_ids:
        release_consumer(forge_out, plan.id, source_id, consumer, consumers_for(plan, source_id))
