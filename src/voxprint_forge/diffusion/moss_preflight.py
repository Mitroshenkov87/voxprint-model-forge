"""Dry-run preflight for the MOSS-Image fit job. It downloads no weights.

Checks: vendored sha and audit; plan parse; author and Comfy revisions with the file list,
sizes and sha256 from the Hub listing; runtime versions (torch>=2.9, diffusers>=0.38,
transformers>=5.3, CUDA); optional fast GDN kernels for the Qwen3.5 text encoder; the
ComfyUI pin; free disk. Exit 0 when every required check passes.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.diffusion.moss import REPOS, VENDORED, audit_source, check_versions, verify_vendored
from voxprint_forge.diffusion.moss_fit import FitPlan, load_plan

# Unique bytes per run: one author snapshot (27.23 GB) + Comfy reference files (15.50 GB) + outputs.
DISK_NEED_GB = 50.0
EXPECTED_FILES = 37  # files in each author repo at the pinned revision


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict[str, object]:
    return {"check": name, "ok": ok, "required": required, "detail": detail}


def _version_of(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "0"


def hub_listing(repo: str, revision: str) -> dict[str, tuple[int, str]]:
    """path -> (size, sha256 or '') at a pinned revision, from the Hub API."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"{repo}: revision resolved to {info.sha}")
    return {s.rfilename: (s.lfs.size if s.lfs else (s.size or 0), s.lfs.sha256 if s.lfs else "")
            for s in info.siblings}


def run(
    plan: FitPlan,
    *,
    lister: Callable[[str, str], dict[str, tuple[int, str]]] = hub_listing,
    versions: Callable[[str], str] = _version_of,
    cuda: Callable[[], str] | None = None,
    disk_free_gb: Callable[[], float] | None = None,
    vm: bool = False,
) -> list[dict[str, object]]:
    """All checks. ``vm`` makes runtime checks required; on a laptop/CI they are advisory."""
    out = []
    try:
        verify_vendored()
        findings = audit_source(VENDORED.read_text(encoding="utf-8"))
        detail = "sha256 pinned; " + ("; ".join(findings) or "audit clean")
        out.append(_check("vendored pipeline", not findings, detail))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        out.append(_check("vendored pipeline", False, str(exc)))
    out.append(_check("plan", True, f"{len(plan.arms)} arms x {len(plan.caps_gb)} caps, {plan.steps} steps, "
                      f"CFG {plan.guidance}, {plan.width}x{plan.height}"))
    model = REPOS[plan.model]
    targets = [(model.repo, model.revision, None)]
    targets += [(a.repo, a.revision, dict(zip(a.files, a.sha256))) for a in plan.arms if a.engine == "comfyui"]
    sizes = []
    for repo, rev, pins in targets:
        try:
            listing = lister(repo, rev)
        except Exception as exc:  # noqa: BLE001
            out.append(_check(f"hub {repo}", False, str(exc)[:200]))
            continue
        if pins is None:
            weights = [p for p in listing if p.endswith(".safetensors")]
            unsafe = [p for p in listing if p.endswith((".bin", ".pt", ".pth", ".pkl", ".ckpt"))]
            ok = len(listing) == EXPECTED_FILES and not unsafe and all(listing[p][1] for p in weights)
            total = sum(v[0] for v in listing.values())
            out.append(_check(f"hub {repo}@{rev[:8]}", ok,
                              f"{len(listing)} files, {len(weights)} safetensors, {total / 1e9:.2f} GB"))
            sizes.append(total)
        else:
            bad = [p for p, sha in pins.items() if listing.get(p, (0, ""))[1] != sha]
            total = sum(listing.get(p, (0, ""))[0] for p in pins)
            detail = f"{len(pins)} pinned files, {total / 1e9:.2f} GB" + (f"; sha mismatch {bad}" if bad else "")
            out.append(_check(f"hub {repo}@{rev[:8]}", not bad, detail))
            sizes.append(total)
    problems = check_versions(versions("torch"), versions("diffusers"), versions("transformers"))
    out.append(_check("runtime versions", not problems, "; ".join(problems) or
                      f"torch {versions('torch')}, diffusers {versions('diffusers')}, "
                      f"transformers {versions('transformers')}", required=vm))
    for dist in ("flash-linear-attention", "causal-conv1d"):
        have = versions(dist) != "0"
        out.append(_check(f"optional {dist}", have,
                          "installed" if have else "missing: Qwen3.5 text encoder falls back to slow PyTorch kernels",
                          required=False))
    if cuda is not None:
        name = cuda()
        out.append(_check("cuda", bool(name), name or "no CUDA device", required=vm))
    for arm in plan.arms:
        if arm.engine == "comfyui":
            out.append(_check("comfyui pin", len(arm.comfyui_commit) == 40 and bool(arm.comfy_kitchen),
                              f"ComfyUI {arm.comfyui_commit[:8]}, comfy-kitchen=={arm.comfy_kitchen}"))
    if disk_free_gb is not None:
        free = disk_free_gb()
        out.append(_check("disk", free >= DISK_NEED_GB, f"{free:.0f} GB free, need {DISK_NEED_GB:.0f} GB",
                          required=vm))
    return out


def _cuda_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m voxprint_forge.diffusion.moss_preflight", description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--vm", action="store_true", help="runtime, CUDA and disk checks are required")
    ap.add_argument("--disk-path", type=Path, default=Path.home())
    args = ap.parse_args(argv)
    checks = run(load_plan(args.plan), cuda=_cuda_name if args.vm else None,
                 disk_free_gb=lambda: shutil.disk_usage(args.disk_path).free / 1e9, vm=args.vm)
    for c in checks:
        flag = "ok  " if c["ok"] else ("FAIL" if c["required"] else "warn")
        print(f"{flag} {c['check']}: {c['detail']}")
    failed = [c for c in checks if c["required"] and not c["ok"]]
    print(json.dumps({"preflight": "pass" if not failed else "fail", "failed": [c["check"] for c in failed]}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
