"""Peak RSS and wall time of ``lora prepare`` with CPU stand-in models. No downloads, no GPU.

The judge, the age estimator and the captioner are fakes. Each allocates and touches a buffer of
a known size when it loads (standing in for weights) and frees it on ``unload``. The photos are
generated: smooth random colour fields with coarse grain, no people.

Each measured run is a fresh child process, so ``ru_maxrss`` is that run's own peak. Pass
``--before-src`` with the ``src`` folder of another checkout (for example a worktree of ``main``)
to measure it and this checkout on the same photos, one after the other::

    git worktree add /tmp/forge-main main
    python scripts/bench_prepare.py --before-src /tmp/forge-main/src

The output digest covers drops.jsonl, gate_trace.jsonl, captions.jsonl and the prepared PNGs,
so equal digests mean both versions wrote the same files. Linux and macOS only (``resource``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
MB = 2**20


# --- photos -----------------------------------------------------------------------------------


def make_photos(folder: Path, count: int, width: int, height: int, quality: int) -> list[Path]:
    """``count`` JPEGs: a smooth random colour field plus coarse grain, so they pass the blur filter."""
    import numpy as np
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for index in range(count):
        rng = np.random.default_rng(1000 + index)
        field = Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
        # Grain in 8 px cells, so it survives the 512 px copy the blur filter looks at.
        grain = Image.fromarray(rng.integers(0, 97, (height // 8, width // 8), dtype=np.uint8)).resize(
            (width, height), Image.Resampling.NEAREST
        )
        pixels = np.asarray(field, dtype=np.int16) + (np.asarray(grain, dtype=np.int16) - 48)[:, :, None]
        path = folder / f"synthetic{index:03d}.jpg"
        Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path, quality=quality)
        out.append(path)
    return out


# --- stand-in models (child process) ------------------------------------------------------------


class Weights:
    """A touched buffer of ``mb`` MiB and the largest total held at once across all stand-ins."""

    held = 0
    peak = 0
    overlap: list[tuple[str, ...]] = []
    loaded: set[str] = set()

    def __init__(self, name: str, mb: int) -> None:
        import numpy as np

        self.name = name
        self.buffer = np.ones(mb * MB, dtype=np.uint8)
        self.size = mb * MB
        Weights.held += self.size
        Weights.peak = max(Weights.peak, Weights.held)
        Weights.loaded.add(name)
        Weights.overlap.append(tuple(sorted(Weights.loaded)))

    def free(self) -> None:
        if self.buffer is not None:
            self.buffer = None
            Weights.held -= self.size
            Weights.loaded.discard(self.name)


def models(judge_mb: int, estimator_mb: int, captioner_mb: int):
    """The fake gate and the captioner loader."""
    from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict

    clean = Verdict("no", "no", None, 0.9, "no_person", False, {key: "absent" for key in INDICATORS})

    class Judge:
        name = "bench-judge"

        def __init__(self) -> None:
            self.weights = Weights("judge", judge_mb)

        def judge(self, image):
            image.convert("RGB").resize((448, 448))
            return clean

        def unload(self) -> None:
            self.weights.free()

    class Estimator:
        name = "bench-estimator"

        def __init__(self) -> None:
            self.weights = Weights("estimator", estimator_mb)

        def estimate(self, image):
            image.convert("RGB").resize((224, 224))
            return AgeReading(0, ())

        def unload(self) -> None:
            self.weights.free()

    class Captioner:
        name = "bench-captioner"

        def __init__(self) -> None:
            self.weights = Weights("captioner", captioner_mb)

        def caption(self, image):
            image.resize((448, 448))
            return {"long": "a smooth colour field with fine grain", "short": "colour field"}

        def unload(self) -> None:
            self.weights.free()

    return Gate(Judge(), Estimator()), (lambda choice: Captioner())


def rss_now_mb() -> float:
    """Current RSS in MiB (Linux ``/proc``; falls back to the peak elsewhere)."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / MB
    except OSError:
        return peak_rss_mb()


def peak_rss_mb() -> float:
    """``ru_maxrss`` in MiB (KiB on Linux, bytes on macOS)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / MB if sys.platform == "darwin" else raw / 1024


def digest(root: Path) -> str:
    """sha256 over the files prepare wrote that must not change."""
    h = hashlib.sha256()
    for name in ("drops.jsonl", "gate_trace.jsonl", "captions.jsonl"):
        path = root / name
        h.update(name.encode() + b"\0" + (path.read_bytes() if path.is_file() else b"-") + b"\0")
    for path in sorted((root / "prepared").glob("*.png")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def child(args: argparse.Namespace) -> None:
    """One measured run of ``prepare`` in this process; prints one JSON line."""
    import voxprint_forge
    from voxprint_forge.lora.consent import check_subject
    from voxprint_forge.lora.events import Progress
    from voxprint_forge.lora.photos import collect
    from voxprint_forge.lora.prepare import prepare
    from voxprint_forge.lora.project import Project

    files = collect(Path(args.photos), [], [])
    subject = check_subject("object", name="Colour field", trigger="vxpbench", rights_confirmed=True)
    project = Project(Path(args.project))
    if args.tracemalloc:
        tracemalloc.start()
    base_mb = rss_now_mb()
    started = time.perf_counter()
    gate, load_captioner = models(args.judge_mb, args.estimator_mb, args.captioner_mb)
    summary = prepare(
        project, files, subject, gate=gate, load_captioner=load_captioner, progress=Progress(), mode="expert"
    )
    wall = time.perf_counter() - started
    traced = tracemalloc.get_traced_memory()[1] / MB if args.tracemalloc else None
    peak = peak_rss_mb()
    print(
        json.dumps(
            {
                "src": str(Path(voxprint_forge.__file__).resolve().parent.parent),
                "photos": len(files),
                "prepared": summary["prepared"],
                "wall_s": round(wall, 2),
                "base_rss_mb": round(base_mb, 1),
                "peak_rss_mb": round(peak, 1),
                "prepare_rss_mb": round(peak - base_mb, 1),
                "tracemalloc_peak_mb": None if traced is None else round(traced, 1),
                "weights_peak_mb": Weights.peak // MB,
                "captioner_with_gate": any("captioner" in names and len(names) > 1 for names in Weights.overlap),
                "digest": digest(project.root),
            }
        )
    )


# --- driver ---------------------------------------------------------------------------------------


def run_child(src: Path, photos: Path, workdir: Path, args: argparse.Namespace, tag: str) -> dict:
    project = workdir / f"project-{tag}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--photos",
        str(photos),
        "--project",
        str(project),
        "--judge-mb",
        str(args.judge_mb),
        "--estimator-mb",
        str(args.estimator_mb),
        "--captioner-mb",
        str(args.captioner_mb),
    ]
    if args.tracemalloc:
        cmd.append("--tracemalloc")
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if done.returncode:
        sys.stderr.write(done.stderr)
        raise SystemExit(f"{tag} run failed ({done.returncode})")
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--count", type=int, default=24, help="Synthetic photos (default 24).")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality of the synthetic photos.")
    parser.add_argument("--judge-mb", type=int, default=512)
    parser.add_argument("--estimator-mb", type=int, default=256)
    parser.add_argument("--captioner-mb", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1, help="Runs per version; the table shows each run.")
    parser.add_argument("--tracemalloc", action="store_true", help="Also report the tracemalloc peak (slower).")
    parser.add_argument("--before-src", type=Path, default=None, help="src folder of the version to compare.")
    parser.add_argument("--keep-dir", type=Path, default=None, help="Work here and keep the files.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--photos", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    workdir = args.keep_dir or Path(tempfile.mkdtemp(prefix="vxp-bench-prepare-"))
    workdir.mkdir(parents=True, exist_ok=True)
    photos = workdir / "photos"
    if not photos.is_dir():
        make_photos(photos, args.count, args.width, args.height, args.quality)
    on_disk = sum(p.stat().st_size for p in photos.iterdir()) / MB
    print(f"{args.count} photos {args.width}x{args.height}, {on_disk:.0f} MiB on disk; ", end="")
    print(f"stand-in weights judge={args.judge_mb} estimator={args.estimator_mb} captioner={args.captioner_mb} MiB")
    versions = ([("before", args.before_src)] if args.before_src else []) + [("after", SRC)]
    rows = []
    for round_no in range(args.repeat):
        for tag, src in versions:
            row = run_child(Path(src), photos, workdir, args, f"{tag}{round_no}")
            rows.append((tag, row))
    cols = ("wall_s", "peak_rss_mb", "prepare_rss_mb", "tracemalloc_peak_mb", "weights_peak_mb", "prepared")
    print(f"{'version':8} " + " ".join(f"{c:>19}" for c in cols) + "  captioner_with_gate  digest")
    for tag, row in rows:
        values = " ".join(f"{'-' if row[c] is None else row[c]!s:>19}" for c in cols)
        print(f"{tag:8} {values}  {row['captioner_with_gate']!s:>19}  {row['digest']}")
    if len({row["digest"] for _, row in rows}) > 1:
        raise SystemExit("outputs differ between versions")


if __name__ == "__main__":
    main()
