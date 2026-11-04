"""Time ``lora prepare`` photo I/O on synthetic photos, with every model stubbed out.

Writes ``--count`` synthetic ``--size`` JPEGs (gradient plus noise, a few with an EXIF
rotation) to a temp folder, then runs :func:`voxprint_forge.lora.prepare.prepare` in a
child process with a stand-in gate and captioner. Prints wall time, the child's peak
RSS and one sha256 over every file in ``prepared/`` plus ``captions.jsonl`` and
``drops.jsonl``, so two code versions can be compared for speed and identical output.
``--photos DIR`` keeps the generated photos for the next run.

No model is loaded or downloaded.

Usage::

    python scripts/bench_prepare_io.py --count 60 --size 4000x3000 --repeat 3
    PYTHONPATH=/path/to/other/checkout/src python scripts/bench_prepare_io.py ...  # compare
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
from pathlib import Path


def make_photos(folder: Path, count: int, width: int, height: int) -> list[Path]:
    """``count`` distinct JPEGs: a per-photo colour gradient with noise; every fifth is EXIF-rotated."""
    import numpy as np
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    for index in range(count):
        rng = np.random.default_rng(index)
        a, b, c = rng.uniform(40, 215, 3)
        base = np.stack([a * xs + 30 * ys, b * ys + 20 * xs, c * (1 - xs) + 10 * ys], axis=2)
        grain = rng.integers(-40, 41, (height // 2, width // 2, 3), dtype=np.int16)
        grain = grain.repeat(2, axis=0).repeat(2, axis=1)
        image = Image.fromarray(np.clip(base.astype(np.int16) + grain, 0, 255).astype(np.uint8))
        path = folder / f"photo{index:03d}.jpg"
        exif = Image.Exif()
        if index % 5 == 0:
            exif[0x0112] = 6  # rotate 90 on display
        image.save(path, quality=90, exif=exif)
        paths.append(path)
    return paths


def _child(folder: Path, project_dir: Path) -> None:
    """Run ``prepare`` on ``folder`` with stand-ins and print one JSON line."""
    from voxprint_forge.lora.consent import check_subject
    from voxprint_forge.lora.events import Progress
    from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict
    from voxprint_forge.lora.photos import collect
    from voxprint_forge.lora.prepare import prepare
    from voxprint_forge.lora.project import Project

    clean = Verdict("no", "no", None, 0.9, "no_person", False, {k: "absent" for k in INDICATORS})

    class Judge:
        name = "stub-judge"

        def judge(self, image):
            return clean

        def unload(self):
            pass

    class Estimator:
        name = "stub-estimator"

        def estimate(self, image):
            return AgeReading(0, ())

    class Captioner:
        name = "stub-captioner"

        def caption(self, image):
            return {"long": "a textured square on a plain background", "short": "textured square"}

        def unload(self):
            pass

    files = collect(folder, (), ())
    project = Project(project_dir)
    start = time.perf_counter()
    summary = prepare(
        project,
        files,
        check_subject("object", name="Bench mug", trigger="vxpbenchq", rights_confirmed=True),
        gate=Gate(Judge(), Estimator()),
        load_captioner=lambda choice: Captioner(),
        progress=Progress(),
        mode="expert",
    )
    wall = time.perf_counter() - start
    digest = hashlib.sha256()
    for path in sorted(project.prepared.iterdir()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    digest.update(project.captions.read_bytes())
    digest.update(project.drops.read_bytes() if project.drops.is_file() else b"")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import voxprint_forge

    src = str(Path(voxprint_forge.__file__).parent)
    print(
        json.dumps(
            {"src": src, "wall_s": wall, "peak_rss_mib": peak_kib / 1024, "sha256": digest.hexdigest(), **summary}
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Generate the photos once, run ``prepare`` ``--repeat`` times in fresh child processes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--size", default="4000x3000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--photos", type=Path, help="reuse (or create) the photos in this folder")
    parser.add_argument("--child", nargs=2, metavar=("PHOTOS", "PROJECT"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        _child(Path(args.child[0]), Path(args.child[1]))
        return 0
    width, height = (int(v) for v in args.size.lower().split("x"))
    with tempfile.TemporaryDirectory(prefix="bench-prepare-") as tmp:
        root = Path(tmp)
        photos = args.photos or root / "photos"
        if not (photos.is_dir() and len(list(photos.glob("*.jpg"))) == args.count):
            t0 = time.perf_counter()
            make_photos(photos, args.count, width, height)
            print(f"generated {args.count} photos {width}x{height} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        runs = []
        for index in range(args.repeat):
            out = subprocess.run(
                [sys.executable, __file__, "--child", str(photos), str(root / f"project{index}")],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            if out.returncode:
                sys.stderr.write(out.stderr)
                return out.returncode
            row = json.loads(out.stdout.strip().splitlines()[-1])
            runs.append(row)
            print(json.dumps(row))
        walls = sorted(r["wall_s"] for r in runs)
        peaks = sorted(r["peak_rss_mib"] for r in runs)
        print(
            json.dumps(
                {
                    "median_wall_s": round(walls[len(walls) // 2], 2),
                    "median_peak_rss_mib": round(peaks[len(peaks) // 2], 1),
                    "outputs_identical": len({r["sha256"] for r in runs}) == 1,
                    "sha256": runs[0]["sha256"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
