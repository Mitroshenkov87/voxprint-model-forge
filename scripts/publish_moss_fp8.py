"""Write the MOSS FP8 model card and print the Hugging Face upload steps.

Dry-run is the only mode. This script never uploads. The upload happens after
the picture-quality check passes, from the built ``model-forge/moss-image-*-fp8``
folder (the one that contains ``dit-fp8.safetensors``). The files written here
are the git template: card, LICENSE, NOTICE and a manifest whose mirror
revision and weight sha256 are still placeholders.

    python scripts/publish_moss_fp8.py
    python scripts/publish_moss_fp8.py --out manifests/moss-fp8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxprint_forge.diffusion.moss_fp8 import upload_steps, write_templates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "manifests" / "moss-fp8"


def main(argv: list[str] | None = None) -> int:
    """Write the template and print the upload steps. ``--upload`` is refused."""
    parser = argparse.ArgumentParser(description="Write the MOSS FP8 card and print upload steps. Does not upload.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directory for the card and manifest.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write local files and print the steps. This is the default.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Refused. Upload is manual, after the picture-quality check.",
    )
    args = parser.parse_args(argv)
    write_templates(args.out)
    print(upload_steps(), end="")
    if args.upload:
        print("Refusing --upload. This script never uploads.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
