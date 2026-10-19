"""``--help``, ``--version`` and argparse errors do not import ML libraries."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("torch", "diffusers", "transformers", "accelerate", "bitsandbytes")
# Prints sorted sys.modules after the parser accepts or rejects argv.
_HELPER = (
    "import sys\n"
    "from voxprint_forge.cli import build_parser\n"
    "try:\n"
    "    build_parser().parse_args(sys.argv[1:])\n"
    "except SystemExit:\n"
    "    pass\n"
    "sys.stdout.write('MODULES\\n')\n"
    "sys.stdout.write('\\n'.join(sorted(sys.modules)))\n"
    "sys.stdout.write('\\n')\n"
)


def _command_paths() -> list[list[str]]:
    from voxprint_forge.cli import build_parser

    paths: list[list[str]] = [[]]

    def walk(parser: argparse.ArgumentParser, prefix: list[str]) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    path = [*prefix, name]
                    paths.append(path)
                    walk(sub, path)

    walk(build_parser(), [])
    return paths


def _present(name: str, modules: set[str]) -> bool:
    prefix = name + "."
    return name in modules or any(item.startswith(prefix) for item in modules)


def _banned(modules: set[str]) -> list[str]:
    return [name for name in BANNED if _present(name, modules)]


def _modules_after_parse(argv: list[str]) -> set[str]:
    """Run the helper and return the module names present after parse."""
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", _HELPER, *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    marker = proc.stdout.rfind("MODULES\n")
    assert marker >= 0, proc.stdout
    body = proc.stdout[marker + len("MODULES\n"):]
    return {line.strip() for line in body.splitlines() if line.strip()}


def _importtime_modules(stderr: str) -> set[str]:
    names: set[str] = set()
    for line in stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3 and parts[-1] and parts[-1] != "imported package":
            names.add(parts[-1])
    return names


def _run_module(argv: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "voxprint_forge", *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc, time.perf_counter() - start


def test_help_version_and_errors_import_no_ml_libraries() -> None:
    """Every help path, --version, and argparse errors stay free of ML libraries."""
    cases: list[tuple[list[str], int]] = [(["--version"], 0), (["--not-a-flag"], 2)]
    cases.extend(([*path, "--help"], 0) for path in _command_paths())
    cases.append((["lora", "train", "--nope"], 2))
    for argv, code in cases:
        proc, elapsed = _run_module(argv)
        assert proc.returncode == code, f"{argv} exit {proc.returncode}\n{proc.stderr}"
        assert "import time:" in proc.stderr
        assert not _banned(_importtime_modules(proc.stderr)), argv
        assert not _banned(_modules_after_parse(argv)), argv
        if argv in (["--help"], ["lora", "train", "--help"]):
            warnings.warn(f"{' '.join(argv)} {elapsed:.3f}s", stacklevel=1)
    help_paths = [path for path, _code in cases if path[-1:] == ["--help"]]
    assert ["--help"] in help_paths
    for name in ("prepare", "train", "extend", "export", "refset"):
        assert ["lora", name, "--help"] in help_paths
    assert ["models", "--help"] in help_paths
    assert ["models", "status", "--help"] in help_paths
    assert ["models", "fetch", "--help"] in help_paths
    assert ["index", "--help"] in help_paths
    assert ["index", "build", "--help"] in help_paths
