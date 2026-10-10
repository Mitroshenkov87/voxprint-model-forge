"""Command lines for llama.cpp b11476. The runner is injected so tests do not need the binary."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.errors import StageError, ToolMissingError

TAG = "b11476"
COMMIT = "988190680d5a89fce97de3c20df2c2813731fd61"


@dataclass
class Completed:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class Runner:
    """Subprocess runner. Tests subclass this and skip the binary."""

    def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ToolMissingError(f"missing {argv[0]}", code="tool_missing") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-2000:]
            raise StageError(f"{Path(argv[0]).name} exited {proc.returncode}: {tail}", code="tool_failed")
        return Completed(argv=argv, returncode=0, stdout=proc.stdout, stderr=proc.stderr)

    def capture_until(self, argv: list[str], *, pattern: str, timeout: float = 180) -> Completed:
        """Run a long-lived tool until stdout contains pattern, then stop it."""
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ToolMissingError(f"missing {argv[0]}", code="tool_missing") from exc
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        assert proc.stdout is not None
        try:
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break
                lines.append(line)
                if pattern in "".join(lines):
                    break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        text = "".join(lines)
        if pattern not in text:
            raise StageError(f"{Path(argv[0]).name} did not report {pattern}", code="tool_output")
        return Completed(argv=argv, returncode=0, stdout=text, stderr="")


def bin_dir() -> Path | None:
    raw = os.environ.get("FORGE_LLAMA_BIN", "")
    if raw:
        return Path(raw)
    return None


def src_dir() -> Path | None:
    raw = os.environ.get("FORGE_LLAMA_SRC", "")
    if raw:
        return Path(raw)
    return None


def tool(name: str) -> str:
    folder = bin_dir()
    if folder is not None:
        candidate = folder / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise ToolMissingError(
        f"{name} not on PATH. Build llama.cpp {TAG} and set FORGE_LLAMA_BIN.",
        code="tool_missing",
    )


def convert_script() -> Path:
    folder = src_dir()
    if folder is None:
        raise ToolMissingError("FORGE_LLAMA_SRC is not set (needs convert_hf_to_gguf.py)", code="tool_missing")
    path = folder / "convert_hf_to_gguf.py"
    if not path.is_file():
        raise ToolMissingError(f"missing {path}", code="tool_missing")
    return path


def convert_argv(script: Path, source: Path, outfile: Path) -> list[str]:
    return [
        os.environ.get("FORGE_PYTHON", "python3"),
        str(script),
        str(source),
        "--outtype",
        "f16",
        "--outfile",
        str(outfile),
    ]


def imatrix_argv(binary: str, model: Path, corpus: Path, outfile: Path, *, chunks: int, ctx: int) -> list[str]:
    return [
        binary,
        "-m",
        str(model),
        "-f",
        str(corpus),
        "-o",
        str(outfile),
        "--output-format",
        "gguf",
        "--chunks",
        str(chunks),
        "-c",
        str(ctx),
        "-ngl",
        "99",
        "-fa",
        "on",
        "--no-ppl",
    ]


def quantize_argv(binary: str, source: Path, outfile: Path, quant: str, imatrix: Path) -> list[str]:
    return [binary, "--imatrix", str(imatrix), str(source), str(outfile), quant]


def perplexity_argv(binary: str, model: Path, corpus: Path, *, chunks: int, ctx: int) -> list[str]:
    return [
        binary,
        "-m",
        str(model),
        "-f",
        str(corpus),
        "--chunks",
        str(chunks),
        "-c",
        str(ctx),
        "-ngl",
        "99",
        "-fa",
        "on",
    ]


def server_argv(
    binary: str,
    model: Path,
    *,
    ctx: int,
    port: int,
    cache: str,
) -> list[str]:
    return [
        binary,
        "-m",
        str(model),
        "-c",
        str(ctx),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-ctk",
        cache,
        "-ctv",
        cache,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "1",
    ]


def cli_argv(binary: str, model: Path, prompt: str, *, n_predict: int, ctx: int, cache: str) -> list[str]:
    return [
        binary,
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-ctk",
        cache,
        "-ctv",
        cache,
        "--no-conversation",
        "--no-display-prompt",
        "--temp",
        "0.7",
    ]
