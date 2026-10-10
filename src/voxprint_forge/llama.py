"""Command lines for llama.cpp b11476. The runner is injected so tests do not need the binary."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from voxprint_forge.errors import StageError, ToolMissingError

TAG = "b11476"
COMMIT = "988190680d5a89fce97de3c20df2c2813731fd61"


def _pump(stream, sink: Queue[str | None]) -> None:
    try:
        for line in stream:
            sink.put(line)
    except (OSError, ValueError):
        return
    finally:
        sink.put(None)


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop the process and anything it spawned. A blocked pipe must not keep it alive."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@dataclass
class Completed:
    """Argv, return code, and captured stdout and stderr."""
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class Runner:
    """Subprocess runner. Tests subclass this and skip the binary."""

    def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
        """Run argv to completion and raise when the return code is not 0."""
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
        """Run until stdout contains pattern. The read never blocks past timeout.

        A reader thread fills a queue, so a silent process still hits the
        deadline. On timeout the whole process group is killed.
        """
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ToolMissingError(f"missing {argv[0]}", code="tool_missing") from exc
        lines: list[str] = []
        sink: Queue[str | None] = Queue()
        assert proc.stdout is not None
        thread = threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = sink.get(timeout=remaining)
                except Empty:
                    timed_out = True
                    break
                if line is None:
                    break
                lines.append(line)
                if pattern in line or pattern in "".join(lines):
                    break
        finally:
            _kill_tree(proc)
        text = "".join(lines)
        if pattern not in text:
            tail = text[-500:]
            if timed_out:
                raise StageError(
                    f"{Path(argv[0]).name} timed out after {timeout:.0f}s before {pattern}: {tail}",
                    code="tool_timeout",
                )
            raise StageError(f"{Path(argv[0]).name} did not report {pattern}: {tail}", code="tool_output")
        return Completed(argv=argv, returncode=0, stdout=text, stderr="")


def bin_dir() -> Path | None:
    """``FORGE_LLAMA_BIN`` when it is set, otherwise None."""
    raw = os.environ.get("FORGE_LLAMA_BIN", "")
    if raw:
        return Path(raw)
    return None


def src_dir() -> Path | None:
    """``FORGE_LLAMA_SRC`` when it is set, otherwise None."""
    raw = os.environ.get("FORGE_LLAMA_SRC", "")
    if raw:
        return Path(raw)
    return None


def tool(name: str) -> str:
    """Path of a llama.cpp binary, from ``FORGE_LLAMA_BIN`` or ``PATH``."""
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
    """Path of ``convert_hf_to_gguf.py`` under ``FORGE_LLAMA_SRC``."""
    folder = src_dir()
    if folder is None:
        raise ToolMissingError("FORGE_LLAMA_SRC is not set (needs convert_hf_to_gguf.py)", code="tool_missing")
    path = folder / "convert_hf_to_gguf.py"
    if not path.is_file():
        raise ToolMissingError(f"missing {path}", code="tool_missing")
    return path


def convert_argv(script: Path, source: Path, outfile: Path) -> list[str]:
    """Argv that converts a safetensors directory to an F16 GGUF."""
    return [
        os.environ.get("FORGE_PYTHON") or sys.executable,
        str(script),
        str(source),
        "--outtype",
        "f16",
        "--outfile",
        str(outfile),
    ]


def imatrix_argv(binary: str, model: Path, corpus: Path, outfile: Path, *, chunks: int, ctx: int) -> list[str]:
    """Argv for llama-imatrix. ``--fit`` stays off."""
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
    """Argv for llama-quantize with an importance matrix."""
    return [binary, "--imatrix", str(imatrix), str(source), str(outfile), quant]


def perplexity_argv(binary: str, model: Path, corpus: Path, *, chunks: int, ctx: int) -> list[str]:
    """Argv for llama-perplexity. ``--fit`` stays off."""
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
        "--fit",
        "off",
    ]


def server_argv(
    binary: str,
    model: Path,
    *,
    ctx: int,
    port: int,
    cache: str,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Argv for llama-server. Extra split flags sit before ``-v --fit off``."""
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
        *extra,
        "-v",
        "--fit",
        "off",
    ]


def cli_argv(binary: str, model: Path, prompt: str, *, n_predict: int, ctx: int, cache: str) -> list[str]:
    """Argv for one llama-cli sample. ``--fit`` stays off."""
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
        "--single-turn",
        "--no-display-prompt",
        "--temp",
        "0.7",
        "--fit",
        "off",
    ]
