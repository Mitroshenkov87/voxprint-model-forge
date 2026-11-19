"""b11476 fit-stage parser and a capture timeout that fires with no output."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from voxprint_forge.errors import StageError
from voxprint_forge.evaluate import data_dir
from voxprint_forge.llama import Runner, cli_argv, perplexity_argv, server_argv
from voxprint_forge.measure import parse_buffers

# Real b11476 -v layout from llama.cpp 98819068:
#   load_tensors: %12s model buffer size = %8.2f MiB
#   llama_kv_cache: %10s KV buffer size = %8.2f MiB
# Figures are the A100 IQ3_M measurement at ctx 8192, q8 KV.
# CPU_Mapped 521 MiB is the line the old parser kept. CUDA_Host is not the card.
B11476_IQ3_M_CTX8192 = """\
load_tensors: offloading 63 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 64/64 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   521.00 MiB
load_tensors:        CUDA0 model buffer size = 11466.58 MiB
load_tensors:    CUDA_Host model buffer size =    12.50 MiB
llama_kv_cache:      CUDA0 KV buffer size =   272.00 MiB
llama_kv_cache:  CUDA_Host KV buffer size =     4.00 MiB
llama_context:      CUDA0 compute buffer size =   300.00 MiB
llama_context:  CUDA_Host compute buffer size =    18.01 MiB
"""

_MIB = 1024 * 1024


def test_iq3_m_fixture_uses_cuda0_not_the_mapped_host() -> None:
    report = parse_buffers(B11476_IQ3_M_CTX8192)
    assert report.devices == {"CUDA0": {"weights_mib": 11466.58, "kv_mib": 272.0}}
    assert report.weights_bytes == int(round(11466.58 * _MIB))
    assert report.kv_bytes == int(round(272.0 * _MIB))
    assert "CPU_Mapped" not in report.devices
    assert "CUDA_Host" not in report.devices


def test_cuda_devices_sum_and_host_is_left_out() -> None:
    log = """\
load_tensors:        CUDA0 model buffer size =  1000.00 MiB
load_tensors:        CUDA1 model buffer size =   250.50 MiB
load_tensors:   CPU_Mapped model buffer size =   521.00 MiB
load_tensors:    CUDA_Host model buffer size =    12.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   200.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =    72.00 MiB
llama_kv_cache:  CUDA_Host KV buffer size =     4.00 MiB
"""
    report = parse_buffers(log)
    assert report.devices["CUDA0"] == {"weights_mib": 1000.0, "kv_mib": 200.0}
    assert report.devices["CUDA1"] == {"weights_mib": 250.5, "kv_mib": 72.0}
    assert set(report.devices) == {"CUDA0", "CUDA1"}
    assert report.weights_bytes == int(round(1250.5 * _MIB))
    assert report.kv_bytes == int(round(272.0 * _MIB))


def test_missing_cuda_buffers_are_an_error() -> None:
    log = "load_tensors:   CPU_Mapped model buffer size =   521.00 MiB\n"
    with pytest.raises(StageError) as caught:
        parse_buffers(log)
    assert caught.value.code == "measure_log"


# CPU log from llama-server b11476 -v --fit off on a machine with no CUDA device.
# CPU_REPACK is a second model buffer. It is not a card, and neither is CPU KV.
B11476_CPU = """\
0.00.076.753 I load_tensors:   CPU_Mapped model buffer size =    98.87 MiB
0.00.076.757 I load_tensors:   CPU_REPACK model buffer size =     7.59 MiB
0.00.085.158 I llama_context:        CPU  output buffer size =     0.19 MiB
0.00.085.248 I llama_kv_cache:        CPU KV buffer size =     5.98 MiB
"""


def test_cpu_log_from_b11476_is_not_treated_as_a_card() -> None:
    with pytest.raises(StageError) as caught:
        parse_buffers(B11476_CPU)
    assert caught.value.code == "measure_log"


def test_cli_argv_is_one_shot_and_does_not_auto_fit() -> None:
    # b11476 llama-cli rejects --no-conversation (that flag belongs to llama-completion).
    # --single-turn exits after the prompt. --fit off keeps the inner server from auto-fitting.
    argv = cli_argv("llama-cli", Path("model.gguf"), "Hello", n_predict=8, ctx=512, cache="q8_0")
    assert "--single-turn" in argv
    assert "--no-conversation" not in argv
    assert "--no-display-prompt" in argv
    assert argv[-2:] == ["--fit", "off"]


def test_perplexity_argv_does_not_auto_fit() -> None:
    argv = perplexity_argv("llama-perplexity", Path("model.gguf"), Path("ppl.txt"), chunks=1, ctx=512)
    assert argv[-2:] == ["--fit", "off"]
    assert "--chunks" in argv and "1" in argv


def test_parse_ppl_reads_the_b11476_final_line() -> None:
    # Real llama-perplexity b11476 line. The ETA prefix is glued on; the number is still the estimate.
    from voxprint_forge.evaluate import parse_ppl

    log = (
        "0.00.739.824 I perplexity: 0.62 seconds per pass - ETA "
        "0.00.750.515 I Final estimate: PPL = 27.1206 +/- 4.62090\n"
    )
    assert parse_ppl(log) == pytest.approx(27.1206)


def test_ppl_corpus_is_long_enough_for_ctx_512() -> None:
    # llama-perplexity at b11476 errors unless the file tokenizes to at least 2*n_ctx.
    # A Llama-family tokenizer turned the old paragraph into 87 tokens. 1200 words
    # is the floor that stayed above 1024 tokens on SmolLM2-135M.
    words = data_dir().joinpath("ppl.txt").read_text(encoding="utf-8").split()
    assert len(words) >= 1200


def test_server_argv_is_verbose_and_does_not_auto_fit() -> None:
    argv = server_argv("llama-server", Path("model.gguf"), ctx=8192, port=8080, cache="q8_0")
    assert argv[-3:] == ["-v", "--fit", "off"]
    assert "-c" in argv and "8192" in argv
    assert "q8_0" in argv


def test_capture_until_returns_when_the_pattern_appears() -> None:
    code = (
        "import time; "
        "print('llama_kv_cache: CUDA0 KV buffer size = 1.00 MiB', flush=True); "
        "time.sleep(30)"
    )
    started = time.monotonic()
    done = Runner().capture_until([sys.executable, "-c", code], pattern="KV buffer size", timeout=5)
    assert time.monotonic() - started < 5
    assert "KV buffer size" in done.stdout


def test_capture_until_times_out_and_kills_the_process_tree(tmp_path: Path) -> None:
    beat = tmp_path / "beat"
    child = tmp_path / "child.py"
    parent = tmp_path / "parent.py"
    child.write_text(
        "import pathlib, time\n"
        f"path = pathlib.Path({str(beat)!r})\n"
        "while True:\n"
        "    path.write_text('x', encoding='utf-8')\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    parent.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    with pytest.raises(StageError) as caught:
        Runner().capture_until([sys.executable, str(parent)], pattern="KV buffer size", timeout=1.2)
    assert time.monotonic() - started < 5
    assert caught.value.code == "tool_timeout"
    deadline = time.monotonic() + 3
    while not beat.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert beat.is_file()
    time.sleep(0.6)
    first = beat.stat().st_mtime_ns
    time.sleep(0.6)
    assert beat.stat().st_mtime_ns == first
