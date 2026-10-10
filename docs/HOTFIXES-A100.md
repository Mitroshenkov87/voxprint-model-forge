# A100 hotfixes (4th live fit hang, plus the next eval failures)

Apply from the repo root. The diff is against the current main text of these
files (`ffae320` plus the coordinator hot-fixes). `patch` accepts the pipeline
hunks on that tree and on the tree that also fetches the imatrix corpus:

    sed -n '/^diff --git/,$p' docs/HOTFIXES-A100.md | patch -p1 --forward

## What the live VM will hit

1. **Fit hang.** `server_argv` did not pass `-v --fit off`. `--fit` defaults to on, and that auto-fit sat for about 25 minutes. The argv now ends with `-v --fit off`.
2. **Wrong buffer.** `parse_buffers` kept the first `model buffer size` match, which is `CPU_Mapped` at 521 MiB, and the regex did not record the device. It now sums every `CUDA*` model buffer and every `CUDA*` KV buffer, skips names ending in `_Host`, and reports `devices[name] = {weights_mib, kv_mib}`. Reference, checked as a fixture: IQ3_M, ctx 8192, q8 KV = 11466.58 MiB weights + 272 MiB KV on CUDA0.
3. **Timeout never fired.** `capture_until` blocked in `readline`, so a silent process never reached the deadline. A reader thread fills a queue. On timeout the process group is killed (`taskkill /F /T` on Windows, `SIGKILL` to the group on POSIX). The error code is `tool_timeout`.
4. **Eval dies on the first sample.** b11476 `llama-cli` rejects `--no-conversation` (`error: invalid argument: --no-conversation`; that flag belongs to `llama-completion`). Refusal and prose now pass `--single-turn`, which generates the prompt and exits. `llama-cli` and `llama-perplexity` also pass `--fit off`, so eval does not start another auto-fit.
5. **Perplexity has no final estimate.** `data/ppl.txt` tokenizes to 87 tokens on a Llama tokenizer. b11476 `llama-perplexity` requires at least `2 * n_ctx` tokens and the eval context is 512, so it needs 1024. It then exits 0 and prints no `Final estimate` line. The corpus is now 1710 words of the same original Apache-2.0 text. On SmolLM2-135M that file produces `Final estimate: PPL = 27.1206`.
6. **A failed quantize is reused.** `llama-quantize` leaves a partial GGUF and the next run treated any existing file as finished. A file is kept only when its `.sha256` sidecar is present. A failed run deletes the partial.

## Checked on CPU against llama.cpp b11476 (`98819068`)

Command shape, same flag order as `scripts/vm_run.sh`:

    forge --plan <tiny plan> --root <repo> --forge-out <out> run-all

Model: `bartowski/SmolLM2-135M-Instruct-GGUF` Q4_K_M (101 MiB). `llama-server -v --fit off` printed KV buffer size and `capture_until` returned. The CPU log is `CPU_Mapped` / `CPU_REPACK` / `CPU` and `parse_buffers` raises `measure_log`. That is the right result on a machine with no CUDA device. The A100 line is CUDA0, covered by the fixture. No CPU fallback.

`llama-imatrix` completed with the existing argv. Eval (perplexity 27.1206, one English refusal, one Russian refusal, one prose sample), card, and upload all finished. Upload used an in-memory private remote, not Hugging Face or Kaggle.

`llama-cli` prints its banner in the prose sample and still exits. The sample is usable. No further patch.

## Canvas overlay

No patch. `scripts/canvas_job.sh` and `src/voxprint_forge/diffusion` do not block on a pipe read. Nothing there matches the fit hang.

## Windows CI

Already fixed on this branch at `79b31ad` (large reported disk in the tiny-plan tests; production margin unchanged). Do not reapply that here.

diff --git a/src/voxprint_forge/data/ppl.txt b/src/voxprint_forge/data/ppl.txt
index c0716dc..593bac8 100644
--- a/src/voxprint_forge/data/ppl.txt
+++ b/src/voxprint_forge/data/ppl.txt
@@ -1,2 +1,34 @@
 # Original perplexity text. Apache-2.0.
 The workshop smelled of cedar and cold tea. She set the plane on the bench, checked the grain, and took one thin shaving from the edge of the board. Outside, a tram bell marked the hour. He copied the measurement twice, because the first number had been written in a hurry, and the drawer they were building had to close without a scrape.
+
+Mira kept the second copy in a tin under the bench. The tin also held a stub of pencil, three brass screws, and a ticket from a tram that no longer ran past the canal. She liked the ticket because the number was printed twice, once in ink and once as a row of holes, and the two versions still agreed. When the drawer stuck, she did not sand the face. She eased the side until the runner met the groove along its whole length, then tried the drawer again with both hands and with one.
+
+The room above the workshop was a kitchen with a window onto the yard. A kettle sat on the stove and clicked as it cooled. Rain had left a dark line on the brick, and the line stopped exactly where the awning began. Mira wiped the line with a cloth, not because the brick needed it, but because the measurement of that line was the first thing she had written down that morning, and she wanted the wall to match the note.
+
+Pavel came in with a paper bag of rolls and a folded newspaper he did not open. He set the bag on the only clear corner of the bench and asked whether the drawer was for the clinic or for the school. Mira said it was for the clinic, for the cards that recorded who had come and who had not, and that the school job was a shelf that could wait until the tram was less crowded. Pavel nodded, ate half a roll, and read the measurement aloud so that both of them heard the same number.
+
+They worked until the light moved off the bench and onto the floor. The plane left a curl of wood that held its shape until Mira pressed it flat. She numbered the curls in her head, not on paper, and stopped when the edge felt the same from one end to the other. Pavel swept the curls into a box that would go to the stove later. Nothing from the cedar board went into the yard. The yard was for rain and for the bicycle, and the bicycle leaned where it always leaned, against the pipe that sang when the wind came down the alley.
+
+At dusk the tram bell came again, farther away, and a dog answered it from a balcony. Mira closed the drawer, opened it, and closed it once more. The sound was a soft knock, not a scrape. She wrote that down too, because a silence can be a result, and a result that is not written is a result that gets argued about later. Pavel put the newspaper under the kettle so the heat would not mark the bench, then forgot it there, which they both noticed and neither of them moved.
+
+The next morning the clinic sent a girl with a list. The list asked for two drawers, not one, and for a label slot on the front of each. Mira measured the slot with a scrap of card the girl had brought, a card that already had a name written on it and then crossed out. They agreed the slot would take a new card without bending it, and that the old card would go back with the girl. The girl waited on the stool, counting the screws in a jar, and said the count was one hundred and twelve. Mira counted them again and got the same number, which pleased them both more than it should have.
+
+Pavel started the second drawer from a board that had a knot near one end. He moved the knot into the back, where a hand would not catch it, and he told Mira before he cut so that the decision was shared. The saw was sharp. The cut was square. They checked the square with a small metal triangle that had lived in the workshop longer than either of them, and whose edge was bright only where thumbs had worn the dust away. When the triangle sat flush, they did not celebrate. They turned the piece and checked the other corner.
+
+By noon the rain had stopped and the brick line had faded. Mira opened the window. The alley smelled of wet dust and of bread from the shop at the corner. A tram passed, then another, and the gap between them was long enough for a bicycle to cross without hurry. Pavel watched the gap and did not cross. He was holding the front of the second drawer, waiting for the glue on the label slot to lose its shine. Glue that still shines will print onto a card, and a card with glue on it is a card that sticks to the next card, and the clinic did not want that.
+
+They ate the rest of the rolls standing up. Mira described the shelf for the school, which was not a hard shelf: four boards, two brackets, and a rule that the bottom board sit high enough for a bucket to pass under it. Pavel said the brackets should be the plain ones, not the ones with a curl, because a curl collects dust and the school already had dust. They wrote the plain brackets into the note, and they wrote the height of the bucket they had seen in the corridor, which was a different bucket from the one in the yard, and they marked the two buckets with different words so the numbers would not be swapped.
+
+In the afternoon a man from the canal came to ask about a window that would not shut. Mira told him the window was not a drawer and that she would look on Thursday. The man left a coin on the bench for the looking, and Pavel put the coin in the tin with the ticket, because a coin on a bench becomes a coin in a drawer, and this drawer was not finished. The man had also left a description: the sash dropped a finger's width, and the finger was his own, which Mira did not trust, so she wrote "measure on site" beside the Thursday line.
+
+The first drawer dried. Mira rubbed the runners with a candle end, a small white stub that smelled faintly of the kitchen and not of the shop. Wax is a quiet material. It does not announce itself. The drawer moved, stopped where she asked it to stop, and stayed shut when she tilted the carcass a little, which is what a floor does in that building when a tram goes by. She tilted it with a wedge of scrap and listened. The knock stayed soft.
+
+The girl came back for the drawers before the second one was waxed. She was early, and she said the clinic wanted them before the evening list. Mira let her take the finished one and kept the other, and she wrote a receipt on the back of the crossed-out card: one drawer, label slot, no scrape, balance on Thursday with the window. The girl read it, folded it into her coat, and carried the drawer with both arms as if it were a cake. Pavel held the door. The alley was bright. The bicycle was still against the pipe.
+
+Thursday smell was different: sun on wet wood, and the kettle empty because they had forgotten it. They finished the second drawer, waxed it, and carried it themselves, since the girl had said the evening list was heavy and a second trip would be kinder. The clinic corridor had the bucket they had measured, and the height matched the note. A nurse thanked them without looking up from a page, then looked up and thanked them again, which Mira preferred. They set the drawer in the cabinet. The card slid in. The card slid out. The nurse wrote a name on a fresh card and the card still slid.
+
+On the way back they stopped for the window. The sash dropped less than a finger, closer to the thickness of the card the girl had carried. Mira planed a single pass from the side rail, no more, and the sash met the sill with a sound like the drawer. The man from the canal was not there. They left the coin's worth of work done, and they left a note in the frame: measured, eased, try it tonight. Pavel put the pencil back in his pocket. The pencil was the stub from the tin. It still wrote.
+
+They walked home along the canal because the tram was full. The water held a long reflection of a wire and a gull that was not fishing, only sitting. Mira said the school shelf could be cut on Friday if the plain brackets were in the drawer under the bench, the small drawer, not the new ones. Pavel said he had counted the brackets on Monday and the count was four, which was exactly the count they needed, and that he would count them again in the morning because Monday was far enough away to be wrong. They agreed that a count you do not repeat is a story, and a count you repeat is a measurement.
+
+At the workshop the newspaper under the kettle had gone yellow at the edge. Pavel moved it to the box of curls, not to burn the news, only to keep the bench clear for Friday. Mira opened her note and added the window, the card that slid, and the four brackets still to be counted. The tram bell marked the hour. The dog on the balcony did not answer. The drawer under the bench, the old one, opened without a scrape, and the brackets were there, four of them, plain, with a little dust in the screw holes and no curl at the ends.
diff --git a/src/voxprint_forge/llama.py b/src/voxprint_forge/llama.py
index 05d36c2..f151d3b 100644
--- a/src/voxprint_forge/llama.py
+++ b/src/voxprint_forge/llama.py
@@ -4,10 +4,13 @@ from __future__ import annotations
 
 import os
 import shutil
+import signal
 import subprocess
+import threading
 import time
 from dataclasses import dataclass
 from pathlib import Path
+from queue import Empty, Queue
 
 from voxprint_forge.errors import StageError, ToolMissingError
 
@@ -15,6 +18,44 @@ TAG = "b11476"
 COMMIT = "988190680d5a89fce97de3c20df2c2813731fd61"
 
 
+def _pump(stream, sink: Queue[str | None]) -> None:
+    try:
+        for line in stream:
+            sink.put(line)
+    except (OSError, ValueError):
+        return
+    finally:
+        sink.put(None)
+
+
+def _kill_tree(proc: subprocess.Popen[str]) -> None:
+    """Stop the process and anything it spawned. A blocked pipe must not keep it alive."""
+    if proc.poll() is not None:
+        return
+    if os.name == "nt":
+        subprocess.run(
+            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
+            check=False,
+            capture_output=True,
+            text=True,
+        )
+        try:
+            proc.wait(timeout=10)
+        except subprocess.TimeoutExpired:
+            proc.kill()
+            proc.wait(timeout=10)
+        return
+    try:
+        os.killpg(proc.pid, signal.SIGKILL)
+    except (ProcessLookupError, PermissionError):
+        proc.kill()
+    try:
+        proc.wait(timeout=10)
+    except subprocess.TimeoutExpired:
+        proc.kill()
+        proc.wait(timeout=10)
+
+
 @dataclass
 class Completed:
     argv: list[str]
@@ -43,38 +84,55 @@ class Runner:
         return Completed(argv=argv, returncode=0, stdout=proc.stdout, stderr=proc.stderr)
 
     def capture_until(self, argv: list[str], *, pattern: str, timeout: float = 180) -> Completed:
-        """Run a long-lived tool until stdout contains pattern, then stop it."""
+        """Run until stdout contains pattern. The read never blocks past timeout.
+
+        A reader thread fills a queue, so a silent process still hits the
+        deadline. On timeout the whole process group is killed.
+        """
         try:
             proc = subprocess.Popen(
                 argv,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT,
                 text=True,
+                start_new_session=True,
             )
         except FileNotFoundError as exc:
             raise ToolMissingError(f"missing {argv[0]}", code="tool_missing") from exc
         lines: list[str] = []
-        deadline = time.monotonic() + timeout
+        sink: Queue[str | None] = Queue()
         assert proc.stdout is not None
+        thread = threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True)
+        thread.start()
+        deadline = time.monotonic() + timeout
+        timed_out = False
         try:
-            while time.monotonic() < deadline:
-                line = proc.stdout.readline()
-                if not line:
+            while True:
+                remaining = deadline - time.monotonic()
+                if remaining <= 0:
+                    timed_out = True
+                    break
+                try:
+                    line = sink.get(timeout=remaining)
+                except Empty:
+                    timed_out = True
+                    break
+                if line is None:
                     break
                 lines.append(line)
-                if pattern in "".join(lines):
+                if pattern in line or pattern in "".join(lines):
                     break
         finally:
-            if proc.poll() is None:
-                proc.terminate()
-                try:
-                    proc.wait(timeout=10)
-                except subprocess.TimeoutExpired:
-                    proc.kill()
-                    proc.wait(timeout=10)
+            _kill_tree(proc)
         text = "".join(lines)
         if pattern not in text:
-            raise StageError(f"{Path(argv[0]).name} did not report {pattern}", code="tool_output")
+            tail = text[-500:]
+            if timed_out:
+                raise StageError(
+                    f"{Path(argv[0]).name} timed out after {timeout:.0f}s before {pattern}: {tail}",
+                    code="tool_timeout",
+                )
+            raise StageError(f"{Path(argv[0]).name} did not report {pattern}: {tail}", code="tool_output")
         return Completed(argv=argv, returncode=0, stdout=text, stderr="")
 
 
@@ -171,6 +229,8 @@ def perplexity_argv(binary: str, model: Path, corpus: Path, *, chunks: int, ctx:
         "99",
         "-fa",
         "on",
+        "--fit",
+        "off",
     ]
 
 
@@ -202,6 +262,9 @@ def server_argv(
         str(port),
         "-np",
         "1",
+        "-v",
+        "--fit",
+        "off",
     ]
 
 
@@ -224,8 +287,10 @@ def cli_argv(binary: str, model: Path, prompt: str, *, n_predict: int, ctx: int,
         cache,
         "-ctv",
         cache,
-        "--no-conversation",
+        "--single-turn",
         "--no-display-prompt",
         "--temp",
         "0.7",
+        "--fit",
+        "off",
     ]
diff --git a/src/voxprint_forge/measure.py b/src/voxprint_forge/measure.py
index 8c2c0c0..611220b 100644
--- a/src/voxprint_forge/measure.py
+++ b/src/voxprint_forge/measure.py
@@ -10,8 +10,16 @@ from voxprint_forge.errors import StageError
 from voxprint_forge.fitmath import Arch, fits_budget, gib, min_vram_gb, quant_rank
 from voxprint_forge.plan import Plan, Tier
 
-_MODEL = re.compile(r"model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB", re.IGNORECASE)
-_KV = re.compile(r"KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB", re.IGNORECASE)
+# b11476 prints "%12s model buffer size" and "%10s KV buffer size" with the backend name.
+# CPU_Mapped is often first. CUDA_Host is host memory and is not the card.
+_DEVICE_MODEL = re.compile(
+    r"(?P<device>[A-Za-z0-9_]+)\s+model buffer size\s*=\s*(?P<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB",
+    re.IGNORECASE,
+)
+_DEVICE_KV = re.compile(
+    r"(?P<device>[A-Za-z0-9_]+)\s+KV buffer size\s*=\s*(?P<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB",
+    re.IGNORECASE,
+)
 _MIB = 1024 * 1024
 
 
@@ -23,6 +31,15 @@ class Candidate:
     origin: str
 
 
+@dataclass(frozen=True)
+class BufferReport:
+    """CUDA device totals. `devices` is one entry per CUDA* buffer, excluding *_Host."""
+
+    weights_bytes: int
+    kv_bytes: int
+    devices: dict[str, dict[str, float]]
+
+
 @dataclass(frozen=True)
 class Measurement:
     ctx: int
@@ -91,12 +108,43 @@ def choose_cache(
     return "q4_0", []
 
 
-def parse_buffers(log: str) -> tuple[int, int]:
-    model = _MODEL.search(log)
-    kv = _KV.search(log)
-    if model is None or kv is None:
-        raise StageError("llama-server log has no model/KV buffer sizes", code="measure_log")
-    return int(float(model.group(1)) * _MIB), int(float(kv.group(1)) * _MIB)
+def _cuda_device(name: str) -> bool:
+    folded = name.lower()
+    if folded.endswith("_host"):
+        return False
+    return folded.startswith("cuda")
+
+
+def _mib_bytes(mib: float) -> int:
+    return int(round(mib * _MIB))
+
+
+def parse_buffers(log: str) -> BufferReport:
+    """Sum CUDA* model and KV buffers. CPU_Mapped and *_Host are not the card."""
+    weights: dict[str, float] = {}
+    kv: dict[str, float] = {}
+    for match in _DEVICE_MODEL.finditer(log):
+        device = match.group("device")
+        if not _cuda_device(device):
+            continue
+        weights[device] = weights.get(device, 0.0) + float(match.group("mib"))
+    for match in _DEVICE_KV.finditer(log):
+        device = match.group("device")
+        if not _cuda_device(device):
+            continue
+        kv[device] = kv.get(device, 0.0) + float(match.group("mib"))
+    if not weights or not kv:
+        raise StageError("llama-server log has no CUDA model/KV buffer sizes", code="measure_log")
+    names = sorted(set(weights) | set(kv))
+    devices = {
+        name: {"weights_mib": weights.get(name, 0.0), "kv_mib": kv.get(name, 0.0)}
+        for name in names
+    }
+    return BufferReport(
+        weights_bytes=_mib_bytes(sum(weights.values())),
+        kv_bytes=_mib_bytes(sum(kv.values())),
+        devices=devices,
+    )
 
 
 def measurement_fits(
diff --git a/src/voxprint_forge/pipeline.py b/src/voxprint_forge/pipeline.py
index 71afd59..b6a6958 100644
--- a/src/voxprint_forge/pipeline.py
+++ b/src/voxprint_forge/pipeline.py
@@ -356,9 +356,18 @@ def _file_record(dest: Path, tier: str, quant: str) -> dict[str, object]:
 
 def _run_quantize(host: Host, gguf: Path, imatrix: Path, dest: Path, quant: str) -> None:
     dest.parent.mkdir(parents=True, exist_ok=True)
-    if dest.is_file():
+    sidecar = Path(str(dest) + ".sha256")
+    # A failed llama-quantize leaves a partial GGUF. The next attempt must not keep it.
+    if dest.is_file() and sidecar.is_file():
         return
-    host.runner.run(quantize_argv(tool("llama-quantize"), gguf, dest, quant, imatrix))
+    if dest.exists():
+        dest.unlink()
+    try:
+        host.runner.run(quantize_argv(tool("llama-quantize"), gguf, dest, quant, imatrix))
+    except StageError:
+        if dest.exists():
+            dest.unlink()
+        raise
     if not dest.is_file():
         raise StageError(f"quantize did not write {dest.name}", code="quantize")
     write_sha256(dest, sha256_file(dest))
@@ -510,10 +519,10 @@ def _measure(
             cache=cache,
         )
         done = host.runner.capture_until(argv, pattern="KV buffer size")
-        weights, kv = parse_buffers(done.stdout + done.stderr)
+        report = parse_buffers(done.stdout + done.stderr)
         fits = measurement_fits(
-            weights,
-            kv,
+            report.weights_bytes,
+            report.kv_bytes,
             budget_gb=tier.budget_gb,
             headroom_gb=plan.headroom_gb,
         )
@@ -521,8 +530,9 @@ def _measure(
             {
                 "ctx": ctx,
                 "cache": cache,
-                "weights_bytes": weights,
-                "kv_bytes": kv,
+                "weights_bytes": report.weights_bytes,
+                "kv_bytes": report.kv_bytes,
+                "devices": report.devices,
                 "fits": fits,
                 "nvidia_used_bytes": host.nvidia_used(),
             }
