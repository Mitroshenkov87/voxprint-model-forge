"""Every gate stage reports what happened per photo (ids and codes only); a failure still drops the photo."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
from PIL import Image

from voxprint_forge.lora.backends import JudgeParseError, MivoloEstimator, WorkerError, parse_verdict
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

ADULT = Verdict("yes", "yes", 34.0, 0.9, "clearly_adult", False, {k: "absent" for k in INDICATORS})
IMAGE = Image.new("RGB", (16, 16))


class Judge:
    name = "judge"

    def __init__(self, verdict=ADULT, exc=None, stats=None):
        self.verdict, self.exc = verdict, exc
        self.last_stats = stats or {}

    def judge(self, image):
        if self.exc is not None:
            raise self.exc
        return self.verdict


class Estimator:
    name = "estimator"

    def __init__(self, reading=AgeReading(1, (31.0,)), exc=None):
        self.reading, self.exc = reading, exc

    def estimate(self, image):
        if self.exc is not None:
            raise self.exc
        return self.reading


def test_keep_records_both_stages() -> None:
    decision = Gate(Judge(stats={"new_tokens": 120, "hit_token_limit": False}), Estimator()).check(IMAGE)
    assert decision.keep
    assert decision.trace["judge"]["ok"] and decision.trace["judge"]["new_tokens"] == 120
    assert decision.trace["judge"]["verdict"]["indicators"] == {k: "absent" for k in INDICATORS}
    assert decision.trace["estimator"] == {"ok": True, "n_faces": 1, "ages": [31.0]}


def test_judge_parse_error_is_a_drop_with_its_code() -> None:
    decision = Gate(Judge(exc=JudgeParseError("no_json"), stats={"hit_token_limit": True}), Estimator()).check(IMAGE)
    assert not decision.keep and decision.reasons == ("judge_error",)
    assert decision.trace["judge"] == {
        "ok": False,
        "stage": "judge.parse",
        "error": "JudgeParseError",
        "code": "no_json",
        "hit_token_limit": True,
    }


def test_judge_crash_records_type_not_message() -> None:
    decision = Gate(Judge(exc=RuntimeError("/secret/path/frame.png exploded")), Estimator()).check(IMAGE)
    assert not decision.keep
    assert decision.trace["judge"] == {"ok": False, "stage": "judge", "error": "RuntimeError"}
    assert "secret" not in json.dumps(decision.trace)


def test_estimator_failure_is_recorded_and_still_drops_a_face_photo() -> None:
    decision = Gate(Judge(), Estimator(exc=WorkerError("OutOfMemoryError"))).check(IMAGE)
    assert not decision.keep and "estimator_missing" in decision.reasons
    assert decision.trace["estimator"] == {
        "ok": False,
        "stage": "estimator.worker",
        "error": "WorkerError",
        "code": "OutOfMemoryError",
    }


def test_trace_cuts_free_text_from_the_verdict() -> None:
    chatty = replace(ADULT, reason_code="a long sentence describing the whole picture in detail")
    decision = Gate(Judge(chatty), Estimator()).check(IMAGE)
    assert len(decision.trace["judge"]["verdict"]["reason_code"]) <= 24


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("no braces here", "no_json"),
        ('{"a": 1,}', "bad_json"),
        ('{"min_apparent_age": "x"}', "bad_age"),
        ('{"min_apparent_age": 30}', "no_indicators"),
        ('{"face_visible": "yes", "indicators": {"child_face"', "no_json"),
    ],
)
def test_parse_codes(raw, code) -> None:
    with pytest.raises(JudgeParseError) as err:
        parse_verdict(raw)
    assert err.value.code == code and isinstance(err.value, ValueError)


def test_prepare_writes_the_trace_and_keeps_drops_jsonl_neutral(tmp_path, capsys) -> None:
    import numpy as np

    files = []
    (tmp_path / "in").mkdir()
    for index in range(6):
        path = tmp_path / "in" / f"img{index}.png"
        rng = np.random.default_rng(index)
        Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)).save(path)
        files.append(path)
    trace = tmp_path / "out" / "gate.jsonl"
    subject = check_subject("person", name="Demo", consent_obtained=True, subject_adult=True)
    with pytest.raises(Exception):
        prepare(
            Project(tmp_path / "proj"),
            files,
            subject,
            gate=Gate(Judge(), Estimator(exc=WorkerError("bad_reply", "estimator.ipc"))),
            load_captioner=lambda choice: None,
            progress=Progress(),
            mode="expert",
            trace=trace,
        )
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 6 and all(not r["keep"] and r["estimator"]["code"] == "bad_reply" for r in rows)
    assert all("img" not in json.dumps(r) for r in rows)
    drops = [json.loads(line) for line in (tmp_path / "proj" / "drops.jsonl").read_text().splitlines()]
    assert {d["reason"] for d in drops} == {"lora.drop.safety"} and all(
        set(d) == {"id", "name", "reason"} for d in drops
    )
    err = capsys.readouterr().err
    assert err.count("safety gate estimator.ipc failed on") == 6 and "WorkerError code=bad_reply" in err


NOISY = """#!{python}
import json, sys
print(json.dumps({{"ready": True}}), flush=True)
for n, line in enumerate(sys.stdin):
    err = json.dumps({{"error": "OutOfMemoryError: CUDA out of memory"}})
    print("a stray library line" if n == 0 else err, flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shebang script as the worker interpreter")
def test_worker_ipc_errors_have_codes(tmp_path) -> None:
    fake = tmp_path / "fake-python"
    fake.write_text(NOISY.format(python=sys.executable))
    fake.chmod(0o755)
    estimator = MivoloEstimator(tmp_path, tmp_path / "yunet.onnx", "cpu", python=str(fake))
    with pytest.raises(WorkerError) as bad:
        estimator.estimate(IMAGE)
    assert (bad.value.code, bad.value.stage) == ("bad_reply", "estimator.ipc")
    with pytest.raises(WorkerError) as oom:
        estimator.estimate(IMAGE)
    assert (oom.value.code, oom.value.stage) == ("OutOfMemoryError", "estimator.worker")
    estimator.unload()


STUBS = {
    "cv2.py": "class FaceDetectorYN:\n    @staticmethod\n    def create(*a):\n        return None\n",
    "torch.py": (
        "float16 = 'f16'\nfloat32 = 'f32'\n"
        "class cuda:\n    @staticmethod\n    def is_available():\n        return False\n"
    ),
    "transformers.py": (
        "import os\n"
        "class _M:\n"
        "    def to(self, *a):\n        return self\n"
        "    def eval(self):\n"
        "        print('weights banner on stdout')\n"
        "        os.write(1, b'C-level banner\\n')\n"
        "        return self\n"
        "class AutoModelForImageClassification:\n"
        "    @staticmethod\n    def from_pretrained(*a, **k):\n        return _M()\n"
        "class AutoImageProcessor:\n"
        "    @staticmethod\n    def from_pretrained(*a, **k):\n        return None\n"
    ),
}


def test_real_worker_keeps_replies_off_library_stdout(tmp_path) -> None:
    """Python- and C-level prints during load go to stderr; stdout carries only protocol lines."""
    import os
    import subprocess

    from voxprint_forge.lora import mivolo_worker

    for name, text in STUBS.items():
        (tmp_path / name).write_text(text)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, mivolo_worker.__file__, "dir", "yunet", "cpu"],
        input=json.dumps({"path": str(tmp_path / "missing.png")}) + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    lines = done.stdout.splitlines()
    assert lines[0] == '{"ready": true}' and len(lines) == 2
    assert json.loads(lines[1])["error"].startswith("AttributeError")
    assert "weights banner" in done.stderr and "C-level banner" in done.stderr
