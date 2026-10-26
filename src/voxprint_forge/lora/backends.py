"""Model backends for the gate and the captioner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict
from voxprint_forge.lora.presets import detect_vram_gb
from voxprint_forge.lora.te_quant import check_available, quant_config
from voxprint_forge.lora.vram_stage import (
    TIER_GB,
    allocated_gb,
    assert_stage_clear,
    hold_heavy,
    release_heavy,
    require_quantized,
)
from voxprint_forge.model_park import drop_vram, park, stage
from voxprint_forge.model_views import ensure_view

JUDGE_ID = "qwen3-vl-4b-instruct"
# Folder names installed before the #47 rename. Same weights; the new name wins when both exist.
LEGACY_IDS: dict[str, tuple[str, ...]] = {JUDGE_ID: ("qwen3-vl-4b",)}
ESTIMATOR_ID = "mivolo-v2"
FACE_DETECTOR_ID = "yunet-2023mar"
FACE_SCORE = 0.6
MAX_SIDE = 1280
_CUES = {
    "child_face": "childlike or adolescent facial proportions",
    "child_body": "childlike body proportions (short limbs, head large relative to body, small stature)",
    "undeveloped_figure": "undeveloped or prepubescent figure",
    "school_context": "child or school context (school uniform, classroom, school bag)",
    "child_props": "child props (toys, dolls, pacifier, childlike hair accessories)",
    "child_clothing": "childlike clothing styling",
    "minor_age_text": "text or a caption stating a minor age",
    "infant_or_putto": "an infant, baby, putto, cherub or cupid anywhere in the image",
    "childlike_body": "the visible body reads as childlike or adolescent",
    "lacks_adult_features": "clear adult body proportions are lacking",
}
assert tuple(_CUES) == INDICATORS
SYSTEM = (
    "You are a strict age-verification checker for a training-photo safety filter. Judge the apparent age of every "
    "person and report concrete age cues. Answer with a single JSON object and nothing else."
)
USER = (
    "Look at every person in this image, including partially visible or background people. Estimate the apparent age "
    'in years of the youngest-looking person. Answer each age cue with exactly one of "present", "absent" or '
    '"unsure":\n'
    + "".join(f"- {key}: {value}\n" for key, value in _CUES.items())
    + 'face_visible and person are "yes", "no" or "unsure". reason_code is one of "clearly_adult", "youthful_adult", '
    '"could_be_under_20", "child_or_adolescent", "no_person", "image_unclear". confidence is how sure you are of the '
    "cue answers (0 to 1). Return exactly this JSON: "
    '{"face_visible": ..., "person": ..., "min_apparent_age": <integer or null>, "confidence": ..., '
    '"any_person_could_be_under_20": <true or false>, "reason_code": ..., "indicators": {'
    + ", ".join(f'"{key}": ...' for key in INDICATORS)
    + "}}"
)


def model_dir(mdl_id: str) -> Path:
    """``<models dir>/model-forge/<mdl_id>``."""
    from voxprint_common.paths import models_dir

    ensure_view(mdl_id)
    return models_dir() / "model-forge" / mdl_id


def installed_dir(mdl_id: str) -> Path | None:
    """The installed folder for ``mdl_id``: the current name first, then any legacy name; ``None`` if neither."""
    ensure_view(mdl_id)
    for name in (mdl_id, *LEGACY_IDS.get(mdl_id, ())):
        folder = model_dir(name)
        if folder.is_dir():
            return folder
    return None


class JudgeParseError(ValueError):
    """The judge answered, but not with a usable verdict. ``code`` says why (no model text is kept)."""

    stage = "judge.parse"

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class WorkerError(RuntimeError):
    """The age worker failed. ``code`` is the worker's exception type or an IPC code; ``stage`` where."""

    def __init__(self, code: str, stage: str = "estimator.worker") -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage


def parse_verdict(raw: str) -> Verdict:
    """Parse the judge's JSON. Anything malformed raises ``JudgeParseError`` (the photo then drops)."""
    match = re.search(r"\{.*\}", raw, re.S)
    if match is None:
        raise JudgeParseError("no_json")
    try:
        data = json.loads(match.group(0))
    except ValueError:
        raise JudgeParseError("bad_json") from None
    if not isinstance(data, dict):
        raise JudgeParseError("bad_json")
    age = data.get("min_apparent_age")
    if age is not None and (isinstance(age, bool) or not isinstance(age, (int, float))):
        raise JudgeParseError("bad_age")
    indicators = data.get("indicators")
    if not isinstance(indicators, dict):
        raise JudgeParseError("no_indicators")
    return Verdict(
        face_visible=str(data.get("face_visible", "")),
        person=str(data.get("person", "")),
        min_age=None if age is None else float(age),
        confidence=data.get("confidence"),
        reason_code=data.get("reason_code"),
        could_be_under_20=data.get("any_person_could_be_under_20"),
        indicators={str(k): str(v) for k, v in indicators.items()},
    )


def _qwen_key(folder: Path, mode: str) -> str:
    return f"qwen3-vl:{Path(folder).resolve()}:{mode}"


def _load_qwen(folder: Path, device: str, mode: str) -> dict[str, object]:
    """Read Qwen3-VL once. The caller parks the result for the rest of the session."""
    # Imported here so ``voxprint-forge --version`` does not pull torch or transformers.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(folder))
    kwargs: dict[str, object] = {"torch_dtype": torch.bfloat16}
    if mode in ("int8", "nf4"):
        kwargs["quantization_config"] = quant_config(mode)
        if str(device).startswith("cuda"):
            kwargs["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(str(folder), **kwargs)
    if mode == "none":
        model = model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "quant": mode}


class VlmJudge:
    """Qwen3-VL-4B-Instruct (Apache-2.0) as the age-cue judge.

    The judge and the captioner share one parked copy, keyed by folder and quant.
    ``unload`` drops the graphics copy and leaves the RAM copy in the park.
    On a 16 GB card the weights load int8; bf16 and a CPU device raise.
    """

    name = JUDGE_ID
    max_new_tokens = 320

    def __init__(self, folder: Path, device: str = "cuda", quant: str | None = None) -> None:
        tier = detect_vram_gb()
        mode = quant if quant is not None else ("int8" if tier == TIER_GB else "none")
        if tier == TIER_GB and mode == "none":
            require_quantized(tier, False, JUDGE_ID)
        if tier == TIER_GB and not str(device).startswith("cuda"):
            raise LoraStageError("lora.err.cpu_fallback", tier=TIER_GB, reason="no CUDA device")
        if mode in ("int8", "nf4"):
            check_available(mode)
        # Before the weights exist, so the guard can subtract the CUDA context.
        baseline_gb = allocated_gb()
        key = _qwen_key(folder, mode)
        hold_heavy("qwen", tier=tier)
        try:
            payload = park(key, lambda: _load_qwen(folder, device, mode), stage_name="caption")
        except Exception:
            release_heavy("qwen")
            raise
        self.__dict__.update(payload)
        self._park_key = key
        self._tier = tier
        self._baseline_gb = baseline_gb
        self.last_stats = {}
        # The first int8 load is already on the card. A later stage has to move
        # the parked CPU copy back, including state.CB which module.to skips.
        stage(self.model, device, key=key, stage_name="caption")
        if mode != "none":
            stage(self.processor, device, key=key, stage_name="caption")

    def ask(self, image: Any, system: str, user: str, max_new_tokens: int) -> str:
        """One greedy answer for ``image``."""
        # Imported here so ``voxprint-forge --version`` does not pull torch.
        import torch

        image = _limit(image)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(new, skip_special_tokens=True)[0]
        self.last_stats = {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "new_tokens": int(new.shape[1]),
            "hit_token_limit": int(new.shape[1]) >= max_new_tokens,
            "chars": len(text),
        }
        return text

    def judge(self, image: Any) -> Verdict:
        """The verdict for one photo."""
        self.last_stats = {}
        return parse_verdict(self.ask(image, SYSTEM, USER, self.max_new_tokens))

    def unload(self) -> None:
        """Drop the graphics copy. The parked weights stay in RAM for the next stage."""
        model = getattr(self, "model", None)
        if model is not None:
            drop_vram(model, key=getattr(self, "_park_key", JUDGE_ID), stage_name="caption")
        release_heavy("qwen")
        assert_stage_clear(
            "caption",
            tier=getattr(self, "_tier", 24),
            baseline_gb=getattr(self, "_baseline_gb", 0.0),
        )


def mivolo_python() -> str:
    """Interpreter for the MiVOLO worker.

    ``VOXPRINT_MIVOLO_PYTHON`` wins. Otherwise a ``venv-mivolo`` folder beside
    this env (the Windows installer layout: ``<app>/venv`` and
    ``<app>/venv-mivolo``) is used when it has a Python. Else this interpreter.
    """
    import os
    import sys

    explicit = os.environ.get("VOXPRINT_MIVOLO_PYTHON", "").strip()
    if explicit:
        return explicit
    sibling = Path(sys.prefix).parent / "venv-mivolo"
    for candidate in (sibling / "Scripts" / "python.exe", sibling / "bin" / "python"):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


class MivoloEstimator:
    """MiVOLO v2 (Apache-2.0) on YuNet face crops (OpenCV Zoo, MIT), in a worker process.

    MiVOLO pins transformers 4.51, which Qwen3-VL cannot use, so the worker runs
    in its own env, found by :func:`mivolo_python`.
    """

    name = ESTIMATOR_ID

    def __init__(self, folder: Path, detector: Path, device: str = "cuda", python: str | None = None) -> None:
        import subprocess
        import tempfile

        worker = Path(__file__).resolve().parent / "mivolo_worker.py"
        exe = python or mivolo_python()
        self.tmp = Path(tempfile.mkdtemp(prefix="vxp-mivolo-"))
        self.proc = subprocess.Popen(
            [exe, str(worker), str(folder), str(detector), device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self.proc.stdout.readline() if self.proc.stdout else ""
        if not ready or not json.loads(ready).get("ready"):
            self.unload()
            raise RuntimeError("MiVOLO worker did not start")

    def estimate(self, image: Any) -> AgeReading:
        """Ages of every detected face. Any worker error raises (the photo then drops)."""
        path = self.tmp / "frame.png"
        image.convert("RGB").save(path)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps({"path": str(path)}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise WorkerError(f"worker_exited_rc{self.proc.poll()}", "estimator.ipc")
        try:
            reply = json.loads(line)
        except ValueError:
            raise WorkerError("bad_reply", "estimator.ipc") from None
        if not isinstance(reply, dict):
            raise WorkerError("bad_reply", "estimator.ipc")
        if "error" in reply:
            raise WorkerError(str(reply["error"]).split(":", 1)[0][:64] or "worker_error")
        if "n_faces" not in reply:
            raise WorkerError("no_reply", "estimator.ipc")
        return AgeReading(int(reply["n_faces"]), tuple(float(a) for a in reply["ages"]))

    def unload(self) -> None:
        """Stop the worker."""
        import shutil

        if self.proc.poll() is None:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)


def load_gate(device: str = "cuda") -> Gate:
    judge = estimator = None
    try:
        folder = installed_dir(JUDGE_ID)
        if folder is not None:
            judge = VlmJudge(folder, device)
    except LoraStageError:
        raise
    except Exception as exc:  # noqa: BLE001
        _report("judge", exc)
        judge = None
    try:
        folder, detector = model_dir(ESTIMATOR_ID), model_dir(FACE_DETECTOR_ID) / "face_detection_yunet_2023mar.onnx"
        if folder.is_dir() and detector.is_file():
            estimator = MivoloEstimator(folder, detector, device)
    except Exception as exc:  # noqa: BLE001
        _report("estimator", exc)
        estimator = None
    return Gate(judge, estimator)


def _report(part: str, exc: Exception) -> None:
    import sys

    print(f"safety gate {part} did not load: {type(exc).__name__}", file=sys.stderr)


def _limit(image: Any) -> Any:
    if max(image.size) <= MAX_SIDE:
        return image
    scale = MAX_SIDE / max(image.size)
    return image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))))
