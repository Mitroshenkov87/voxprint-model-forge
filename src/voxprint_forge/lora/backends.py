"""Model backends for the safety gate and the captioner. Weights come from MDL; nothing is downloaded here.

These run on the user's GPU; CI tests use fakes. A backend that cannot load
returns ``None`` from its loader, which makes the gate unavailable and blocks
the run (fail closed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict

JUDGE_ID = "qwen3-vl-4b"
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

    return models_dir() / "model-forge" / mdl_id


def parse_verdict(raw: str) -> Verdict:
    """Parse the judge's JSON. Anything malformed raises ``ValueError`` (the photo then drops)."""
    match = re.search(r"\{.*\}", raw, re.S)
    if match is None:
        raise ValueError("no JSON object")
    data = json.loads(match.group(0))
    age = data.get("min_apparent_age")
    if age is not None and (isinstance(age, bool) or not isinstance(age, (int, float))):
        raise ValueError("bad age")
    indicators = data.get("indicators")
    if not isinstance(indicators, dict):
        raise ValueError("no indicators")
    return Verdict(
        face_visible=str(data.get("face_visible", "")),
        person=str(data.get("person", "")),
        min_age=None if age is None else float(age),
        confidence=data.get("confidence"),
        reason_code=data.get("reason_code"),
        could_be_under_20=data.get("any_person_could_be_under_20"),
        indicators={str(k): str(v) for k, v in indicators.items()},
    )


class VlmJudge:
    """Qwen3-VL-4B-Instruct (Apache-2.0) as the age-cue judge."""

    name = JUDGE_ID

    def __init__(self, folder: Path, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(folder))
        self.model = AutoModelForImageTextToText.from_pretrained(str(folder), torch_dtype=torch.bfloat16).to(device)
        self.model.eval()

    def ask(self, image: Any, system: str, user: str, max_new_tokens: int) -> str:
        """One greedy answer for ``image``."""
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
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    def judge(self, image: Any) -> Verdict:
        """The verdict for one photo."""
        return parse_verdict(self.ask(image, SYSTEM, USER, 320))

    def unload(self) -> None:
        """Free graphics memory."""
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MivoloEstimator:
    """MiVOLO v2 (Apache-2.0) on YuNet face crops (OpenCV Zoo, MIT)."""

    name = ESTIMATOR_ID

    def __init__(self, folder: Path, detector: Path, device: str = "cuda") -> None:
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.cv2 = cv2
        self.model = AutoModelForImageClassification.from_pretrained(
            str(folder), trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        self.processor = AutoImageProcessor.from_pretrained(str(folder), trust_remote_code=True)
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), FACE_SCORE, 0.3, 50)

    def estimate(self, image: Any) -> AgeReading:
        """Ages of every detected face."""
        import numpy as np
        import torch

        bgr = np.asarray(_limit(image).convert("RGB"))[:, :, ::-1].copy()
        height, width = bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(bgr)
        crops = []
        for face in [] if faces is None else faces:
            x, y, w, h = (int(round(v)) for v in face[:4])
            crop = bgr[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]
            if crop.size:
                crops.append(crop)
        if not crops:
            return AgeReading(0, ())
        pixels = self.processor(images=crops)["pixel_values"].to(self.model.device, self.model.dtype)
        with torch.no_grad():
            out = self.model(faces_input=pixels, body_input=None)
        return AgeReading(len(crops), tuple(float(a) for a in out.age_output.flatten().tolist()))

    def unload(self) -> None:
        """Free graphics memory."""
        del self.model


def load_gate(device: str = "cuda") -> Gate:
    """The installed gate. A missing model or library leaves that part ``None`` (the run is then blocked)."""
    judge = estimator = None
    try:
        folder = model_dir(JUDGE_ID)
        if folder.is_dir():
            judge = VlmJudge(folder, device)
    except Exception:  # noqa: BLE001 - an unloadable judge means no gate
        judge = None
    try:
        folder, detector = model_dir(ESTIMATOR_ID), model_dir(FACE_DETECTOR_ID) / "face_detection_yunet_2023mar.onnx"
        if folder.is_dir() and detector.is_file():
            estimator = MivoloEstimator(folder, detector, device)
    except Exception:  # noqa: BLE001 - an unloadable estimator means no gate
        estimator = None
    return Gate(judge, estimator)


def _limit(image: Any) -> Any:
    if max(image.size) <= MAX_SIDE:
        return image
    scale = MAX_SIDE / max(image.size)
    return image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))))
