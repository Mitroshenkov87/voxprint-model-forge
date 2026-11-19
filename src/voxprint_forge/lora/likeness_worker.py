"""SFace face embeddings for the likeness score (YuNet detector + SFace recognizer, OpenCV Zoo).

Runs outside the caller's process, one of two ways:

* as a ``voxprint_common.worker`` handler, ``voxprint_forge.lora.likeness_worker:run_job``;
* as a script, ``python likeness_worker.py`` with one JSON object on stdin,
  ``{"detector": ONNX, "recognizer": ONNX, "paths": [...]}``, and one JSON object on stdout,
  ``{"rows": [...]}`` or ``{"error": "..."}``.

Each row is ``{"embedding": [128 floats, L2-normalised] or null, "face_px": N}``. The largest
face in a picture is used. A picture with no face, or one that cannot be read, has a null
embedding; it is never an error. Imports nothing from voxprint_forge, so it can run under any
Python that has OpenCV.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence

FACE_SCORE = 0.6
RETRY_SCORE = 0.3
MAX_SIDE = 1600


def _read(cv2, path: str):
    image = cv2.imread(path)
    if image is not None:
        return image
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as opened:
            return cv2.cvtColor(np.array(opened.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001 - an unreadable picture is a row with no face
        return None


def _largest(faces):
    return max(faces, key=lambda row: float(row[2]) * float(row[3]))


def embed_one(cv2, detector, recognizer, path: str) -> dict[str, object]:
    """One row for ``path``. A second, lower detector score is tried before "no face"."""
    image = _read(cv2, path)
    if image is None:
        return {"embedding": None, "face_px": 0}
    scale = MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    detector.setInputSize((image.shape[1], image.shape[0]))
    faces = None
    for score in (FACE_SCORE, RETRY_SCORE):
        detector.setScoreThreshold(score)
        _, faces = detector.detect(image)
        if faces is not None and len(faces):
            break
    detector.setScoreThreshold(FACE_SCORE)
    if faces is None or not len(faces):
        return {"embedding": None, "face_px": 0}
    face = _largest(faces)
    feature = recognizer.feature(recognizer.alignCrop(image, face)).flatten()
    norm = float((feature * feature).sum()) ** 0.5
    if norm <= 0:
        return {"embedding": None, "face_px": 0}
    return {
        "embedding": [float(value) / norm for value in feature],
        "face_px": round((float(face[2]) * float(face[3])) ** 0.5),
    }


def embed_paths(detector_path: str, recognizer_path: str, paths: Sequence[str]) -> list[dict[str, object]]:
    """One row per path, in order."""
    import cv2

    detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), FACE_SCORE)
    recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
    return [embed_one(cv2, detector, recognizer, str(path)) for path in paths]


def run_job(ctx) -> dict[str, object]:
    """``voxprint_common.worker`` handler. ``ctx.payload`` is the same object the script reads."""
    payload = ctx.payload
    if not isinstance(payload, dict):
        raise TypeError("likeness payload must be an object")
    return {"rows": embed_paths(str(payload["detector"]), str(payload["recognizer"]), list(payload["paths"]))}


def main() -> int:
    """Read one request on stdin, write one reply on stdout."""
    # The reply goes on a private copy of stdout; anything a library prints goes to stderr.
    reply = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        request = json.loads(sys.stdin.read())
        rows = embed_paths(str(request["detector"]), str(request["recognizer"]), list(request["paths"]))
        reply.write(json.dumps({"rows": rows}))
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which records a skipped score
        reply.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}))
    reply.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
