"""MiVOLO v2 age worker, run in its own Python env (MiVOLO pins transformers 4.51).

Reads one JSON object per line on stdin, ``{"path": "<png>"}``, and writes one
per line on stdout, ``{"n_faces": N, "ages": [...]}`` or ``{"error": "..."}``.
Arguments: ``MIVOLO_DIR YUNET_ONNX DEVICE``. Imports nothing from voxprint_forge.
"""

from __future__ import annotations

import json
import os
import sys

FACE_SCORE = 0.6
MAX_SIDE = 1280


def main(argv: list[str]) -> int:
    """Serve age requests until stdin closes."""
    folder, detector, device = argv[1], argv[2], argv[3]
    # Replies go on a private copy of stdout; anything a library prints (Python or C level) goes to stderr,
    # so a stray line can never be read as, or shift, a reply.
    reply = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"  # the Windows installer gives this env a CPU-only torch
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForImageClassification.from_pretrained(folder, trust_remote_code=True, torch_dtype=dtype)
    model = model.to(device).eval()
    processor = AutoImageProcessor.from_pretrained(folder, trust_remote_code=True)
    yunet = cv2.FaceDetectorYN.create(detector, "", (320, 320), FACE_SCORE, 0.3, 50)
    print(json.dumps({"ready": True}), file=reply, flush=True)
    for line in sys.stdin:
        try:
            image = cv2.imread(json.loads(line)["path"])
            if image is None:
                raise ValueError("unreadable image")
            height, width = image.shape[:2]
            scale = min(1.0, MAX_SIDE / max(height, width))
            small = image if scale == 1.0 else cv2.resize(image, (round(width * scale), round(height * scale)))
            yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = yunet.detect(small)
            crops = []
            for face in [] if faces is None else faces:
                x, y, w, h = (int(round(v / scale)) for v in face[:4])
                crop = image[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]
                if crop.size:
                    crops.append(crop)
            if not crops:
                print(json.dumps({"n_faces": 0, "ages": []}), file=reply, flush=True)
                continue
            face_px = processor(images=crops)["pixel_values"].to(device, dtype)
            body_px = processor(images=[None] * len(crops))["pixel_values"].to(device, dtype)
            with torch.inference_mode():
                out = model(faces_input=face_px, body_input=body_px)
            ages = [round(float(a), 2) for a in out.age_output.float().flatten().tolist()]
            print(json.dumps({"n_faces": len(crops), "ages": ages}), file=reply, flush=True)
        except Exception as exc:  # noqa: BLE001 - reported per request; the caller drops the photo
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=reply, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
