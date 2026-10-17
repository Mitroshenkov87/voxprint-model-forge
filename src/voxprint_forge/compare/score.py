"""LPIPS, CLIP, and DINOv2 scorers. Models load on the GPU path only."""

from __future__ import annotations

import torch
from PIL import Image

from voxprint_forge.compare.cases import Case
from voxprint_forge.compare.metrics import PairScore, finish_score, ssim_psnr


def make_scorer():
    """Build the live metric bundle once and return a pair scorer."""
    bundle = LiveMetrics()

    def score(reference: Image.Image, candidate: Image.Image, case: Case) -> PairScore:
        return bundle.score(reference, candidate, case.prompt)

    return score


class LiveMetrics:
    """Alex LPIPS, CLIP ViT-B/32, and DINOv2-base on one device."""

    def __init__(self) -> None:
        import lpips
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.lpips = lpips.LPIPS(net="alex").to(self.device)
        self.lpips.eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino.eval()

    def score(self, reference: Image.Image, candidate: Image.Image, prompt: str) -> PairScore:
        """Score one pair. ``prompt`` is used for CLIP-T and is not returned."""
        left = reference.convert("RGB")
        right = candidate.convert("RGB")
        ssim, psnr, saturated = ssim_psnr(_tensor(left), _tensor(right))
        return finish_score(
            lpips=self._lpips(left, right),
            ssim=ssim,
            psnr=psnr,
            psnr_saturated=saturated,
            dino_cosine=self._dino(left, right),
            clip_i=self._clip_image(left, right),
            clip_t_reference=self._clip_text(left, prompt),
            clip_t_candidate=self._clip_text(right, prompt),
        )

    def _lpips(self, left: Image.Image, right: Image.Image) -> float:
        def prep(image: Image.Image) -> torch.Tensor:
            return _tensor(image).unsqueeze(0).to(self.device) * 2 - 1

        with torch.no_grad():
            return float(self.lpips(prep(left), prep(right)).mean())

    def _clip_text(self, image: Image.Image, text: str) -> float:
        image_inputs = self.clip_processor(images=image, return_tensors="pt")
        text_inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        pixels = image_inputs["pixel_values"].to(self.device)
        input_ids = text_inputs["input_ids"].to(self.device)
        mask = text_inputs.get("attention_mask")
        attention = mask.to(self.device) if mask is not None else None
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=pixels)
            text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention)
        return _cosine(_feature(image_features), _feature(text_features))

    def _clip_image(self, left: Image.Image, right: Image.Image) -> float:
        inputs = self.clip_processor(images=[left, right], return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.clip.get_image_features(pixel_values=pixels)
        rows = _rows(features)
        return _cosine(rows[0], rows[1])

    def _dino(self, left: Image.Image, right: Image.Image) -> float:
        inputs = self.dino_processor(images=[left, right], return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            hidden = self.dino(pixel_values=pixels).last_hidden_state[:, 0]
        return _cosine(hidden[0], hidden[1])


def _tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np

    array = np.asarray(image.convert("RGB"))
    copied = torch.from_numpy(np.array(array, copy=True))
    return copied.permute(2, 0, 1).float() / 255.0


def _feature(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value[0]
    pooled = getattr(value, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled[0]
    embeds = getattr(value, "image_embeds", None)
    if isinstance(embeds, torch.Tensor):
        return embeds[0]
    raise RuntimeError("CLIP returned no features")


def _rows(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    pooled = getattr(value, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    embeds = getattr(value, "image_embeds", None)
    if isinstance(embeds, torch.Tensor):
        return embeds
    raise RuntimeError("CLIP returned no features")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_n = torch.nn.functional.normalize(left.float().flatten(), dim=0)
    right_n = torch.nn.functional.normalize(right.float().flatten(), dim=0)
    return float((left_n * right_n).sum())
