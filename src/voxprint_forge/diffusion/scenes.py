"""Synthetic CC0-style scenes and control maps. No photographs and no real people."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def render_scene(seed: int, size: int = 1024) -> torch.Tensor:
    """RGB float tensor, shape (3, H, W), values in 0..1. Geometric shapes only."""
    gen = _generator(seed)
    yy = torch.linspace(0, 1, size).view(size, 1)
    xx = torch.linspace(0, 1, size).view(1, size)
    sky = torch.tensor([0.45, 0.62, 0.78]).view(3, 1, 1) * (1.0 - 0.35 * yy)
    ground = torch.tensor([0.35, 0.40, 0.28]).view(3, 1, 1)
    image = torch.where(yy > 0.62, ground.expand(3, size, size), sky.expand(3, size, size)).clone()
    for index in range(4):
        cy = 0.35 + 0.5 * torch.rand((), generator=gen).item()
        cx = 0.1 + 0.8 * torch.rand((), generator=gen).item()
        half_h = 0.06 + 0.12 * torch.rand((), generator=gen).item()
        half_w = 0.05 + 0.14 * torch.rand((), generator=gen).item()
        colour = torch.rand(3, generator=gen).view(3, 1, 1) * 0.6 + 0.2
        mask = (yy - cy).abs().le(half_h) & (xx - cx).abs().le(half_w)
        image = torch.where(mask, colour.expand_as(image), image)
    # A clothed figure is two rectangles, not a photograph.
    torso = (yy - 0.58).abs().le(0.08) & (xx - 0.5).abs().le(0.04)
    legs = (yy - 0.72).abs().le(0.06) & (xx - 0.5).abs().le(0.035)
    coat = torch.tensor([0.15, 0.22, 0.38]).view(3, 1, 1)
    image = torch.where(torso | legs, coat.expand_as(image), image)
    return image.clamp(0, 1)


def canny_map(rgb: torch.Tensor) -> torch.Tensor:
    """Sobel magnitude, repeated to three channels."""
    gray = rgb.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=gray.dtype)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=gray.dtype)
    gx = functional.conv2d(gray, kernel_x.view(1, 1, 3, 3), padding=1)
    gy = functional.conv2d(gray, kernel_y.view(1, 1, 3, 3), padding=1)
    mag = torch.sqrt(gx.square() + gy.square()).squeeze(0)
    peak = mag.max().clamp(min=1e-6)
    return (mag / peak).repeat(3, 1, 1).clamp(0, 1)


def depth_map(seed: int, size: int = 1024) -> torch.Tensor:
    """Closer shapes are brighter. This is a rendered depth image, not a photo."""
    gen = _generator(seed + 17)
    yy = torch.linspace(0, 1, size).view(size, 1)
    xx = torch.linspace(0, 1, size).view(1, size)
    depth = torch.full((size, size), 0.12)
    for _ in range(5):
        cy = torch.rand((), generator=gen).item()
        cx = torch.rand((), generator=gen).item()
        rad = 0.08 + 0.18 * torch.rand((), generator=gen).item()
        level = 0.35 + 0.6 * torch.rand((), generator=gen).item()
        mask = (yy - cy).square() + (xx - cx).square() <= rad * rad
        depth = torch.where(mask, torch.full_like(depth, level), depth)
    return depth.unsqueeze(0).repeat(3, 1, 1)


def _stamp(canvas: torch.Tensor, x: int, y: int, colour: torch.Tensor) -> None:
    height = canvas.shape[1]
    width = canvas.shape[2]
    if 0 <= x < width and 0 <= y < height:
        canvas[:, y, x] = colour


def _line(canvas: torch.Tensor, x0: int, y0: int, x1: int, y1: int, colour: torch.Tensor) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        _stamp(canvas, int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)), colour)


def pose_map(seed: int, size: int = 1024) -> torch.Tensor:
    """A coloured stick figure on black, drawn in-job. Not a photograph."""
    gen = _generator(seed + 29)
    canvas = torch.zeros(3, size, size)
    jitter = int(torch.randint(-size // 20, size // 20 + 1, (), generator=gen).item())
    cx = size // 2 + jitter
    head = size // 5
    shoulder = size // 3
    hip = size // 2
    knee = int(size * 0.72)
    foot = int(size * 0.9)
    span = size // 6
    limb = torch.tensor([0.95, 0.25, 0.2])
    torso = torch.tensor([0.2, 0.85, 0.35])
    _line(canvas, cx, head, cx, hip, torso)
    _line(canvas, cx - span, shoulder, cx + span, shoulder, limb)
    _line(canvas, cx - span, shoulder, cx - span - size // 10, hip, limb)
    _line(canvas, cx + span, shoulder, cx + span + size // 12, int(size * 0.62), limb)
    _line(canvas, cx, hip, cx - span // 2, knee, limb)
    _line(canvas, cx - span // 2, knee, cx - span // 3, foot, limb)
    _line(canvas, cx, hip, cx + span // 2, knee, limb)
    _line(canvas, cx + span // 2, knee, cx + span // 2, foot, limb)
    return canvas


def inpaint_mask(size: int = 1024) -> torch.Tensor:
    """White rectangle where the inpaint should repaint. Shape (3, H, W)."""
    mask = torch.zeros(3, size, size)
    y0, y1 = size // 3, size // 2
    x0, x1 = size // 2, int(size * 0.75)
    mask[:, y0:y1, x0:x1] = 1
    return mask


def downscale(rgb: torch.Tensor, size: int = 128) -> torch.Tensor:
    scaled = functional.interpolate(rgb.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
    return scaled.squeeze(0).clamp(0, 1)


def contact_sheet(cells: list[torch.Tensor], columns: int) -> torch.Tensor:
    if not cells:
        raise ValueError("contact sheet has no cells")
    columns = max(1, columns)
    channels, height, width = cells[0].shape
    rows = (len(cells) + columns - 1) // columns
    sheet = torch.zeros(channels, rows * height, columns * width)
    for index, cell in enumerate(cells):
        row, col = divmod(index, columns)
        sheet[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = cell
    return sheet
