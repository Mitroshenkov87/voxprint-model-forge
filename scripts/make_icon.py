"""Draw the VOXPrint AI Model Forge icon master (suite 'Icon style') and pack the .ico.

Tile: rounded square, 4% margin, radius 18% of side, #1b1b23 -> #2c2c38 lift.
Glyph: flat anvil, ~60% of the tile, identity colour proposed #f58a3c (ember orange),
one soft highlight band on the anvil face. No text.

Outputs (repo assets, not the wheel):
  assets/voxprint-model-forge.png
  assets/voxprint-model-forge.ico
  assets/icon/frames/{16,24,32,48,64,128,256}.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
S = 4  # supersample
N = 1024 * S
IDENTITY = (0xF5, 0x8A, 0x3C)
HILITE = (0xFF, 0xB3, 0x78)


def bez(p0, p1, p2, p3, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3; b = 3 * (1 - t) ** 2 * t; c = 3 * (1 - t) * t * t; d = t ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def anvil(dy=-8, k=1.1):
    p = []
    p += bez((214, 366), (250, 352), (290, 348), (340, 348))  # horn top up to the face
    p += [(802, 348)]                                    # flat top
    p += [(802, 446)]                                    # right face
    p += bez((802, 446), (740, 452), (690, 480), (690, 528))  # right shoulder into waist
    p += [(690, 566)]
    p += bez((690, 566), (690, 610), (740, 628), (790, 636))  # right foot flare
    p += [(790, 694), (334, 694), (334, 636)]            # base
    p += bez((334, 636), (384, 628), (434, 610), (434, 566))  # left foot flare
    p += [(434, 528)]
    p += bez((434, 528), (434, 486), (404, 462), (366, 452))  # left shoulder under the horn
    p += bez((366, 452), (296, 440), (244, 410), (214, 366))  # horn underside back to tip
    cx, cy = 508, 521
    return [((cx + (x - cx) * k) * S, (cy + (y - cy) * k + dy) * S) for x, y in p]


def tile():
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    grad = Image.new("RGBA", (N, N))
    top, bot = (0x1B, 0x1B, 0x23), (0x2C, 0x2C, 0x38)
    gd = ImageDraw.Draw(grad)
    for y in range(N):
        t = y / (N - 1)
        gd.line([(0, y), (N, y)], fill=tuple(round(top[i] + (bot[i] - top[i]) * t) for i in range(3)) + (255,))
    m = round(0.04 * N); side = N - 2 * m
    mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(mask).rounded_rectangle([m, m, N - m - 1, N - m - 1], radius=round(0.18 * side), fill=255)
    img.paste(grad, (0, 0), mask)
    return img


def master():
    img = tile()
    glyph = Image.new("L", (N, N), 0)
    ImageDraw.Draw(glyph).polygon(anvil(), fill=255)
    img.paste(Image.new("RGBA", (N, N), IDENTITY + (255,)), (0, 0), glyph)
    # one soft highlight: a lighter band along the anvil face, clipped to the glyph
    band = Image.new("L", (N, N), 0)
    ImageDraw.Draw(band).rectangle([150 * S, 300 * S, 900 * S, 352 * S], fill=170)
    band = band.filter(ImageFilter.GaussianBlur(6 * S))
    band = Image.composite(band, Image.new("L", (N, N), 0), glyph)
    img.paste(Image.new("RGBA", (N, N), HILITE + (255,)), (0, 0), band)
    return img.resize((1024, 1024), Image.LANCZOS)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    m = master()
    m.save(ASSETS / "voxprint-model-forge.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = []
    frame_dir = ASSETS / "icon" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for s in sizes:
        f = m.resize((s, s), Image.LANCZOS)
        f.save(frame_dir / f"{s}.png")
        frames.append(f)
    m.save(ASSETS / "voxprint-model-forge.ico", format="ICO", sizes=[(s, s) for s in sizes])
    print("ok")
