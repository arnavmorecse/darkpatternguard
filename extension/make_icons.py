"""Draws the extension icons (red rounded square, white magnifier with an exclamation mark).
Run: python make_icons.py   (writes icons/icon16.png ... icon128.png next to this file)"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "icons"
OUT.mkdir(exist_ok=True)
RED, WHITE = (214, 41, 62, 255), (255, 255, 255, 255)

for size in (16, 32, 48, 128):
    S = size * 8                                   # draw big, then downsample for smooth edges
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=RED)
    lw = int(S * (0.11 if size <= 16 else 0.085))  # thicker strokes at tiny sizes
    cx, cy, r = S * 0.44, S * 0.42, S * 0.25       # magnifier lens
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=lw)
    hx, hy = cx + r * 0.72, cy + r * 0.72          # handle
    d.line([hx, hy, S * 0.80, S * 0.78], fill=WHITE, width=int(lw * 1.25))
    if size >= 32:                                 # "!" inside the lens
        w = lw * 0.7
        d.rounded_rectangle([cx - w / 2, cy - r * 0.55, cx + w / 2, cy + r * 0.15], radius=w / 2, fill=WHITE)
        d.ellipse([cx - w / 2, cy + r * 0.32, cx + w / 2, cy + r * 0.32 + w], fill=WHITE)
    im.resize((size, size), Image.LANCZOS).save(OUT / f"icon{size}.png")
print("icons written to", OUT)
