"""The place behind removed things (the net, on-screen graphics), filled in by LaMa (big-lama, Apache-2.0) on the
scene stage's background image; cv2's inpainting alongside for comparison.

usage: fill.py <work dir> <out dir> <mask.png> [<mask.png> ...]   (lab venv)
Writes filled.jpg (LaMa), filled_cv2.jpg, fill_mask.png and compare.jpg.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(4)
WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MASKS = [Path(p) for p in sys.argv[3:]]
LAMA = Path(__file__).resolve().parent.parent / "models" / "lama" / "big-lama.pt"
GROW = 9  # px: masks grow by this, so edges and soft shadows go too

image = cv2.imread(str(WORK / "scene" / "background.jpg"))
H, W = image.shape[:2]
mask = np.zeros((H, W), np.uint8)
for p in MASKS:
    mask |= (cv2.imread(str(p), 0) > 127).astype(np.uint8)
mask = cv2.dilate(mask, np.ones((2 * GROW + 1,) * 2, np.uint8))
cv2.imwrite(str(OUT / "fill_mask.png"), mask * 255)

model = torch.jit.load(str(LAMA), map_location="cpu").eval()
ph, pw = -H % 8, -W % 8  # LaMa wants sides divisible by 8
rgb = np.pad(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), ((0, ph), (0, pw), (0, 0)), mode="reflect")
m = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
t = time.time()
with torch.inference_mode():
    out = model(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255, torch.from_numpy(m)[None, None].float())
print(f"LaMa: {time.time() - t:.0f} s for {W}x{H}", flush=True)
filled = (out[0].permute(1, 2, 0).numpy()[:H, :W] * 255).clip(0, 255).astype(np.uint8)
filled = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)
filled = np.where(mask[..., None] > 0, filled, image)  # outside the mask, keep the original exactly
cv2.imwrite(str(OUT / "filled.jpg"), filled, [cv2.IMWRITE_JPEG_QUALITY, 95])
simple = cv2.inpaint(image, mask, 9, cv2.INPAINT_TELEA)
cv2.imwrite(str(OUT / "filled_cv2.jpg"), simple, [cv2.IMWRITE_JPEG_QUALITY, 95])

# The masked areas' bounding box, three ways: original, cv2, LaMa
ys, xs = np.nonzero(mask)
crops = []
for img in (image, simple, filled):
    crops.append(cv2.resize(img, (960, 540)))
cv2.imwrite(str(OUT / "compare.jpg"), np.vstack(crops))
