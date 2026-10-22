"""Fill the floor behind removed things in the floor's own top-down view, where painted lines are straight and evenly
spaced, then put it back into the image. Pixels whose ray meets a wall before the floor keep fill.py's fill.

usage: fill_floor.py <work dir> <run dir>   (lab venv; after fill.py and walls.py)
Writes filled_floor.jpg, topdown_fill.jpg (the top-down crop before and after) and compare_floor.jpg.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(4)
WORK, RUN = Path(sys.argv[1]), Path(sys.argv[2])
LAMA = Path(__file__).resolve().parent.parent / "models" / "lama" / "big-lama.pt"
PER_M = 50.0  # texels per meter
PAD = 3.0  # m of floor kept around the masked floor
MAX_DIST = 30.0  # m from the camera

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
walls = [w for w in json.loads((RUN / "walls.json").read_text()) if w["kind"] == "wall"]
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
H, W = image.shape[:2]
mask = cv2.imread(str(RUN / "fill_mask.png"), 0) > 127
filled = cv2.imread(str(RUN / "filled.jpg"))

# Each image pixel's ray meets the floor at s; it's floor if that comes before every wall
u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
d = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], -1) @ R_cw.T
s = -t_cw[1] / d[..., 1]
p = t_cw + d * s[..., None]
floor = (s > 0) & (np.linalg.norm(p - t_cw, axis=-1) < MAX_DIST)
for w in walls:
    n, off = np.array(w["normal"]), w["offset"]
    side = np.sign(off - t_cw @ n)  # the camera's side of the wall
    floor &= side * (off - p @ n) > 0.05
target = mask & floor
print(f"masked pixels: {mask.sum()}, of them floor: {target.sum()}", flush=True)

# The top-down grid around the masked floor
xz = p[target][:, [0, 2]]
x0, z0 = xz.min(0) - PAD
x1, z1 = xz.max(0) + PAD
X, Z = np.meshgrid(np.arange(x0, x1, 1 / PER_M), np.arange(z0, z1, 1 / PER_M))
world = np.stack([X, np.zeros_like(X), Z], -1)
c = (world - t_cw) @ R_cw
px = np.stack([cam["fx"] * c[..., 0] / c[..., 2] + cam["cx"], cam["fy"] * c[..., 1] / c[..., 2] + cam["cy"]], -1).astype(np.float32)
behind = c[..., 2] < 0.1
px[behind] = -1
top = cv2.remap(image, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderValue=0)
# A texel is unknown where its pixel is masked, or not floor, or outside the image
look = lambda a: cv2.remap(a.astype(np.uint8), px[..., 0], px[..., 1], cv2.INTER_NEAREST, borderValue=1)
unknown = (look(mask) > 0) | (look(~floor) > 0) | behind
unknown = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8))
th, tw = top.shape[:2]
print(f"top-down grid {tw}x{th} texels, {unknown.mean():.0%} unknown", flush=True)

model = torch.jit.load(str(LAMA), map_location="cpu").eval()
ph, pw = -th % 8, -tw % 8
rgb = np.pad(cv2.cvtColor(top, cv2.COLOR_BGR2RGB), ((0, ph), (0, pw), (0, 0)), mode="reflect")
m = np.pad(unknown, ((0, ph), (0, pw)), mode="reflect")
t = time.time()
with torch.inference_mode():
    out = model(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255, torch.from_numpy(m)[None, None].float())
print(f"LaMa: {time.time() - t:.0f} s", flush=True)
top_filled = cv2.cvtColor((out[0].permute(1, 2, 0).numpy()[:th, :tw] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(str(RUN / "topdown_fill.jpg"), np.vstack([np.where(unknown[..., None] > 0, 0, top), top_filled]))

# Back into the image: each target pixel's floor point, looked up in the filled top-down grid
mx = ((p[..., 0] - x0) * PER_M).astype(np.float32)
mz = ((p[..., 2] - z0) * PER_M).astype(np.float32)
back = cv2.remap(top_filled, mx, mz, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
out_img = filled.copy()
out_img[target] = back[target]
cv2.imwrite(str(RUN / "filled_floor.jpg"), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
crop = lambda i: i[360:600, 900:1700]
cv2.imwrite(str(RUN / "compare_floor.jpg"), np.vstack([crop(image), crop(filled), crop(out_img)]))
