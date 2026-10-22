"""The floor seen from straight above: each floor point (y = 0) on a grid, looked up in the background image.

usage: topdown.py <work dir> <out dir> [texels per meter=50]   (lab venv)
Writes topdown.png (x right, z down in the image) and topdown.json (the grid's world extent).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PER_M = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
OUT.mkdir(parents=True, exist_ok=True)
cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
H, W = image.shape[:2]

# Where the bottom and side image edges meet the floor bounds the grid; far rows are cut at 30 m
u, v = np.meshgrid(np.linspace(0, W, 64), np.linspace(0, H, 36))
d = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], -1) @ R_cw.T
s = -t_cw[1] / d[..., 1]
p = t_cw + d * s[..., None]
ok = (s > 0) & (np.linalg.norm(p - t_cw, axis=-1) < 30)
x0, z0 = p[ok][:, [0, 2]].min(0)
x1, z1 = p[ok][:, [0, 2]].max(0)
X, Z = np.meshgrid(np.arange(x0, x1, 1 / PER_M), np.arange(z0, z1, 1 / PER_M))
world = np.stack([X, np.zeros_like(X), Z], -1)
c = (world - t_cw) @ R_cw
front = c[..., 2] > 0.1
px = np.stack([cam["fx"] * c[..., 0] / c[..., 2] + cam["cx"], cam["fy"] * c[..., 1] / c[..., 2] + cam["cy"]], -1)
px[~front] = -1
top = cv2.remap(image, px[..., 0].astype(np.float32), px[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
cv2.imwrite(str(OUT / "topdown.png"), top)
info = dict(x=[float(x0), float(x1)], z=[float(z0), float(z1)], per_m=PER_M, camera=t_cw.round(3).tolist())
(OUT / "topdown.json").write_text(json.dumps(info, indent=1))
print(info, top.shape)
