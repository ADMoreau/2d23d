"""Render scene.npz meshes (painted from their texture, one color per triangle) from another viewpoint, side by side.

usage: render.py <out.jpg> <eye x,y,z> <target x,y,z> <scene dir> [<scene dir> ...]   (lab venv)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = sys.argv[1]
eye, target = (np.array([float(x) for x in a.split(",")]) for a in sys.argv[2:4])
W, H, F = 960, 540, 560.0
fwd = (target - eye) / np.linalg.norm(target - eye)
right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
views = []
for d in sys.argv[4:]:
    z = np.load(Path(d) / "scene.npz")
    tex = cv2.imread(str(Path(d) / "background.jpg"))
    img = np.full((H, W, 3), 40, np.uint8)
    for prefix in ("backdrop_", ""):
        V, Fc, UV = z[prefix + "vertices"], z[prefix + "faces"], z[prefix + "uvs"]
        c = (V - eye) @ np.stack([right, up, fwd], 1)
        ok = c[:, 2] > 0.1
        px = np.stack([F * c[:, 0] / np.maximum(c[:, 2], 0.1) + W / 2, -F * c[:, 1] / np.maximum(c[:, 2], 0.1) + H / 2], -1)
        faces = Fc[ok[Fc].all(1)]
        depth = c[faces, 2].mean(1)
        faces = faces[np.argsort(-depth)]
        uv = UV[faces].mean(1)
        color = tex[np.clip((uv[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1), np.clip((uv[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)]
        tri = np.round(px[faces] * 4).astype(np.int32)  # 2 bits of subpixel precision
        for t, col in zip(tri, color):
            cv2.fillConvexPoly(img, t, tuple(int(x) for x in col), cv2.LINE_8, 2)
    cv2.putText(img, d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    views.append(img)
cv2.imwrite(OUT, np.vstack(views))
