"""The setting's big flat surfaces from the scene mesh: walls (vertical planes) and the ceiling (a horizontal one),
found one at a time by RANSAC on the scene's points above the floor.

usage: walls.py <work dir> <out dir>   (lab venv)
Writes walls.json (each plane: unit normal n and offset d with n·p = d, kind, share of points) and walls.jpg (the
background with each plane's points colored).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TOL = 0.10  # m, a point within this of a plane belongs to it
MIN_SHARE = 0.03  # a plane needs this share of the points
MAX_TILT = 10.0  # degrees: a wall's normal is this close to horizontal, a ceiling's to vertical

z = np.load(WORK / "scene" / "scene.npz")
V, uv = z["vertices"], z["uvs"]
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
H, W = image.shape[:2]
above = np.flatnonzero(V[:, 1] > 0.3)  # the floor layer is y = 0 exactly; low clutter (net base, chairs) is left out
P = V[above].astype(np.float64)
rng = np.random.default_rng(0)
left = np.ones(len(P), bool)
planes = []
while left.sum() > MIN_SHARE * len(P):
    idx = np.flatnonzero(left)
    best, best_n, best_d = None, None, None
    for _ in range(2000):
        a, b, c = P[rng.choice(idx, 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        tilt = np.degrees(np.arcsin(min(1.0, abs(n[1]))))
        if not (tilt < MAX_TILT or tilt > 90 - MAX_TILT):
            continue
        inl = idx[np.abs(P[idx] @ n - a @ n) < TOL]
        if best is None or len(inl) > len(best):
            best, best_n, best_d = inl, n, a @ n
    if best is None or len(best) < MIN_SHARE * len(P):
        break
    # Refit by least squares on the inliers, keeping the plane exactly vertical or horizontal
    q = P[best]
    if abs(best_n[1]) < 0.5:
        c2 = q[:, [0, 2]].mean(0)
        _, _, vt = np.linalg.svd(q[:, [0, 2]] - c2)
        n = np.array([vt[1][0], 0.0, vt[1][1]])
        d = float(np.r_[c2[0], 0, c2[1]] @ n)
        kind = "wall"
    else:
        n, d, kind = np.array([0.0, 1.0, 0.0]), float(np.median(q[:, 1])), "ceiling"
    inl = idx[np.abs(P[idx] @ n - d) < TOL]
    if d < 0:
        n, d = -n, -d
    left[inl] = False
    rms = float(np.sqrt(np.mean((P[inl] @ n - d) ** 2)))
    planes.append(dict(kind=kind, normal=n.round(4).tolist(), offset=round(d, 3), share=round(len(inl) / len(P), 3),
                       rms_m=round(rms, 3), points=above[inl].tolist()))
    print(f"{kind}: n={n.round(3)} d={d:.2f} m, {len(inl) / len(P):.1%} of points, rms {rms * 100:.1f} cm", flush=True)

view = (image * 0.5).astype(np.uint8)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 128, 255), (255, 128, 0)]
for k, pl in enumerate(planes):
    px = (uv[pl["points"]] * [W, H]).astype(int)
    for x, y in px:
        cv2.circle(view, (x, y), 3, colors[k % len(colors)], -1)
    cx, cy = px.mean(0).astype(int)
    cv2.putText(view, f"{k} {pl['kind']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite(str(OUT / "walls.jpg"), cv2.resize(view, (960, 540)))
(OUT / "walls.json").write_text(json.dumps([{k: v for k, v in p.items() if k != "points"} for p in planes], indent=1))
