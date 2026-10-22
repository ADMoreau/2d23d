"""Is the ball's shadow visible? Where each fitted light (lights.json) would cast the ball's shadow (ball.json's 3D
path), average the floor's darkening (1 - frame / background) over many frames; compare with rings around those spots.

usage: ball_shadow.py <work dir> <lights.json> <ball.json> <video>   (lab venv)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORK, LIGHTS, BALL, VIDEO = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
lights = json.loads(LIGHTS.read_text())["lights"]
ball = json.loads(BALL.read_text())
pos, frames = np.array(ball["position"]), np.array(ball["frames"])
bg = cv2.cvtColor(cv2.imread(str(WORK / "scene" / "background.jpg")), cv2.COLOR_BGR2GRAY).astype(np.float32)


def pixel(p):
    c = (p - t_cw) @ R_cw
    return c[0] / c[2] * cam["fx"] + cam["cx"], c[1] / c[2] * cam["fy"] + cam["cy"]


cap = cv2.VideoCapture(VIDEO)
patches = {k: [] for k in ("low", "mid", "high")}
R = 24  # px half-size of the patch around each predicted shadow
for i in range(0, len(frames), 3):
    h = pos[i][1]
    band = "low" if h < 0.3 else "mid" if h < 1.0 else "high"
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[i]))
    ok, img = cap.read()
    if not ok:
        continue
    ratio = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / np.maximum(bg, 1)
    bu, bv = pixel(pos[i])
    for L in lights[:3]:
        Lp = np.array(L["position"])
        q = Lp + (pos[i] - Lp) * (Lp[1] / (Lp[1] - pos[i][1]))
        u, v = pixel(q)
        u, v = int(round(u)), int(round(v))
        if not (R <= u < img.shape[1] - R and R <= v < img.shape[0] - R) or np.hypot(u - bu, v - bv) < 2 * R:
            continue  # off screen, or the ball itself is in the patch
        patch = ratio[v - R:v + R + 1, u - R:u + R + 1]
        patches[band].append(patch / np.median(patch))
for band, ps in patches.items():
    if not ps:
        continue
    mean = np.mean(ps, 0)
    c = mean[R - 3:R + 4, R - 3:R + 4].mean()
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    ring = mean[(np.hypot(yy, xx) > 12) & (np.hypot(yy, xx) < R)].mean()
    print(f"ball height {band}: {len(ps)} predicted shadow spots; mean brightness at the spot {c:.3f} vs around {ring:.3f}")
    cv2.imwrite(f"run/ball_shadow_{band}.png", cv2.resize(np.clip((mean - 0.85) / 0.3 * 255, 0, 255).astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
