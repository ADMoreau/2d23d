"""Lights the camera sees: bright blobs above the walls' tops, placed where their rays meet the fitted ceiling.
Draws them and lights.json's fitted lights on the background image.

usage: visible_lights.py <work dir> <setting run dir> <lights.json> <out dir>   (lab venv)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORK, SET, LIGHTS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
fit = json.loads(LIGHTS.read_text())
ceiling = fit["ceiling"]
img = cv2.imread(str(WORK / "scene" / "background.jpg"))
v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
bright = ((v > 250) & (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1] < 40)).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(bright)
seen = []
for k in range(1, n):
    if stats[k, cv2.CC_STAT_AREA] < 20:
        continue
    u, vv = cent[k]
    d = np.array([(u - cam["cx"]) / cam["fx"], (vv - cam["cy"]) / cam["fy"], 1.0]) @ R_cw.T
    if d[1] <= 0:
        continue  # below the horizon: not a ceiling light
    s = (ceiling - t_cw[1]) / d[1]
    p = t_cw + s * d
    seen.append(dict(pixel=[round(u), round(vv)], position=p.round(2).tolist(), area=int(stats[k, cv2.CC_STAT_AREA])))
print("visible ceiling lights (x, z):", [s["position"][0::2] for s in seen])
view = img.copy()
for s in seen:
    cv2.circle(view, tuple(s["pixel"]), 14, (0, 255, 0), 3)
for L in fit["lights"]:
    c = (np.array(L["position"]) - t_cw) @ R_cw
    if c[2] > 0:
        q = (int(cam["fx"] * c[0] / c[2] + cam["cx"]), int(cam["fy"] * c[1] / c[2] + cam["cy"]))
        cv2.circle(view, q, 6 + int(300 * L["share"]), (0, 0, 255), 3)
    else:
        print("fitted light behind the camera:", L["position"])
cv2.imwrite(str(OUT / "lights_view.jpg"), cv2.resize(view, (960, 540)))
(OUT / "visible.json").write_text(json.dumps(seen, indent=1))
