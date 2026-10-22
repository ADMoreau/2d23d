"""Measure the painted lines on the top-down floor: white lines' directions and the spacing of parallel ones.

usage: measure_lines.py <run dir>
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RUN = Path(sys.argv[1])
top = cv2.imread(str(RUN / "topdown.png"))
info = json.loads((RUN / "topdown.json").read_text())
per_m = info["per_m"]
hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
# Painted lines are thin and brighter than the floor on both sides: a white top-hat (brightness minus its opening by a
# 25 cm disc) keeps them and drops wide bright patches
gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.25 * per_m) | 1,) * 2)
hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, disc)
white = ((hat > 40) & (hsv[..., 1] < 90)).astype(np.uint8)
# Keep only lines on the floor near the court: the camera's 15 m
H, W = white.shape
Z, X = np.mgrid[0:H, 0:W]
x = info["x"][0] + X / per_m
z = info["z"][0] + Z / per_m
cam = np.array(info["camera"])
white &= (np.hypot(x - cam[0], z - cam[2]) < 15).astype(np.uint8)
cv2.imwrite(str(RUN / "white.png"), white * 255)
lines = cv2.HoughLinesP(white, 1, np.pi / 720, 80, minLineLength=int(1.5 * per_m), maxLineGap=int(0.3 * per_m)).reshape(-1, 4)
ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180
length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]) / per_m
hist = np.zeros(180)
for a, l in zip(ang, length):
    hist[int(a) % 180] += l
main = int(np.argmax(np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid")))
print(f"{len(lines)} segments; main direction {main} deg")
view = top.copy()
for fam, a0 in (("A", main), ("B", (main + 90) % 180)):
    sel = np.abs((ang - a0 + 90) % 180 - 90) < 4
    t = np.radians(a0)
    n = np.array([-np.sin(t), np.cos(t)])  # normal in image (x right, z down)
    mids = (lines[sel, :2] + lines[sel, 2:]) / 2
    off = mids @ n / per_m
    w = length[sel]
    order = np.argsort(off)
    # Group offsets within 10 cm into one painted line (its two edges merge too)
    groups, cur = [], [order[0]]
    for i, j in zip(order[:-1], order[1:]):
        if off[j] - off[i] < 0.10:
            cur.append(j)
        else:
            groups.append(cur); cur = [j]
    groups.append(cur)
    stats = [(float(np.average(off[g], weights=w[g])), float(w[g].sum())) for g in groups]
    stats = [s for s in stats if s[1] > 2.0]  # at least 2 m of line found
    base = stats[0][0]
    print(f"family {fam} (angle {a0}): line offsets (m) " + ", ".join(f"{o - base:.2f} [{l:.0f} m]" for o, l in stats))
    for k in np.flatnonzero(sel):
        cv2.line(view, tuple(lines[k, :2]), tuple(lines[k, 2:]), (0, 0, 255) if fam == "A" else (0, 255, 255), 2)
cv2.imwrite(str(RUN / "lines.jpg"), view)

# Every segment: its offset across each family, and where it lies along it
for a0 in (main, (main + 90) % 180):
    sel = np.abs((ang - a0 + 90) % 180 - 90) < 4
    t = np.radians(a0)
    dvec, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    print(f"-- angle {a0}")
    for k in np.flatnonzero(sel)[np.argsort((lines[sel, :2] @ n))]:
        a, b = lines[k, :2], lines[k, 2:]
        print(f"offset {a @ n / per_m:6.2f}  along {min(a @ dvec, b @ dvec) / per_m:6.2f}..{max(a @ dvec, b @ dvec) / per_m:6.2f}  len {length[k]:.2f}")
