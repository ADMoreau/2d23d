"""Do on-screen graphics (scoreboards, logos) stand out over time from the place itself? Graphics are drawn on every
frame identically, so their pixels don't carry the camera's noise; the place's pixels do. Measures each pixel's
frame-to-frame change over a stretch of consecutive frames, where nothing in the place moves except people.

usage: overlay.py <video> <out dir> [start frame=0] [frames=60]   (lab venv)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO, OUT = sys.argv[1], Path(sys.argv[2])
START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, START)
prev, change, still = None, None, None
for i in range(N):
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16)
    if prev is not None:
        d = np.abs(g - prev)
        change = d.astype(np.float32) if change is None else change + d
        still = (d == 0).astype(np.float32) if still is None else still + (d == 0)
    prev = g
change /= N - 1
still /= N - 1
cv2.imwrite(str(OUT / "change.png"), np.clip(change * 60, 0, 255).astype(np.uint8))
cv2.imwrite(str(OUT / "still.png"), (still * 255).astype(np.uint8))
print("share of pixels identical in every pair of frames:", float((still == 1).mean()).__round__(3))
print("mean change (gray levels): 50th/90th percentile", np.percentile(change, [50, 90]).round(2))

# Graphics' edges are as sharp as the place's but don't move: the camera shakes slightly and compression flickers
# along the place's edges. Score each strong edge pixel by its change, relative to edges in general.
cap.set(cv2.CAP_PROP_POS_FRAMES, START + N // 2)
_, mid = cap.read()
gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY).astype(np.float32)
edge = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
strong = edge > np.percentile(edge, 95)
print("change on strong edges: 10th/50th percentile", np.percentile(change[strong], [10, 50]).round(2))
frozen = strong & (change < 0.2)
frozen = cv2.morphologyEx(frozen.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
view = mid.copy()
view[frozen > 0] = (0.4 * view[frozen > 0] + [0, 0, 150]).astype(np.uint8)
cv2.imwrite(str(OUT / "frozen.jpg"), cv2.resize(view, (960, 540)))
