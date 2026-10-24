"""All ball candidates over time (x: frame, y: image u and v), sized by box, with the linked path on top.

usage: linkplot.py <detections dir> <ball.json> <out.png> [min score=0.15]
"""
import json
import sys

import cv2
import numpy as np

det = [json.loads(l) for l in open(sys.argv[1] + "/detections.jsonl")]
linked = json.loads(open(sys.argv[2]).read()).get("linked", {})
MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
n = len(det)
W, H = 1800, 500
xs = lambda f: int(40 + f / n * (W - 60))
img = np.full((2 * H + 20, W, 3), 255, np.uint8)
umax = max(b["box"][2] for d in det for b in d["balls"]) + 1
vmax = max(b["box"][3] for d in det for b in d["balls"]) + 1
for row, (axis, top) in enumerate(((0, umax), (1, vmax))):
    y0 = row * (H + 20)
    cv2.putText(img, "u" if axis == 0 else "v", (5, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for d in det:
        for b in d["balls"]:
            if b["score"] < MIN:
                continue
            c = (b["box"][axis] + b["box"][axis + 2]) / 2
            s = min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])
            g = int(200 - 150 * min(1, b["score"]))
            cv2.circle(img, (xs(d["frame"]), y0 + int(c / top * H)), max(1, int(s / 10)), (g, g, g), -1)
    pts = sorted((int(f), v) for f, v in linked.items())
    for (f0, a), (f1, b) in zip(pts, pts[1:]):
        col = (0, 0, 255) if f1 - f0 == 1 else (255, 0, 0)
        cv2.line(img, (xs(f0), y0 + int(a[axis] / top * H)), (xs(f1), y0 + int(b[axis] / top * H)), col, 1)
for f in range(0, n, 50):
    cv2.putText(img, str(f), (xs(f), 2 * H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
cv2.imwrite(sys.argv[3], img)
