"""How often is each player's paddle found, and does it sit at a hand?

Projects SAM 3D Body's wrists (keypoints 41 right, 62 left; camera coordinates, meters) into the frame and
measures the distance from each OWLv2 box to the nearer wrist, in units of the person's box height.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK, D = Path(sys.argv[1]), Path(sys.argv[2])
RIGHT, LEFT = 41, 62
recs = json.loads((D / "results.json").read_text())
cam = json.loads((WORK / "bodies" / "camera.json").read_text())
with np.load(WORK / "bodies" / "bodies.npz") as f:
    frame, track, kp = f["frame"], f["track_id"], f["keypoints"]
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(frame, track))}


def project(p):
    return np.array([cam["fx"] * p[0] / p[2] + cam["cx"], cam["fy"] * p[1] / p[2] + cam["cy"]])


def box_distance(box, pt):
    dx = max(box[0] - pt[0], 0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0, pt[1] - box[3])
    return float(np.hypot(dx, dy))


stats = defaultdict(lambda: defaultdict(list))
for r in recs:
    s = stats[r["track_id"]]
    s["height"].append(r["person_h"])
    o = r["owl"]
    if not o:
        s["score"].append(0.0)
        continue
    s["score"].append(o["score"])
    ox, oy = r["origin"]
    box = [ox + o["box"][0] / r["scale"], oy + o["box"][1] / r["scale"], ox + o["box"][2] / r["scale"], oy + o["box"][3] / r["scale"]]
    s["length"].append(max(box[2] - box[0], box[3] - box[1]))
    i = row.get((r["frame"], r["track_id"]))
    if i is not None:
        d = [box_distance(box, project(kp[i, k])) / r["person_h"] for k in (RIGHT, LEFT)]
        s["hand_dist"].append(min(d))
        s["hand"].append("R" if d[0] <= d[1] else "L")
        s["scored_at_hand"].append((o["score"], min(d)))

print("player  n   height  found@0.3  found@0.5  at-hand(<0.1h)  paddle-len-px  right-hand")
for t, s in sorted(stats.items()):
    sc = np.array(s["score"])
    pairs = np.array(s["scored_at_hand"]) if s["scored_at_hand"] else np.zeros((0, 2))
    good = pairs[(pairs[:, 0] >= 0.3)] if len(pairs) else pairs
    at_hand = (good[:, 1] < 0.1).mean() if len(good) else float("nan")
    hands = [h for h, (scr, dist) in zip(s["hand"], s["scored_at_hand"]) if scr >= 0.3 and dist < 0.1]
    print(f"{t:>6} {len(sc):>3} {np.median(s['height']):>7.0f} {(sc >= 0.3).mean():>9.0%} {(sc >= 0.5).mean():>9.0%}"
          f" {at_hand:>14.0%} {np.median(s['length']) if s['length'] else 0:>14.0f}"
          f" {hands.count('R') / len(hands) if hands else float('nan'):>10.0%}")
