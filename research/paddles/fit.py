"""Fit one grip per player: where a standard paddle sits in the hand, so its outline matches SAM's cutouts.

Hand frame (from SAM 3D Body's keypoints, camera coordinates, meters): origin at the knuckles' center, x from the
wrist to the middle knuckle, z the palm's normal, y = z × x. The grip is a rotation (axis-angle) and a
translation of the paddle in that frame. The paddle is two thin boxes: the handle from y=0 to 13 cm, the face
above it. Its outline in a frame is the union of each box's projected convex hull.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

WORK, D = Path(sys.argv[1]), Path(sys.argv[2])
OUT = D / "fit"
OUT.mkdir(exist_ok=True)
HANDS = {"R": dict(wrist=41, index=28, middle=32, pinky=40), "L": dict(wrist=62, index=49, middle=53, pinky=61)}
MIN_SCORE = 0.3
NEAR_HAND = 0.1  # of the person's box height
SCALE = 0.5  # render masks at half the crop's resolution


def box(x0, x1, y0, y1, t):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (-t / 2, t / 2)])


HANDLE = box(-0.016, 0.016, -0.02, 0.13, 0.025)  # the hand holds it around y = 0
FACE = box(-0.095, 0.095, 0.13, 0.40, 0.015)
PADDLE = [HANDLE, FACE]

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
with np.load(WORK / "bodies" / "bodies.npz") as f:
    frame, track, kp = f["frame"], f["track_id"], f["keypoints"].astype(np.float64)
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(frame, track))}
recs = json.loads((D / "results.json").read_text())


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


def hand_frame(k, hand):
    h = HANDS[hand]
    w, i, m, p = k[h["wrist"]], k[h["index"]], k[h["middle"]], k[h["pinky"]]
    x = m - w
    x /= np.linalg.norm(x)
    z = np.cross(i - w, p - w)
    z -= x * (z @ x)
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1), (i + m + p) / 3  # rotation (columns are axes), origin


def box_distance(b, pt):
    return float(np.hypot(max(b[0] - pt[0], 0, pt[0] - b[2]), max(b[1] - pt[1], 0, pt[1] - b[3])))


# Which detections are this person's own paddle: near one of their wrists, and nearer it than any other wrist
samples = defaultdict(list)
for r in recs:
    o, i = r["owl"], row.get((r["frame"], r["track_id"]))
    if not o or o["score"] < MIN_SCORE or i is None:
        continue
    ox, oy = r["origin"]
    b = [ox + o["box"][0] / r["scale"], oy + o["box"][1] / r["scale"], ox + o["box"][2] / r["scale"], oy + o["box"][3] / r["scale"]]
    best = None
    for j in np.flatnonzero(frame == r["frame"]):
        for hand, h in HANDS.items():
            d = box_distance(b, project(kp[j, h["wrist"]]))
            if best is None or d < best[0]:
                best = (d, int(j), hand)
    if best[1] != i or best[0] > NEAR_HAND * r["person_h"]:
        continue
    mask = cv2.imread(str(D / "crops" / f"{r['track_id']}_{r['frame']:06d}_owlmask.png"), 0)
    mask = cv2.resize(mask, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) > 127
    samples[r["track_id"]].append(dict(rec=r, row=i, hand=best[2], mask=mask))


def silhouette(R, origin, grip, s, shape):
    """The paddle's outline in a sample's (downscaled) crop."""
    gR, gt = Rotation.from_rotvec(grip[:3]).as_matrix(), grip[3:]
    out = np.zeros(shape, np.uint8)
    r = s["rec"]
    for part in PADDLE:
        pts = (part @ gR.T + gt) @ R.T + origin
        if (pts[:, 2] < 0.1).any():
            return out
        px = (project(pts) - r["origin"]) * r["scale"] * SCALE
        cv2.fillConvexPoly(out, cv2.convexHull(px.astype(np.float32)).astype(np.int32), 1)
    return out.astype(bool)


def ious(grip, items):
    res = []
    for s in items:
        R, origin = hand_frame(kp[s["row"]], s["hand"])
        sil = silhouette(R, origin, grip, s, s["mask"].shape)
        union = (sil | s["mask"]).sum()
        res.append((sil & s["mask"]).sum() / union if union else 0.0)
    return np.array(res)


def fit(items, rng, starts=24):
    best = None
    for _ in range(starts):
        g0 = np.concatenate([Rotation.random(random_state=rng).as_rotvec(), rng.normal(0, 0.03, 3)])
        res = minimize(lambda g: 1 - ious(g, items).mean(), g0, method="Powell", options=dict(maxiter=3000, xtol=1e-3, ftol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, 1 - best.fun


rng = np.random.default_rng(0)
report = {}
for t, items in sorted(samples.items()):
    hand = Counter(s["hand"] for s in items).most_common(1)[0][0]
    items = [s for s in items if s["hand"] == hand]
    grip, score = fit(items, rng)
    half = len(items) // 2
    g1, s1 = fit(items[:half], rng, 8)
    g2, s2 = fit(items[half:], rng, 8)
    per = ious(grip, items)
    angle = np.degrees((Rotation.from_rotvec(g1[:3]).inv() * Rotation.from_rotvec(g2[:3])).magnitude())
    report[t] = dict(hand=hand, frames=len(items), mean_iou=round(float(score), 3), median_iou=round(float(np.median(per)), 3),
                     halves_iou=[round(float(s1), 3), round(float(s2), 3)], halves_angle_deg=round(float(angle), 1),
                     halves_shift_cm=round(float(100 * np.linalg.norm(g1[3:] - g2[3:])), 1), grip=[float(v) for v in grip])
    print(t, {k: v for k, v in report[t].items() if k != "grip"}, flush=True)
    # A sheet: fitted outline (green) over the crop, SAM's mask (red)
    tiles = []
    for s, v in list(zip(items, per))[:: max(1, len(items) // 18)][:18]:
        r = s["rec"]
        img = cv2.imread(str(D / "crops" / f"{r['track_id']}_{r['frame']:06d}.jpg"))
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        R, origin = hand_frame(kp[s["row"]], s["hand"])
        sil = silhouette(R, origin, grip, s, s["mask"].shape).astype(np.uint8)
        img[s["mask"]] = (0.5 * img[s["mask"]] + [0, 0, 127]).astype(np.uint8)
        cv2.drawContours(img, cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 1)
        cv2.putText(img, f"f{r['frame']} {v:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tile = np.zeros((320, 320, 3), np.uint8)
        tile[: img.shape[0], : img.shape[1]] = img
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT / f"fit_p{t}.jpg"), np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]))
(OUT / "grips.json").write_text(json.dumps(report, indent=1))
