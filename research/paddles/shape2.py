"""Each paddle's real outline and colors, from SAM's cutouts, in the paddle's own plane.

A paddle is flat: in its frame (fit2.py's grips) it lies in z = 0. Every cell of a grid on that plane is projected
into each well-fit detection, and votes on whether it's paddle (SAM's mask) weighted by how squarely the face
looks at the camera. Cells that are paddle in most views form the outline; their median video color, the texture.
Writes shape/{track}.png (RGBA: alpha is the outline), shape/shape.json (the grid's extent) and a sheet.
Poses (fit with a two-box paddle) and outline are then improved in turn: each view's grip is refit to make the
outline match SAM's mask, and the outline rebuilt from the refit views, ROUNDS times.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

WORK, D = Path(sys.argv[1]), Path(sys.argv[2])
OUT = D / "shape"
OUT.mkdir(exist_ok=True)
CELL = 0.003  # m
X0, X1, Y0, Y1 = -0.15, 0.15, -0.10, 0.52  # m, in the paddle's frame (handle along +y from 0)
MIN_IOU = 0.5  # detections whose refined grip fits at least this well
MIN_FACING = 0.3  # |cos| between the face's normal and the view ray
UPSAMPLE = 4  # texture pixels per cell side
ROUNDS = 2
SCALE = 0.5  # masks at half the crop's resolution, for refitting
PRIOR_DEG, PRIOR_M, PRIOR_COST = 15.0, 0.03, 0.1  # refit: a 15° or 3 cm move from the first grip costs 0.1 IoU

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
recs = {(r["frame"], r["track_id"]): r for r in json.loads((D / "results.json").read_text())}
with np.load(D / "fit2" / "grips.npz") as f:
    data = dict(f)
players = sorted({int(k.split("_")[0]) for k in data})
xs = np.arange(X0 + CELL / 2, X1, CELL)
ys = np.arange(Y1 - CELL / 2, Y0, -CELL)  # image rows run down the paddle
gx, gy = np.meshgrid(xs, ys)
cells = np.stack([gx, gy, np.zeros_like(gx)], -1).reshape(-1, 3)


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


def views(t):
    """Each well-fit detection: its frame, the joint's rotation and origin, grip, crop record, mask, crop image."""
    frames = list(data[f"{t}_frames"])
    out = []
    for f, grip, fit in zip(data[f"{t}_det_frames"], data[f"{t}_det_grips"], data[f"{t}_det_iou"]):
        if fit < MIN_IOU:
            continue
        k = frames.index(f)
        mask = cv2.imread(str(D / "crops" / f"{t}_{int(f):06d}_owlmask.png"), 0)
        out.append(dict(frame=int(f), R=data[f"{t}_R"][k], origin=data[f"{t}_origin"][k], grip=grip.copy(), first=grip.copy(), rec=recs[(int(f), t)],
                        mask=mask.astype(np.float32) / 255, small=cv2.resize(mask, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) > 127,
                        crop=cv2.imread(str(D / "crops" / f"{t}_{int(f):06d}.jpg"))))
    return out


def to_camera(v, grip, points):
    gR = Rotation.from_rotvec(grip[:3]).as_matrix()
    return (points @ gR.T + grip[3:]) @ v["R"].T + v["origin"], v["R"] @ gR @ [0, 0, 1]


def build(vs):
    """Occupancy, outline (solid cells) and colors on the grid, from the views' current grips."""
    votes, weights, colors = np.zeros(len(cells)), np.zeros(len(cells)), []
    for v in vs:
        points, normal = to_camera(v, v["grip"], cells)
        facing = abs(normal @ points.mean(0)) / np.linalg.norm(points.mean(0))
        if facing < MIN_FACING:
            continue
        px = ((project(points) - v["rec"]["origin"]) * v["rec"]["scale"]).astype(np.float32)
        mask = v["mask"]
        inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < mask.shape[1] - 1) & (px[:, 1] < mask.shape[0] - 1)
        m = cv2.remap(mask, px[None, :, 0], px[None, :, 1], cv2.INTER_LINEAR)[0]
        c = cv2.remap(v["crop"], px[None, :, 0], px[None, :, 1], cv2.INTER_LINEAR)[0].astype(np.float32)
        votes += np.where(inside, facing * m, 0)
        weights += np.where(inside, facing, 0)
        colors.append(np.where(((m > 0.5) & inside & (facing > 0.5))[:, None], c, np.nan))
    occupancy = np.where(weights > 0, votes / np.maximum(weights, 1e-9), 0).reshape(gx.shape)
    occupancy = cv2.GaussianBlur(occupancy.astype(np.float32), (0, 0), 1.0)
    solid = (occupancy > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    if n > 1:
        solid = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # e.g. under the hand
    return occupancy, solid, colors


def outline_points(solid):
    """The outline as a polygon in the paddle's plane (m)."""
    contour = max(cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)[:, 0].astype(float)
    return np.stack([X0 + (contour[:, 0] + 0.5) * CELL, Y1 - (contour[:, 1] + 0.5) * CELL, np.zeros(len(contour))], 1)


def iou(v, grip, polygon):
    points, _ = to_camera(v, grip, polygon)
    if (points[:, 2] < 0.1).any():
        return 0.0
    sil = np.zeros(v["small"].shape, np.uint8)
    cv2.fillPoly(sil, [((project(points) - v["rec"]["origin"]) * v["rec"]["scale"] * SCALE).astype(np.int32)], 1)
    sil = sil.astype(bool)
    union = (sil | v["small"]).sum()
    return (sil & v["small"]).sum() / union if union else 0.0


def refit(v, polygon):
    R0, t0 = Rotation.from_rotvec(v["first"][:3]), v["first"][3:]

    def cost(g):
        angle = np.degrees((R0.inv() * Rotation.from_rotvec(g[:3])).magnitude())
        return 1 - iou(v, g, polygon) + PRIOR_COST * ((angle / PRIOR_DEG) ** 2 + (np.linalg.norm(g[3:] - t0) / PRIOR_M) ** 2)

    v["grip"] = minimize(cost, v["grip"], method="Powell", options=dict(maxiter=1500, xtol=1e-3, ftol=1e-4)).x
    return iou(v, v["grip"], polygon)


report, sheet = {}, []
for t in players:
    vs = views(t)
    occupancy, solid, colors = build(vs)
    rows = [occupancy]
    for k in range(ROUNDS):
        polygon = outline_points(solid)
        before = np.mean([iou(v, v["grip"], polygon) for v in vs])
        after = np.mean([refit(v, polygon) for v in vs])
        occupancy, solid, colors = build(vs)
        rows.append(occupancy)
        print(t, f"round {k + 1}: outline IoU {before:.3f} -> {after:.3f}", flush=True)
    # Colors: median over views; cells never seen take their neighbors' (so edges don't fade to black)
    color = np.nanmedian(np.array(colors), 0).reshape(*gx.shape, 3) if colors else np.full((*gx.shape, 3), np.nan)
    missing = np.isnan(color).any(-1).astype(np.uint8)
    color = cv2.inpaint(np.nan_to_num(color).clip(0, 255).astype(np.uint8), missing, 5, cv2.INPAINT_TELEA)
    big = (UPSAMPLE * gx.shape[1], UPSAMPLE * gx.shape[0])
    alpha = cv2.resize(cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 0.7), big, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / f"{t}.png"), np.dstack([cv2.resize(color, big, interpolation=cv2.INTER_CUBIC), (alpha > 0.5).astype(np.uint8) * 255]))
    ys_in, xs_in = np.nonzero(solid)
    report[t] = dict(views=len(vs), area_cm2=round(float(solid.sum() * CELL**2 * 1e4), 1),
                     width_cm=round(float((np.ptp(xs_in) + 1) * CELL * 100), 1), length_cm=round(float((np.ptp(ys_in) + 1) * CELL * 100), 1))
    print(t, report[t], flush=True)
    tiles = [cv2.cvtColor((o * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for o in rows] + [np.where(solid[..., None] > 0, color, 40)]
    sheet.append(cv2.resize(np.hstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
(OUT / "shape.json").write_text(json.dumps(dict(extent=[X0, X1, Y0, Y1], players=report), indent=1))
cv2.imwrite(str(OUT / "sheet.png"), np.vstack([np.pad(s, ((0, 10), (0, 0), (0, 0))) for s in sheet]))
