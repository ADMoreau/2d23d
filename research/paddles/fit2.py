"""Grips in MHR's wrist-joint frame: one per player, refined per detection, then smoothed over time.

The skeleton is the one export puts in the GLB (check/bodies.npz: smoothed, on the floor), so a grip fit here is
exactly the paddle's transform under its wrist joint. Steps, per player:
1. One grip for all frames, fit on mean silhouette IoU with SAM's masks (as fit.py, but in the joint's frame).
2. Each detection nudges that grip toward its own mask, with a prior that keeps it near the shared grip.
3. Every frame of the track takes a Gaussian-weighted average of nearby refined grips (and a little of the shared
   one, so frames far from any detection fall back to it). Leave-one-out IoU says how well that fills a frame
   that had no detection.
The paddle looks the same turned 180° about its long axis, so grips are compared and averaged modulo that flip.
Writes fit2/grips.npz, fit2/report.json, contact sheets, and a debug video of every frame.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from twod23d.stages import export

torch.set_num_threads(4)
WORK, D = Path(sys.argv[1]), Path(sys.argv[2])
OUT = D / "fit2"
OUT.mkdir(exist_ok=True)
KEYPOINT = {"R": 41, "L": 62}  # SAM 3D Body's wrist keypoints
JOINT = {"R": 41, "L": 77}  # MHR's wrist joints (at those keypoints)
MIN_SCORE, NEAR_HAND, SCALE = 0.3, 0.1, 0.5
PRIOR_DEG, PRIOR_M, PRIOR_COST = 30.0, 0.05, 0.1  # refinement: a 30° or 5 cm move costs 0.1 IoU
MIN_IOU = 0.3  # a refined grip below this is dropped
SIGMA = 6.0  # frames, for the temporal average
PRIOR_WEIGHT = 0.05  # of the shared grip in that average
FLIP = Rotation.from_euler("y", np.pi)


def box(x0, x1, y0, y1, t):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (-t / 2, t / 2)])


HANDLE, FACE = box(-0.016, 0.016, -0.02, 0.13, 0.025), box(-0.095, 0.095, 0.13, 0.40, 0.015)

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
fps = json.loads((WORK / "ingest" / "meta.json").read_text())["fps"]
world_to_camera = np.linalg.inv(np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"]))
with np.load(WORK / "bodies" / "bodies.npz") as f:
    raw = {k: f[k] for k in ("frame", "track_id", "keypoints")}
with np.load(WORK / "check" / "bodies.npz") as f:
    bodies = dict(f)
raw_row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(raw["frame"], raw["track_id"]))}
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(bodies["frame"], bodies["track_id"]))}
recs = json.loads((D / "results.json").read_text())
rig = export.load_rig()


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


def joint_frames(rows, j):
    """The wrist joint's rotation (n, 3, 3), origin (n, 3) in camera coordinates (meters), and scale (n,)."""
    out = []
    for chunk in np.array_split(rows, max(1, len(rows) // 64)):
        inputs = [torch.from_numpy(bodies[k][chunk]).float() for k in ("mhr_shape", "mhr_model_params", "mhr_expression")]
        with torch.inference_mode():
            vertices, state = rig.model(*inputs, True)
        rotation = np.linalg.inv(world_to_camera)[:3, :3] @ export.SAM_AXES / 100
        offset = bodies["vertices"][chunk].astype(np.float64).mean(1) - vertices.double().numpy().mean(1) @ rotation.T
        place = np.tile(np.eye(4), (len(chunk), 1, 1))
        place[:, :3, :3], place[:, :3, 3] = rotation, offset
        out.append(world_to_camera @ place @ export.tqs_to_matrix(state.double().numpy()[:, j]))
    m = np.concatenate(out)
    s = np.cbrt(np.linalg.det(m[:, :3, :3]))
    return m[:, :3, :3] / s[:, None, None], m[:, :3, 3], s


def box_distance(b, pt):
    return float(np.hypot(max(b[0] - pt[0], 0, pt[0] - b[2]), max(b[1] - pt[1], 0, pt[1] - b[3])))


# Each player's own detections: nearest wrist (of anyone) is theirs, and near enough
samples = defaultdict(list)
for r in recs:
    o, i = r["owl"], raw_row.get((r["frame"], r["track_id"]))
    if not o or o["score"] < MIN_SCORE or i is None or (r["frame"], r["track_id"]) not in row:
        continue
    ox, oy = r["origin"]
    b = [ox + o["box"][0] / r["scale"], oy + o["box"][1] / r["scale"], ox + o["box"][2] / r["scale"], oy + o["box"][3] / r["scale"]]
    best = min((box_distance(b, project(raw["keypoints"][j, k])), int(j), hand)
               for j in np.flatnonzero(raw["frame"] == r["frame"]) for hand, k in KEYPOINT.items())
    if best[1] != i or best[0] > NEAR_HAND * r["person_h"]:
        continue
    mask = cv2.imread(str(D / "crops" / f"{r['track_id']}_{r['frame']:06d}_owlmask.png"), 0)
    mask = cv2.resize(mask, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) > 127
    samples[r["track_id"]].append(dict(rec=r, row=row[(r["frame"], r["track_id"])], hand=best[2], mask=mask))


def as_rigid(grip):
    return Rotation.from_rotvec(grip[:3]), np.asarray(grip[3:])


def outline(R, origin, grip, s, parts=(HANDLE, FACE)):
    gR, gt = as_rigid(grip)
    out = np.zeros(s["mask"].shape, np.uint8)
    r = s["rec"]
    for part in parts:
        pts = (gR.apply(part) + gt) @ R.T + origin
        if (pts[:, 2] < 0.1).any():
            return out.astype(bool)
        px = (project(pts) - r["origin"]) * r["scale"] * SCALE
        cv2.fillConvexPoly(out, cv2.convexHull(px.astype(np.float32)).astype(np.int32), 1)
    return out.astype(bool)


def iou(grip, s):
    sil = outline(s["R"], s["origin"], grip, s)
    union = (sil | s["mask"]).sum()
    return (sil & s["mask"]).sum() / union if union else 0.0


def fit_shared(items, rng, starts=24):
    best = None
    for _ in range(starts):
        g0 = np.concatenate([Rotation.random(random_state=rng).as_rotvec(), rng.normal(0, 0.03, 3)])
        res = minimize(lambda g: 1 - np.mean([iou(g, s) for s in items]), g0, method="Powell",
                       options=dict(maxiter=3000, xtol=1e-3, ftol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def closest(rotation, reference):
    """rotation or its flipped twin, whichever is nearer reference."""
    flipped = rotation * FLIP
    return flipped if (reference.inv() * flipped).magnitude() < (reference.inv() * rotation).magnitude() else rotation


def refine(shared, s, rng):
    R0, t0 = as_rigid(shared)

    def cost(g):
        R, t = as_rigid(g)
        prior = (np.degrees((R0.inv() * closest(R, R0)).magnitude()) / PRIOR_DEG) ** 2 + (np.linalg.norm(t - t0) / PRIOR_M) ** 2
        return 1 - iou(g, s) + PRIOR_COST * prior

    best = None
    for k in range(4):
        jitter = Rotation.from_rotvec(rng.normal(0, np.radians(20), 3)) if k else Rotation.identity()
        g0 = np.concatenate([(jitter * R0).as_rotvec(), t0 + (rng.normal(0, 0.02, 3) if k else 0)])
        res = minimize(cost, g0, method="Powell", options=dict(maxiter=2000, xtol=1e-3, ftol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    R, t = as_rigid(best.x)
    return np.concatenate([closest(R, R0).as_rotvec(), t])


def average(frames, grips, shared, at, exclude=None):
    """The grip at frame `at`: Gaussian-weighted over refined grips, plus the shared grip."""
    w = np.exp(-0.5 * ((frames - at) / SIGMA) ** 2)
    if exclude is not None:
        w[exclude] = 0
    rotations = Rotation.concatenate([Rotation.from_rotvec(shared[:3])] + [Rotation.from_rotvec(g[:3]) for g in grips])
    weights = np.concatenate([[PRIOR_WEIGHT], w])
    t = (weights[:, None] * np.vstack([shared[3:], grips[:, 3:]])).sum(0) / weights.sum()
    return np.concatenate([rotations.mean(weights).as_rotvec(), t])


rng = np.random.default_rng(0)
report, saved = {}, {}
for t, items in sorted(samples.items()):
    hand = Counter(s["hand"] for s in items).most_common(1)[0][0]
    items = sorted((s for s in items if s["hand"] == hand), key=lambda s: s["rec"]["frame"])
    R, origin, _ = joint_frames(np.array([s["row"] for s in items]), JOINT[hand])
    for s, a, b in zip(items, R, origin):
        s["R"], s["origin"] = a, b
    shared = fit_shared(items, rng)
    shared_iou = np.array([iou(shared, s) for s in items])
    refined = np.array([refine(shared, s, rng) for s in items])
    refined_iou = np.array([iou(g, s) for g, s in zip(refined, items)])
    keep = refined_iou >= MIN_IOU
    frames = np.array([s["rec"]["frame"] for s in items])
    kept_frames, kept = frames[keep], refined[keep]
    # Leave one out: each detected frame's grip from its neighbors only, as if it had no detection
    loo = np.array([iou(average(kept_frames, kept, shared, f, exclude=np.flatnonzero(kept_frames == f)), s)
                    for f, s in zip(frames, items)])
    moves = np.degrees([(Rotation.from_rotvec(shared[:3]).inv() * Rotation.from_rotvec(g[:3])).magnitude() for g in refined])
    report[t] = dict(hand=hand, joint=JOINT[hand], detections=len(items), kept=int(keep.sum()),
                     iou_shared=round(float(shared_iou.mean()), 3), iou_refined=round(float(refined_iou.mean()), 3),
                     iou_leave_one_out=round(float(loo.mean()), 3), refine_move_deg_median=round(float(np.median(moves)), 1),
                     shared=[float(v) for v in shared])
    print(t, {k: v for k, v in report[t].items() if k != "shared"}, flush=True)

    # Every frame of the track
    rows = np.flatnonzero(bodies["track_id"] == t)
    rows = rows[np.argsort(bodies["frame"][rows])]
    grips = np.array([average(kept_frames, kept, shared, f) for f in bodies["frame"][rows]])
    jR, jo, js = joint_frames(rows, JOINT[hand])
    # Color: video pixels inside both SAM's mask and the fitted face (not the hand or handle)
    pixels = []
    for s, g in zip(items, refined):
        r = s["rec"]
        img = cv2.resize(cv2.imread(str(D / "crops" / f"{r['track_id']}_{r['frame']:06d}.jpg")), None, fx=SCALE, fy=SCALE)
        inside = s["mask"] & cv2.erode(outline(s["R"], s["origin"], g, s, parts=(FACE,)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        pixels.append(img[inside])
    pixels = np.concatenate(pixels)
    saved[t] = dict(frames=bodies["frame"][rows], grips=grips, R=jR, origin=jo, scale=js, joint=JOINT[hand],
                    det_frames=frames, det_grips=refined, det_iou=refined_iou,
                    color=np.median(pixels, 0) if len(pixels) else np.array([128.0, 128, 128]))
    report[t]["color_bgr"] = [float(v) for v in saved[t]["color"]]
    report[t]["color_pixels"] = len(pixels)

    # Sheet: shared grip (blue), refined (green), leave-one-out (yellow) over SAM's mask (red)
    tiles = []
    for k in range(0, len(items), max(1, len(items) // 18))[:18]:
        s, r = items[k], items[k]["rec"]
        img = cv2.resize(cv2.imread(str(D / "crops" / f"{r['track_id']}_{r['frame']:06d}.jpg")), None, fx=SCALE, fy=SCALE)
        img[s["mask"]] = (0.5 * img[s["mask"]] + [0, 0, 127]).astype(np.uint8)
        loo_grip = average(kept_frames, kept, shared, frames[k], exclude=np.flatnonzero(kept_frames == frames[k]))
        for g, color in ((shared, (255, 128, 0)), (loo_grip, (0, 255, 255)), (refined[k], (0, 255, 0))):
            sil = outline(s["R"], s["origin"], g, s).astype(np.uint8)
            cv2.drawContours(img, cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, color, 1)
        cv2.putText(img, f"f{r['frame']} {shared_iou[k]:.2f}/{loo[k]:.2f}/{refined_iou[k]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tile = np.zeros((320, 320, 3), np.uint8)
        tile[: min(320, img.shape[0]), : min(320, img.shape[1])] = img[:320, :320]
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT / f"fit_p{t}.jpg"), np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]))

(OUT / "report.json").write_text(json.dumps(report, indent=1))
np.savez(OUT / "grips.npz", **{f"{t}_{k}": v for t, d in saved.items() for k, v in d.items()})

# Debug video: every frame, each paddle from its smoothed grip; a dot where the frame had a kept detection
detected = {(int(s["rec"]["frame"]), t) for t, items in samples.items() for s in items}
frames_dir = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
writer = cv2.VideoWriter(str(OUT / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
index = {t: {int(f): k for k, f in enumerate(d["frames"])} for t, d in saved.items()}
for n, path in enumerate(frames_dir):
    img = cv2.imread(str(path))
    for t, d in saved.items():
        k = index[t].get(n)
        if k is None:
            continue
        gR, gt = as_rigid(d["grips"][k])
        for part in (HANDLE, FACE):
            pts = (gR.apply(part) + gt) @ d["R"][k].T + d["origin"][k]
            hull = cv2.convexHull(project(pts).astype(np.float32)).astype(np.int32)
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)
        if (n, t) in detected:
            cv2.circle(img, tuple(project(d["origin"][k]).astype(int)), 6, (0, 0, 255), -1)
    writer.write(cv2.resize(img, (960, 540)))
writer.release()
print("done", flush=True)
