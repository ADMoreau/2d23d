"""Where the lights are, from people's shadows on the floor.

The floor darkens where bodies block lights. With the background image (no people) as the unshadowed floor, each
frame's darkening is 1 - frame / background. Model: darkening = sum over lights of (the light's share of the floor's
light) x (whether a body blocks it there). Candidate lights sit on the fitted ceiling (walls.json) on a grid; each
one's shadows are the bodies' vertices (check/bodies.npz, world) cast from it onto y = 0 and seen by the camera.
Non-negative least squares picks a few candidates and their shares; a finer grid around each refines it.

Frames: every STEP-th of the video (the fit needs variety, not every frame); images at 1/SCALE size.

usage: lights.py <work dir> <setting run dir> <video> <out dir>   (lab venv)
Writes lights.json and check.jpg (observed vs. modeled darkening on a few frames).
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

WORK, SET, VIDEO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 8  # image downscale
STEP = 25  # frames
VERTS = 6  # every VERTS-th body vertex casts a shadow
COARSE = 1.5  # m between coarse candidates
FINE = 0.25  # m between fine candidates
FINE_REACH = 1.5  # m around each coarse pick
MIN_SHARE = 0.01  # a light needs this share to count
MAX_ROWS = 60000  # pixels sampled for the fit (memory)
SOFT = 1.5  # px (low res): shadow edge blur
CONTACT = 0.4  # m: floor this close to a foot is darkened by contact (ambient occlusion), not by one light

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
planes = json.loads((SET / "walls.json").read_text())
ceiling = [p for p in planes if p["kind"] == "ceiling"][0]["offset"]
walls = [p for p in planes if p["kind"] == "wall"]
bg = cv2.imread(str(WORK / "scene" / "background.jpg"))
H, W = bg.shape[0] // SCALE, bg.shape[1] // SCALE
gray = lambda img: cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
bg_small = gray(bg)
fx, fy, cx, cy = cam["fx"] / SCALE, cam["fy"] / SCALE, cam["cx"] / SCALE, cam["cy"] / SCALE


def to_pixels(p):
    c = (p - t_cw) @ R_cw
    return np.stack([fx * c[..., 0] / c[..., 2] + cx, fy * c[..., 1] / c[..., 2] + cy], -1), c[..., 2]


# Floor pixels: the ray meets the floor before any wall, within 25 m
u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1) @ R_cw.T
with np.errstate(divide="ignore"):
    s = np.where(rays[..., 1] < -1e-9, -t_cw[1] / rays[..., 1], np.inf)
floor = np.isfinite(s) & (s * np.linalg.norm(rays, axis=-1) < 25)
ground = t_cw + np.where(np.isfinite(s), s, 0)[..., None] * rays
for w in walls:
    n, d = np.array(w["normal"]), w["offset"]
    floor &= np.sign(d - t_cw @ n) * (d - ground @ n) > 0.2
floor &= bg_small > 30  # not the black corners

bodies = np.load(WORK / "check" / "bodies.npz")
frames_all, verts_all = bodies["frame"], bodies["vertices"]
frames = np.arange(STEP // 2, frames_all.max() + 1, STEP)
cap = cv2.VideoCapture(VIDEO)
obs, keep_px, people_pts = [], [], []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    rows = np.flatnonzero(frames_all == f)
    pts = verts_all[rows][:, ::VERTS].astype(np.float32).reshape(-1, 3)
    people_pts.append(pts)
    # Pixels showing a person (or right next to one) don't show the floor
    px, depth = to_pixels(pts)
    covered = np.zeros((H, W), np.uint8)
    good = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    covered[px[good, 1].astype(int), px[good, 0].astype(int)] = 1
    covered = cv2.dilate(covered, np.ones((7, 7), np.uint8))
    ratio = gray(img) / np.maximum(bg_small, 1)
    low = pts[pts[:, 1] < 0.3][:, [0, 2]]
    near_feet = np.zeros((H, W), bool)
    if len(low):
        d, _ = cKDTree(low).query(ground[floor][:, [0, 2]], distance_upper_bound=CONTACT)
        near_feet[floor] = np.isfinite(d)
    use = floor & (covered == 0) & ~near_feet
    ratio /= np.median(ratio[use])  # exposure changes between frames
    obs.append(np.clip(1 - ratio, -0.3, 1))
    keep_px.append(use)
obs, keep_px = np.array(obs), np.array(keep_px)
print(f"{len(frames)} frames, {keep_px.sum()} floor pixels in all", flush=True)


def shadows(L):
    """Each frame's shadow mask (len(frames), H, W) of all bodies, from a light at L."""
    out = np.zeros((len(frames), H, W), np.float16)
    for k, pts in enumerate(people_pts):
        q = L + (pts - L) * (L[1] / np.maximum(L[1] - pts[:, 1], 1e-3))[:, None]
        px, depth = to_pixels(q)
        good = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        out[k, px[good, 1].astype(int), px[good, 0].astype(int)] = 1
        mask = cv2.morphologyEx(out[k].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        out[k] = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), SOFT)  # lights are panels: soft shadows
    return out


def fit(candidates, frame_set=None):
    t = time.time()
    S = np.array([shadows(L) for L in candidates])  # (C, F, H, W)
    use = keep_px.copy()
    if frame_set is not None:
        use[~np.isin(np.arange(len(frames)), frame_set)] = False
    rows = np.flatnonzero(use & ((S > 0.05).any(0) | (obs > 0.08)))  # where some light could shadow, or the floor is darker
    rows = np.sort(np.random.default_rng(0).choice(rows, min(MAX_ROWS, len(rows)), replace=False))
    A = S.reshape(len(S), -1)[:, rows].T.astype(np.float32)
    b = obs.reshape(-1)[rows]
    w, _ = nnls(A, b, maxiter=2000)
    resid = np.sqrt(np.mean((A @ w - b) ** 2))
    print(f"  {len(candidates)} candidates, {len(rows)} rows, rms {resid:.3f} (none: {np.sqrt(np.mean(b ** 2)):.3f}), {time.time() - t:.0f} s", flush=True)
    return w, S


def score(w, S, frame_set):
    """rms darkening error on the given frames' shadow-relevant pixels, with these lights and with none."""
    use = keep_px.copy()
    use[~np.isin(np.arange(len(frames)), frame_set)] = False
    on = w > 0
    rows = np.flatnonzero(use & ((S[on] > 0.05).any(0) | (obs > 0.08)))
    pred = np.tensordot(w[on], S[on].reshape(on.sum(), -1)[:, rows].astype(np.float32), 1)
    b = obs.reshape(-1)[rows]
    return float(np.sqrt(np.mean((pred - b) ** 2))), float(np.sqrt(np.mean(b ** 2)))


if len(sys.argv) > 5 and sys.argv[5] == "holdout":
    even, odd = np.arange(0, len(frames), 2), np.arange(1, len(frames), 2)
    xs = np.arange(-15, 20.01, COARSE)
    zs = np.arange(-25, 12.01, COARSE)
    coarse = np.array([[x, ceiling, z] for x in xs for z in zs])
    w, S = fit(coarse, even)
    print("fit on even frames; held-out odd frames: rms %.4f with lights, %.4f without" % score(w, S, odd))
    rng = np.random.default_rng(1)
    for trial in range(3):  # baseline: the same number of lights at random spots, refit only their shares
        idx = rng.choice(len(coarse), int((w > MIN_SHARE).sum()), replace=False)
        wr, Sr = fit(coarse[idx], even)
        print("random lights, held out: rms %.4f / %.4f" % score(wr, Sr, odd))
    sys.exit()


xs = np.arange(-15, 20.01, COARSE)
zs = np.arange(-25, 12.01, COARSE)
coarse = np.array([[x, ceiling, z] for x in xs for z in zs])
w, _ = fit(coarse)
picks = coarse[w > MIN_SHARE]
print("coarse picks:", [(p[[0, 2]].round(1).tolist(), round(float(x), 3)) for p, x in zip(picks, w[w > MIN_SHARE])], flush=True)

offsets = np.arange(-FINE_REACH, FINE_REACH + 1e-6, FINE)
fine = np.unique(np.round(np.array([[p[0] + dx, ceiling, p[2] + dz] for p in picks for dx in offsets for dz in offsets]), 3), axis=0)
w, S = fit(fine)
order = np.argsort(-w)
lights = [dict(position=fine[i].round(2).tolist(), share=round(float(w[i]), 3)) for i in order if w[i] > MIN_SHARE]
print("lights:", lights, flush=True)
(OUT / "lights.json").write_text(json.dumps(dict(ceiling=ceiling, frames=frames.tolist(), lights=lights), indent=1))

on = w > 0
model = np.tensordot(w[on], S[on].astype(np.float32), 1)
panels = []
for k in np.linspace(0, len(frames) - 1, 3).astype(int):
    o = np.where(keep_px[k], obs[k], 0)
    m = np.where(keep_px[k], model[k], 0)
    to = lambda a: cv2.applyColorMap(np.clip(a / 0.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    panels.append(np.hstack([to(o), to(m)]))
cv2.imwrite(str(OUT / "check.jpg"), cv2.resize(np.vstack(panels), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
