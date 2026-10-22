"""A shape prior for a held object from image-to-3D (TripoSR, MIT) on its clearest frames, in the object's own frame.

Frames are picked where the object faces the camera, its mask is big, and the hand covers little of it (joint.py's
poses). Each is cut out on gray and given to TripoSR. Its density field is then queried at joint.py's grid cells:
cell -> camera (that frame's pose) -> TripoSR's input camera (a similarity that maps the cutout onto the frame, at the
object's depth) -> TripoSR's world. The occupancy is averaged over the frames.

TripoSR's input camera looks at the origin from +x (distance 1.9, fov 40°, z up). That is checked here: its
reconstruction is projected back into each input and compared with the cutout's mask (input IoU).

usage: prior.py <work dir> <paddles run dir> <track> [frames=3]   (i23d venv, with src and shim on sys.path)
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from tsr.system import TSR

torch.set_num_threads(int(os.environ.get("THREADS", 3)))
WORK, D, track = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
COUNT = int(sys.argv[4]) if len(sys.argv) > 4 else 3
OUT = D / "prior"
OUT.mkdir(exist_ok=True)
TRACK_DIR = os.environ.get("TRACK_DIR", "track")
SEPARATION = 30  # frames between picks
MAX_SPEED = 0.03  # object lengths per frame, in the image
MIN_AGREE = 0.6  # IoU of the mask with joint.py's fitted shape
THRESHOLD = 25.0  # TripoSR's surface density
FILL = 0.85  # object's share of TripoSR's input
TSR_DISTANCE, TSR_FOV = 1.9, 40.0

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
with np.load(D / "fit3" / "grips.npz") as f:
    fit = {k.split("_", 1)[1]: f[k] for k in f.files if k.startswith(f"{track}_")}
with np.load(D / TRACK_DIR / f"{track}.npz") as f:
    tracked = dict(f)
with np.load(D / "joint" / f"{track}.npz") as f:
    joint = dict(f)
TSCALE = float(tracked["scale"])
frames_dir = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
bounds, cell = joint["bounds"], float(joint["cell"])
shape = joint["occupancy"].shape
axes = [bounds[i, 0] + cell / 2 + cell * np.arange(shape[i]) for i in range(3)]
cells = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


def pose(n):
    """joint.py's object -> camera pose for its n-th frame."""
    x = int(joint["frames"][n])
    k = list(fit["frames"]).index(x)
    Rw, ow, g = fit["R"][k], fit["origin"][k], fit["grips"][k]
    Rg = cv2.Rodrigues(g[:3])[0]
    c = joint["correction"][n]
    return Rw @ Rg @ cv2.Rodrigues(c[:3].astype(np.float64))[0], Rw @ (Rg @ c[3:] + g[3:]) + ow


# Pick frames: sharp (the object barely moves in the image), its mask agrees with joint.py's fitted shape, and it faces
# the camera with a big mask. Fast frames are blurred into fans and drifted masks are big, so both must be excluded first.
occupied = cells[joint["occupancy"].reshape(-1) > 0.5]
center = occupied.mean(0) if len(occupied) else np.array([0, 0.25, 0])
poses = [pose(n) for n in range(len(joint["frames"]))]
centers = np.array([project(R @ center + t) for R, t in poses])
lengths = np.array([cam["fx"] * 0.4 / (R @ center + t)[2] for R, t in poses])  # rough object length, px
scores = []
for n, x in enumerate(joint["frames"]):
    R, t = poses[n]
    near = [m for m in (n - 1, n + 1) if 0 <= m < len(poses) and abs(int(joint["frames"][m]) - int(x)) == 1]
    speed = max([np.linalg.norm(centers[m] - centers[n]) for m in near], default=np.inf) / lengths[n]
    mask = np.unpackbits(tracked["masks"][int(x)], axis=-1).astype(bool)
    sil = np.zeros((640, 640), np.uint8)
    q = np.round((project(occupied @ R.T + t) - tracked["origin"][int(x)]) * TSCALE).astype(int)
    q = q[(q >= 0).all(1) & (q < 640).all(1)]
    sil[q[:, 1], q[:, 0]] = 1
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    agree = (sil & mask).sum() / max((sil | mask).sum(), 1)
    ray = R @ center + t
    facing = abs(R[:, 2] @ ray) / np.linalg.norm(ray)
    scores.append(facing * mask.sum() * agree if speed < MAX_SPEED and agree > MIN_AGREE else 0)
print(f"{sum(v > 0 for v in scores)} of {len(scores)} frames are sharp and agree with the fit", flush=True)
picks = []
for n in np.argsort(scores)[::-1]:
    if scores[n] > 0 and all(abs(int(joint["frames"][n]) - int(joint["frames"][m])) >= SEPARATION for m in picks):
        picks.append(n)
    if len(picks) == COUNT:
        break

model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192)
radius = model.renderer.cfg.radius
f_tsr = 256 / math.tan(math.radians(TSR_FOV / 2))
votes, report, tiles = np.zeros(len(cells)), dict(track=track, frames=[]), []

# The TripoSR world's grid, for the input check
g = np.linspace(-radius, radius, 96)
world_grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def tsr_camera(P):
    """TripoSR world -> its input camera (x right, y down, z forward)."""
    return np.stack([P[:, 1], -P[:, 2], TSR_DISTANCE - P[:, 0]], -1)


def tsr_world(c):
    return np.stack([TSR_DISTANCE - c[:, 2], c[:, 0], -c[:, 1]], -1)


for n in picks:
    x = int(joint["frames"][n])
    R, t = pose(n)
    M = np.array([[TSCALE, 0, -tracked["origin"][x][0] * TSCALE], [0, TSCALE, -tracked["origin"][x][1] * TSCALE]])
    crop = cv2.warpAffine(cv2.imread(str(frames_dir[x])), M, (640, 640), flags=cv2.INTER_CUBIC)
    mask = np.unpackbits(tracked["masks"][x], axis=-1).astype(bool)
    ys, xs = np.nonzero(mask)
    side = max(np.ptp(xs), np.ptp(ys)) / FILL
    x0, y0 = (xs.min() + xs.max()) / 2 - side / 2, (ys.min() + ys.max()) / 2 - side / 2
    A = np.array([[512 / side, 0, -x0 * 512 / side], [0, 512 / side, -y0 * 512 / side]])  # crop px -> input px
    rgb = np.where(mask[..., None], cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 128).astype(np.uint8)
    image = cv2.warpAffine(rgb, A, (512, 512), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
    input_mask = cv2.warpAffine(mask.astype(np.uint8), A, (512, 512), flags=cv2.INTER_NEAREST) > 0
    start = time.time()
    with torch.no_grad():
        code = model([Image.fromarray(image)], device="cpu")[0]

        def density(P):
            return np.concatenate([model.renderer.query_triplane(model.decoder, torch.tensor(c, dtype=torch.float32), code)["density_act"].numpy().reshape(-1)
                                   for c in np.array_split(P, max(1, len(P) // 200000))])

        # Check the camera: TripoSR's reconstruction projected into its own input
        solid = world_grid[density(world_grid) > THRESHOLD]
        c = tsr_camera(solid)
        uv = np.round(f_tsr * c[:, :2] / c[:, 2:] + 256).astype(int)
        back = np.zeros((512, 512), np.uint8)
        ok = (uv >= 0).all(1) & (uv < 512).all(1)
        back[uv[ok, 1], uv[ok, 0]] = 1
        back = cv2.morphologyEx(back, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(bool)  # the grid's points are ~7 px apart
        input_iou = (back & input_mask).sum() / max((back | input_mask).sum(), 1)

        # Object-frame cells -> camera -> TripoSR's camera -> its world
        a = side / 512 / TSCALE  # input px -> image px
        b = np.array([x0, y0]) / TSCALE + tracked["origin"][x]
        Zc = (R @ center + t)[2]
        s = a * f_tsr * Zc / (cam["fx"] * TSR_DISTANCE)  # TripoSR units -> m
        Xc = np.array([(a * 256 + b[0] - cam["cx"]) * Zc / cam["fx"], (a * 256 + b[1] - cam["cy"]) * Zc / cam["fy"], Zc])
        in_camera = cells @ R.T + t
        in_tsr = tsr_world((in_camera - Xc) / s + [0, 0, TSR_DISTANCE])
        inside = (np.abs(in_tsr) < radius).all(1)
        d = np.zeros(len(cells))
        d[inside] = density(in_tsr[inside])
    occupancy = (d > THRESHOLD).astype(float)
    votes += occupancy
    # How well this frame's prior alone explains the object's mask in that frame
    sil = np.zeros((640, 640), np.uint8)
    q = np.round((project(cells[occupancy > 0] @ R.T + t) - tracked["origin"][x]) * TSCALE).astype(int)
    q = q[(q >= 0).all(1) & (q < 640).all(1)]
    sil[q[:, 1], q[:, 0]] = 1
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    frame_iou = (sil & mask).sum() / max((sil | mask).sum(), 1)
    solid_cells = cells[occupancy > 0]
    thick = float(np.ptp(solid_cells[:, 2]) * 100) if len(solid_cells) else 0.0
    report["frames"].append(dict(frame=x, input_iou=round(float(input_iou), 3), frame_iou=round(float(frame_iou), 3), scale_m_per_unit=round(float(s), 4),
                                 cells=int(occupancy.sum()), z_extent_cm=round(thick, 1), seconds=round(time.time() - start)))
    print(report["frames"][-1], flush=True)
    view = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(view, cv2.findContours(back.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 0), 2)
    tiles.append(cv2.resize(view, (256, 256)))

prior = (votes / max(len(picks), 1)).reshape(shape)
np.savez(OUT / f"{track}.npz", occupancy=prior, bounds=bounds, cell=cell, frames=[int(joint["frames"][n]) for n in picks])
solid = prior > 0.5
report["cells_solid"] = int(solid.sum())
report["extent_cm_xyz"] = [round(float((np.ptp(a) + 1) * cell * 100), 1) for a in np.nonzero(solid)] if solid.any() else [0, 0, 0]
(OUT / f"report_{track}.json").write_text(json.dumps(report, indent=1))
print(report, flush=True)
a = (prior[:, :, shape[2] // 2].T[::-1] * 255).astype(np.uint8)
b = (prior[shape[0] // 2][::-1] * 255).astype(np.uint8)
slices = cv2.resize(np.hstack([a, np.zeros((a.shape[0], 4), np.uint8), b]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
slices = cv2.cvtColor(cv2.resize(slices, (int(slices.shape[1] * 256 / slices.shape[0]), 256)), cv2.COLOR_GRAY2BGR)
cv2.imwrite(str(OUT / f"sheet_{track}.png"), np.hstack(tiles + [slices]))
