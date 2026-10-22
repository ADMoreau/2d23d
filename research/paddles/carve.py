"""A 3D shape from the video's cutouts alone: soft space carving in the held object's frame.

Every voxel of a box around the grip is projected into each detection (pose = wrist joint × that view's grip, from
fit2.py) and samples SAM's mask there. A voxel's occupancy is the mean over the views that see it; above THRESHOLD
it's solid. Views where the paddle is edge-on are what pin down its thickness. Marching cubes gives the mesh, and
each vertex takes the median color of the views that see it face-on enough.

usage: carve.py <paddles run dir> <out dir> <track>   (needs scikit-image: run with the i23d venv)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, label
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes

D, OUT, track = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
CELL = 0.004  # m
BOUNDS = np.array([[-0.16, 0.16], [-0.10, 0.52], [-0.10, 0.10]])  # m, x y z in the grip frame
MIN_IOU = 0.5
THRESHOLD = 0.5

cam = json.loads((D.parent.parent / "work-smooth" / "bodies" / "camera.json").read_text())
recs = {(r["frame"], r["track_id"]): r for r in json.loads((D / "results.json").read_text())}
with np.load(D / "fit2" / "grips.npz") as f:
    data = {k.split("_", 1)[1]: f[k] for k in f.files if k.startswith(f"{track}_")}
axes = [np.arange(lo + CELL / 2, hi, CELL) for lo, hi in BOUNDS]
grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
voxels = grid.reshape(-1, 3)


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


frames = list(data["frames"])
views = []
for f, grip, fit in zip(data["det_frames"], data["det_grips"], data["det_iou"]):
    if fit < MIN_IOU:
        continue
    k = frames.index(f)
    R = data["R"][k] @ Rotation.from_rotvec(grip[:3]).as_matrix()
    t = data["R"][k] @ grip[3:] + data["origin"][k]
    r = recs[(int(f), track)]
    views.append(dict(R=R, t=t, rec=r, mask=cv2.imread(str(D / "crops" / f"{track}_{int(f):06d}_owlmask.png"), 0).astype(np.float32) / 255,
                      crop=cv2.imread(str(D / "crops" / f"{track}_{int(f):06d}.jpg"))))


def sample(v, points, image):
    px = ((project(points @ v["R"].T + v["t"]) - v["rec"]["origin"]) * v["rec"]["scale"]).astype(np.float32)
    h, w = image.shape[:2]
    inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < w - 1) & (px[:, 1] < h - 1)
    n = len(px)
    rows = np.pad(px, ((0, -n % 1024), (0, 0)), constant_values=-1).reshape(-1, 1024, 2)  # remap takes < 32767 per row
    values = cv2.remap(image, rows[..., 0], rows[..., 1], cv2.INTER_LINEAR)
    return values.reshape(-1, *image.shape[2:])[:n], inside


votes, seen = np.zeros(len(voxels)), np.zeros(len(voxels))
for v in views:
    m, inside = sample(v, voxels, v["mask"])
    votes += np.where(inside, m, 0)
    seen += inside
occupancy = np.where(seen > 0, votes / np.maximum(seen, 1), 0).reshape(grid.shape[:3])
occupancy = gaussian_filter(occupancy, 0.7)
solid = occupancy > THRESHOLD
labels, n = label(solid)
if n > 1:
    solid = labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])
verts, faces, normals, _ = marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
verts = (verts - 1) * CELL + BOUNDS[:, 0] + CELL / 2

# Vertex colors: median over views where the surface faces the camera
colors = []
for v in views:
    c, inside = sample(v, verts, v["crop"])
    to_camera = -(verts @ v["R"].T + v["t"])
    facing = np.einsum("ij,ij->i", normals @ v["R"].T, to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True))
    colors.append(np.where((inside & (facing > 0.3))[:, None], c.astype(float), np.nan))
colors = np.nanmedian(np.array(colors), 0)
colors = np.where(np.isnan(colors), 128, colors)[:, ::-1]
mesh = trimesh.Trimesh(verts, faces[:, ::-1], vertex_colors=np.hstack([colors, np.full((len(colors), 1), 255)]).astype(np.uint8))
mesh.export(OUT / f"carve_{track}.glb")

ext = (np.array(np.nonzero(solid)).ptp(1) + 1) * CELL * 100 if solid.any() else np.zeros(3)
report = dict(views=len(views), voxels_solid=int(solid.sum()), extent_cm_xyz=[round(float(e), 1) for e in ext],
              volume_cm3=round(float(solid.sum() * CELL**3 * 1e6), 1))
print(report, flush=True)
# Slices through the occupancy: the paddle's plane (z = 0) and a cut across its thickness (x = 0)
zmid, xmid = grid.shape[2] // 2, grid.shape[0] // 2
a = (occupancy[:, :, zmid].T[::-1] * 255).astype(np.uint8)
b = (occupancy[xmid, :, :][::-1] * 255).astype(np.uint8)
cv2.imwrite(str(OUT / f"slices_{track}.png"), cv2.resize(np.hstack([a, np.zeros((a.shape[0], 4), np.uint8), b]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
