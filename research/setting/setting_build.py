"""The setting layer from the scene stage's mesh: the net's stand-in geometry replaced by what's behind it, walls and
ceiling made flat, everything painted with the cleaned image (see_through.py: net and on-screen graphics removed).

The scene mesh is a grid over the image (every STEP-th pixel), in two layers: floor (on y = 0) and the rest (MoGe's
points). Here, per grid node:
- In the net's region (its mask plus its fitted quad), the rest layer goes; the node's ray meets the floor or, if a
  wall comes first, that wall: what's behind the net.
- Rest nodes within SNAP of a wall or the ceiling move onto it along their ray, so the texture stays put.

usage: setting_build.py <work dir> <run dir> <net mask.png> <net.json> <out dir>   (lab venv)
Writes <out>/scene/{scene.npz, background.jpg, scene.json} in the scene stage's format, and <out>/views.jpg.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

WORK, RUN, NET_MASK, NET, OUT = (Path(p) for p in sys.argv[1:6])
(OUT / "scene").mkdir(parents=True, exist_ok=True)
GRID = 320  # as the scene stage
MAX_STRETCH = 1.1
SNAP = 0.20  # m: rest points this close to a plane are on it

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
planes = json.loads((RUN / "walls.json").read_text())
z = dict(np.load(WORK / "scene" / "scene.npz"))
H, W = 1080, 1920
step = max(1, math.ceil(max(H, W) / GRID))
rows, cols = (np.unique(np.append(np.arange(0, n, step), n - 1)) for n in (H, W))

# Rays and floor points at the grid nodes
u, v = np.meshgrid(cols + 0.5, rows + 0.5)
rays = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], -1) @ R_cw.T
with np.errstate(divide="ignore"):
    s_floor = np.where(rays[..., 1] < -1e-9, -t_cw[1] / rays[..., 1], np.inf)
s_planes = []
for p in planes:  # where each ray meets each plane, if in front
    n, d = np.array(p["normal"]), p["offset"]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (d - t_cw @ n) / (rays @ n)
    s_planes.append(np.where(s > 0, s, np.inf))
s_wall = np.min(s_planes, axis=0)

# The scene's two layers back onto the grid
r_idx = np.clip(np.searchsorted(rows, z["uvs"][:, 1] * H - 0.5 - 0.01), 0, len(rows) - 1)
c_idx = np.clip(np.searchsorted(cols, z["uvs"][:, 0] * W - 0.5 - 0.01), 0, len(cols) - 1)
is_floor = np.abs(z["vertices"][:, 1]) < 1e-4
floor_ok = np.zeros(u.shape, bool)
floor_ok[r_idx[is_floor], c_idx[is_floor]] = True
rest = np.full(u.shape + (3,), np.nan)
rest[r_idx[~is_floor], c_idx[~is_floor]] = z["vertices"][~is_floor]
rest_ok = np.isfinite(rest).all(-1)
xz = z["vertices"][is_floor][:, [0, 2]]
low, high = xz.min(0), xz.max(0)  # the floor layer's extent, as the scene stage kept it

# The net's region
region = (cv2.imread(str(NET_MASK), 0) > 127).astype(np.uint8)
net = json.loads(NET.read_text())
a, b = np.array(net["ends_world"])
up = np.array([0.0, net["height_m"], 0.0])
c = (np.array([a, b, b + up, a + up]) - t_cw) @ R_cw
quad = np.stack([cam["fx"] * c[:, 0] / c[:, 2] + cam["cx"], cam["fy"] * c[:, 1] / c[:, 2] + cam["cy"]], -1)
cv2.fillPoly(region, [quad.round().astype(np.int32)], 1)
region = cv2.dilate(region, np.ones((2 * step + 1,) * 2, np.uint8))[rows][:, cols] > 0
ground = t_cw + np.where(np.isfinite(s_floor), s_floor, 0)[..., None] * rays
floor_first = np.isfinite(s_floor) & (s_floor < s_wall) & np.all((ground[..., [0, 2]] > low) & (ground[..., [0, 2]] < high), -1)
rest_ok &= ~region
floor_ok |= region & floor_first
behind_wall = region & ~floor_first & np.isfinite(s_wall)
rest[behind_wall] = t_cw + s_wall[behind_wall][:, None] * rays[behind_wall]
rest_ok |= behind_wall
print(f"net region: {region.sum()} nodes; now floor {(region & floor_first).sum()}, wall {behind_wall.sum()}", flush=True)

# Flat walls and ceiling: each rest node near a plane moves onto it along its ray
moved = np.zeros(u.shape, bool)
best = np.full(u.shape, np.inf)
for p, s in zip(planes, s_planes):
    n, d = np.array(p["normal"]), p["offset"]
    dist = np.abs(np.nan_to_num(rest @ n - d, nan=np.inf))
    take = rest_ok & (dist < SNAP) & (dist < best) & np.isfinite(s)
    rest[take] = t_cw + s[take][:, None] * rays[take]
    best[take] = dist[take]
    moved |= take
print(f"rest nodes: {rest_ok.sum()}, moved onto a plane: {moved.sum()} ({moved.sum() / rest_ok.sum():.0%}), "
      f"by {np.median(best[moved]) * 100:.1f} cm median", flush=True)


def grid_mesh(points, keep, max_stretch=np.inf):
    """The scene stage's grid_mesh, on grid nodes already picked."""
    index = np.full(keep.shape, -1)
    index[keep] = np.arange(keep.sum())
    tl, tr, bl, br = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = np.concatenate([np.stack([tl, bl, tr], -1).reshape(-1, 3), np.stack([tr, bl, br], -1).reshape(-1, 3)])
    faces = faces[(faces >= 0).all(axis=1)]
    vertices = points[keep]
    distance = np.linalg.norm(vertices - t_cw, axis=1)[faces]
    faces = faces[distance.max(axis=1) <= max_stretch * distance.min(axis=1)]
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    away = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), t_cw - p0) < 0
    faces[away] = faces[away][:, [0, 2, 1]]
    used = np.zeros(len(vertices), bool)
    used[faces] = True
    uvs = np.stack([((u - 0.5 + 0.5) / W)[keep], ((v - 0.5 + 0.5) / H)[keep]], -1)
    renumber = np.cumsum(used) - 1
    return vertices[used].astype(np.float32), renumber[faces].astype(np.uint32), uvs[used].astype(np.float32)


layers = [grid_mesh(ground, floor_ok), grid_mesh(rest, rest_ok, MAX_STRETCH)]
offsets = np.cumsum([0] + [len(l[0]) for l in layers])
vertices = np.concatenate([l[0] for l in layers])
faces = np.concatenate([l[1] + o for l, o in zip(layers, offsets)]).astype(np.uint32)
uvs = np.concatenate([l[2] for l in layers])
np.savez_compressed(OUT / "scene" / "scene.npz", vertices=vertices, faces=faces, uvs=uvs,
                    backdrop_vertices=z["backdrop_vertices"], backdrop_faces=z["backdrop_faces"], backdrop_uvs=z["backdrop_uvs"])
shutil.copy(RUN / "filled_through.jpg", OUT / "scene" / "background.jpg")
report = json.loads((WORK / "scene" / "scene.json").read_text())
report |= {"vertices": len(vertices), "triangles": len(faces), "setting": {"snapped": int(moved.sum()), "net_nodes": int(region.sum())}}
(OUT / "scene" / "scene.json").write_text(json.dumps(report, indent=2))
print(f"old: {len(z['faces'])} triangles; new: {len(faces)}", flush=True)
