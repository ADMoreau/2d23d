"""Color a fitted object's mesh the way a camera sees it: each vertex looks in along its normal from just outside, and
the fitted color grid is blended along that ray exactly as joint.py's renderer does (cells weighted by how much of the
ray reaches and stops in them). Reading at a fixed depth instead lands between a thin plate's front and back colors.

usage: recolor.py <joint dir with {track}.npz and {track}_colors.npz> <track>   (writes joint_{track}.glb there)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

OUT, track = Path(sys.argv[1]), int(sys.argv[2])
with np.load(OUT / f"{track}.npz") as f:
    occ, bounds, cell = f["occupancy"], f["bounds"], float(f["cell"])
with np.load(OUT / f"{track}_colors.npz") as f:
    colors = torch.tensor(f["colors"])
LOOK = 4  # cells: start this far outside the surface, march this far inside
SAMPLES = 32

v, faces, _, _ = marching_cubes(np.pad(gaussian_filter(occ, 0.6), 1), 0.5)
v = (v - 1) * cell + bounds[:, 0] + cell / 2
mesh = trimesh.Trimesh(v, faces[:, ::-1])
if mesh.volume < 0:
    mesh.invert()
trimesh.smoothing.filter_taubin(mesh, iterations=10)

density = torch.tensor(-np.log1p(-np.clip(occ, 0, 1 - 1e-6)), dtype=torch.float32).permute(2, 1, 0)[None, None]  # z y x
lo, hi = torch.tensor(bounds[:, 0], dtype=torch.float32), torch.tensor(bounds[:, 1], dtype=torch.float32)
vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
t = torch.linspace(-LOOK * cell, LOOK * cell, SAMPLES)
points = vertices[:, None] + normals[:, None] * (-t)[None, :, None]  # outside to inside
g = (2 * (points - lo) / (hi - lo) - 1).reshape(1, -1, 1, 1, 3)
with torch.no_grad():
    sigma = F.grid_sample(density, g, align_corners=False, padding_mode="zeros").reshape(len(vertices), SAMPLES) / cell
    stop = 1 - torch.exp(-sigma * (2 * LOOK * cell / SAMPLES))
    reach = torch.cumprod(torch.cat([torch.ones_like(stop[:, :1]), 1 - stop[:, :-1] + 1e-7], -1), -1)
    w = reach * stop
    c = torch.sigmoid(F.grid_sample(colors, g, align_corners=False, padding_mode="border")).reshape(3, len(vertices), SAMPLES)
    rgb = (w[None] * c).sum(-1) / w.sum(-1).clamp_min(1e-6)[None]
mesh.visual.vertex_colors = np.hstack([rgb.T.numpy() * 255, np.full((len(vertices), 1), 255)]).astype(np.uint8)
mesh.export(OUT / f"joint_{track}.glb")
print(f"{len(vertices)} vertices, rays that hit: {float((w.sum(-1) > 0.5).float().mean()):.3f}")
