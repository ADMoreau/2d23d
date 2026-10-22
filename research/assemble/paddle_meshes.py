"""Paddle meshes (joint.py/recolor.py GLBs) as plain arrays for assemble_all.py, which runs without trimesh.

usage: paddle_meshes.py <joint dir> <out dir>   (lab venv)
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

JOINT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
for path in sorted(JOINT.glob("joint_*.glb")):
    mesh = trimesh.load(path, force="mesh")
    np.savez(OUT / f"paddle_{path.stem.split('_')[1]}.npz", vertices=mesh.vertices, faces=mesh.faces,
             normals=mesh.vertex_normals, colors=mesh.visual.vertex_colors[:, :3])
    print(path.name, len(mesh.vertices), "vertices")
