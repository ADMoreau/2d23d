"""Prototype: the export stage's GLB plus each player's paddle, animated under their wrist joint (fit2.py's grips).

A grip is the paddle's rotation and offset (meters) in the joint's unscaled frame. The joint node in the GLB is
scaled (the person's root scales MHR's centimeters to meters), so the paddle node undoes that scale per frame.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from twod23d import glb as glb_module
from twod23d.pipeline import Context, Settings
from twod23d.stages import export

WORK, D, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
fps = json.loads((WORK / "ingest" / "meta.json").read_text())["fps"]
with np.load(D / "fit2" / "grips.npz") as f:
    data = dict(f)
players = sorted({int(k.split("_")[0]) for k in data})


def box_mesh(x0, x1, y0, y1, t):
    """A box with flat-shaded faces: vertices, faces, normals."""
    z0, z1 = -t / 2, t / 2
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    vertices, faces, normals = [], [], []
    for q in quads:
        v = corners[list(q)]
        n = np.cross(v[1] - v[0], v[2] - v[0])
        n /= np.linalg.norm(n)
        if n @ (v.mean(0) - corners.mean(0)) < 0:  # face outward
            v, n = v[::-1], -n
        base = len(vertices)
        vertices += list(v)
        normals += [n] * 4
        faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(vertices), np.array(faces), np.array(normals)


def paddle_mesh():
    parts = [box_mesh(-0.016, 0.016, -0.02, 0.13, 0.025), box_mesh(-0.095, 0.095, 0.13, 0.40, 0.015)]
    v, f, n = [], [], []
    for pv, pf, pn in parts:
        f.append(pf + sum(len(x) for x in v))
        v.append(pv)
        n.append(pn)
    return np.concatenate(v), np.concatenate(f), np.concatenate(n)


names = {}
original_add_node, original_add_animation = glb_module.GLB.add_node, glb_module.GLB.add_animation


def add_node(self, name, parent=None, **properties):
    index = original_add_node(self, name, parent, **properties)
    names[name] = index
    return index


def add_animation(self, name, channels):
    vertices, faces, normals = paddle_mesh()
    for t in players:
        grips, scale, frames = data[f"{t}_grips"], data[f"{t}_scale"], data[f"{t}_frames"]
        color = export.linear(data[f"{t}_color"][::-1][None])[0]
        mesh = self.add_mesh(f"paddle {t}", vertices, faces, self.add_material(tuple(color) + (1.0,)), normals)
        rotation = Rotation.from_rotvec(grips[:, :3]).as_quat()  # x y z w, as glTF
        for k in range(1, len(rotation)):  # no sign flips between keys
            if rotation[k] @ rotation[k - 1] < 0:
                rotation[k] *= -1
        translation = grips[:, 3:] / scale[:, None]
        scales = np.repeat(1 / scale[:, None], 3, 1)
        node = original_add_node(self, f"person {t} paddle", parent=names[f"person {t} joint {int(data[f'{t}_joint'])}"], mesh=mesh,
                                 translation=translation[0], rotation=rotation[0], scale=scales[0])
        times = frames / fps
        channels = channels + [(node, "rotation", times, rotation, "LINEAR"), (node, "translation", times, translation, "LINEAR"),
                               (node, "scale", times, scales, "LINEAR")]
    return original_add_animation(self, name, channels)


glb_module.GLB.add_node, glb_module.GLB.add_animation = add_node, add_animation
export.GLB = glb_module.GLB
export.write_debug_video = lambda *a, **k: None  # not needed for the prototype
export.run(Context(video=Path("pickleball.mp4"), work=WORK, settings=Settings(), device="cpu"), OUT)
print("wrote", OUT / "scene.glb", flush=True)
