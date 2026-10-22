"""Prototype: the export stage's GLB plus each player's paddle, animated under their wrist joint (fit2.py's grips).

The paddle is its real outline and colors (shape2.py): a textured panel, cut out by the texture's alpha, on each face.
The video barely shows the handle (it's in the hand), so that is a plain box from the grip to the face's lower edge.

A grip is the paddle's rotation and offset (meters) in the joint's unscaled frame. The joint node in the GLB is
scaled (the person's root scales MHR's centimeters to meters), so the paddle node undoes that scale per frame.
"""

import json
import sys
from pathlib import Path

import cv2
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


shape = json.loads((D / "shape" / "shape.json").read_text())
X0, X1, Y0, Y1 = shape["extent"]
HALF = 0.0075  # half the paddle's thickness, m


def panel():
    """Two textured quads, one per face: vertices, faces, normals, uvs."""
    corners = np.array([[X0, Y1], [X1, Y1], [X1, Y0], [X0, Y0]])
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    vertices, normals, faces = [], [], []
    for z, quad in ((HALF, [0, 3, 2, 0, 2, 1]), (-HALF, [0, 1, 2, 0, 2, 3])):
        base = len(vertices)
        vertices += [[x, y, z] for x, y in corners]
        normals += [[0, 0, np.sign(z)]] * 4
        faces += [base + i for i in quad]
    return np.array(vertices), np.array(faces).reshape(-1, 3), np.array(normals), np.vstack([uvs, uvs])


def handle(t):
    """A box from the grip (y = -2 cm) up into the face's lower edge: vertices, faces, normals."""
    alpha = cv2.imread(str(D / "shape" / f"{t}.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 127
    rows = np.flatnonzero(alpha[:, alpha.shape[1] // 2])
    bottom = Y1 - (rows.max() + 1) / alpha.shape[0] * (Y1 - Y0) if len(rows) else 0.13
    corners = np.array([[x, y, z] for x in (-0.016, 0.016) for y in (-0.02, bottom + 0.02) for z in (-0.0125, 0.0125)])
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


names = {}
original_add_node, original_add_animation = glb_module.GLB.add_node, glb_module.GLB.add_animation


def add_node(self, name, parent=None, **properties):
    index = original_add_node(self, name, parent, **properties)
    names[name] = index
    return index


def add_animation(self, name, channels):
    vertices, faces, normals, uvs = panel()
    for t in players:
        grips, scale, frames = data[f"{t}_grips"], data[f"{t}_scale"], data[f"{t}_frames"]
        material = self.add_material((1.0, 1.0, 1.0, 1.0), self.add_texture((D / "shape" / f"{t}.png").read_bytes(), "image/png"))
        self.gltf["materials"][material] |= {"alphaMode": "MASK", "alphaCutoff": 0.5}
        mesh = self.add_mesh(f"paddle {t}", vertices, faces, material, normals, uvs=uvs)
        hv, hf, hn = handle(t)
        self.gltf["meshes"][mesh]["primitives"] += self.gltf["meshes"][self.add_mesh("handle", hv, hf, self.add_material((0.05, 0.05, 0.05, 1.0)), hn)]["primitives"]
        self.gltf["meshes"].pop()
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
