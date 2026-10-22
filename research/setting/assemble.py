"""The export stage's GLB from a work folder whose scene is the setting layer (setting_build.py), plus the net as its
own see-through quad (net_build.py's texture), unlit like the scene.

usage: assemble.py <work dir> <net out dir> <out dir>   (repo venv)
"""

import json
import sys
from pathlib import Path

import numpy as np

from twod23d import glb as glb_module
from twod23d.pipeline import Context, Settings
from twod23d.stages import export

WORK, NET, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
net = json.loads((NET / "net.json").read_text())
a, b = np.array(net["ends_world"])
up = np.array([0.0, net["height_m"], 0.0])
original_save = glb_module.GLB.save


def save(self, path):
    texture = self.add_texture((NET / "texture.png").read_bytes(), "image/png")
    material = self.add_material((1.0, 1.0, 1.0, 1.0), texture=texture, unlit=True, double_sided=True)
    self.gltf["materials"][material]["alphaMode"] = "BLEND"
    vertices = np.array([a, b, b + up, a + up])
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)  # glTF: (0, 0) is the texture's top left
    mesh = self.add_mesh("net", vertices, np.array([[0, 1, 2], [0, 2, 3]]), material, uvs=uvs)
    self.add_node("net", mesh=mesh)
    original_save(self, path)


glb_module.GLB.save = save
export.GLB = glb_module.GLB
export.write_debug_video = lambda *a, **k: None
export.run(Context(video=Path("pickleball.mp4"), work=WORK, settings=Settings(), device="cpu"), OUT)
print("wrote", OUT / "scene.glb", flush=True)
