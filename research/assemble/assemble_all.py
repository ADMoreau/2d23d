"""Every layer in one GLB, in the scene hierarchy's order:
1. people: the export stage's animated bodies (work dir's check/bodies.npz, smoothed)
2. objects directly interacted with: each player's paddle (joint.py's shape and per-frame poses, recolor.py's colors)
   and the ball (ball3d.py's path)
3. pertinent objects: the net (net_build.py's see-through quad)
4. the setting: the work dir's scene (setting_build.py's, with the net and graphics removed and walls flat)
All in the floor's world (meters, y up). Layer 2's poses are fit in the camera's frame and moved to the world here.

usage: assemble_all.py <work dir> <lab dir> <out dir>   (repo venv; after paddle_meshes.py into <out dir>/meshes)
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

WORK, LAB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
fps = json.loads((WORK / "ingest" / "meta.json").read_text())["fps"]
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
PADDLES = LAB / "paddles" / "run"
JOINT = PADDLES / "joint_smooth_color"
MAX_GAP = 5  # frames: a paddle missing longer than this is hidden, not interpolated
FLIP = np.diag([-1.0, 1.0, -1.0])


def linear(srgb):
    c = np.asarray(srgb, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def paddle_track(t):
    """Frames, world rotations (n, 3, 3) and translations (n, 3) of player t's paddle, as joint.py's pose()."""
    with np.load(PADDLES / "fit3" / "grips.npz") as f:
        fit = {k.split("_", 1)[1]: f[k] for k in f.files if k.startswith(f"{t}_")}
    with np.load(JOINT / f"{t}.npz") as f:
        frames, correction = f["frames"], f["correction"].astype(np.float64)
    with np.load(JOINT / f"{t}_colors.npz") as f:
        flipped, fc = f["flipped"], f["flip_center"].astype(np.float64)
    index = {int(x): k for k, x in enumerate(fit["frames"])}
    k = np.array([index[int(x)] for x in frames])
    R_w, o_w, grips = fit["R"][k], fit["origin"][k], fit["grips"][k]
    R_g = Rotation.from_rotvec(grips[:, :3]).as_matrix()
    dR = Rotation.from_rotvec(correction[:, :3]).as_matrix()
    R = R_w @ R_g @ dR
    t_cam = np.einsum("nij,nj->ni", R_w, np.einsum("nij,nj->ni", R_g, correction[:, 3:]) + grips[:, 3:]) + o_w
    for i in np.flatnonzero(flipped):  # p -> R (FLIP (p - c) + c) + t
        t_cam[i] = t_cam[i] + R[i] @ (fc - FLIP @ fc)
        R[i] = R[i] @ FLIP
    return frames, M[:3, :3] @ R, t_cam @ M[:3, :3].T + M[:3, 3]


def keys(frames, rotations, translations):
    """Animation keys; a gap longer than MAX_GAP hides the object (scale 0, stepped) from just after to just before."""
    q = Rotation.from_matrix(rotations).as_quat()  # x y z w, as glTF
    for k in range(1, len(q)):  # no sign flips between keys
        if q[k] @ q[k - 1] < 0:
            q[k] *= -1
    times = frames / fps
    s_times, s_values = [0.0], [[0.0] * 3 if frames[0] > 0 else [1.0] * 3]
    s_times.append(times[0]); s_values.append([1.0] * 3)
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a > MAX_GAP:
            s_times += [(a + 1) / fps, b / fps]
            s_values += [[0.0] * 3, [1.0] * 3]
    s_times.append((frames[-1] + 1) / fps); s_values.append([0.0] * 3)
    return times, q, translations, np.array(s_times), np.array(s_values)


def sphere(radius, n=16):
    """A UV sphere: vertices, faces (outward, counter-clockwise), normals."""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n + 1), np.linspace(0, 2 * np.pi, 2 * n + 1), indexing="ij")
    normals = np.stack([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)], -1).reshape(-1, 3)
    idx = np.arange((n + 1) * (2 * n + 1)).reshape(n + 1, 2 * n + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return normals * radius, faces, normals


original_add_animation, original_save = glb_module.GLB.add_animation, glb_module.GLB.save


def add_animation(self, name, channels):
    # Layer 2: paddles
    for t in range(4):
        if not (OUT / "meshes" / f"paddle_{t}.npz").exists():
            continue
        with np.load(OUT / "meshes" / f"paddle_{t}.npz") as f:
            mesh = dict(f)
        material = self.add_material((1.0, 1.0, 1.0, 1.0))
        m = self.add_mesh(f"paddle {t}", mesh["vertices"], mesh["faces"], material, mesh["normals"], colors=linear(mesh["colors"]))
        frames, R, tr = paddle_track(t)
        times, q, tr, s_times, s_values = keys(frames, R, tr)
        node = self.add_node(f"paddle {t}", mesh=m, translation=tr[0], rotation=q[0], scale=s_values[0])
        channels = channels + [(node, "rotation", times, q, "LINEAR"), (node, "translation", times, tr, "LINEAR"),
                               (node, "scale", s_times, s_values, "STEP")]
    # Layer 2: the ball
    ball = json.loads((LAB / "ball" / "out3d" / "ball.json").read_text())
    frames, pos = np.array(ball["frames"]), np.array(ball["position"])
    v, f, n = sphere(ball["summary"]["diameter_m"] / 2)
    m = self.add_mesh("ball", v, f, self.add_material((0.85, 0.9, 0.1, 1.0)), n)
    times, _, tr, s_times, s_values = keys(frames, np.repeat(np.eye(3)[None], len(frames), 0), pos)
    node = self.add_node("ball", mesh=m, translation=tr[0], scale=s_values[0])
    channels = channels + [(node, "translation", times, tr, "LINEAR"), (node, "scale", s_times, s_values, "STEP")]
    return original_add_animation(self, name, channels)


def save(self, path):
    # Layer 3: the net
    net = json.loads((LAB / "net" / "out" / "net.json").read_text())
    a, b = np.array(net["ends_world"])
    up = np.array([0.0, net["height_m"], 0.0])
    texture = self.add_texture((LAB / "net" / "out" / "texture.png").read_bytes(), "image/png")
    material = self.add_material((1.0, 1.0, 1.0, 1.0), texture=texture, unlit=True, double_sided=True)
    self.gltf["materials"][material]["alphaMode"] = "BLEND"
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)  # glTF: (0, 0) is the texture's top left
    m = self.add_mesh("net", np.array([a, b, b + up, a + up]), np.array([[0, 1, 2], [0, 2, 3]]), material, uvs=uvs)
    self.add_node("net", mesh=m)
    original_save(self, path)


glb_module.GLB.add_animation, glb_module.GLB.save = add_animation, save
export.GLB = glb_module.GLB
export.write_debug_video = lambda *a, **k: None
export.run(Context(video=Path("pickleball.mp4"), work=WORK, settings=Settings(), device="cpu"), OUT)
print("wrote", OUT / "scene.glb", f"{(OUT / 'scene.glb').stat().st_size / 1e6:.1f} MB", flush=True)
