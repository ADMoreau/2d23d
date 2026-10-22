"""Prototype: the export stage's GLB, plus each player's paddle on their wrist bone with the fitted grip.

The grip (fit.py) places the paddle in a hand frame built from SAM 3D Body's keypoints. Here that becomes a fixed
transform relative to MHR's wrist joint, averaged over the person's frames, so the paddle can be a child node of
the joint and follow the exported (smoothed) skeleton. The spread of that transform over frames shows how rigidly
the keypoint hand frame moves with the joint.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from twod23d import glb as glb_module
from twod23d.pipeline import Context, Settings
from twod23d.stages import export

WORK, D, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
HANDS = {"R": dict(wrist=41, index=28, middle=32, pinky=40), "L": dict(wrist=62, index=49, middle=53, pinky=61)}
grips = json.loads((D / "fit" / "grips.json").read_text())

with np.load(WORK / "bodies" / "bodies.npz") as f:
    raw = dict(f)


def hand_frame(k, hand):
    h = HANDS[hand]
    w, i, m, p = k[h["wrist"]], k[h["index"]], k[h["middle"]], k[h["pinky"]]
    x = (m - w) / np.linalg.norm(m - w)
    z = np.cross(i - w, p - w)
    z -= x * (z @ x)
    z /= np.linalg.norm(z)
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = np.stack([x, np.cross(z, x), z], 1), (i + m + p) / 3
    return H


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


def paddle_color(track_id):
    pixels = []
    for m in sorted((D / "crops").glob(f"{track_id}_*_owlmask.png"))[::3]:
        mask = cv2.imread(str(m), 0) > 127
        crop = cv2.imread(str(m).replace("_owlmask.png", ".jpg"))
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # keep away from the edges
        pixels.append(crop[mask])
    bgr = np.median(np.concatenate(pixels), axis=0)
    return export.linear(bgr[::-1][None])[0]


rig = export.load_rig()
attachments = {}  # track_id -> (wrist joint, local matrix, color)
for t, g in grips.items():
    t = int(t)
    rows = np.flatnonzero(raw["track_id"] == t)
    rows = rows[np.argsort(raw["frame"][rows])][::3]
    inputs = [torch.from_numpy(raw[k][rows]).float() for k in ("mhr_shape", "mhr_model_params", "mhr_expression")]
    with torch.inference_mode():
        vertices, state = rig.model(*inputs, True)
    rotation = export.SAM_AXES / 100
    offset = raw["vertices"][rows].astype(np.float64).mean(1) - vertices.double().numpy().mean(1) @ rotation.T
    joints = export.tqs_to_matrix(state.double().numpy())  # (n, 127, 4, 4) MHR cm
    place = np.tile(np.eye(4), (len(rows), 1, 1))
    place[:, :3, :3], place[:, :3, 3] = rotation, offset
    joints_cam = place[:, None] @ joints
    wrist_kp = raw["keypoints"][rows][:, HANDS[g["hand"]]["wrist"]]
    j = int(np.argmin(np.linalg.norm(joints_cam[:, :, :3, 3] - wrist_kp[:, None], axis=-1).mean(0)))
    G = np.eye(4)
    G[:3, :3], G[:3, 3] = Rotation.from_rotvec(g["grip"][:3]).as_matrix(), g["grip"][3:]
    local = np.array([np.linalg.inv(joints_cam[k, j]) @ hand_frame(raw["keypoints"][r].astype(np.float64), g["hand"]) @ G for k, r in enumerate(rows)])
    scale = np.cbrt(np.linalg.det(local[:, :3, :3]))
    rots = Rotation.from_matrix(local[:, :3, :3] / scale[:, None, None])
    mean_rot = rots.mean()
    spread = np.degrees((mean_rot.inv() * rots).magnitude())
    M = np.eye(4)
    M[:3, :3] = mean_rot.as_matrix() * np.median(scale)
    M[:3, 3] = np.median(local[:, :3, 3], axis=0)
    print(f"person {t}: hand {g['hand']}, joint {j} (wrist keypoint {np.linalg.norm(joints_cam[:, j, :3, 3] - wrist_kp, axis=-1).mean() * 100:.1f} cm away), "
          f"grip vs joint varies {np.median(spread):.1f} deg median, {np.percentile(spread, 90):.1f} deg 90th pct", flush=True)
    attachments[t] = (j, M, paddle_color(t))

# Run the normal export, adding the paddles just before the GLB is saved
names = {}
original_add_node, original_save = glb_module.GLB.add_node, glb_module.GLB.save


def add_node(self, name, parent=None, **properties):
    index = original_add_node(self, name, parent, **properties)
    names[name] = index
    return index


def save(self, path):
    vertices, faces, normals = paddle_mesh()
    for t, (j, M, color) in attachments.items():
        material = self.add_material(tuple(color) + (1.0,))
        mesh = self.add_mesh(f"paddle {t}", vertices, faces, material, normals)
        s = np.cbrt(np.linalg.det(M[:3, :3]))
        original_add_node(self, f"person {t} paddle", parent=names[f"person {t} joint {j}"], mesh=mesh, translation=M[:3, 3],
                          rotation=export.matrix_quaternion(M[:3, :3] / s), scale=[s] * 3)
    original_save(self, path)


glb_module.GLB.add_node, glb_module.GLB.save = add_node, save
export.GLB = glb_module.GLB
export.write_debug_video = lambda *a, **k: None  # not needed for the prototype
export.run(Context(video=Path("pickleball.mp4"), work=WORK, settings=Settings(), device="cpu"), OUT)
print("wrote", OUT / "scene.glb", flush=True)
