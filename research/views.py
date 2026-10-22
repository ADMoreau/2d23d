"""Render a vertex-colored mesh from four sides (painter's algorithm, flat shading) into one image.

usage: views.py <mesh.glb> <out.png> [size=300] [flat: colors only, no shading]
"""

import sys

import cv2
import numpy as np
import trimesh

mesh = trimesh.load(sys.argv[1], force="mesh")
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 300
v = mesh.vertices - mesh.bounds.mean(0)
scale = 0.85 * SIZE / np.ptp(v, 0).max()
colors = mesh.visual.vertex_colors[:, :3].astype(float)
light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
tiles = []
for angle in (0, 90, 180, 270):
    a = np.radians(angle)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])  # about the object's long axis (y)
    p = v @ R.T
    tri = p[mesh.faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    front = normal[:, 2] > 0  # the camera looks along -z
    shade = np.ones(len(normal)) if len(sys.argv) > 4 and sys.argv[4] == "flat" else 0.45 + 0.55 * np.abs(normal @ light)
    order = np.argsort(tri[:, :, 2].mean(1))  # far to near
    image = np.full((SIZE, SIZE, 3), 255, np.uint8)
    xy = np.stack([SIZE / 2 + tri[:, :, 0] * scale, SIZE / 2 - tri[:, :, 1] * scale], -1).astype(np.int32)
    face_color = colors[mesh.faces].mean(1)[:, ::-1]  # RGB -> BGR
    for i in order:
        if front[i]:
            cv2.fillConvexPoly(image, xy[i], (face_color[i] * shade[i]).clip(0, 255).tolist())
    cv2.putText(image, f"{angle} deg", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    tiles.append(image)
cv2.imwrite(sys.argv[2], np.hstack(tiles))
