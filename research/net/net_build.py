"""A thin, upright object standing on the floor (a net, a fence, a board) as a textured, see-through surface in 3D.

1. Where it stands: the mask's bottom edge, column by column, meets the floor; rays through those pixels hit the fitted
   floor (y = 0) along a line (a robust fit). The object is the vertical plane on that line.
2. How tall: the mask's top edge, column by column, where its rays meet that plane.
3. See-through: what's behind it is estimated by filling its mask in from the surroundings (inpainting). Each pixel's
   opacity is how far its color moved from that toward black or white (netting, tape), and its own color is unmixed
   from it. Posts come out opaque, the holes clear.
4. A rectangle on the plane with that texture (RGBA), as a GLB; and the object drawn back over the filled-in background,
   which should look like the original image.

usage: net_build.py <work dir> <mask dir> <out dir>   (lab venv)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

WORK, MASK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
TEX_PER_M = 200  # texels per meter
LINE_TOL = 0.05  # m, inliers of the floor line

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
mask = cv2.imread(str(MASK / "mask.png"), 0) > 127
H, W = mask.shape


def ray(u, v):
    """World direction of the camera ray through pixel (u, v)."""
    d = np.stack([(np.asarray(u, float) - cam["cx"]) / cam["fx"], (np.asarray(v, float) - cam["cy"]) / cam["fy"], np.ones(np.shape(u))], -1)
    return d @ R_cw.T


def project(p):
    c = (p - t_cw) @ R_cw
    return np.stack([cam["fx"] * c[..., 0] / c[..., 2] + cam["cx"], cam["fy"] * c[..., 1] / c[..., 2] + cam["cy"]], -1)


# 1. The floor line, from the mask's bottom edge
cols = np.flatnonzero(mask.any(0))
bottom = np.array([np.flatnonzero(mask[:, c]).max() for c in cols])
top = np.array([np.flatnonzero(mask[:, c]).min() for c in cols])
d = ray(cols + 0.5, bottom + 0.5)
s = -t_cw[1] / d[:, 1]  # where each ray meets y = 0
floor_pts = t_cw + d * s[:, None]
ok = s > 0
xz = floor_pts[ok][:, [0, 2]]
rng = np.random.default_rng(0)
best = None
for _ in range(500):  # RANSAC: posts' wheels and the base bar touch the floor; the mask's lowest pixels mostly do
    a, b = xz[rng.choice(len(xz), 2, replace=False)]
    direction = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    normal = np.array([-direction[1], direction[0]])
    inliers = np.abs((xz - a) @ normal) < LINE_TOL
    if best is None or inliers.sum() > best.sum():
        best = inliers
center = xz[best].mean(0)
_, _, vt = np.linalg.svd(xz[best] - center)
direction = vt[0]
normal2 = np.array([-direction[1], direction[0]])
along = (xz - center) @ direction  # every column, not only the floor line's inliers: the posts' columns count for its ends
t0, t1 = np.percentile(along, [0.5, 99.5])
print(f"floor line: {best.sum()}/{len(xz)} columns within {LINE_TOL * 100:.0f} cm; length {t1 - t0:.2f} m", flush=True)

# The plane: points c + a·dir + h·up (world), c on the floor
c3 = np.array([center[0], 0.0, center[1]])
dir3 = np.array([direction[0], 0.0, direction[1]])
n3 = np.array([normal2[0], 0.0, normal2[1]])
up = np.array([0.0, 1.0, 0.0])

# 2. Heights: the top edge's rays meet the plane
dt = ray(cols + 0.5, top + 0.5)
s_top = ((c3 - t_cw) @ n3) / (dt @ n3)
top_pts = t_cw + dt * s_top[:, None]
heights = top_pts[:, 1]
pos_along = (top_pts - c3) @ dir3
inside = (pos_along > t0) & (pos_along < t1)
profile = np.percentile(heights[inside], [5, 50, 95])
height = float(np.percentile(heights[inside], 98))
print(f"top edge heights: 5th {profile[0]:.2f} m, median {profile[1]:.2f} m, 95th {profile[2]:.2f} m", flush=True)

# 3. See-through: behind = the mask filled in from its surroundings; opacity toward black or white
grown = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
behind = cv2.inpaint(image, grown, 9, cv2.INPAINT_TELEA).astype(np.float32)
I = image.astype(np.float32)
darker = (behind - I).mean(-1)
alpha_black = np.clip(darker / np.maximum(behind.mean(-1), 1), 0, 1)
alpha_white = np.clip(-darker / np.maximum(255 - behind.mean(-1), 1), 0, 1)
alpha = np.maximum(alpha_black, alpha_white)
alpha = np.clip(alpha * 1.5, 0, 1)  # thin strings rarely fill a whole pixel
alpha = alpha * cv2.GaussianBlur(grown.astype(np.float32) * mask, (0, 0), 1.0)
# Color: nearly opaque pixels (posts, tape, signs) keep theirs, unmixed from what's behind; see-through ones (strings) take
# the plain black or white that explained their opacity, since unmixing a faint string amplifies the fill-in's errors
unmixed = np.clip((I - (1 - alpha[..., None]) * behind) / np.maximum(alpha[..., None], 0.05), 0, 255)
plain = np.where((alpha_white > alpha_black)[..., None], 255.0, 20.0) * np.ones(3)
opaque = np.clip((alpha - 0.6) / 0.3, 0, 1)[..., None]
fg = opaque * unmixed + (1 - opaque) * plain

# 4. The texture on the plane: each texel's world point, seen in the image
tw, th = int((t1 - t0) * TEX_PER_M), int(height * TEX_PER_M)
a_s = np.linspace(t0, t1, tw)
h_s = np.linspace(height, 0, th)  # image rows run down
A, Hh = np.meshgrid(a_s, h_s)
world = c3 + A[..., None] * dir3 + Hh[..., None] * up
px = project(world).astype(np.float32)
sample = lambda img: cv2.remap(img, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
tex_rgb = sample(fg.astype(np.float32))
tex_a = sample(alpha.astype(np.float32))
rgba = np.dstack([cv2.cvtColor(tex_rgb.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB), (tex_a * 255).clip(0, 255).astype(np.uint8)])
Image.fromarray(rgba).save(OUT / "texture.png")

corners = np.array([c3 + t0 * dir3, c3 + t1 * dir3, c3 + t1 * dir3 + height * up, c3 + t0 * dir3 + height * up])
uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)  # glTF: v=1 at the top here since the texture's first row is the top
faces = np.array([[0, 1, 2], [0, 2, 3]])
material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(rgba), alphaMode="BLEND", doubleSided=True, metallicFactor=0.0, roughnessFactor=1.0)
mesh = trimesh.Trimesh(corners, faces, visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)
mesh.export(OUT / "net.glb")

# Check: the object drawn back over the filled-in background should look like the original image
recomposed = behind.copy()
inside_px = mask
recomposed[inside_px] = alpha[inside_px, None] * fg[inside_px] + (1 - alpha[inside_px, None]) * behind[inside_px]
x0, x1 = max(0, cols.min() - 30), min(W, cols.max() + 30)
y0, y1 = max(0, top.min() - 30), min(H, bottom.max() + 30)
outline = image.copy()
q = project(corners).astype(np.int32)
cv2.polylines(outline, [q], True, (0, 255, 255), 2)
rows = [image[y0:y1, x0:x1], behind.astype(np.uint8)[y0:y1, x0:x1], (alpha[y0:y1, x0:x1, None] * 255).repeat(3, -1).astype(np.uint8), outline[y0:y1, x0:x1]]
cv2.imwrite(str(OUT / "check.jpg"), np.vstack(rows))
report = dict(length_m=round(float(t1 - t0), 3), height_m=round(height, 3), height_median_m=round(float(profile[1]), 3),
              floor_line_inliers=int(best.sum()), columns=int(len(xz)), texture=[tw, th],
              ends_world=[np.round(c3 + t0 * dir3, 3).tolist(), np.round(c3 + t1 * dir3, 3).tolist()])
(OUT / "net.json").write_text(json.dumps(report, indent=1))
print(report, flush=True)
