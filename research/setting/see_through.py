"""The place behind a see-through object (netting, a fence), from what shows through its holes.

At video resolution netting is resolved: thin dark strings, with holes between them that show what's behind. A
morphological closing (the local max, then the local min, over a few pixels) replaces each string with the holes
beside it: what's behind at nearly full resolution. Solid parts (posts, tape, signs, base) survive the closing; they
are told apart by differing a lot from a fill-in of the whole object (fill.py's LaMa), and LaMa fills just them,
given the recovered netting around them.

The object's region is its mask plus, if given, its fitted 3D shape drawn into the image (masks miss thin parts).

usage: see_through.py <work dir> <run dir> <object mask.png> [<other mask.png> ...] [--shape <net.json>]   (lab venv)
Writes filled_through.jpg, through_mask.png and compare_through.jpg.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(4)
import json

args = sys.argv[1:]
SHAPE = None
if "--shape" in args:
    i = args.index("--shape"); SHAPE = Path(args[i + 1]); del args[i:i + 2]
WORK, RUN, OBJ = Path(args[0]), Path(args[1]), Path(args[2])
OTHERS = [Path(p) for p in args[3:]]  # removed entirely (e.g. on-screen graphics)
LAMA = Path(__file__).resolve().parent.parent / "models" / "lama" / "big-lama.pt"
THIN = 5  # px: strings are thinner than this, and holes are this close
OPAQUE = 50  # gray levels a solid part differs from the fill-in, after the closing
DARK = 60  # gray level below which a part that survived the closing is solid

image = cv2.imread(str(WORK / "scene" / "background.jpg"))
H, W = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
lama_color = cv2.imread(str(RUN / "filled.jpg")).astype(np.float32)
obj = (cv2.imread(str(OBJ), 0) > 127).astype(np.uint8)
if SHAPE:  # an upright rectangle on the floor (net_build.py's net.json), projected
    net = json.loads(SHAPE.read_text())
    cam = json.loads((WORK / "bodies" / "camera.json").read_text())
    M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
    a, b = np.array(net["ends_world"])
    up = np.array([0.0, net["height_m"], 0.0])
    c = (np.array([a, b, b + up, a + up]) - M[:3, 3]) @ M[:3, :3]
    quad = np.stack([cam["fx"] * c[:, 0] / c[:, 2] + cam["cx"], cam["fy"] * c[:, 1] / c[:, 2] + cam["cy"]], -1)
    cv2.fillPoly(obj, [quad.round().astype(np.int32)], 1)
obj = cv2.dilate(obj, np.ones((5, 5), np.uint8)) > 0

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THIN, THIN))
closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)  # per channel; removes strings darker than what's behind
through = np.where(obj[..., None], closed, image)
closed_gray = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY).astype(np.float32)
lama_gray = cv2.cvtColor(lama_color.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
# Solid: far brighter or darker than the fill-in; or dark and thick (a black post against a dark wall)
solid = obj & ((np.abs(closed_gray - lama_gray) > OPAQUE) | ((closed_gray < DARK) & (lama_gray - closed_gray > 20)))
solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # specks aren't solid
solid = cv2.dilate(solid, np.ones((9, 9), np.uint8)) > 0  # with the blur at their edges
mask = solid & obj
for p in OTHERS:
    mask |= cv2.dilate((cv2.imread(str(p), 0) > 127).astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
print(f"object pixels {obj.sum()}: {(solid & obj).sum() / obj.sum():.0%} solid, the rest recovered through the holes", flush=True)
cv2.imwrite(str(RUN / "through_mask.png"), mask.astype(np.uint8) * 255)
image = through

model = torch.jit.load(str(LAMA), map_location="cpu").eval()
ph, pw = -H % 8, -W % 8
rgb = np.pad(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), ((0, ph), (0, pw), (0, 0)), mode="reflect")
m = np.pad(mask.astype(np.uint8), ((0, ph), (0, pw)), mode="reflect")
t = time.time()
with torch.inference_mode():
    out = model(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255, torch.from_numpy(m)[None, None].float())
print(f"LaMa: {time.time() - t:.0f} s", flush=True)
filled = cv2.cvtColor((out[0].permute(1, 2, 0).numpy()[:H, :W] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
filled = np.where(mask[..., None], filled, image)
cv2.imwrite(str(RUN / "filled_through.jpg"), filled, [cv2.IMWRITE_JPEG_QUALITY, 95])
crop = lambda i: i[360:600, 900:1700]
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
full = cv2.imread(str(RUN / "filled.jpg"))
cv2.imwrite(str(RUN / "compare_through.jpg"), np.vstack([crop(image), crop(full), crop(filled)]))
