"""Fixture candidates: every instance of the given names in the scene's background image (Grounding DINO), cut out with
SAM 2.1, each turned into a floor footprint (the mask's lowest pixel per column, where its ray meets the floor) and a
height (the mask's top over that footprint).

Objects that hang in the air (a hoop, a backboard) have no floor line under them. For those, the scene's MoGe points
inside the mask (already scaled to the bodies) place them instead: if the mask's lowest points are RAISED above the
floor, the footprint is the points' extent on the floor and the object gets a bottom as well as a top.

usage: candidates.py <work dir> <out dir> <name> [<name> ...]   (repo venv: transformers)
Writes candidates.json (name, score, box, footprint ends in world x, z, height) and overlay.jpg.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

torch.set_num_threads(4)
WORK, OUT, NAMES = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
OUT.mkdir(parents=True, exist_ok=True)
MIN_SCORE, MAX_AREA = 0.15, 0.25  # boxes over a quarter of the image are the whole place, not an object
FOOT_TOL = 0.05  # of the mask's height: columns whose lowest pixel is this close to the bottom line are on the floor
MAX_RANGE = 80.0  # m from the camera: beyond it, the floor is too grazing to place anything
RAISED = 0.5  # m: a mask whose lowest scene points (10th percentile) are this high hangs in the air...
MIN_COVER = 0.5  # ...if the scene (kept only near the play) covers this much of it; else it's far behind
MIN_POINTS = 20
SCENE_GRID = 320  # the scene mesh's vertices across the image's longer side
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
height_px, width_px = image.shape[:2]
pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cam = json.loads((WORK / "bodies" / "camera.json").read_text())
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
scale = width_px / json.loads((WORK / "ingest" / "meta.json").read_text())["width"]  # background may be smaller
scene_points, scene_px = np.zeros((0, 3)), np.zeros((0, 2), int)
if (WORK / "scene" / "scene.npz").exists():
    with np.load(WORK / "scene" / "scene.npz") as sc:
        scene_points = sc["vertices"]
        scene_px = np.minimum((sc["uvs"] * [width_px, height_px]).astype(int), [width_px - 1, height_px - 1])


def ray(u, v):
    d = np.array([(u / scale - cam["cx"]) / cam["fx"], (v / scale - cam["cy"]) / cam["fy"], 1.0])
    return R_cw @ d


gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").eval()
found = []
for name in NAMES:
    print(name, flush=True)
    with torch.inference_mode():
        inputs = gd_proc(images=pil, text=f"{name}.", return_tensors="pt")
        r = gd_proc.post_process_grounded_object_detection(gd(**inputs), inputs.input_ids, threshold=MIN_SCORE, text_threshold=MIN_SCORE,
                                                           target_sizes=[pil.size[::-1]])[0]
    boxes = [(float(s), [float(v) for v in b]) for s, b in zip(r["scores"], r["boxes"])]
    boxes = [(s, b) for s, b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < MAX_AREA * width_px * height_px]
    keep = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in boxes], [s for s, _ in boxes], MIN_SCORE, 0.5)
    for i in np.array(keep).flatten():
        s, box = boxes[i]
        with torch.inference_mode():
            inputs = sam_proc(images=pil, input_boxes=[[box]], return_tensors="pt")
            out = sam(**inputs, multimask_output=False)
        mask = sam_proc.post_process_masks(out.pred_masks, inputs["original_sizes"])[0][0, 0].numpy() > 0
        cols = np.flatnonzero(mask.any(0))
        if len(cols) < 3:
            continue
        inside = scene_points[mask[scene_px[:, 1], scene_px[:, 0]]]
        cover = len(inside) / (mask.sum() / np.ceil(max(width_px, height_px) / SCENE_GRID) ** 2)
        if len(inside) >= MIN_POINTS and cover >= MIN_COVER and np.percentile(inside[:, 1], 10) > RAISED:
            xz = inside[:, [0, 2]]
            c = np.median(xz, 0)
            axis = np.linalg.svd(xz - c)[2][0]
            along = np.percentile((xz - c) @ axis, [5, 95])
            found.append({"name": name, "score": round(s, 3), "box": [round(v) for v in box],
                          "ends": [(c + a * axis).round(2).tolist() for a in along],
                          "height": round(float(np.percentile(inside[:, 1], 95)), 2),
                          "bottom": round(float(np.percentile(inside[:, 1], 5)), 2), "cover": round(float(cover), 2)})
            print(found[-1], flush=True)
            continue
        # The bottom line: RANSAC over each column's lowest pixel. Columns that only hold the top of the object (a
        # crossbar seen past a post) sit far above it and are left out.
        cols = cols[:: max(1, len(cols) // 80)]
        low = np.array([np.flatnonzero(mask[:, u]).max() for u in cols], float)
        rows_any = np.flatnonzero(mask.any(1))
        tol = max(4.0, FOOT_TOL * (rows_any.max() - rows_any.min()))
        rng = np.random.default_rng(0)
        best_in = np.zeros(len(cols), bool)
        for _ in range(300):
            i, j = rng.choice(len(cols), 2, replace=False)
            line = low[i] + (low[j] - low[i]) * (cols - cols[i]) / (cols[j] - cols[i])
            inliers = np.abs(low - line) < tol
            if inliers.sum() > best_in.sum():
                best_in = inliers
        foot, tops = [], []
        for u in cols[best_in]:
            rows = np.flatnonzero(mask[:, u])
            d = ray(u, rows.max())
            if d[1] >= 0:
                continue  # above the horizon
            p = t_cw + d * (-t_cw[1] / d[1])
            if np.linalg.norm(p - t_cw) > MAX_RANGE:
                continue
            # the top of the mask in this column, over the same floor point (the object stands upright)
            dt = ray(u, rows.min())
            s_top = np.dot(p[[0, 2]] - t_cw[[0, 2]], dt[[0, 2]]) / np.dot(dt[[0, 2]], dt[[0, 2]])
            foot.append(p[[0, 2]])
            tops.append(t_cw[1] + s_top * dt[1])
        if len(foot) < 3:
            continue
        foot = np.array(foot)
        # the footprint as a segment: its principal direction, from end to end
        c = foot.mean(0)
        axis = np.linalg.svd(foot - c)[2][0]
        along = (foot - c) @ axis
        ends = [c + along.min() * axis, c + along.max() * axis]
        found.append({"name": name, "score": round(s, 3), "box": [round(v) for v in box],
                      "ends": [e.round(2).tolist() for e in ends], "height": round(float(np.median(tops)), 2)})
        print(found[-1], flush=True)
(OUT / "candidates.json").write_text(json.dumps(found, indent=1))
view = image.copy()
for n, c in enumerate(found):
    x0, y0, x1, y1 = c["box"]
    cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.putText(view, f"{n} {c['name']} {c['score']:.2f}", (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
cv2.imwrite(str(OUT / "overlay.jpg"), cv2.resize(view, None, fx=1600 / width_px, fy=1600 / width_px))
