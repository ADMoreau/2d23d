"""Ball feasibility: on every STEP-th frame, what finds the ball, and how big is it?

Three finders:
- RF-DETR's COCO "sports ball" class on the whole frame (the model shrinks it, so a ~10 px ball may vanish);
- the same on full-resolution tiles;
- motion against the scene stage's background image (people removed): small, compact blobs that differ from it and lie
  outside the people's boxes (a fixed camera only; generic for any small moving object).

usage: probe.py <work dir> <out dir> [step=3]   (repo venv: needs rfdetr)
Writes probe.json and a contact sheet of the best candidates per finder.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(3)
BALL = 37  # COCO "sports ball" in RF-DETR's output (1-based ids; person is 1)
TILE = 640
DIFF = 40  # background difference (max over channels) that counts as changed
AREA = (6, 900)  # px, a blob's size range
COMPACT = 0.35  # blob area / its min enclosing circle's area: a ball is round, a streak is long

import rfdetr  # noqa: E402

model = rfdetr.RFDETRMedium(device="cpu")
frames = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
background = cv2.imread(str(WORK / "scene" / "background.jpg")).astype(np.int16)
people = {}
for d in json.loads((WORK / "track" / "tracks.json").read_text())["detections"]:
    people.setdefault(d["frame"], []).append(d["box"])


def detect(image, offset=(0, 0)):
    found = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=0.05, include_source_image=False)
    keep = found.class_id == BALL
    return [dict(box=[float(b[0] + offset[0]), float(b[1] + offset[1]), float(b[2] + offset[0]), float(b[3] + offset[1])], score=float(s))
            for b, s in zip(found.xyxy[keep], found.confidence[keep])]


def motion(image, i):
    if background.shape != image.shape:
        return []
    changed = (np.abs(image.astype(np.int16) - background).max(-1) > DIFF).astype(np.uint8)
    for x0, y0, x1, y1 in people.get(i, []):
        changed[int(y0):int(y1), int(x0):int(x1)] = 0
    n, labels, stats, centers = cv2.connectedComponentsWithStats(changed)
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if not AREA[0] <= area <= AREA[1]:
            continue
        pts = np.column_stack(np.nonzero(labels == k))[:, ::-1].astype(np.float32)
        (_, _), r = cv2.minEnclosingCircle(pts)
        compact = area / (np.pi * max(r, 0.5) ** 2)
        if compact >= COMPACT:
            out.append(dict(center=[float(c) for c in centers[k]], area=int(area), compact=round(float(compact), 2)))
    return out


results, start = [], time.time()
for i in range(0, len(frames), STEP):
    image = cv2.imread(str(frames[i]))
    h, w = image.shape[:2]
    whole = detect(image)
    tiles = []
    for y in range(0, h, TILE - 64):
        for x in range(0, w, TILE - 64):
            tiles += detect(image[y:y + TILE, x:x + TILE], (x, y))
    results.append(dict(frame=i, whole=whole, tiles=tiles, motion=motion(image, i)))
    if len(results) % 25 == 0:
        print(f"frame {i}: {time.time() - start:.0f} s", flush=True)
(OUT / "probe.json").write_text(json.dumps(results))

# Summary and contact sheets: each finder's most confident candidates
for name in ("whole", "tiles"):
    best = [(r["frame"], max(r[name], key=lambda d: d["score"])) for r in results if r[name]]
    scores = [d["score"] for _, d in best]
    sizes = [max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for _, d in best]
    print(name, f"frames with a ball box: {len(best)}/{len(results)}, score>0.3: {sum(s > 0.3 for s in scores)}, size median {np.median(sizes) if sizes else 0:.0f} px", flush=True)
    tiles_img = []
    for f, d in sorted(best, key=lambda fd: -fd[1]["score"])[:24]:
        image = cv2.imread(str(frames[f]))
        cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
        x0, y0 = int(np.clip(cx - 40, 0, image.shape[1] - 80)), int(np.clip(cy - 40, 0, image.shape[0] - 80))
        crop = cv2.resize(image[y0:y0 + 80, x0:x0 + 80], (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{f} {d['score']:.2f}", (3, 14), 0, 0.45, (255, 255, 255), 1)
        tiles_img.append(crop)
    if tiles_img:
        while len(tiles_img) % 8:
            tiles_img.append(np.zeros_like(tiles_img[0]))
        cv2.imwrite(str(OUT / f"sheet_{name}.jpg"), np.vstack([np.hstack(tiles_img[k:k + 8]) for k in range(0, len(tiles_img), 8)]))
blobs = sum(len(r["motion"]) for r in results)
print("motion", f"frames with a candidate blob: {sum(bool(r['motion']) for r in results)}/{len(results)}, blobs per frame {blobs / len(results):.1f}", flush=True)
