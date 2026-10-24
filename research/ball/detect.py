"""Ball candidates on every frame: RF-DETR's COCO "sports ball" class on full-resolution tiles (whole frames shrink a
~20 px ball away). Overlapping tiles' duplicates are merged. Resumable: frames already in the output are skipped.

usage: detect.py <work dir> <out dir>   (repo venv: needs rfdetr)
Writes detections.jsonl: one line per frame, {"frame", "balls": [{"box", "score"}]} (score >= 0.05).
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", 3)))
BALL = 37  # COCO "sports ball" in RF-DETR's output (1-based ids)
TILE, OVERLAP = 640, 64
MIN_SCORE = 0.05
SAME = 10  # px between centers: one ball seen in two tiles

import rfdetr  # noqa: E402

model = rfdetr.RFDETRMedium(device="cpu")
frames = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
path = OUT / "detections.jsonl"
done = {json.loads(line)["frame"] for line in path.open()} if path.exists() else set()
start = time.time()
with path.open("a") as out:
    for i, file in enumerate(frames):
        if i in done:
            continue
        image = cv2.imread(str(file))
        h, w = image.shape[:2]
        found = []
        for y in range(0, h, TILE - OVERLAP):
            for x in range(0, w, TILE - OVERLAP):
                tile = cv2.cvtColor(image[y:y + TILE, x:x + TILE], cv2.COLOR_BGR2RGB)
                d = model.predict(tile, threshold=MIN_SCORE, include_source_image=False)
                keep = d.class_id == BALL
                found += [([float(b[0] + x), float(b[1] + y), float(b[2] + x), float(b[3] + y)], float(s)) for b, s in zip(d.xyxy[keep], d.confidence[keep])]
        balls = []
        for box, score in sorted(found, key=lambda bs: -bs[1]):
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            if all(np.linalg.norm(c - np.array([(b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2])) > SAME for b in balls):
                balls.append(dict(box=[round(v, 1) for v in box], score=round(score, 3)))
        out.write(json.dumps(dict(frame=i, balls=balls)) + "\n")
        out.flush()
        if i % 50 == 0:
            print(f"frame {i}: {time.time() - start:.0f} s", flush=True)
print("done", flush=True)
