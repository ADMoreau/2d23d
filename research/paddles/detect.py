"""Paddle feasibility test: how often do open-vocabulary detectors find each player's paddle?

For every 9th frame and every tracked person, crop around the person (padded, since a paddle reaches past the
body), upscale, and ask OWLv2 and Grounding DINO for a paddle. Keep each detector's best box, then cut the best
one out with SAM 2.1. Writes results.json and per-player contact sheets.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection,
                          Owlv2Processor, Sam2Model, Sam2Processor)

torch.set_num_threads(4)
WORK = Path(sys.argv[1])
OUT = Path(sys.argv[2])
(OUT / "crops").mkdir(parents=True, exist_ok=True)
STEP = 9
PAD = 0.4  # of the person's box height, on every side
CROP = 640  # longer side of the upscaled crop
OWL_QUERIES = ["a pickleball paddle", "a paddle", "a tennis racket"]
GDINO_PROMPT = "pickleball paddle. paddle. racket."

frames = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
dets = json.loads((WORK / "track" / "tracks.json").read_text())["detections"]
by_frame = {}
for d in dets:
    by_frame.setdefault(d["frame"], []).append(d)

owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").eval()


def crop_person(image, box):
    x0, y0, x1, y1 = box
    p = PAD * (y1 - y0)
    H, W = image.shape[:2]
    cx0, cy0, cx1, cy1 = int(max(0, x0 - p)), int(max(0, y0 - p)), int(min(W, x1 + p)), int(min(H, y1 + p))
    crop = image[cy0:cy1, cx0:cx1]
    s = CROP / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), (cx0, cy0), s


@torch.inference_mode()
def owl_best(pil):
    inputs = owl_proc(text=[OWL_QUERIES], images=pil, return_tensors="pt")
    outputs = owl(**inputs)
    # OWLv2 pads to a square: boxes are relative to the padded size
    size = max(pil.size)
    r = owl_proc.post_process_grounded_object_detection(outputs, threshold=0.0, target_sizes=torch.tensor([[size, size]]))[0]
    if not len(r["scores"]):
        return None
    i = int(r["scores"].argmax())
    return {"score": float(r["scores"][i]), "box": [float(v) for v in r["boxes"][i]], "query": OWL_QUERIES[int(r["labels"][i])]}


@torch.inference_mode()
def gdino_best(pil):
    inputs = gd_proc(images=pil, text=GDINO_PROMPT, return_tensors="pt")
    outputs = gd(**inputs)
    r = gd_proc.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.05, text_threshold=0.05,
                                                       target_sizes=[pil.size[::-1]])[0]
    if not len(r["scores"]):
        return None
    i = int(r["scores"].argmax())
    labels = r.get("text_labels", r.get("labels"))
    return {"score": float(r["scores"][i]), "box": [float(v) for v in r["boxes"][i]], "query": str(labels[i])}


@torch.inference_mode()
def sam_mask(pil, box):
    inputs = sam_proc(images=pil, input_boxes=[[box]], return_tensors="pt")
    outputs = sam(**inputs, multimask_output=False)
    masks = sam_proc.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]
    return masks[0, 0].numpy() > 0, float(outputs.iou_scores[0, 0, 0])


results = []
start = time.time()
picked = list(range(4, len(frames), STEP))[: int(sys.argv[3]) if len(sys.argv) > 3 else None]
for n, f in enumerate(picked):
    image = cv2.imread(str(frames[f]))
    for d in by_frame.get(f, []):
        crop, origin, scale = crop_person(image, d["box"])
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        rec = {"frame": f, "track_id": d["track_id"], "person_h": d["box"][3] - d["box"][1], "origin": origin, "scale": scale,
               "owl": owl_best(pil), "gdino": None}
        best = max((x for x in (rec["owl"], rec["gdino"]) if x), key=lambda x: x["score"], default=None)
        name = f"{d['track_id']}_{f:06d}"
        cv2.imwrite(str(OUT / "crops" / f"{name}.jpg"), crop)
        if rec["owl"]:
            mask, iou = sam_mask(pil, rec["owl"]["box"])
            rec["owl"]["mask_px"] = int(mask.sum() / scale**2)  # in original-frame pixels
            rec["owl"]["sam_iou"] = iou
            cv2.imwrite(str(OUT / "crops" / f"{name}_owlmask.png"), mask.astype(np.uint8) * 255)
        results.append(rec)
    if n % 10 == 0:
        print(f"{n + 1}/{len(picked)} frames, {time.time() - start:.0f} s", flush=True)
        (OUT / "results.json").write_text(json.dumps(results))
(OUT / "results.json").write_text(json.dumps(results))
print("done", time.time() - start, flush=True)
