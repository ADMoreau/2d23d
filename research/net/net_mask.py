"""The net's mask in the scene stage's background image (people removed): Grounding DINO finds it by name, SAM 2.1 cuts it
out. Generic for any named, static object in the scene.

usage: net_mask.py <work dir> <out dir> [name=net]   (repo venv: transformers)
Writes mask.png (the best-scoring box's mask), candidates.json and overlay.jpg.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

torch.set_num_threads(2)
WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NAME = sys.argv[3] if len(sys.argv) > 3 else "net"
OUT.mkdir(parents=True, exist_ok=True)
image = cv2.imread(str(WORK / "scene" / "background.jpg"))
pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
with torch.inference_mode():
    inputs = gd_proc(images=pil, text=f"{NAME}.", return_tensors="pt")
    r = gd_proc.post_process_grounded_object_detection(gd(**inputs), inputs.input_ids, threshold=0.1, text_threshold=0.1,
                                                       target_sizes=[pil.size[::-1]])[0]
candidates = sorted(({"score": round(float(s), 3), "box": [round(float(v), 1) for v in b]} for s, b in zip(r["scores"], r["boxes"])),
                    key=lambda c: -c["score"])
(OUT / "candidates.json").write_text(json.dumps(candidates, indent=1))
print(candidates[:5], flush=True)
box = candidates[0]["box"]

sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").eval()
with torch.inference_mode():
    inputs = sam_proc(images=pil, input_boxes=[[box]], return_tensors="pt")
    outputs = sam(**inputs, multimask_output=False)
mask = sam_proc.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0][0, 0].numpy() > 0
print(f"mask: {mask.sum()} px, SAM's own IoU estimate {float(outputs.iou_scores[0, 0, 0]):.2f}", flush=True)
cv2.imwrite(str(OUT / "mask.png"), mask.astype(np.uint8) * 255)
view = image.copy()
view[mask] = (0.5 * view[mask] + [0, 0, 127]).astype(np.uint8)
for c in candidates[:5]:
    x0, y0, x1, y1 = map(int, c["box"])
    cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(view, f"{c['score']:.2f}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite(str(OUT / "overlay.jpg"), cv2.resize(view, (960, 540)))
