"""Grounding DINO boxes for some names on one image, drawn. usage: peek_boxes.py <image> <out.jpg> <name> [...]"""
import sys, cv2, torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
torch.set_num_threads(2)
img = cv2.imread(sys.argv[1]); pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
for name in sys.argv[3:]:
    inputs = proc(images=pil, text=name + ".", return_tensors="pt")
    with torch.no_grad(): out = model(**inputs)
    r = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[pil.size[::-1]])[0]
    for box, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
        x0, y0, x1, y1 = map(int, box); print(f"{name:20s} {s:.2f} {x0} {y0} {x1} {y1}")
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2); cv2.putText(img, f"{name} {s:.2f}", (x0, y0 + 20), 0, 0.7, (0, 0, 255), 2)
cv2.imwrite(sys.argv[2], img)
