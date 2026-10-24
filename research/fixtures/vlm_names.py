"""Ask a vision-language model which objects a game is played with or around, from a video frame (people in view).

usage: vlm_names.py <model id> <image> [<image> ...]   (repo venv: transformers)
"""
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.set_num_threads(6)
MODEL = sys.argv[1]
QUESTION = ("This frame is from a video of people playing a sport. Which sport is it? Then list the objects in the "
            "scene, other than the ball and anything the players hold or wear, that this game is played with or "
            "around. Answer in two lines:\nsport: <name>\nobjects: <short object names, comma separated>")
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
for path in sys.argv[2:]:
    image = Image.open(path).convert("RGB")
    image.thumbnail((1280, 1280))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUESTION}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                      enable_thinking=False)
    start = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"== {path} ({time.time() - start:.0f} s)\n{answer}", flush=True)
