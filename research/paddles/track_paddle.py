"""Follow one player's paddle through every frame with SAM 2.1's video mode, seeded by sparse detections.

The video SAM 2 sees is a crop that follows the player (fixed size, centered on their smoothed box), so the paddle
is as big as in the detection crops. It's streamed frame by frame (memory stays flat). Seeds are the player's
own OWLv2 boxes (assigned to their hand by fit2.py); every SEED_EVERY-th one is used, and the rest are held out
to score the tracked masks against.

usage: track_paddle.py <work dir> <paddles run dir> <track> [seed_every]
Writes track/{track}.npz (per frame: crop origin and scale, packed mask, score) and track/{track}.mp4. Saves progress to
track/{track}.partial.npz every 50 frames; a rerun resumes from it.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from transformers import Sam2VideoModel, Sam2VideoProcessor

torch.set_num_threads(3)
WORK, D, track = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
SEED_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 2
OUT = D / os.environ.get("TRACK_OUT", "track")
OUT.mkdir(exist_ok=True)
HIDE_OTHERS = bool(os.environ.get("HIDE_OTHERS"))  # gray out everyone else, except within reach of this player's hand
REACH = 0.6  # m from the wrist that stays visible (a paddle and a bit)
PAD = 0.4  # of the person's height, as in detect.py
SIZE = 640  # crop side, px

frames = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
boxes = {d["frame"]: d["box"] for d in json.loads((WORK / "track" / "tracks.json").read_text())["detections"] if d["track_id"] == track}
known = np.array(sorted(boxes))
centers = np.array([[(boxes[f][0] + boxes[f][2]) / 2, (boxes[f][1] + boxes[f][3]) / 2] for f in known])
heights = np.array([boxes[f][3] - boxes[f][1] for f in known])
everywhere = np.arange(len(frames))
center = np.stack([np.interp(everywhere, known, uniform_filter1d(centers[:, i], 9)) for i in range(2)], 1)
side = (1 + 2 * PAD) * np.median(heights)
scale = SIZE / side

# Seeds: this player's detections at their hand (fit2's), held out alternately
recs = {(r["frame"], r["track_id"]): r for r in json.loads((D / "results.json").read_text())}
with np.load(D / "fit2" / "grips.npz") as f:
    det_frames = [int(x) for x in f[f"{track}_det_frames"]]
seeds, held_out = det_frames[::SEED_EVERY], [x for i, x in enumerate(det_frames) if i % SEED_EVERY]


def crop(image, i):
    x0, y0 = center[i] - side / 2
    M = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    return cv2.warpAffine(image, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128)), (x0, y0)


if HIDE_OTHERS:
    cam = json.loads((WORK / "bodies" / "camera.json").read_text())
    with np.load(WORK / "bodies" / "bodies.npz") as f:
        people = {k: f[k] for k in ("frame", "track_id", "vertices", "keypoints", "faces")}
    with np.load(D / "fit2" / "grips.npz") as f:
        WRIST = {41: 41, 77: 62}[int(f[f"{track}_joint"])]  # the holding hand's keypoint


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


def hide_others(image, i, origin):
    """Other people painted gray: their bodies, and their paddles' reach around both their wrists. Near this player's
    holding wrist stays visible, where it's nearer that wrist than any other (so a teammate behind the paddle doesn't
    hide it)."""
    rows = np.flatnonzero(people["frame"] == i)
    mine = [r for r in rows if people["track_id"][r] == track]
    if not mine:
        return image
    others = np.zeros(image.shape[:2], np.uint8)
    other_wrists = []
    for r in rows:
        if people["track_id"][r] != track:
            px = ((project(people["vertices"][r]) - origin) * scale)[people["faces"]]
            cv2.fillPoly(others, list(np.round(px).astype(np.int32)), 1)
            other_wrists += [people["keypoints"][r, k] for k in (41, 62)]
    others = cv2.dilate(others, np.ones((7, 7), np.uint8))
    ys, xs = np.mgrid[:SIZE, :SIZE]

    def distance(wrist):  # px from the wrist, and its reach in px
        c = (project(wrist) - origin) * scale
        return np.hypot(xs - c[0], ys - c[1]), cam["fx"] * REACH / wrist[2] * scale

    mine_d, mine_reach = distance(people["keypoints"][mine[0], WRIST])
    nearest_other = np.full(others.shape, np.inf)
    for w in other_wrists:
        d, reach = distance(w)
        others[d < reach] = 1
        nearest_other = np.minimum(nearest_other, d)
    hidden = others.astype(bool) & ~((mine_d < mine_reach) & (mine_d < nearest_other))
    image = image.copy()
    image[hidden] = 128
    return image


def to_crop(rec, box, origin):
    """A box from a detection crop (detect.py) into this frame's following crop."""
    x = [rec["origin"][0] + box[0] / rec["scale"], rec["origin"][1] + box[1] / rec["scale"], rec["origin"][0] + box[2] / rec["scale"], rec["origin"][1] + box[3] / rec["scale"]]
    return [(x[0] - origin[0]) * scale, (x[1] - origin[1]) * scale, (x[2] - origin[0]) * scale, (x[3] - origin[1]) * scale]


def detection_mask(rec, origin):
    """The held-out SAM mask (detect.py's crop) resampled into this frame's following crop."""
    mask = cv2.imread(str(D / "crops" / f"{track}_{rec['frame']:06d}_owlmask.png"), 0)
    s = scale / rec["scale"]
    M = np.array([[s, 0, (rec["origin"][0] - origin[0]) * scale], [0, s, (rec["origin"][1] - origin[1]) * scale]])
    return cv2.warpAffine(mask, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 127


processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-small")
model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-small").eval()
session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
CHECKPOINT, EVERY = OUT / f"{track}.partial.npz", 50


def prune(session, fed, keep=16):
    """Streaming keeps every frame's pixels and outputs (memory grows ~12 MB a frame); SAM 2 only looks back
    num_maskmem (7) frames, plus the prompted ones, which are kept."""
    if session.processed_frames:  # emptied, not deleted: the session numbers new frames by how many it holds
        for k in [k for k, v in session.processed_frames.items() if k < fed - 1 and v is not None]:
            session.processed_frames[k] = None
    for out in session.output_dict_per_obj.values():
        for k in [k for k in out["non_cond_frame_outputs"] if k < fed - keep]:
            del out["non_cond_frame_outputs"][k]


def save(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def saved(done):
    return dict(masks=np.array(masks), shape=[SIZE, SIZE], origin=np.array(origins), scale=scale, score=np.array(scores), seeds=seeds,
                held_out=list(ious), held_out_iou=list(ious.values()), done=done)


masks, scores, origins, ious, first = [], [], [], {}, 0  # masks: bit-packed rows
if CHECKPOINT.exists():  # SAM 2's memory isn't saved: go back to the last seed before the checkpoint and follow from there
    with np.load(CHECKPOINT) as c:
        first = max([s for s in seeds if s < int(c["done"])], default=0)
        masks, scores, origins = list(c["masks"][:first]), list(c["score"][:first]), [tuple(o) for o in c["origin"][:first]]
        ious = {int(f): float(v) for f, v in zip(c["held_out"], c["held_out_iou"]) if f < first}
    print(f"resuming at frame {first}", flush=True)
started, start, fed = False, time.time(), 0  # fed: the session numbers frames by how many it has seen
LAST = int(__import__('os').environ.get('LAST_FRAME', len(frames) - 1))
for i in range(first, LAST + 1):
    image, origin = crop(cv2.imread(str(frames[i])), i)
    if HIDE_OTHERS:
        image = hide_others(image, i, origin)
    origins.append(origin)
    if not started and i not in seeds:  # nothing to follow before the first seed
        masks.append(np.packbits(np.zeros((SIZE, SIZE), bool), axis=-1))
        scores.append(0.0)
        continue
    inputs = processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device="cpu", return_tensors="pt")
    if i in seeds:
        rec = recs[(i, track)]
        processor.add_inputs_to_inference_session(session, frame_idx=fed, obj_ids=1, input_boxes=[[to_crop(rec, rec["owl"]["box"], origin)]],
                                                  original_size=inputs.original_sizes[0])
        started = True
    with torch.inference_mode():
        out = model(inference_session=session, frame=inputs.pixel_values[0])
    fed += 1
    prune(session, fed)
    logits = processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0][0, 0]
    mask = logits.numpy() > 0
    masks.append(np.packbits(mask, axis=-1))
    scores.append(float(out.object_score_logits.squeeze()) if getattr(out, "object_score_logits", None) is not None else 0.0)
    if i in held_out:
        truth = detection_mask(recs[(i, track)], origin)
        union = (mask | truth).sum()
        ious[i] = float((mask & truth).sum() / union) if union else 0.0
    if i % EVERY == 0:
        save(CHECKPOINT, **saved(i + 1))
        print(f"frame {i}: {time.time() - start:.0f} s, held-out IoU so far {np.mean(list(ious.values())) if ious else float('nan'):.3f}", flush=True)
save(OUT / f"{track}.npz", **saved(LAST + 1))
CHECKPOINT.unlink(missing_ok=True)

# Debug video, from the saved masks
writer = cv2.VideoWriter(str(OUT / f"{track}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (SIZE, SIZE))
for i, packed in enumerate(masks):
    view, origin = crop(cv2.imread(str(frames[i])), i)
    if HIDE_OTHERS:
        view = hide_others(view, i, origin)
    mask = np.unpackbits(packed, axis=-1)[:, :SIZE].astype(bool)
    view[mask] = (0.5 * view[mask] + [0, 0, 127]).astype(np.uint8)
    cv2.putText(view, f"{i}{' seed' if i in seeds else ''}{f' held-out IoU {ious[i]:.2f}' if i in ious else ''}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(view)
writer.release()
v = np.array(list(ious.values()))
print(json.dumps(dict(track=track, seeds=len(seeds), held_out=len(v), iou_mean=round(float(v.mean()), 3), iou_median=round(float(np.median(v)), 3),
                      iou_below_0_3=int((v < 0.3).sum()), frames_with_mask=int(sum(m.any() for m in masks)), seconds=round(time.time() - start))), flush=True)
