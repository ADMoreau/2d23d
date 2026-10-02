"""Track: find the people in each frame and follow each one with an ID. Roadmap step 2.

RF-DETR finds person boxes in each frame, and OC-SORT from the `trackers` library links them into tracks.
Output:
- tracks.json: {"detections": [{"frame": 0, "track_id": 1, "box": [x1, y1, x2, y2], "score": 0.9}, ...]},
  boxes in pixels of the ingested frames
- debug.mp4 with each track's box and ID drawn
"""

import json
import logging
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.video import iter_frames, read_meta, write_debug_video

log = logging.getLogger(__name__)

PERSON = 1  # COCO class id in RF-DETR's output
MIN_SCORE = 0.1  # the tracker uses low-score boxes too, to keep tracks alive through occlusions
CONFIDENT = 0.5  # a detection at least this sure counts toward MIN_TRACK_SECONDS
MIN_TRACK_SECONDS = 1.0  # tracks with less confident time are false detections or people passing the edge
LOST_SECONDS = 2.0  # how long a hidden person keeps their ID, e.g. while standing behind their partner
MIN_OVERLAP = 0.2  # box overlap (IoU) to re-attach a hidden person; the default 0.3 split one on the sample


def run(ctx: Context, out: Path) -> None:
    meta = read_meta(ctx.work)
    model, tracker = load(ctx.settings.detector, ctx.device, meta["fps"])
    rows = []
    for i, frame in enumerate(iter_frames(ctx.work)):
        found = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=MIN_SCORE, include_source_image=False)
        tracked = tracker.update(found[found.class_id == PERSON])
        rows += [
            {"frame": i, "track_id": int(t), "box": [round(float(v), 1) for v in box], "score": round(float(s), 3)}
            for box, t, s in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence)
            if t >= 0  # -1: not (yet) part of a track
        ]
    kept = keep_real_tracks(rows, min_frames=round(MIN_TRACK_SECONDS * meta["fps"]))
    (out / "tracks.json").write_text(json.dumps({"detections": kept}))
    n_all, n_kept = len({r["track_id"] for r in rows}), len({r["track_id"] for r in kept})
    log.info(
        "track: %d tracks (%d with under %g s of confident detections dropped), %.1f people per frame",
        n_kept, n_all - n_kept, MIN_TRACK_SECONDS, len(kept) / meta["num_frames"],
    )
    write_debug_video(ctx.work, out, "track", draw_tracks(kept))


def load(size: str, device: str, fps: float):
    """The RF-DETR model and an OC-SORT tracker, without RF-DETR's known, harmless warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit's deprecation, raised when RF-DETR is imported
        try:
            import rfdetr
            import trackers
        except ImportError as e:
            raise ImportError('the track stage needs the ML extras: pip install -e ".[ml]"') from e
    logging.getLogger("rf-detr").setLevel(logging.ERROR)  # hides misleading notes about DINOv2 weights
    models = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}
    tracker = trackers.OCSORTTracker(
        frame_rate=fps,
        lost_track_buffer=round(LOST_SECONDS * 30),  # counted in 30 fps frames
        minimum_consecutive_frames=1,  # keep each track's first frames; keep_real_tracks drops the noise
        minimum_iou_threshold=MIN_OVERLAP,
    )
    return models[size](device=device), tracker


def keep_real_tracks(rows: list[dict], min_frames: int) -> list[dict]:
    """Keep only the detections of tracks that are confident in at least min_frames frames."""
    confident = Counter(row["track_id"] for row in rows if row["score"] >= CONFIDENT)
    return [row for row in rows if confident[row["track_id"]] >= min_frames]


def draw_tracks(rows: list[dict]) -> Callable[[np.ndarray, int], None]:
    """A draw(frame, index) function that draws each track's box and ID in its own color."""
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)

    def draw(frame: np.ndarray, i: int) -> None:
        for row in by_frame[i]:
            x1, y1, x2, y2 = (round(v) for v in row["box"])
            color = track_color(row["track_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(row["track_id"]), (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return draw


def track_color(track_id: int) -> tuple[int, int, int]:
    """A bright BGR color that stays the same for an ID and differs between nearby IDs."""
    hsv = np.uint8([[[track_id * 47 % 180, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
