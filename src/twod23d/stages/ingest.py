"""Ingest: copy the input video into the work folder as frames, so no later stage reads the video.

Output: frames/000000.jpg, ...; meta.json with fps, width, height, num_frames and source; debug.mp4.
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.video import DebugVideo, put_text

log = logging.getLogger(__name__)


def run(ctx: Context, out: Path) -> None:
    video = cv2.VideoCapture(str(ctx.video))
    if not video.isOpened():
        raise RuntimeError(f"cannot open {ctx.video}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps > 0:  # not reported (0 or NaN): assume a phone's 30
        fps = 30.0
    max_seconds = ctx.settings.max_seconds
    max_frames = math.inf if max_seconds is None else round(max_seconds * fps)
    (out / "frames").mkdir()
    n = 0
    with DebugVideo(out / "debug.mp4", fps) as debug:
        while n < max_frames:
            ok, frame = video.read()
            if not ok:
                break
            frame = shrink(frame, ctx.settings.max_side)
            height, width = frame.shape[:2]
            cv2.imwrite(str(out / "frames" / f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            put_text(frame, f"ingest  {n}")
            debug.write(frame)
            n += 1
    video.release()
    if n == 0:
        raise RuntimeError(f"no frames in {ctx.video}")
    meta = {"fps": fps, "width": width, "height": height, "num_frames": n, "source": str(ctx.video)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("ingest: %d frames, %dx%d at %.2f fps", n, width, height, fps)


def shrink(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so the longer side is at most max_side."""
    height, width = frame.shape[:2]
    if max_side is None or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
