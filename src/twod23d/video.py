"""Reading the ingested frames and writing debug videos. Shared by all stages."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np


def read_meta(work: Path) -> dict:
    """fps, width, height and num_frames of the ingested video."""
    return json.loads((work / "ingest" / "meta.json").read_text())


def iter_frames(work: Path) -> Iterator[np.ndarray]:
    """The ingested frames in order, as BGR images."""
    for path in sorted((work / "ingest" / "frames").glob("*.jpg")):
        yield cv2.imread(str(path))


def read_frame(work: Path, index: int) -> np.ndarray:
    """One ingested frame, as a BGR image."""
    return cv2.imread(str(work / "ingest" / "frames" / f"{index:06d}.jpg"))


class DebugVideo:
    """Writes BGR frames to an .mp4 file. The first frame sets the size."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.writer = path, fps, None

    def __enter__(self) -> "DebugVideo":
        return self

    def __exit__(self, *exc) -> None:
        if self.writer is not None:
            self.writer.release()

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot write {self.path}")
        self.writer.write(frame)


def put_text(frame: np.ndarray, text: str) -> None:
    """Write white text on a black box in the top-left corner."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(frame, (0, 0), (width + 16, height + baseline + 16), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, height + 8), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def write_debug_video(
    work: Path, out: Path, title: str, draw: Callable[[np.ndarray, int], None] | None = None
) -> None:
    """Write out/debug.mp4: the ingested frames, drawn on by draw(frame, index), with a title and frame number."""
    with DebugVideo(out / "debug.mp4", read_meta(work)["fps"]) as video:
        for i, frame in enumerate(iter_frames(work)):
            if draw is not None:
                draw(frame, i)
            put_text(frame, f"{title}  {i}")
            video.write(frame)
