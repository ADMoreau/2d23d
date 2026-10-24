"""Ingest: copy the input video into the work folder as frames, so no later stage reads the video.

Wide-angle lenses bend straight lines outward from the middle (barrel distortion). Every later stage assumes a
pinhole camera, and on a distorted video that puts people near the sides of the picture off the floor. So the
frames are undistorted here, by the radial distortion that makes the scene's straight edges straightest (the
plumb-line idea): field lines, poles, fences and walls are straight wherever the video was shot. Edges are cut
into long pieces, and the distortion that brings the pieces closest to straight lines wins. Truly curved edges
never come close, so they don't count. A video whose edges are as straight uncorrected, or that has too few long
edges to tell, is left as it is.

The undistorted frames keep the resolution of the middle and grow to keep everything the camera saw; their
corners, outside what the camera saw, are black.

Output: frames/000000.jpg, ...; meta.json with fps, width, height, num_frames, source and lens (the distortion
taken out, or null); debug.mp4.
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

LENS_FRAMES = 5  # frames spread over the video, to measure the lens on
LENS_WIDTH = 1920  # px: the lens is measured on frames shrunk to at most this wide
PIECE = 500  # px of edge per test piece, at LENS_WIDTH
MAX_OFF_LINE = 3.0  # px: a piece's distance from its line (RMS) counts up to this, so curved edges add a constant
MIN_PIECES = 20  # fewer long edge pieces than this are too few to tell
MIN_GAIN = 0.2  # the best distortion must bring the pieces this much closer to lines than none, or none is taken out
# Candidates, in coordinates where the image's half width is 1: a coarse grid, then a fine one around its best
K1, K2 = np.round(np.arange(-0.5, 0.21, 0.05), 3), np.round(np.arange(-0.1, 0.11, 0.05), 3)
FINE = np.round(np.arange(-0.05, 0.051, 0.01), 3)


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
    paths = []
    while len(paths) < max_frames:
        ok, frame = video.read()
        if not ok:
            break
        paths.append(out / "frames" / f"{len(paths):06d}.jpg")
        cv2.imwrite(str(paths[-1]), shrink(frame, ctx.settings.max_side), [cv2.IMWRITE_JPEG_QUALITY, 95])
    video.release()
    if not paths:
        raise RuntimeError(f"no frames in {ctx.video}")

    picks = np.linspace(0, len(paths) - 1, min(LENS_FRAMES, len(paths))).round().astype(int)
    lens = measure_lens([cv2.imread(str(paths[i])) for i in picks])
    height, width = cv2.imread(str(paths[0])).shape[:2]
    maps = None
    if lens is not None:
        maps, (width, height) = undistort_maps(lens, cv2.imread(str(paths[0])).shape[:2])
    with DebugVideo(out / "debug.mp4", fps) as debug:
        for n, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if maps is not None:
                frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            put_text(frame, f"ingest  {n}")
            debug.write(frame)
    meta = {"fps": fps, "width": width, "height": height, "num_frames": len(paths), "source": str(ctx.video), "lens": lens}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if lens is None:
        log.info("ingest: %d frames, %dx%d at %.2f fps; lens left as it is", len(paths), width, height, fps)
    else:
        log.info(
            "ingest: %d frames, %dx%d at %.2f fps; lens distortion k1 %.2f, k2 %.2f taken out (long edges %.1f px off straight on average, was %.1f)",
            len(paths), width, height, fps, lens["k1"], lens["k2"], lens["off_line"], lens["off_line_before"],
        )


def shrink(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so the longer side is at most max_side."""
    height, width = frame.shape[:2]
    if max_side is None or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def measure_lens(frames: list[np.ndarray]) -> dict | None:
    """The radial distortion (k1, k2) that straightens the frames' long edges best, or None if none does clearly.

    k1 and k2 are OpenCV's, in coordinates centered on the image where its half width is 1.
    """
    height, width = frames[0].shape[:2]
    scale = min(1.0, LENS_WIDTH / width)
    pieces = np.concatenate([edge_pieces(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)) for f in frames])
    if len(pieces) < MIN_PIECES:
        return None
    half = width * scale / 2
    pieces = (pieces - [half, height * scale / 2]) / half  # centered, half width 1

    def off_line(k1: float, k2: float) -> float:
        """The pieces' mean RMS distance from their own best lines, in pixels at LENS_WIDTH, once undistorted."""
        if not monotonic(k1, k2, np.hypot(1, height / width)):
            return math.inf
        u = undistort_points(pieces.reshape(-1, 2), k1, k2).reshape(pieces.shape)
        u = u - u.mean(1, keepdims=True)
        rms = np.linalg.svd(u, compute_uv=False)[:, 1] / np.sqrt(pieces.shape[1]) * half
        return float(np.minimum(rms, MAX_OFF_LINE).mean())

    scores = {(float(k1), float(k2)): off_line(k1, k2) for k1 in K1 for k2 in K2}
    coarse = min(scores, key=scores.get)
    scores |= {(float(round(coarse[0] + a, 3)), float(round(coarse[1] + b, 3))): off_line(coarse[0] + a, coarse[1] + b) for a in FINE for b in FINE}
    best = min(scores, key=scores.get)
    before = off_line(0.0, 0.0)
    if scores[best] > (1 - MIN_GAIN) * before:
        return None
    return {"k1": best[0], "k2": best[1], "off_line": round(scores[best], 2), "off_line_before": round(before, 2), "pieces": len(pieces)}


def edge_pieces(image: np.ndarray) -> np.ndarray:
    """Long, nearly straight runs of edge pixels (n, PIECE·scale, 2), from Canny edges traced as contours."""
    length = round(PIECE * image.shape[1] / LENS_WIDTH)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pieces = []
    for contour in contours:
        contour = contour[:, 0].astype(np.float64)
        for start in range(0, len(contour) - length, length // 4):
            piece = contour[start : start + length]
            if np.linalg.norm(piece[-1] - piece[0]) > 0.9 * length:  # no hairpins where a contour turns back
                pieces.append(piece)
    return np.array(pieces).reshape(-1, max(length, 1), 2)


def undistort_points(points: np.ndarray, k1: float, k2: float, iterations: int = 10) -> np.ndarray:
    """Where distorted points (n, 2) would be without the distortion: r_d = r (1 + k1 r² + k2 r⁴), inverted by Newton's
    method on each point's radius."""
    r_d = np.linalg.norm(points, axis=-1)
    r = r_d.copy()
    for _ in range(iterations):
        r -= (r * (1 + k1 * r**2 + k2 * r**4) - r_d) / (1 + 3 * k1 * r**2 + 5 * k2 * r**4)
    with np.errstate(invalid="ignore", divide="ignore"):
        return points * np.where(r_d > 0, r / r_d, 1.0)[:, None]


def monotonic(k1: float, k2: float, r_corner: float) -> bool:
    """Whether the distortion can be inverted out to the image's corner: r_d grows with r until it gets there."""
    r = np.linspace(0, 4, 4000)
    reached = r * (1 + k1 * r**2 + k2 * r**4) >= r_corner
    if not reached.any():
        return False
    return bool((1 + 3 * k1 * r[: np.argmax(reached) + 1] ** 2 + 5 * k2 * r[: np.argmax(reached) + 1] ** 4 > 0).all())


def undistort_maps(lens: dict, shape: tuple[int, int]) -> tuple[tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """cv2.remap maps from undistorted frames to the video's, and the undistorted frames' (width, height).

    The middle keeps its resolution, and the frame grows until the middle of each side is still in it.
    """
    height, width = shape
    half = width / 2
    K = np.array([[half, 0, width / 2], [0, half, height / 2], [0, 0, 1]])
    sides = undistort_points(np.array([[1.0, 0.0], [0.0, height / width]]), lens["k1"], lens["k2"])
    new_width = max(width, 2 * math.ceil(sides[0, 0] * half))
    new_height = max(height, 2 * math.ceil(sides[1, 1] * half))
    new_K = np.array([[half, 0, new_width / 2], [0, half, new_height / 2], [0, 0, 1]])
    maps = cv2.initUndistortRectifyMap(K, np.array([lens["k1"], lens["k2"], 0, 0]), None, new_K, (new_width, new_height), cv2.CV_32FC1)
    return maps, (new_width, new_height)
