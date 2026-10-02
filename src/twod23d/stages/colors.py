"""Colors: each person's body painted with their colors from the video. Roadmap step 9.

Each kept body's vertices are projected into its frame, using the bodies stage's meshes, which line up
with the video. A vertex is seen in a frame when it faces the camera, nothing nearer covers it (the person's
own arm, or someone else), and its pixel differs from the scene stage's background without people, so the
court behind a slightly misplaced arm isn't taken for the arm. Each vertex gets the median color of the
frames that saw it, and vertices never seen, such as the soles of the feet, get their neighbors' colors.
The colors come with the place's light and shadows, as the video shows them.
Output:
- colors.npz: track_id (P,), colors (P, V, 3) RGB as in the video (sRGB), seen (P, V) whether a vertex was seen
- colors.json: per person, how many frames were sampled and the share of their vertices seen
- debug.mp4: each kept body drawn in its colors over the video
"""

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.stages.export import vertex_normals
from twod23d.stages.scene import median_of_seen
from twod23d.video import read_frame, read_meta, write_debug_video

log = logging.getLogger(__name__)

COLOR_FRAMES = 200  # frames sampled, spread over the video
MIN_FACING = 0.3  # a vertex faces the camera if its normal is within about 70 degrees of the way to the camera
CELL = 270  # depth test cells: the image's height is split into this many rows of cells
DEPTH_TOLERANCE = 0.03  # meters: a vertex this far behind the nearest one in its cell still counts as seen
BACKGROUND_DIFFERENCE = 30  # a pixel is the person if a color channel differs from the background by more
MIN_SAMPLES = 3  # frames that must see a vertex for its own color; others take their neighbors'


def run(ctx: Context, out: Path) -> None:
    meta = read_meta(ctx.work)
    camera = json.loads((ctx.work / "bodies" / "camera.json").read_text())
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    with np.load(ctx.work / "check" / "bodies.npz") as f:
        kept = set(zip(f["frame"].tolist(), f["track_id"].tolist()))
    with np.load(ctx.work / "bodies" / "bodies.npz") as f:
        frames, track_ids, faces = f["frame"], f["track_id"], f["faces"]
        rows = np.array([i for i, key in enumerate(zip(frames.tolist(), track_ids.tolist())) if key in kept], int)
        vertices = f["vertices"][rows] if len(rows) else np.zeros((0, 0, 3), np.float16)  # camera coordinates
    frames, track_ids = frames[rows], track_ids[rows]
    background_file = ctx.work / "scene" / "background.jpg"
    background = cv2.imread(str(background_file)) if background_file.exists() else None
    cell = max(1, round(meta["height"] / CELL))

    picks = np.unique(frames)
    picks = picks[np.linspace(0, len(picks) - 1, min(COLOR_FRAMES, len(picks))).round().astype(int)] if len(picks) else picks
    samples = defaultdict(list)  # track: [(seen (V,), colors (V, 3))]
    for frame in picks:
        here = np.flatnonzero(frames == frame)
        bodies = [vertices[i].astype(np.float64) for i in here]
        for i, sample in zip(here, sample_frame(read_frame(ctx.work, int(frame)), background, bodies, faces, K, cell)):
            samples[int(track_ids[i])].append(sample)

    people, report = {}, {}
    for track_id, found in sorted(samples.items()):
        seen = np.stack([s for s, _ in found])
        colors = median_of_seen(np.stack([c for _, c in found]), seen)
        enough = seen.sum(axis=0) >= MIN_SAMPLES
        people[track_id] = (fill_unseen(colors, enough, faces).round().astype(np.uint8), enough)
        report[str(track_id)] = {"frames": len(found), "seen": round(float(enough.mean()), 3)}
    n = vertices.shape[1]
    np.savez_compressed(
        out / "colors.npz",
        track_id=np.array(list(people), int),
        colors=np.array([c for c, _ in people.values()], np.uint8).reshape(len(people), n, 3),
        seen=np.array([s for _, s in people.values()], bool).reshape(len(people), n),
    )
    (out / "colors.json").write_text(json.dumps(report, indent=2))
    log.info("colors: %d people from %d frames; share of each one's body seen: %s", len(people), len(picks), {t: r["seen"] for t, r in report.items()} or "none")
    write_debug_video(ctx.work, out, "colors", draw_colors(frames, track_ids, vertices, {t: c for t, (c, _) in people.items()}, K))


def sample_frame(image: np.ndarray, background: np.ndarray | None, bodies: list[np.ndarray], faces: np.ndarray, K: np.ndarray, cell: int) -> list:
    """For each body (V, 3) in camera coordinates: which vertices the frame shows, (V,), and their colors there, (V, 3) RGB."""
    height, width = image.shape[:2]
    nearest = np.full((height // cell + 1, width // cell + 1), np.inf)  # the nearest body point in each cell
    projected = []
    for v in bodies:
        z = np.maximum(v[:, 2], 1e-6)
        x, y = v[:, 0] / z * K[0, 0] + K[0, 2], v[:, 1] / z * K[1, 1] + K[1, 2]
        inside = (v[:, 2] > 0.1) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = np.where(inside, x, 0).astype(int), np.where(inside, y, 0).astype(int)
        np.minimum.at(nearest, (y[inside] // cell, x[inside] // cell), v[inside, 2])
        projected.append((x, y, inside))
    if background is not None:
        person = np.abs(image.astype(np.int16) - background.astype(np.int16)).max(axis=2) > BACKGROUND_DIFFERENCE
    else:
        person = np.ones((height, width), bool)
    results = []
    for v, (x, y, inside) in zip(bodies, projected):
        facing = np.einsum("ij,ij->i", vertex_normals(v, faces), -v) > MIN_FACING * np.linalg.norm(v, axis=1)
        seen = inside & facing & (v[:, 2] <= nearest[y // cell, x // cell] + DEPTH_TOLERANCE) & person[y, x]
        colors = np.zeros((len(v), 3), np.uint8)
        colors[seen] = image[y[seen], x[seen], ::-1]  # BGR to RGB
        results.append((seen, colors))
    return results


def fill_unseen(colors: np.ndarray, seen: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Colors (V, 3) with each vertex not seen given the mean of its seen neighbors, spreading out along the mesh."""
    colors, seen = colors.astype(np.float32), seen.copy()
    a, b = faces.ravel(), np.roll(faces, -1, axis=1).ravel()  # each face's edges
    source, target = np.concatenate([a, b]), np.concatenate([b, a])
    while not seen.all():
        edges = seen[source] & ~seen[target]
        if not edges.any():  # nothing seen is connected to the rest
            break
        total, count = np.zeros_like(colors), np.zeros(len(colors))
        np.add.at(total, target[edges], colors[source[edges]])
        np.add.at(count, target[edges], 1)
        reached = count > 0
        colors[reached] = total[reached] / count[reached, None]
        seen |= reached
    return colors


def draw_colors(frames: np.ndarray, track_ids: np.ndarray, vertices: np.ndarray, colors: dict, K: np.ndarray) -> Callable[[np.ndarray, int], None]:
    """A draw(frame, index) function that paints each kept body's vertices in its colors, nearest last."""
    by_frame = defaultdict(list)
    for i, (frame, track_id) in enumerate(zip(frames.tolist(), track_ids.tolist())):
        if track_id in colors:
            by_frame[frame].append(i)

    def draw(image: np.ndarray, index: int) -> None:
        height, width = image.shape[:2]
        for i in by_frame[index]:
            v = vertices[i].astype(np.float32)
            order = np.argsort(-v[:, 2])  # far to near, so near points end on top
            v, bgr = v[order], colors[int(track_ids[i])][order, ::-1]
            uv = (v @ K.T)[:, :2] / v[:, 2:3]
            inside = (v[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1)
            x, y = uv[inside].astype(int).T
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                image[y + dy, x + dx] = bgr[inside]

    return draw
