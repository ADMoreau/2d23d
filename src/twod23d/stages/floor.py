"""Floor: stand everyone on one flat floor. Roadmap step 4.

Fits one plane to everyone's heels and toes across the whole video, then moves to world coordinates:
meters, y up, floor at y = 0, x along the camera's right, and the origin under the middle of where people
stand. Each body's lowest point is put on the floor, and each person's path is smoothed.
Output:
- bodies.npz like bodies/bodies.npz, but in world coordinates, plus floor_gap (N,): how far above the floor
  each body's lowest point was before it was snapped down (negative: below it)
- floor.json: camera_to_world (4x4), and the camera's height above the floor and downward tilt
- debug.mp4 with a 1 m floor grid and each body's contact with the floor drawn over the frames
"""

import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.stages.track import track_color
from twod23d.video import read_meta, write_debug_video

log = logging.getLogger(__name__)

FEET = [15, 16, 17, 18, 19, 20]  # heels and toe tips among SAM 3D Body's 70 keypoints
ON_FLOOR = 0.05  # meters: a foot point this close to the plane counts as touching it
MIN_FEET = 30  # with fewer foot points than this, fall back to a level camera
FALLBACK_HEIGHT = 1.6  # meters: that level camera's height, about eye level
SMOOTH_SECONDS = 0.2  # each person's position on the floor is averaged over this much time


def run(ctx: Context, out: Path) -> None:
    with np.load(ctx.work / "bodies" / "bodies.npz") as f:
        bodies = dict(f)
    camera = json.loads((ctx.work / "bodies" / "camera.json").read_text())
    fps = read_meta(ctx.work)["fps"]

    feet = bodies["keypoints"][:, FEET].reshape(-1, 3).astype(np.float64)
    up, height, on_floor = fit_plane(feet) if len(feet) >= MIN_FEET else (None, None, None)
    if up is None or height <= 0:
        log.warning("floor: no floor found in %d foot points, assuming a level camera %g m up", len(feet), FALLBACK_HEIGHT)
        up, height = np.array([0.0, -1.0, 0.0]), FALLBACK_HEIGHT
        middle = np.array([0.0, 0.0, 5.0])
    else:
        middle = feet[on_floor].mean(0)
        log.info("floor: %d of %d foot points on one plane", on_floor.sum(), len(feet))
    origin = middle - up * (up @ middle + height)  # straight below the middle, on the floor
    camera_to_world = world_frame(up, origin)

    for points in ("vertices", "keypoints"):
        bodies[points] = to_world(bodies[points], camera_to_world)
    bodies["floor_gap"] = np.zeros(len(bodies["frame"]), np.float32)
    if len(bodies["frame"]):
        bodies["floor_gap"] = bodies["vertices"][:, :, 1].min(axis=1)  # how far each body floats (or sinks, < 0)
        shift = np.zeros((len(bodies["frame"]), 3), np.float32)
        shift[:, 1] = -bodies["floor_gap"]  # snap: each body's lowest point onto the floor
        shift[:, [0, 2]] = smoothed_paths(bodies, round(SMOOTH_SECONDS * fps / 2))
        bodies["vertices"] = (bodies["vertices"] + shift[:, None]).astype(np.float16)
        bodies["keypoints"] = (bodies["keypoints"] + shift[:, None]).astype(np.float32)
    np.savez_compressed(out / "bodies.npz", **bodies)

    tilt = math.degrees(math.asin(-up[2]))
    floor = {"camera_to_world": camera_to_world.tolist(), "camera_height": round(height, 3), "camera_tilt_down": round(tilt, 1)}
    (out / "floor.json").write_text(json.dumps(floor, indent=2))
    log.info("floor: the camera is %.1f m above the floor, looking %.0f degrees down", height, tilt)
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    write_debug_video(ctx.work, out, "floor", draw_floor(bodies, camera_to_world, K))


def fit_plane(points: np.ndarray, iterations: int = 500) -> tuple[np.ndarray, float, np.ndarray]:
    """The plane through the most points, as (up, height, on_plane): a point p is up @ p + height above it.

    RANSAC picks the plane most points touch, and least squares then refines it on those points, so feet
    in the air don't pull it up. `up` points toward the camera's up (-y in camera coordinates).
    """
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        near = np.abs((points - a) @ normal) < ON_FLOOR
        if near.sum() > best.sum():
            best = near
    if best.sum() < 3:
        return None, None, None
    center = points[best].mean(0)
    up = np.linalg.svd(points[best] - center)[2][-1]  # the direction the points spread least in
    if up[1] > 0:
        up = -up
    return up, float(-up @ center), best


def world_frame(up: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """camera_to_world (4x4) for a world with y along `up`, x along the camera's right and the origin at `origin`."""
    x = np.array([1.0, 0.0, 0.0]) - up * up[0]  # the camera's right, laid flat on the floor
    x /= np.linalg.norm(x)
    rotation = np.stack([x, up, np.cross(x, up)])  # rows: the world's axes in camera coordinates
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ origin
    return matrix


def to_world(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    return (points.astype(np.float32) @ camera_to_world[:3, :3].T.astype(np.float32) + camera_to_world[:3, 3]).astype(np.float32)


def smoothed_paths(bodies: dict, half_window: int) -> np.ndarray:
    """How far to move each body along the floor (x, z) to smooth its track's path, by averaging over time."""
    centers = bodies["vertices"][:, :, [0, 2]].astype(np.float32).mean(axis=1)
    moves = np.zeros_like(centers)
    for track_id in np.unique(bodies["track_id"]):
        rows = np.flatnonzero(bodies["track_id"] == track_id)
        frames = bodies["frame"][rows]
        for row, frame in zip(rows, frames):
            near = rows[np.abs(frames - frame) <= half_window]
            moves[row] = centers[near].mean(axis=0) - centers[row]
    return moves


def draw_floor(bodies: dict, camera_to_world: np.ndarray, K: np.ndarray) -> Callable[[np.ndarray, int], None]:
    """A draw(frame, index) function: a 1 m grid on the floor and a dot where each body touches it."""
    world_to_camera = np.linalg.inv(camera_to_world)

    def project(world_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = world_points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        visible = cam[:, 2] > 0.1  # in front of the camera
        return (cam @ K.T)[:, :2] / np.maximum(cam[:, 2:3], 1e-6), visible

    # The camera doesn't move, so the grid looks the same in every frame: project it once
    steps = np.linspace(-15, 15, 121)
    lines = [np.stack([np.full_like(steps, x), np.zeros_like(steps), steps], 1) for x in range(-15, 16)]
    lines += [line[:, [2, 1, 0]] for line in lines]
    grid = []
    for line in lines:
        uv, visible = project(line)
        grid += [np.round(uv[[a, a + 1]]).astype(np.int32) for a in np.flatnonzero(visible[:-1] & visible[1:])]
    contacts = defaultdict(list)
    for frame, track_id, vertices in zip(bodies["frame"], bodies["track_id"], bodies["vertices"]):
        contacts[int(frame)].append((int(track_id), vertices[np.argmin(vertices[:, 1])].astype(np.float64)))

    def draw(image: np.ndarray, i: int) -> None:
        cv2.polylines(image, grid, False, (255, 255, 255), 1, cv2.LINE_AA)
        for track_id, point in contacts[i]:
            uv, visible = project(point[None])
            if visible[0]:
                cv2.circle(image, tuple(np.round(uv[0]).astype(int)), 6, track_color(track_id), -1)

    return draw
