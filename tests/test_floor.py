import json
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context, Settings
from twod23d.stages import floor
from twod23d.stages.floor import FEET, fit_plane, smoothed_paths, to_world, world_frame


def work_folder(tmp_path: Path, vertices: np.ndarray, keypoints: np.ndarray, frames: np.ndarray, tracks: np.ndarray) -> Context:
    """A work folder with ingest and bodies output, as the floor stage expects, and a context for it."""
    (tmp_path / "ingest" / "frames").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(tmp_path / "ingest" / "frames" / f"{i:06d}.jpg"), np.zeros((120, 160, 3), np.uint8))
    meta = {"fps": 30.0, "width": 160, "height": 120, "num_frames": 3}
    (tmp_path / "ingest" / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies" / "camera.json").write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}))
    np.savez(tmp_path / "bodies" / "bodies.npz", frame=frames, track_id=tracks, vertices=vertices, keypoints=keypoints, faces=np.zeros((0, 3), np.int32))
    return Context(video=Path("clip.mp4"), work=tmp_path, settings=Settings(), device="cpu")


def tilted_floor(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Points on a floor 3 m below a camera tilted 20 degrees down, in camera coordinates (y down)."""
    rng = np.random.default_rng(seed)
    tilt = np.radians(20)
    up = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])  # the floor's up, seen from the tilted camera
    along = np.array([0.0, np.sin(tilt), -np.cos(tilt)])  # a direction in the floor, away from the camera
    across = np.array([1.0, 0.0, 0.0])
    points = -3.0 * up + rng.uniform(-5, 5, (n, 1)) * across + rng.uniform(-15, -4, (n, 1)) * along
    return points, up, 3.0


def test_fit_plane_ignores_feet_in_the_air():
    points, up, height = tilted_floor()
    lifted = points[:60] + up * np.random.default_rng(1).uniform(0.2, 0.6, (60, 1))  # a fifth mid-stride
    fitted_up, fitted_height, on_floor = fit_plane(np.concatenate([lifted, points[60:]]))
    assert np.allclose(fitted_up, up, atol=1e-6)
    assert abs(fitted_height - height) < 1e-6
    assert not on_floor[:60].any() and on_floor[60:].all()


def test_world_frame_puts_the_floor_at_zero_with_y_up():
    points, up, height = tilted_floor()
    origin = points.mean(0)
    matrix = world_frame(up, origin)
    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3)) and np.isclose(np.linalg.det(rotation), 1)
    world = to_world(points, matrix)
    assert np.allclose(world[:, 1], 0, atol=1e-4)
    camera = matrix[:3, 3]  # where the camera's own origin lands
    assert np.isclose(camera[1], height, atol=1e-6)
    assert np.allclose(matrix[:3, :3] @ [1, 0, 0], [1, 0, 0], atol=1e-6)  # the camera's right stays +x


def test_smoothed_paths_average_each_track_over_time():
    vertices = np.zeros((6, 2, 3), np.float16)
    vertices[:, :, 0] = np.array([0, 1, 0, 1, 0, 1])[:, None]  # track 1 jitters in x, track 2 holds still
    bodies = {"vertices": vertices, "frame": np.array([0, 1, 2, 0, 1, 2]), "track_id": np.array([1, 1, 1, 2, 2, 2])}
    bodies["vertices"][3:, :, 0] = 5
    moves = smoothed_paths(bodies, half_window=1)
    assert np.allclose(moves[:3, 0], [0.5, -2 / 3, 0.5])  # pulled toward each neighborhood's mean
    assert np.allclose(moves[3:], 0)


def test_floor_stage_stands_bodies_on_the_floor(tmp_path):
    points, up, height = tilted_floor(n=60)
    box = np.array([[x, y, z] for x in (-0.2, 0.2) for y in (0, 1.7) for z in (-0.15, 0.15)])  # a 1.7 m "person"
    vertices, keypoints = [], []
    for foot in points:  # each stands on the floor, but floats 0-10 cm above it, as estimates do
        lift = foot + up * np.random.default_rng(len(vertices)).uniform(0, 0.1)
        body = lift + box[:, :1] * [1, 0, 0] - box[:, 1:2] * up + box[:, 2:] * np.cross([1, 0, 0], up)
        vertices.append(body)
        kp = np.repeat(body[:1], 70, 0)
        kp[FEET] = lift  # heels and toes where the body stands
        keypoints.append(kp)
    ctx = work_folder(tmp_path, np.float16(vertices), np.float32(keypoints), np.arange(60) % 3, np.arange(60) // 3)
    out = tmp_path / "floor.tmp"
    out.mkdir()
    floor.run(ctx, out)
    result = json.loads((out / "floor.json").read_text())
    assert abs(result["camera_height"] - height) < 0.06
    assert abs(result["camera_tilt_down"] - 20) < 1
    with np.load(out / "bodies.npz") as f:
        snapped = f["vertices"].astype(np.float32)
    assert np.allclose(snapped[:, :, 1].min(1), 0, atol=0.01)  # every body touches the floor
    assert np.allclose(snapped[:, :, 1].max(1), 1.7, atol=0.02)  # and stands upright
    assert (out / "debug.mp4").is_file()


def test_floor_stage_falls_back_without_bodies(tmp_path):
    ctx = work_folder(tmp_path, np.zeros((0, 0, 3), np.float16), np.zeros((0, 70, 3), np.float32), np.zeros(0, int), np.zeros(0, int))
    out = tmp_path / "floor.tmp"
    out.mkdir()
    floor.run(ctx, out)
    result = json.loads((out / "floor.json").read_text())
    assert result["camera_height"] == floor.FALLBACK_HEIGHT and result["camera_tilt_down"] == 0
    assert (out / "debug.mp4").is_file()
