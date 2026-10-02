import json
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context, Settings
from twod23d.stages import scene

K = np.array([[100.0, 0, 80], [0, 100, 60], [0, 0, 1]])
LEVEL_CAMERA = np.array([[1.0, 0, 0, 0], [0, -1, 0, 2], [0, 0, -1, 0], [0, 0, 0, 1]])  # 2 m up, looking along -z


def test_background_leaves_out_people():
    place = np.random.default_rng(0).integers(1, 256, (40, 60, 3), dtype=np.uint8)
    frames = np.repeat(place[None], 5, axis=0)
    boxes = []
    for k in range(5):
        x = 5 + 10 * k  # someone walking across
        frames[k, 10:30, x : x + 8] = 0
        frames[k, :6, :6] = 0  # and someone who never moves
        boxes.append([[x, 10, x + 7, 29], [0, 0, 5, 5]])
    image = scene.background(frames, boxes)
    assert (image[6:, 6:] == place[6:, 6:]).all()  # the walker is gone
    assert image[:5, :5].min() > 0  # the one who stood still is painted over from around them


def test_depth_ratios_compare_feet_with_the_scene_on_their_rays():
    u, v = np.meshgrid(np.arange(160), np.arange(120))
    rays = np.stack([(u - 80) / 100, (v - 60) / 100, np.ones(u.shape)], axis=-1)
    points = 4.0 * rays  # a wall 4 m away
    valid = np.ones((120, 160), bool)
    valid[:, :20] = False  # the model isn't sure about the left edge
    feet = 5.0 * np.array([rays[60, 100], rays[30, 40], rays[90, 10], [0, 0, -1]])  # 25% farther; unsure; behind
    assert np.allclose(scene.depth_ratios(feet, K, points, valid), [1.25, 1.25])


def test_check_scale_needs_enough_feet_that_agree():
    report = scene.check_scale(np.full(100, 1.05))
    assert report["scale"] == 1.05 and "reason" not in report
    assert "reason" in scene.check_scale(np.linspace(0.5, 1.5, 100))  # the middle half spans 50%
    assert "reason" in scene.check_scale(np.full(10, 1.0))


def test_floor_depth_puts_rays_on_the_floor():
    rays = scene.pixel_rays(K, LEVEL_CAMERA, (120, 160))
    depth = scene.floor_depth(rays, 2.0)
    assert np.isinf(depth[:60]).all() and np.isfinite(depth[60:]).all()  # rays above the horizon miss the floor
    points = LEVEL_CAMERA[:3, 3] + depth[60:, :, None] * rays[60:]
    assert np.allclose(points[..., 1], 0)
    assert np.allclose(points[..., 2], -depth[60:])  # depth along the camera's view


def test_grid_mesh_faces_the_camera_and_leaves_out_gaps():
    camera = np.zeros(3)
    u, v = np.meshgrid(np.arange(40), np.arange(30))
    points = np.stack([u - 20.0, v - 15.0, np.full(u.shape, 10.0)], axis=-1)  # a wall 10 m away
    points[:, 20:, 2] = 30.0  # and its right half 30 m away
    keep = np.ones((30, 40), bool)
    keep[:5, :5] = False
    vertices, faces, uvs = scene.grid_mesh(points, keep, 2, camera, max_stretch=1.1)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), camera - a) > 0).all()  # counter-clockwise, seen from the camera
    assert (vertices[faces][..., 2].min(axis=1) == vertices[faces][..., 2].max(axis=1)).all()  # no triangle spans the gap
    assert not ((uvs[:, 0] < 5 / 40) & (uvs[:, 1] < 5 / 30)).any()  # nothing where keep is False
    assert ((uvs >= 0) & (uvs <= 1)).all() and len(vertices) == len(np.unique(faces))
    assert uvs[:, 0].max() == 39.5 / 40 and uvs[:, 1].max() == 29.5 / 30  # out to the image's edges


def test_merge_renumbers_faces():
    triangle = (np.zeros((3, 3)), np.array([[0, 1, 2]]), np.zeros((3, 2)))
    vertices, faces, uvs = scene.merge([triangle, triangle])
    assert faces.tolist() == [[0, 1, 2], [3, 4, 5]] and len(vertices) == len(uvs) == 6


def test_scene_stage_without_people_makes_no_scene(tmp_path):
    (tmp_path / "ingest" / "frames").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(tmp_path / "ingest" / "frames" / f"{i:06d}.jpg"), np.full((120, 160, 3), 90, np.uint8))
    (tmp_path / "ingest" / "meta.json").write_text(json.dumps({"fps": 30.0, "width": 160, "height": 120, "num_frames": 3}))
    (tmp_path / "track").mkdir()
    (tmp_path / "track" / "tracks.json").write_text(json.dumps({"detections": []}))
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies" / "camera.json").write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}))
    np.savez(tmp_path / "bodies" / "bodies.npz", frame=np.zeros(0, int), track_id=np.zeros(0, int), keypoints=np.zeros((0, 70, 3)))
    (tmp_path / "floor").mkdir()
    (tmp_path / "floor" / "floor.json").write_text(json.dumps({"camera_to_world": LEVEL_CAMERA.tolist()}))
    out = tmp_path / "scene.tmp"
    out.mkdir()
    scene.run(Context(video=Path("clip.mp4"), work=tmp_path, settings=Settings(), device="cpu"), out)  # no model needed
    report = json.loads((out / "scene.json").read_text())
    assert report["used"] is False and "too few" in report["reason"]
    assert not (out / "scene.npz").exists()
    assert (out / "background.jpg").is_file() and (out / "debug.mp4").is_file()
