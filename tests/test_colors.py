import json
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context, Settings
from twod23d.stages import colors
from twod23d.stages.export import linear

K = np.array([[100.0, 0, 80], [0, 100, 60], [0, 0, 1]])


def square(x: float, z: float, half: float, mirrored: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A 5x5-vertex square facing the camera (or, mirrored, facing away), centered on x at depth z."""
    steps = np.linspace(-half, half, 5)
    u, v = np.meshgrid(steps[::-1] if mirrored else steps, steps)
    vertices = np.stack([x + u.ravel(), v.ravel(), np.full(25, z)], axis=-1)
    faces = []
    for i in range(4):
        for j in range(4):
            tl, tr, bl, br = 5 * i + j, 5 * i + j + 1, 5 * (i + 1) + j, 5 * (i + 1) + j + 1
            faces += [[tl, bl, tr], [tr, bl, br]]
    return vertices, np.array(faces)


def test_sample_frame_sees_what_faces_the_camera_uncovered():
    front, faces = square(0, 2, 0.2)  # covers pixels 70-90 across
    behind, _ = square(0, 3, 0.3)  # the same pixels, but behind it
    beside, _ = square(0.75, 3, 0.15)  # pixels 100-110
    away, _ = square(-0.75, 3, 0.15, mirrored=True)  # pixels 50-60, facing away
    background = np.full((120, 160, 3), 128, np.uint8)
    image = background.copy()
    image[48:73, 68:93] = (0, 0, 255)  # red, as BGR
    image[53:68, 98:113] = (0, 255, 0)
    image[53:68, 48:63] = (255, 0, 0)
    (seen_front, red), (seen_behind, _), (seen_beside, green), (seen_away, _) = colors.sample_frame(
        image, background, [front, behind, beside, away], faces, K, cell=2
    )
    assert seen_front.all() and (red == [255, 0, 0]).all()
    assert seen_beside.all() and (green == [0, 255, 0]).all()
    assert not seen_behind.any() and not seen_away.any()
    (seen_front, _), = colors.sample_frame(background, background, [front], faces, K, cell=2)
    assert not seen_front.any()  # it looks just like the background there, so it may be the court behind the person


def test_fill_unseen_spreads_colors_along_the_mesh():
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [6, 7, 8]])  # a strip, and a triangle apart
    start = np.zeros((9, 3))
    start[0], start[5] = (200, 0, 0), (0, 0, 200)
    seen = np.zeros(9, bool)
    seen[[0, 5]] = True
    filled = colors.fill_unseen(start, seen, faces)
    assert (filled[[0, 5]] == start[[0, 5]]).all()  # seen colors stay
    assert filled[1, 0] > filled[1, 2] and filled[4, 2] > filled[4, 0]  # each end takes after its nearest seen vertex
    assert (filled[6:] == 0).all()  # nothing reaches the separate triangle


def test_linear_matches_srgb():
    assert np.allclose(linear(np.array([0, 128, 255])), [0, 0.2158605, 1])


def run_colors(tmp_path: Path, frames: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """The colors stage on three frames with a red square, and one person's bodies in the given frames; returns its output."""
    (tmp_path / "ingest" / "frames").mkdir(parents=True)
    for i in range(3):
        frame = np.full((120, 160, 3), 128, np.uint8)
        frame[45:76, 65:96] = (0, 0, 255)  # the person, in red
        cv2.imwrite(str(tmp_path / "ingest" / "frames" / f"{i:06d}.jpg"), frame)
    (tmp_path / "ingest" / "meta.json").write_text(json.dumps({"fps": 30.0, "width": 160, "height": 120, "num_frames": 3}))
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies" / "camera.json").write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}))
    track_ids = np.full(len(frames), 7)
    np.savez(tmp_path / "bodies" / "bodies.npz", frame=frames, track_id=track_ids, vertices=np.float16(vertices), faces=faces)
    (tmp_path / "check").mkdir()
    np.savez(tmp_path / "check" / "bodies.npz", frame=frames, track_id=track_ids)
    (tmp_path / "scene").mkdir()
    cv2.imwrite(str(tmp_path / "scene" / "background.jpg"), np.full((120, 160, 3), 128, np.uint8))
    out = tmp_path / "colors.tmp"
    out.mkdir()
    colors.run(Context(video=Path("clip.mp4"), work=tmp_path, settings=Settings(), device="cpu"), out)
    return out


def test_colors_stage_paints_each_person(tmp_path):
    body, faces = square(0, 2, 0.2)
    out = run_colors(tmp_path, np.arange(3), np.array([body] * 3), faces)

    with np.load(out / "colors.npz") as f:
        assert f["track_id"].tolist() == [7] and f["seen"].all()
        assert (np.abs(f["colors"][0].astype(int) - [255, 0, 0]) < 20).all()  # red, give or take the JPEG
    assert json.loads((out / "colors.json").read_text()) == {"7": {"frames": 3, "seen": 1.0}}
    assert (out / "debug.mp4").is_file()


def test_colors_stage_without_people(tmp_path):
    out = run_colors(tmp_path, np.zeros(0, int), np.zeros((0, 0, 3)), np.zeros((0, 3), int))
    with np.load(out / "colors.npz") as f:
        assert f["track_id"].shape == (0,) and f["colors"].shape == (0, 0, 3)
    assert json.loads((out / "colors.json").read_text()) == {}
    assert (out / "debug.mp4").is_file()
