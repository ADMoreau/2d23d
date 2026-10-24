import json
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context, Settings
from twod23d.stages import ingest
from twod23d.stages.ingest import measure_lens, undistort_maps, undistort_points


def straight_lines(width: int = 960, height: int = 540) -> np.ndarray:
    """A picture of long straight lines that don't cross (crossings cut edges short), plus circles, curved in any lens."""
    image = np.full((height, width, 3), 40, np.uint8)
    for y in range(30, height, 90):
        cv2.line(image, (0, y), (width, y + 40), (230, 230, 230), 3)
    for x in range(100, width, 250):
        cv2.circle(image, (x, 120), 25, (230, 230, 230), 3)
    return image


def distort(image: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """The picture as a lens with this distortion would show it (half width = 1, as in ingest)."""
    height, width = image.shape[:2]
    half = width / 2
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    seen = np.stack([(u - half) / half, (v - height / 2) / half], -1).reshape(-1, 2)
    source = undistort_points(seen, k1, k2) * half + [half, height / 2]
    maps = source[:, 0].reshape(height, width).astype(np.float32), source[:, 1].reshape(height, width).astype(np.float32)
    return cv2.remap(image, *maps, cv2.INTER_LINEAR)


def test_undistort_points_inverts_the_distortion():
    points = np.random.default_rng(0).uniform(-1.2, 1.2, (100, 2))
    r2 = (points**2).sum(-1, keepdims=True)
    distorted = points * (1 - 0.2 * r2 + 0.04 * r2**2)
    assert np.allclose(undistort_points(distorted, -0.2, 0.04), points, atol=1e-9)


def test_measure_lens_finds_barrel_distortion():
    lens = measure_lens([distort(straight_lines(), -0.2, 0.03)])
    assert lens is not None
    # k1 and k2 trade off a little; what matters is how far the side of the picture moves
    r = undistort_points(np.array([[1.0, 0.0]]), lens["k1"], lens["k2"])[0, 0]
    assert abs(r - undistort_points(np.array([[1.0, 0.0]]), -0.2, 0.03)[0, 0]) < 0.03
    assert lens["off_line"] < 0.7 * lens["off_line_before"]


def test_measure_lens_leaves_a_straight_picture_alone():
    assert measure_lens([straight_lines()]) is None


def test_measure_lens_needs_long_edges():
    assert measure_lens([np.full((540, 960, 3), 90, np.uint8)]) is None


def test_undistort_maps_straighten_and_keep_the_middle():
    lens = {"k1": -0.2, "k2": 0.03}
    picture = straight_lines()
    maps, (width, height) = undistort_maps(lens, picture.shape[:2])
    assert width > picture.shape[1] and height >= picture.shape[0]
    assert width % 2 == 0 and height % 2 == 0
    restored = cv2.remap(distort(picture, -0.2, 0.03), *maps, cv2.INTER_LINEAR)
    # the middle stays put, at the same scale
    middle = restored[(height - 540) // 2 :][:540, (width - 960) // 2 :][:, :960]
    assert np.abs(middle[200:340, 380:580].astype(int) - picture[200:340, 380:580]).mean() < 8


def test_ingest_writes_frames_and_no_lens_for_a_plain_video(tmp_path):
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(6):
        writer.write(np.full((120, 160, 3), 20 * i, np.uint8))
    writer.release()
    out = tmp_path / "ingest"
    out.mkdir()
    ingest.run(Context(video=clip, work=tmp_path, settings=Settings(), device="cpu"), out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["num_frames"] == 6 and (meta["width"], meta["height"]) == (160, 120)
    assert meta["lens"] is None
    assert len(list((out / "frames").glob("*.jpg"))) == 6
    assert (out / "debug.mp4").exists()
