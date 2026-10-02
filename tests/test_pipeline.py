"""The whole pipeline in --tiny mode, like the README quickstart. Marked ml: it runs the models, so CI skips it."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from twod23d.cli import main
from twod23d.stages import STAGES

pytest.importorskip("rfdetr")
from huggingface_hub import get_token  # noqa: E402  (part of the ml extras)

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(get_token() is None, reason="SAM 3D Body's gated weights need `huggingface-cli login`"),
]

SAMPLE = Path(__file__).parents[1] / "samples" / "pickleball.mp4"


@pytest.fixture
def clip(tmp_path):
    """4 seconds of two boxes sliding across a gray background, 960x540 at 25 fps."""
    path = tmp_path / "clip.mp4"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 540))
    for i in range(100):
        frame = np.full((540, 960, 3), 128, np.uint8)
        cv2.rectangle(frame, (100 + 5 * i, 200), (160 + 5 * i, 400), (0, 0, 255), -1)
        cv2.rectangle(frame, (800 - 4 * i, 180), (860 - 4 * i, 400), (255, 0, 0), -1)
        video.write(frame)
    video.release()
    return path


def test_tiny_run(clip, tmp_path):
    work, glb = tmp_path / "work", tmp_path / "out" / "scene.glb"
    args = ["run", str(clip), "--tiny", "--work", str(work), "-o", str(glb)]
    main(args)

    meta = json.loads((work / "ingest" / "meta.json").read_text())
    assert meta["num_frames"] == 75  # the first 3 of 4 seconds
    assert (meta["width"], meta["height"]) == (640, 360)
    for stage in STAGES:
        debug = cv2.VideoCapture(str(work / stage / "debug.mp4"))
        assert debug.get(cv2.CAP_PROP_FRAME_COUNT) == 75, stage

    assert glb.read_bytes()[:4] == b"glTF"

    ingested = (work / "ingest" / "meta.json").stat().st_mtime_ns
    main(args)  # everything exists, so every stage is skipped
    assert (work / "ingest" / "meta.json").stat().st_mtime_ns == ingested


@pytest.mark.skipif(not SAMPLE.exists(), reason="needs samples/pickleball.mp4")
def test_tracks_the_four_players(tmp_path):
    work = tmp_path / "work"
    main(["run", str(SAMPLE), "--tiny", "--work", str(work), "-o", str(tmp_path / "scene.glb")])
    rows = json.loads((work / "track" / "tracks.json").read_text())["detections"]
    frames_per_track = Counter(row["track_id"] for row in rows)
    assert len(frames_per_track) == 4
    assert min(frames_per_track.values()) >= 0.9 * 90  # --tiny keeps 90 frames

    with np.load(work / "floor" / "bodies.npz") as f:
        vertices, tracks = f["vertices"].astype(np.float32), f["track_id"]
    assert set(tracks) == set(frames_per_track)  # a body for every player
    assert np.allclose(vertices[:, :, 1].min(1), 0, atol=0.01)  # each standing on the floor
    floor = json.loads((work / "floor" / "floor.json").read_text())
    assert 0.5 < floor["camera_height"] < 10
