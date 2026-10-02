import json
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context, Settings
from twod23d.stages import check


def gaps_for(tracks: dict[int, list[float]]) -> dict:
    """Bodies with the given floor gaps per track, one per frame starting at 0."""
    frame, track_id, gap = [], [], []
    for t, values in tracks.items():
        frame += list(range(len(values)))
        track_id += [t] * len(values)
        gap += values
    return {"frame": np.array(frame), "track_id": np.array(track_id), "floor_gap": np.array(gap, np.float32)}


def test_flag_finds_bodies_off_the_floor_and_sudden_jumps():
    steady = [-0.04] * 30
    steady[10] = 0.30  # far off the floor
    steady[20] = 0.12  # only 16 cm off this person's usual -4 cm: a jump
    bodies = gaps_for({1: steady, 2: [-0.35] * 30, 3: [0.05] * 30})
    far, jump = check.flag(bodies, fps=30)
    assert np.flatnonzero(far[:30]).tolist() == [10] and np.flatnonzero(jump[:30] & ~far[:30]).tolist() == [20]
    assert far[30:60].all()  # a person sinking the whole time
    assert not (far[60:] | jump[60:]).any()


def run_check(tmp_path: Path, bodies: dict, settings: Settings = Settings()) -> Path:
    """The check stage on these bodies, in a work folder of 40 frames at 30 fps; returns its output folder."""
    (tmp_path / "ingest" / "frames").mkdir(parents=True)
    for i in range(40):
        cv2.imwrite(str(tmp_path / "ingest" / "frames" / f"{i:06d}.jpg"), np.zeros((120, 160, 3), np.uint8))
    (tmp_path / "ingest" / "meta.json").write_text(json.dumps({"fps": 30.0, "width": 160, "height": 120, "num_frames": 40}))
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies" / "camera.json").write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}))
    (tmp_path / "floor").mkdir()
    camera_to_world = [[1, 0, 0, 0], [0, -1, 0, 1.6], [0, 0, -1, 0], [0, 0, 0, 1]]  # a level camera 1.6 m up
    (tmp_path / "floor" / "floor.json").write_text(json.dumps({"camera_to_world": camera_to_world}))
    n = len(bodies["frame"])
    vertices = np.zeros((n, 3, 3), np.float16)
    vertices[:, :, 2] = -5  # 5 m in front of the camera
    np.savez(tmp_path / "floor" / "bodies.npz", **bodies, vertices=vertices, faces=np.array([[0, 1, 2]]), mhr_shape=np.zeros((n, 45)))
    out = tmp_path / "check.tmp"
    out.mkdir()
    check.run(Context(video=Path("clip.mp4"), work=tmp_path, settings=settings, device="cpu"), out)
    return out


def test_check_stage_drops_bad_bodies_and_short_tracks(tmp_path):
    steady = [-0.04] * 40
    steady[5] = 0.40
    bodies = gaps_for({1: steady, 2: [-0.40] * 30 + [0.0] * 10})  # track 2 has only 10 good frames: too few
    out = run_check(tmp_path, bodies)

    with np.load(out / "bodies.npz") as f:
        assert f["track_id"].tolist() == [1] * 39 and 5 not in f["frame"].tolist()
        assert f["faces"].tolist() == [[0, 1, 2]] and f["mhr_shape"].shape == (39, 45)
    report = json.loads((out / "report.json").read_text())["tracks"]
    assert report["1"] == {"kept": 39, "off_the_floor": 1, "sudden_jumps": 0, "dropped_whole_track": False}
    assert report["2"]["dropped_whole_track"] and report["2"]["kept"] == 0
    assert (out / "debug.mp4").is_file()


def test_check_stage_counts_each_body_for_the_frames_it_stands_for(tmp_path):
    bodies = gaps_for({1: [0.0] * 4})
    bodies["frame"] *= 10  # bodies for every 10th frame only, as in --tiny: 4 of them cover 40 frames
    out = run_check(tmp_path, bodies, Settings(body_every=10))
    assert not json.loads((out / "report.json").read_text())["tracks"]["1"]["dropped_whole_track"]
