import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from twod23d.pipeline import Context, Settings
from twod23d.stages import smooth
from twod23d.stages.smooth import move_keypoints, runs


def test_runs_split_at_gaps():
    assert [r.tolist() for r in runs(np.array([3, 4, 5, 9, 10, 20]))] == [[0, 1, 2], [3, 4], [5]]
    assert runs(np.zeros(0, int)) == []
    assert [r.tolist() for r in runs(np.array([0, 10, 20, 40]), step=10)] == [[0, 1, 2], [3]]


def test_keypoints_follow_their_nearest_vertex():
    before = np.array([[[0.0, 0, 0], [1, 0, 0]]])
    after = before + np.array([[[0.0, 0.1, 0], [0.0, 0, 0.5]]])
    keypoints = np.array([[[0.1, 0, 0], [0.9, 0, 0]]], np.float32)
    moved = move_keypoints(keypoints, before, after)
    assert np.allclose(moved, [[[0.1, 0.1, 0], [0.9, 0, 0.5]]])
    assert moved.dtype == np.float32


def test_without_pymomentum_bodies_pass_through(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pymomentum.solver2", None)  # import fails
    (tmp_path / "ingest" / "frames").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "ingest" / "frames" / "000000.jpg"), np.zeros((120, 160, 3), np.uint8))
    (tmp_path / "ingest" / "meta.json").write_text(json.dumps({"fps": 30.0, "width": 160, "height": 120, "num_frames": 1}))
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies" / "camera.json").write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}))
    vertices = np.float16(np.random.default_rng(0).normal(0, 1, (1, 5, 3)) + [0, 0, 4])
    np.savez(tmp_path / "bodies" / "bodies.npz", frame=np.zeros(1, np.int32), track_id=np.zeros(1, np.int32), vertices=vertices)
    out = tmp_path / "smooth.tmp"
    out.mkdir()
    smooth.run(Context(video=Path("clip.mp4"), work=tmp_path, settings=Settings(), device="cpu"), out)
    with np.load(out / "bodies.npz") as f:
        assert np.array_equal(f["vertices"], vertices)
    assert (out / "debug.mp4").is_file()


@pytest.mark.ml
def test_solve_steadies_a_jittery_root():
    pytest.importorskip("pymomentum.solver2")
    import pymomentum.geometry as geometry

    character = smooth.load_character()
    n = 30
    params = np.zeros((n, character.parameter_transform.size), np.float32)
    params[:, 0] = np.linspace(0, 3, n)  # walking sideways, 30 cm over the clip
    rng = np.random.default_rng(0)
    noisy = params.copy()
    noisy[:, 2] += rng.normal(0, 0.3, n)  # 3 cm of depth jitter
    targets = geometry.model_parameters_to_skeleton_state(character, noisy)[..., :3]
    shared = np.asarray(character.parameter_transform.scaling_parameters, bool)
    solved = smooth.solve(character, targets, noisy, shared)

    def jitter(p):
        return np.abs(np.diff(p[:, 2], 2)).mean()

    assert jitter(solved) < 0.7 * jitter(noisy)  # SMOOTHING is gentle: about 0.55 here, as on the sample
    assert np.abs(solved[:, 0] - params[:, 0]).max() < 0.1  # the walk itself survives, within 1 cm
