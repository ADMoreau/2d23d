import json
from pathlib import Path

import cv2
import numpy as np
from test_glb import read_glb

from twod23d.pipeline import Context, Settings
from twod23d.stages import export
from twod23d.stages.export import local_transforms, matrix_quaternion, quaternion_matrix, visibility


def random_quaternions(rng, shape):
    q = rng.normal(size=shape + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_local_transforms_rebuild_the_world_ones():
    rng = np.random.default_rng(0)
    parents = np.array([-1, 0, 1, 1, 3, 0])
    state = np.concatenate([rng.normal(size=(3, 6, 3)), random_quaternions(rng, (3, 6)), rng.uniform(0.8, 1.2, (3, 6, 1))], -1)
    t, q, s = local_transforms(state, parents)
    for frame in range(3):
        world = []
        for j, parent in enumerate(parents):
            local = np.eye(4)
            local[:3, :3] = quaternion_matrix(q[frame, j]) * s[frame, j]
            local[:3, 3] = t[frame, j]
            world.append(local if parent < 0 else world[parent] @ local)
            expected = quaternion_matrix(state[frame, j, 3:7]) * state[frame, j, 7]
            assert np.allclose(world[j][:3, :3], expected) and np.allclose(world[j][:3, 3], state[frame, j, :3])


def test_matrix_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for q in random_quaternions(rng, (20,)):
        back = matrix_quaternion(quaternion_matrix(q))
        assert np.allclose(back, q) or np.allclose(back, -q)


def test_visibility_shows_a_person_only_while_tracked():
    times, scales = visibility(np.array([5, 6, 20]), fps=10)
    assert np.allclose(times, [0, 0.5, 2.1])
    assert np.allclose(scales[:, 0], [0, 0.01, 0])
    times, scales = visibility(np.array([0, 1]), fps=10)  # there from the start
    assert np.allclose(times, [0, 0.2]) and np.allclose(scales[:, 0], [0.01, 0])


def test_export_puts_the_scene_in_place_of_the_floor(tmp_path):
    (tmp_path / "ingest" / "frames").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "ingest" / "frames" / "000000.jpg"), np.zeros((120, 160, 3), np.uint8))
    (tmp_path / "ingest" / "meta.json").write_text(json.dumps({"fps": 30.0, "width": 160, "height": 120, "num_frames": 1}))
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies" / "camera.json").write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}))
    (tmp_path / "floor").mkdir()
    (tmp_path / "floor" / "floor.json").write_text(json.dumps({"camera_to_world": np.eye(4).tolist()}))
    (tmp_path / "check").mkdir()
    np.savez(tmp_path / "check" / "bodies.npz", frame=np.zeros(0, int), track_id=np.zeros(0, int), vertices=np.zeros((0, 0, 3)))
    (tmp_path / "scene").mkdir()
    cv2.imwrite(str(tmp_path / "scene" / "background.jpg"), np.full((120, 160, 3), 128, np.uint8))
    triangle = {"vertices": np.eye(3, dtype=np.float32), "faces": np.array([[0, 1, 2]], np.uint32), "uvs": np.zeros((3, 2), np.float32)}
    np.savez(tmp_path / "scene" / "scene.npz", **triangle, **{"backdrop_" + key: value for key, value in triangle.items()})
    out = tmp_path / "export.tmp"
    out.mkdir()
    export.run(Context(video=Path("clip.mp4"), work=tmp_path, settings=Settings(), device="cpu"), out)

    gltf, _ = read_glb(out / "scene.glb")
    nodes = {node["name"]: node for node in gltf["nodes"]}
    assert set(nodes) == {"scene", "backdrop", "video camera"}  # no plain floor
    assert nodes["scene"]["extras"]["play_area"] == [-5, -5, 5, 5]  # nobody, so the default floor's area
    assert gltf["meshes"][nodes["backdrop"]["mesh"]]["primitives"][0]["material"] == 0  # one painted material
    assert len(gltf["images"]) == 1
