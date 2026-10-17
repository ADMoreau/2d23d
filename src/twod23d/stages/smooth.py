"""Smooth: each person's motion made steady over the whole video, with Momentum.

SAM 3D Body estimates every frame on its own, so its bodies wobble from frame to frame: the far players in the
sample by several centimeters, most of it in depth, and everyone's bone lengths by about 2%. Momentum, the
solver library MHR is built on, fits each person's whole track at once: every joint stays near where SAM 3D Body
put it, joints are discouraged from accelerating, and MHR's scale parameters, the body's proportions, are shared
by all frames. A gap in a track splits it into runs that are solved apart (with --tiny, bodies 10 frames apart
count as consecutive). Momentum needs MHR's rig files, which are downloaded into the cache on first use. Without
pymomentum (it needs Python 3.12 or newer) the bodies pass through unchanged.
Output:
- bodies.npz like bodies/bodies.npz, with each body's MHR model parameters, vertices and keypoints smoothed, in
  the same camera coordinates
- debug.mp4 with each smoothed body's mesh drawn over the frames
"""

import io
import json
import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from twod23d.pipeline import Context
from twod23d.stages.bodies import CACHE, draw_bodies
from twod23d.video import write_debug_video

log = logging.getLogger(__name__)

MHR_ASSETS = "https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
RIG_FILES = ("lod1.fbx", "compact_v6_1.model", "LICENSE.txt")  # SAM 3D Body's bodies are MHR's LOD 1
SMOOTHING = 0.1  # weight of joint acceleration against distance from SAM 3D Body's joints; 1 damped fast swings
ITERATIONS = 15
ROOT_UNITS = 10.0  # centimeters per unit of MHR's root translation parameters
CHUNK = 64  # bodies posed at once by the MHR rig
SAM_AXES = np.diag([1.0, -1.0, -1.0])  # SAM 3D Body flips MHR's y and z to get camera axes; MHR is in centimeters


def run(ctx: Context, out: Path) -> None:
    with np.load(ctx.work / "bodies" / "bodies.npz") as f:
        bodies = dict(f)
    camera = json.loads((ctx.work / "bodies" / "camera.json").read_text())
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], np.float32)
    try:
        import pymomentum.solver2  # noqa: F401
    except ImportError:
        log.warning("smooth: pymomentum isn't installed (it needs Python 3.12 or newer); bodies left as they are")
    else:
        if len(bodies["frame"]):
            from twod23d.stages.export import load_rig

            character, rig = load_character(), load_rig()
            for track_id in np.unique(bodies["track_id"]):
                rows = np.flatnonzero(bodies["track_id"] == track_id)
                rows = rows[np.argsort(bodies["frame"][rows], kind="stable")]
                smooth_track(character, rig, bodies, rows, ctx.settings.body_every)
    np.savez_compressed(out / "bodies.npz", **bodies)
    write_debug_video(ctx.work, out, "smooth", draw_bodies(bodies["frame"], bodies["track_id"], bodies["vertices"], K))


def load_character():
    """MHR's rig as a Momentum character, from its release files, downloaded into the cache once."""
    import pymomentum.geometry as geometry

    folder = CACHE / "mhr-v1.0.1"
    if not all((folder / name).exists() for name in RIG_FILES):
        log.info("smooth: downloading MHR's rig files")
        with urllib.request.urlopen(MHR_ASSETS, timeout=600) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        tmp = folder.with_name(folder.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name in RIG_FILES:
            (tmp / name).write_bytes(archive.read(f"assets/{name}"))
        shutil.rmtree(folder, ignore_errors=True)
        tmp.rename(folder)
    return geometry.Character.load_fbx(str(folder / "lod1.fbx"), str(folder / "compact_v6_1.model"), load_blendshapes=False)


def smooth_track(character, rig, bodies: dict, rows: np.ndarray, step: int = 1) -> None:
    """Smooth one person's bodies (rows, in frame order, one every `step` frames) in place."""
    import pymomentum.geometry as geometry

    placed = place(rig, bodies, rows)
    targets = geometry.model_parameters_to_skeleton_state(character, placed)[..., :3]
    shared = np.asarray(character.parameter_transform.scaling_parameters, bool)
    solved = placed.copy()
    for run in runs(bodies["frame"][rows], step):
        if len(run) < 3:  # too short to have an acceleration
            continue
        start = placed[run].copy()
        start[:, shared] = np.median(placed[:, shared], axis=0)
        solved[run] = solve(character, targets[run], start, shared)
    before = bodies["vertices"][rows].astype(np.float32)
    after = pose(rig, bodies, rows, solved)
    bodies["keypoints"][rows] = move_keypoints(bodies["keypoints"][rows], before, after)
    bodies["vertices"][rows] = after.astype(bodies["vertices"].dtype)
    bodies["mhr_model_params"][rows] = solved
    moved = np.linalg.norm(after - before, axis=-1).mean(axis=1)
    log.info("smooth: person %d, %d bodies moved %.1f cm on average", bodies["track_id"][rows[0]], len(rows), 100 * moved.mean())


def place(rig, bodies: dict, rows: np.ndarray) -> np.ndarray:
    """The rows' MHR model parameters, with the root moved to where SAM 3D Body put each body in the camera's space."""
    params = bodies["mhr_model_params"][rows].astype(np.float32)
    moved = []
    for chunk, posed in zip(chunks(len(rows)), pose_chunks(rig, bodies, rows, params)):
        # SAM 3D Body poses MHR at the origin and moves the result into place: recover the move
        offset = bodies["vertices"][rows[chunk]].astype(np.float64).mean(axis=1) - posed.mean(axis=1)
        moved.append(offset @ SAM_AXES.T * 100 / ROOT_UNITS)
    params[:, :3] += np.concatenate(moved).astype(np.float32)
    return params


def pose(rig, bodies: dict, rows: np.ndarray, params: np.ndarray) -> np.ndarray:
    """MHR's mesh for the rows' shapes and expressions with these model parameters, in camera meters."""
    return np.concatenate(list(pose_chunks(rig, bodies, rows, params))).astype(np.float32)


def pose_chunks(rig, bodies: dict, rows: np.ndarray, params: np.ndarray) -> Iterator[np.ndarray]:
    import torch

    for chunk in chunks(len(rows)):
        inputs = [torch.from_numpy(np.asarray(a, np.float32)) for a in (bodies["mhr_shape"][rows[chunk]], params[chunk], bodies["mhr_expression"][rows[chunk]])]
        with torch.inference_mode():
            vertices, _ = rig.model(*inputs, True)
        yield vertices.double().numpy() @ (SAM_AXES / 100).T


def chunks(n: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n), max(1, n // CHUNK))


def solve(character, targets: np.ndarray, start: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """Model parameters for consecutive frames: joints near their targets (n, J, 3), with little acceleration."""
    import pymomentum.solver2 as solver2

    function = solver2.SequenceSolverFunction(character, len(targets), shared)
    for k, frame_targets in enumerate(targets):
        near = solver2.PositionErrorFunction(character)
        for joint, target in enumerate(frame_targets):
            near.add_constraint(parent=joint, target=target.astype(np.float32), weight=1.0)
        function.add_error_function(k, near)
    function.add_sequence_error_function_all_frames(solver2.AccelerationSequenceErrorFunction(character, weight=SMOOTHING))
    options = solver2.SequenceSolverOptions()
    options.max_iterations = ITERATIONS
    return solver2.solve_sequence(function, start.astype(np.float32), options)


def runs(frames: np.ndarray, step: int = 1) -> list[np.ndarray]:
    """Indices of each stretch of frames `step` apart, in order."""
    return np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != step) + 1) if len(frames) else []


def move_keypoints(keypoints: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Keypoints (n, K, 3) moved along with the mesh vertex nearest each one, from before (n, V, 3) to after."""
    nearest = np.stack([np.linalg.norm(before[i][None] - keypoints[i][:, None], axis=-1).argmin(axis=1) for i in range(len(keypoints))])
    take = np.arange(len(keypoints))[:, None]
    return (keypoints + after[take, nearest] - before[take, nearest]).astype(keypoints.dtype)
