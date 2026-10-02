"""Export: the scene as a GLB for the viewer. Roadmap step 6.

Each person becomes one skinned MHR body in their median body shape, in the colors stage's colors if it has
them, or else in their track's color. It is driven by MHR's 127-joint
skeleton in every frame they were seen, placed in the floor's world coordinates, and hidden before they
first appear and after they are last seen. They stand in the scene stage's mesh of the place, painted with
the video and unlit, since the video's light is in the picture, in front of its "backdrop"; without a scene,
on a plain floor. The scene or floor node holds extras.play_area, [x0, z0, x1, z1] around where people go,
for a viewer to size the scene by. The
skinning is MHR's own linear blend skinning without its pose correctives, which move the mesh about 2 mm on average.
Output:
- scene.glb: y up, meters, one animation named "motion"
- debug.mp4: the exported bodies, skinned the way a viewer skins them, drawn back over the video
"""

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twod23d.glb import GLB
from twod23d.pipeline import Context
from twod23d.stages.track import track_color
from twod23d.video import read_meta, write_debug_video

log = logging.getLogger(__name__)

FLOOR_MARGIN = 3.0  # meters of floor around where people go
FLOOR_SIZE = 10.0  # meters on a side when nobody was found
STATIC_TRANSLATION = 0.1  # cm: a joint offset that varies less than this is stored once, not per frame
STATIC_SCALE = 0.002  # likewise for a joint's scale
SAM_AXES = np.diag([1.0, -1.0, -1.0])  # SAM 3D Body flips MHR's y and z to get camera axes


def run(ctx: Context, out: Path) -> None:
    fps = read_meta(ctx.work)["fps"]
    with np.load(ctx.work / "check" / "bodies.npz") as f:
        bodies = dict(f)
    camera_to_world = np.array(json.loads((ctx.work / "floor" / "floor.json").read_text())["camera_to_world"])
    glb = GLB()
    rig, people = None, []
    if len(bodies["frame"]):
        rig = load_rig()
        people = [pose_person(rig, bodies, track_id, camera_to_world) for track_id in np.unique(bodies["track_id"])]
        colors = {}
        if (ctx.work / "colors" / "colors.npz").exists():
            with np.load(ctx.work / "colors" / "colors.npz") as f:
                colors = {int(t): linear(c) for t, c in zip(f["track_id"], f["colors"])}
        channels = []
        for person in people:
            channels += add_person(glb, rig, person, fps, colors.get(person.track_id))
        glb.add_animation("motion", channels)
        errors = np.concatenate([person_errors(rig, person, bodies) for person in people])
        log.info("export: %d people; skinned bodies are %.1f cm from the estimated meshes on average (worst %.1f cm)", len(people), 100 * errors.mean(), 100 * errors.max())
        xz = bodies["vertices"][:, ::50][..., [0, 2]].reshape(-1, 2).astype(np.float32)
        center, half = (xz.min(0) + xz.max(0)) / 2, (xz.max(0) - xz.min(0)).max() / 2 + FLOOR_MARGIN
        low, high = center - half, center + half  # square, so a viewer's square grid fits it
    else:
        log.warning("export: nobody to export, just a floor")
        low, high = np.full(2, -FLOOR_SIZE / 2), np.full(2, FLOOR_SIZE / 2)
    play_area = {"play_area": [float(low[0]), float(low[1]), float(high[0]), float(high[1])]}
    if (ctx.work / "scene" / "scene.npz").exists():
        with np.load(ctx.work / "scene" / "scene.npz") as f:
            scene = dict(f)
        texture = glb.add_texture((ctx.work / "scene" / "background.jpg").read_bytes())
        material = glb.add_material((1.0, 1.0, 1.0, 1.0), texture=texture, unlit=True, double_sided=False)
        for name, prefix, extras in (("scene", "", play_area), ("backdrop", "backdrop_", None)):
            mesh = glb.add_mesh(name, scene[prefix + "vertices"], scene[prefix + "faces"], material, uvs=scene[prefix + "uvs"])
            glb.add_node(name, mesh=mesh, extras=extras)
    else:
        floor = glb.add_mesh("floor", *floor_mesh(low, high), glb.add_material((0.55, 0.57, 0.6, 1.0)))
        glb.add_node("floor", mesh=floor, extras=play_area)
    camera = json.loads((ctx.work / "bodies" / "camera.json").read_text())
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    # The video's camera, so a viewer can start from the same view. glTF cameras look down -z with y up.
    pose = camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0])
    video_camera = glb.add_camera(aspect=camera["cx"] / camera["cy"], yfov=2 * np.arctan(camera["cy"] / camera["fy"]))
    glb.add_node("video camera", camera=video_camera, translation=pose[:3, 3], rotation=matrix_quaternion(pose[:3, :3]))
    glb.save(out / "scene.glb")
    log.info("export: scene.glb is %.1f MB", (out / "scene.glb").stat().st_size / 1e6)
    write_debug_video(ctx.work, out, "export", draw_people(rig, people, np.linalg.inv(camera_to_world), K))


@dataclass
class Rig:
    """What MHR's model file holds about its skeleton and skin, as numpy arrays (centimeters)."""

    model: object  # the TorchScript MHR model, for posing
    parents: np.ndarray  # (127,), -1 for the root; parents come before their children
    inverse_bind: np.ndarray  # (127, 4, 4)
    base_shape: np.ndarray  # (V, 3)
    shape_vectors: np.ndarray  # (45, V, 3)
    faces: np.ndarray  # (F, 3)
    joints: np.ndarray  # (V, 4) the joints that move each vertex
    weights: np.ndarray  # (V, 4) and how much


@dataclass
class Person:
    track_id: int
    frames: np.ndarray  # (n,)
    shape: np.ndarray  # (45,) median body shape
    translation: np.ndarray  # (n, 127, 3) each joint relative to its parent, MHR centimeters
    rotation: np.ndarray  # (n, 127, 4) quaternions, x y z w
    scale: np.ndarray  # (n, 127)
    placement: np.ndarray  # (n, 4, 4) MHR's space (centimeters) to the world (meters) in each frame
    rows: np.ndarray  # (n,) this person's rows in bodies.npz


def load_rig() -> Rig:
    """MHR's rig, from the model file that comes with SAM 3D Body's weights."""
    import torch
    from huggingface_hub import snapshot_download

    from twod23d.stages.bodies import BODY_WEIGHTS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load's deprecation; the file is TorchScript
        model = torch.jit.load(str(Path(snapshot_download(BODY_WEIGHTS)) / "assets" / "mhr_model.pt"), map_location="cpu")
    buffers = {name: value.numpy() for name, value in model.named_buffers()}
    skin = "character_torch.linear_blend_skinning."
    vertex, joint, weight = (buffers[skin + key] for key in ("vert_indices_flattened", "skin_indices_flattened", "skin_weights_flattened"))
    order = np.argsort(vertex, kind="stable")
    vertex, joint, weight = vertex[order], joint[order], weight[order]
    count = len(buffers["character_torch.mesh.rest_vertices"])
    slot = np.arange(len(vertex)) - np.searchsorted(vertex, vertex)  # nth influence of its vertex
    joints, weights = np.zeros((count, 4), np.int64), np.zeros((count, 4), np.float32)
    joints[vertex, slot], weights[vertex, slot] = joint, weight  # MHR uses at most 4 per vertex, as glTF does
    parents = buffers["character_torch.skeleton.joint_parents"].astype(np.int64)
    assert (parents[1:] < np.arange(1, len(parents))).all()  # so a single pass can chain joint transforms
    return Rig(
        model=model,
        parents=parents,
        inverse_bind=tqs_to_matrix(buffers[skin + "inverse_bind_pose"].astype(np.float64)),
        base_shape=buffers["character_torch.blend_shape.base_shape"],
        shape_vectors=buffers["character_torch.blend_shape.shape_vectors"],
        faces=buffers["character_torch.mesh.faces"],
        joints=joints,
        weights=weights / weights.sum(1, keepdims=True),
    )


def pose_person(rig: Rig, bodies: dict, track_id: int, camera_to_world: np.ndarray) -> Person:
    """The skeleton of one track in each of its frames, and where it stands in the world."""
    import torch

    rows = np.flatnonzero(bodies["track_id"] == track_id)
    rows = rows[np.argsort(bodies["frame"][rows])]
    skeleton, placement = [], []
    rotation = camera_to_world[:3, :3] @ SAM_AXES / 100  # MHR centimeters to world meters
    for chunk in np.array_split(rows, max(1, len(rows) // 64)):
        inputs = [torch.from_numpy(bodies[key][chunk]).float() for key in ("mhr_shape", "mhr_model_params", "mhr_expression")]
        with torch.inference_mode():
            vertices, state = rig.model(*inputs, True)  # as SAM 3D Body posed them, correctives and all
        skeleton.append(state.double().numpy())
        # The saved world vertices are this pose, moved into place: recover the move
        offset = bodies["vertices"][chunk].astype(np.float64).mean(1) - vertices.double().numpy().mean(1) @ rotation.T
        for t in offset:
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = rotation, t
            placement.append(matrix)
    state = np.concatenate(skeleton)  # (n, 127, 8): each joint's world translation, quaternion, scale
    translation, quaternion, scale = local_transforms(state, rig.parents)
    return Person(
        track_id=int(track_id),
        frames=bodies["frame"][rows],
        shape=np.median(bodies["mhr_shape"][rows], axis=0),
        translation=translation,
        rotation=quaternion,
        scale=scale,
        placement=np.array(placement),
        rows=rows,
    )


def local_transforms(state: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint transforms relative to their parents, from world ones (translation, xyzw quaternion, uniform scale)."""
    t, q, s = state[..., :3], state[..., 3:7], state[..., 7]
    lt, lq, ls = t.copy(), q.copy(), s.copy()
    p = parents[1:]
    inverse = np.swapaxes(quaternion_matrix(q[:, p]), -1, -2)
    lt[:, 1:] = np.einsum("nkij,nkj->nki", inverse, t[:, 1:] - t[:, p]) / s[:, p, None]
    lq[:, 1:] = quaternion_multiply(q[:, p] * [-1, -1, -1, 1], q[:, 1:])
    ls[:, 1:] = s[:, 1:] / s[:, p]
    return lt, lq / np.linalg.norm(lq, axis=-1, keepdims=True), ls


def add_person(glb: GLB, rig: Rig, person: Person, fps: float, colors: np.ndarray | None = None) -> list:
    """Nodes, skinned mesh and animation channels for one person, in linear RGB vertex colors (V, 3) if given."""
    color = (1.0, 1.0, 1.0, 1.0) if colors is not None else tuple(c / 255 for c in track_color(person.track_id)[::-1]) + (1.0,)
    rest = rig.base_shape + np.einsum("k,kvc->vc", person.shape, rig.shape_vectors)
    first = person.placement[0]
    root = glb.add_node(
        f"person {person.track_id}",
        translation=first[:3, 3],
        rotation=matrix_quaternion(first[:3, :3] * 100),
        scale=[0.01] * 3,
    )
    times = person.frames / fps
    channels = [
        (root, "translation", times, person.placement[:, :3, 3], "LINEAR"),
        (root, "scale", *visibility(person.frames, fps), "STEP"),
    ]
    joints = []
    for j, parent in enumerate(rig.parents):
        translation = person.translation[:, j]
        scale = np.repeat(person.scale[:, j, None], 3, 1)
        steady_t = np.abs(translation - np.median(translation, 0)).max() < STATIC_TRANSLATION
        steady_s = np.abs(scale - np.median(scale, 0)).max() < STATIC_SCALE
        joints.append(
            glb.add_node(
                f"person {person.track_id} joint {j}",
                parent=root if parent < 0 else joints[parent],
                translation=np.median(translation, 0) if steady_t else translation[0],
                rotation=person.rotation[0, j],
                scale=np.median(scale, 0) if steady_s else scale[0],
            )
        )
        channels.append((joints[j], "rotation", times, person.rotation[:, j], "LINEAR"))
        if not steady_t:
            channels.append((joints[j], "translation", times, translation, "LINEAR"))
        if not steady_s:
            channels.append((joints[j], "scale", times, scale, "LINEAR"))
    skin = glb.add_skin(joints, rig.inverse_bind, skeleton=joints[0])
    mesh = glb.add_mesh(
        f"person {person.track_id}", rest, rig.faces, glb.add_material(color), vertex_normals(rest, rig.faces), rig.joints, rig.weights, colors=colors
    )
    glb.add_node(f"person {person.track_id} body", mesh=mesh, skin=skin)  # at the root: glTF ignores a skinned mesh's own node
    return channels


def visibility(frames: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale keys that show a person only between their first and last frame (its node scales MHR's cm to m)."""
    times, scales = [frames[0] / fps], [0.01]
    if frames[0] > 0:
        times, scales = [0.0] + times, [0.0] + scales
    times.append((frames[-1] + 1) / fps)
    scales.append(0.0)
    return np.array(times), np.repeat(np.array(scales)[:, None], 3, 1)


def skinned_vertices(rig: Rig, person: Person, k: int) -> np.ndarray:
    """Person's body in world meters in their kth frame, skinned as a glTF viewer does."""
    world = np.zeros((len(rig.parents), 4, 4))
    local = np.zeros((len(rig.parents), 4, 4))
    local[:, :3, :3] = quaternion_matrix(person.rotation[k]) * person.scale[k, :, None, None]
    local[:, :3, 3] = person.translation[k]
    local[:, 3, 3] = 1
    for j, parent in enumerate(rig.parents):
        world[j] = local[j] if parent < 0 else world[parent] @ local[j]
    skin = person.placement[k] @ world @ rig.inverse_bind  # (127, 4, 4)
    rest = rig.base_shape + np.einsum("k,kvc->vc", person.shape, rig.shape_vectors)
    rest = np.concatenate([rest, np.ones((len(rest), 1))], 1)
    return np.einsum("vk,vkij,vj->vi", rig.weights, skin[rig.joints], rest)[:, :3]


def person_errors(rig: Rig, person: Person, bodies: dict, every: int = 10) -> np.ndarray:
    """Mean distance between the exported body and the estimated mesh, in meters, for every 10th frame."""
    return np.array(
        [
            np.linalg.norm(skinned_vertices(rig, person, k) - bodies["vertices"][person.rows[k]].astype(np.float64), axis=1).mean()
            for k in range(0, len(person.frames), every)
        ]
    )


def draw_people(rig: Rig | None, people: list, world_to_camera: np.ndarray, K: np.ndarray):
    """A draw(frame, index) function that paints the exported bodies over the video."""
    index = {(person.track_id, int(frame)): (person, k) for person in people for k, frame in enumerate(person.frames)}

    def draw(image: np.ndarray, i: int) -> None:
        height, width = image.shape[:2]
        for person in people:
            if (person.track_id, i) not in index:
                continue
            v = skinned_vertices(rig, *index[(person.track_id, i)]) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            uv = (v @ K.T)[:, :2] / v[:, 2:3]
            inside = (v[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1)
            x, y = uv[inside].astype(int).T
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                image[y + dy, x + dx] = track_color(person.track_id)

    return draw


def floor_mesh(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A rectangle at y = 0 from (x, z) = low to high, facing up."""
    (x0, z0), (x1, z1) = low, high
    vertices = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], np.float32)
    faces = np.array([[0, 3, 2], [0, 2, 1]], np.uint32)  # counter-clockwise seen from above
    return vertices, faces


def linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB colors (0-255), as in images, to the linear 0-1 colors that glTF vertex colors are."""
    c = srgb.astype(np.float64) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)  # length is twice the area, so bigger faces count more
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from x y z w quaternions, for any leading shape."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation a * b (b first, then a), as x y z w quaternions."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def matrix_quaternion(m: np.ndarray) -> np.ndarray:
    """The x y z w quaternion of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return np.array([x, y, z, w])


def tqs_to_matrix(tqs: np.ndarray) -> np.ndarray:
    """4x4 matrices from MHR's (translation, x y z w quaternion, uniform scale) rows."""
    matrix = np.zeros(tqs.shape[:-1] + (4, 4))
    matrix[..., :3, :3] = quaternion_matrix(tqs[..., 3:7]) * tqs[..., 7:8, None]
    matrix[..., :3, 3] = tqs[..., :3]
    matrix[..., 3, 3] = 1
    return matrix
