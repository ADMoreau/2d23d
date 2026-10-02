"""Scene: the place the video was shot, as a 3D mesh painted with the video. Roadmap step 8.

The camera doesn't move, so the median of frames spread over the video, leaving out the tracked people's
boxes, shows the place without its people. MoGe-2 turns that image into a 3D point per pixel, given the field
of view the bodies stage estimated, so the scene and the bodies share the camera's rays. MoGe-2 and SAM 3D Body
each guess the scale on their own. Where a foot touches the floor, the floor at that pixel is as far away as
the foot, so the median ratio of the two distances scales the scene to the bodies.
The scene has three layers, all painted by projecting the image from the camera:
- The floor: where each pixel's camera ray meets the floor that the floor stage fitted to people's feet, so the
  court lines stay where the camera saw them, under the people's feet. MoGe's own floor only tells which pixels
  show floor; the layer reaches a little past them, under the net and walls, so nothing behind shows through.
- Everything else MoGe saw (net, walls, ceiling), without the stretched triangles that would join surfaces at
  different depths, such as a light and the ceiling behind it.
- A backdrop behind both, along every pixel's ray, that fills their gaps as seen from the camera: from there the
  scene looks exactly like the video. From elsewhere, gaps show the backdrop far behind, so a viewer can hide it.
Anything that never moved, on-screen graphics included, is painted on.
If the scene doesn't fit the bodies (too few feet on visible floor, scale ratios that disagree, or a floor
tilted away from the feet's), there's no scene, and export uses a plain floor.
Output:
- scene.json: whether there is a scene, and if not why; the scale and the checks behind it
- background.jpg: the video without its people, which is the scene's texture
- scene.npz, only with a scene: vertices (V, 3) in the floor's world, faces (F, 3) facing the camera, and
  uvs (V, 2) into background.jpg, with (0, 0) at its top left, for the floor and everything else; and
  backdrop_vertices, backdrop_faces, backdrop_uvs likewise for the backdrop
- debug.mp4: the scene's height over the video, gray on the floor and colored up to 3 m
"""

import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.stages.floor import FEET, ON_FLOOR, fit_plane, to_world
from twod23d.video import read_frame, read_meta, write_debug_video

log = logging.getLogger(__name__)

BACKGROUND_FRAMES = 60  # frames spread over the video, for the median
BOX_PADDING = 0.1  # grow each person's box by this fraction of its size on each side, for limbs and shadows
MIN_CONTACTS = 50  # foot points on visible floor needed to scale the scene
MAX_SPREAD = 0.15  # the middle half of the scale ratios may span at most this fraction of their median
MAX_TILT = 5.0  # degrees between the scene's floor and the floor fitted to feet
FLOOR_NORMAL = 0.8  # a scene point faces up if its normal's world y is above this
FLOOR_BAND = 0.15  # meters: upward-facing points this close to the floor show floor...
FLOOR_BAND_RATIO = 0.1  # ...or this fraction of the camera's height, if less: points more than 11% off along their ray aren't floor
GRID = 320  # mesh vertices across the image's longer side; the backdrop has half as many
MAX_STRETCH = 1.1  # a triangle whose far corner is this many times as far from the camera as its near one spans a gap
MARGIN = 10.0  # meters of scene kept around where people's feet touched the floor
BACKDROP_DEPTH = 1.5  # the backdrop is this many times as deep as the farthest scene points (99th percentile)...
MIN_BACKDROP_DEPTH = 20.0  # ...and at least this many meters
UNSEEN = np.iinfo(np.uint16).max


def run(ctx: Context, out: Path) -> None:
    meta = read_meta(ctx.work)
    camera = json.loads((ctx.work / "bodies" / "camera.json").read_text())
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    camera_to_world = np.array(json.loads((ctx.work / "floor" / "floor.json").read_text())["camera_to_world"])

    boxes = defaultdict(list)
    for row in json.loads((ctx.work / "track" / "tracks.json").read_text())["detections"]:
        boxes[row["frame"]].append(row["box"])
    picks = np.linspace(0, meta["num_frames"] - 1, min(BACKGROUND_FRAMES, meta["num_frames"])).round().astype(int)
    image = background(np.stack([read_frame(ctx.work, int(i)) for i in picks]), [boxes[int(i)] for i in picks])
    cv2.imwrite(str(out / "background.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    with np.load(ctx.work / "bodies" / "bodies.npz") as f:
        feet = f["keypoints"][:, FEET].reshape(-1, 3).astype(np.float64)
    feet = feet[np.abs(to_world(feet, camera_to_world)[:, 1]) < ON_FLOOR]  # touching the fitted floor
    report = {"used": False}
    world, keep = None, None
    if len(feet) < MIN_CONTACTS:
        report["reason"] = f"only {len(feet)} foot points touch the floor, too few to scale the scene"
    else:
        points, normals, valid = estimate_points(image, camera, ctx.settings.fov_model, ctx.device)
        ratios = depth_ratios(feet, K, points, valid)
        report |= check_scale(ratios)
        if "reason" not in report:
            points = points * report["scale"]
            world = to_world(points, camera_to_world).astype(np.float64)
            up = normals @ camera_to_world[1, :3] > FLOOR_NORMAL  # normals' world y
            report |= check_tilt(world[valid & up & (np.abs(world[..., 1]) < 2 * FLOOR_BAND)])
        if "reason" not in report:
            camera_position = camera_to_world[:3, 3]
            band = min(FLOOR_BAND, FLOOR_BAND_RATIO * camera_position[1])
            floor = valid & up & (np.abs(world[..., 1]) < band)
            rays = pixel_rays(K, camera_to_world, image.shape[:2])
            depth = floor_depth(rays, camera_position[1])
            ground = camera_position + np.where(np.isfinite(depth), depth, 0)[..., None] * rays
            low, high = np.percentile(to_world(feet, camera_to_world)[:, [0, 2]], [1, 99], axis=0)
            step = max(1, math.ceil(max(image.shape[:2]) / GRID))
            reach = cv2.dilate(floor.astype(np.uint8), np.ones((2 * step + 1, 2 * step + 1), np.uint8)).astype(bool)
            floor_keep = reach & np.isfinite(depth) & near(ground, low - MARGIN, high + MARGIN)
            rest = valid & ~floor & near(world, low - MARGIN, high + MARGIN)
            layers = [grid_mesh(ground, floor_keep, step, camera_position), grid_mesh(world, rest, step, camera_position, MAX_STRETCH)]
            vertices, faces, uvs = merge(layers)
            # The backdrop: deeper than the scene, and just under the floor where its rays reach the floor first
            depths = np.concatenate([(world[rest] - camera_position) @ camera_to_world[:3, 2], depth[floor_keep]])
            far = max(MIN_BACKDROP_DEPTH, BACKDROP_DEPTH * float(np.percentile(depths, 99)) if len(depths) else 0)
            backdrop = camera_position + np.minimum(far, 1.001 * depth)[..., None] * rays
            backdrop_mesh = grid_mesh(backdrop, np.ones(image.shape[:2], bool), 2 * step, camera_position)
            world = np.where(rest[..., None], world, ground)  # for the debug video
            keep = floor_keep | rest
            if len(faces):
                arrays = dict(zip(("vertices", "faces", "uvs"), (vertices, faces, uvs)))
                arrays |= dict(zip(("backdrop_vertices", "backdrop_faces", "backdrop_uvs"), backdrop_mesh))
                np.savez_compressed(out / "scene.npz", **arrays)
                report |= {"used": True, "floor_pixels": round(float(floor_keep.mean()), 3), "other_pixels": round(float(rest.mean()), 3)}
                report |= {"vertices": len(vertices), "triangles": len(faces), "backdrop_depth": round(far, 1)}
            else:
                report["reason"] = "nothing of the scene is near where people stood"
    (out / "scene.json").write_text(json.dumps(report, indent=2))
    if report["used"]:
        log.info(
            "scene: %d triangles, scaled by %.3f from %d foot contacts, floor %.1f degrees from the feet's",
            report["triangles"], report["scale"], report["foot_contacts"], report["tilt"],
        )
    else:
        log.warning("scene: none, %s; export uses a plain floor", report["reason"])
    write_debug_video(ctx.work, out, "scene", draw_heights(world if report["used"] else None, keep, report))


def background(frames: np.ndarray, boxes: list[list]) -> np.ndarray:
    """The per-pixel median of frames (n, H, W, 3), leaving out each frame's boxes [x1, y1, x2, y2].

    Pixels inside a box in every frame are filled in from the pixels around them.
    """
    n, height, width = frames.shape[:3]
    covered = np.zeros((n, height, width), bool)
    for k, frame_boxes in enumerate(boxes):
        for x1, y1, x2, y2 in frame_boxes:
            pad_x, pad_y = BOX_PADDING * (x2 - x1), BOX_PADDING * (y2 - y1)
            covered[k, max(0, int(y1 - pad_y)) : max(0, int(y2 + pad_y) + 1), max(0, int(x1 - pad_x)) : max(0, int(x2 + pad_x) + 1)] = True
    image = np.zeros((height, width, 3), np.uint8)
    for top in range(0, height, 64):  # in strips, to keep memory down
        image[top : top + 64] = median_of_seen(frames[:, top : top + 64], ~covered[:, top : top + 64])
    never = covered.all(axis=0)
    if never.any():
        image = cv2.inpaint(image, never.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return image


def median_of_seen(values: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """The median along the first axis of uint8 values (n, ..., C), over the entries seen (n, ...); 0 if none were."""
    ordered = values.astype(np.uint16)
    ordered[~seen] = UNSEEN  # sorts after every real value, so the seen ones come first
    ordered.sort(axis=0)
    count = np.broadcast_to(seen.sum(axis=0)[None, ..., None], (1,) + ordered.shape[1:])
    low = np.take_along_axis(ordered, np.maximum(count - 1, 0) // 2, axis=0)[0]
    high = np.take_along_axis(ordered, count // 2, axis=0)[0]
    return np.where(count[0] > 0, (low + high + 1) // 2, 0).astype(np.uint8)


def estimate_points(image: np.ndarray, camera: dict, size: str, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MoGe-2's 3D point and surface normal per pixel, in camera coordinates, and the pixels it trusts."""
    import torch

    from twod23d.stages.bodies import load_moge, quiet

    rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
    fov_x = math.degrees(2 * math.atan(camera["cx"] / camera["fx"]))  # so its points lie on our camera's rays
    with quiet(), torch.inference_mode():
        result = load_moge(size, device).infer(rgb, fov_x=fov_x, use_fp16=device.startswith("cuda"))
    points = result["points"].float().cpu().numpy().astype(np.float64)
    normals = result["normal"].float().cpu().numpy().astype(np.float64)
    valid = result["mask"].cpu().numpy().astype(bool) & np.isfinite(points).all(-1) & (points[..., 2] > 0)
    return points, normals, valid


def depth_ratios(feet: np.ndarray, K: np.ndarray, points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For each foot point (camera coordinates), its distance over the scene's at the same pixel."""
    feet = feet[feet[:, 2] > 0]
    uv = feet @ K.T
    px = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)
    height, width = valid.shape
    inside = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    feet, px = feet[inside], px[inside]
    seen = valid[px[:, 1], px[:, 0]]
    return feet[seen, 2] / points[px[seen, 1], px[seen, 0], 2]


def check_scale(ratios: np.ndarray) -> dict:
    """The scene's scale from the foot ratios, or a reason not to trust it."""
    if len(ratios) < MIN_CONTACTS:
        return {"reason": f"only {len(ratios)} foot points on visible floor, too few to scale the scene"}
    scale = float(np.median(ratios))
    low, high = np.percentile(ratios, [25, 75])
    report = {"scale": round(scale, 4), "foot_contacts": len(ratios), "scale_spread": round(float((high - low) / scale), 3)}
    if report["scale_spread"] > MAX_SPREAD:
        report["reason"] = f"the feet disagree on the scene's scale: the middle half spans {100 * report['scale_spread']:.0f}%"
    return report


def check_tilt(floor_points: np.ndarray) -> dict:
    """The angle between the scene's floor (points in the world near y = 0) and the fitted floor, or a reason."""
    if len(floor_points) < 3:
        return {"reason": "the scene shows no floor near the feet's"}
    sample = floor_points[np.random.default_rng(0).choice(len(floor_points), min(20000, len(floor_points)), replace=False)]
    up, _, _ = fit_plane(sample)
    if up is None:
        return {"reason": "the scene shows no flat floor"}
    tilt = math.degrees(math.acos(min(1.0, abs(up[1]))))
    report = {"tilt": round(tilt, 2)}
    if tilt > MAX_TILT:
        report["reason"] = f"the scene's floor is {tilt:.1f} degrees from the feet's"
    return report


def pixel_rays(K: np.ndarray, camera_to_world: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Each pixel's ray in the world (H, W, 3), as the step that goes 1 m deeper in front of the camera."""
    height, width = shape
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    return pixels @ np.linalg.inv(K).T @ camera_to_world[:3, :3].T


def floor_depth(rays: np.ndarray, camera_height: float) -> np.ndarray:
    """How deep (H, W) each ray meets the floor (y = 0), from a camera that high; inf if it never does."""
    down = rays[..., 1] < -1e-9
    return np.where(down, camera_height / -np.where(down, rays[..., 1], -1), np.inf)


def near(points: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Which points (..., 3) lie inside the box from low to high (x, z) on the floor."""
    xz = points[..., [0, 2]]
    with np.errstate(invalid="ignore"):
        return np.all((xz > low) & (xz < high), axis=-1)


def merge(meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One mesh from several (vertices, faces, uvs)."""
    offsets = np.cumsum([0] + [len(vertices) for vertices, _, _ in meshes])
    return (
        np.concatenate([vertices for vertices, _, _ in meshes]),
        np.concatenate([faces + offset for (_, faces, _), offset in zip(meshes, offsets)]).astype(np.uint32),
        np.concatenate([uvs for _, _, uvs in meshes]),
    )


def grid_mesh(
    points: np.ndarray, keep: np.ndarray, step: int, camera_position: np.ndarray, max_stretch: float = np.inf
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A mesh through every `step`th pixel of a point map (H, W, 3), over the pixels to keep, facing the camera.

    Leaves out triangles whose far corner is over max_stretch times as far from the camera as their near one.
    Returns vertices (V, 3), faces (F, 3) and each vertex's pixel as uvs (V, 2) in [0, 1], (0, 0) at the top left.
    """
    height, width = keep.shape
    rows, cols = (np.unique(np.append(np.arange(0, n, step), n - 1)) for n in (height, width))  # out to the edges
    grid, ok = points[rows][:, cols], keep[rows][:, cols]
    index = np.full(ok.shape, -1)
    index[ok] = np.arange(ok.sum())
    tl, tr, bl, br = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = np.concatenate([np.stack([tl, bl, tr], -1).reshape(-1, 3), np.stack([tr, bl, br], -1).reshape(-1, 3)])
    faces = faces[(faces >= 0).all(axis=1)]
    vertices = grid[ok]
    distance = np.linalg.norm(vertices - camera_position, axis=1)[faces]
    faces = faces[distance.max(axis=1) <= max_stretch * distance.min(axis=1)]
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    away = np.einsum("ij,ij->i", np.cross(b - a, c - a), camera_position - a) < 0
    faces[away] = faces[away][:, [0, 2, 1]]  # glTF's front faces wind counter-clockwise, seen from the camera
    used = np.zeros(len(vertices), bool)
    used[faces] = True
    u, v = np.meshgrid((cols + 0.5) / width, (rows + 0.5) / height)
    uvs = np.stack([u[ok], v[ok]], axis=-1)
    renumber = np.cumsum(used) - 1
    return vertices[used].astype(np.float32), renumber[faces].astype(np.uint32), uvs[used].astype(np.float32)


def draw_heights(world: np.ndarray | None, keep: np.ndarray | None, report: dict) -> Callable[[np.ndarray, int], None]:
    """A draw(frame, index) function: the scene's height over the frames, gray on the floor, colored up to 3 m."""
    if world is None:
        text = f"no scene: {report['reason']}"
    else:
        heights = np.clip(world[..., 1] / 3.0, 0, 1)
        overlay = cv2.applyColorMap((heights * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        overlay[np.abs(world[..., 1]) < 1e-6] = (128, 128, 128)
        text = f"scale {report['scale']:.3f} from {report['foot_contacts']} foot contacts, floor {report['tilt']:.1f} deg off"

    def draw(image: np.ndarray, i: int) -> None:
        if world is not None:
            np.copyto(image, cv2.addWeighted(image, 0.5, overlay, 0.5, 0), where=keep[..., None])
        cv2.putText(image, text, (8, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return draw
