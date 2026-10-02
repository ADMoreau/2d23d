"""Bodies: a 3D body for each tracked person in each frame. Roadmap step 3.

SAM 3D Body (ViT-H, MHR body model) runs on each tracked box. MoGe-2 estimates the camera's field of view
once, from a few frames, since the camera doesn't move. Both run from pinned commits of their code,
downloaded into the cache on first use, with weights from Hugging Face. SAM 3D Body's weights are gated:
request access at https://huggingface.co/facebook/sam-3d-body-vith, then run `huggingface-cli login`.
Output:
- camera.json: fx, fy, cx, cy in pixels of the ingested frames
- bodies.npz with one row per body: frame (N,), track_id (N,), vertices (N, V, 3) and keypoints (N, 70, 3)
  in meters in camera coordinates (x right, y down, z forward, as in OpenCV), the shared mesh faces (F, 3),
  and the MHR parameters behind each body (mhr_*)
- debug.mp4 with each body's mesh drawn over the frames
"""

import contextlib
import io
import json
import logging
import math
import os
import shutil
import sys
import time
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.stages.track import track_color
from twod23d.video import read_frame, read_meta, write_debug_video

log = logging.getLogger(__name__)

# Code that isn't pip-installable, pinned and fetched on first use: (GitHub repo, commit, package folder)
SAM_3D_BODY = ("facebookresearch/sam-3d-body", "b5c765a0d89d789985e186d396315e7590887b94", "sam_3d_body")
MOGE = ("microsoft/MoGe", "74fbce054ebed49800de42d0ad0e83495065719a", "moge")
BODY_WEIGHTS = "facebook/sam-3d-body-vith"  # as accurate as the DINOv3 version, smaller, and one license
FOV_FRAMES = 5  # the camera doesn't move, so a few frames are enough for its field of view
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "2d23d"
PARAMS = {  # saved name: SAM 3D Body output
    "global_rot": "global_rot",
    "body_pose": "body_pose",
    "hand_pose": "hand",
    "scale": "scale",
    "shape": "shape",
    "expression": "face",
    "model_params": "mhr_model_params",
}


def run(ctx: Context, out: Path) -> None:
    meta = read_meta(ctx.work)
    tracks = json.loads((ctx.work / "track" / "tracks.json").read_text())["detections"]
    for code in (SAM_3D_BODY, MOGE):
        path = str(fetch_code(*code))
        if path not in sys.path:
            sys.path.insert(0, path)

    camera = estimate_camera(ctx.work, meta, ctx.settings.fov_model, ctx.device)
    (out / "camera.json").write_text(json.dumps(camera, indent=2))
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], np.float32)
    log.info("bodies: the camera sees %.0f degrees across", math.degrees(2 * math.atan(camera["cx"] / camera["fx"])))

    estimator = load_body_model(ctx.device)
    by_frame = defaultdict(list)
    for row in tracks:
        if row["frame"] % ctx.settings.body_every == 0:
            by_frame[row["frame"]].append(row)
    found = defaultdict(list)
    start = time.perf_counter()
    for done, i in enumerate(sorted(by_frame)):
        if done and done % 50 == 0:  # on a CPU this stage takes hours, so say how it's going
            left = (time.perf_counter() - start) / done * (len(by_frame) - done)
            log.info("bodies: %d of %d frames done, about %d min left", done, len(by_frame), left / 60)
        rgb = cv2.cvtColor(read_frame(ctx.work, i), cv2.COLOR_BGR2RGB)
        mhr = infer(estimator, rgb, np.array([row["box"] for row in by_frame[i]], np.float32), K, ctx.device)
        cam_t = mhr["pred_cam_t"][:, None, :]
        found["frame"] += [i] * len(by_frame[i])
        found["track_id"] += [row["track_id"] for row in by_frame[i]]
        found["vertices"].append((mhr["pred_vertices"] + cam_t).astype(np.float16))
        found["keypoints"].append((mhr["pred_keypoints_3d"] + cam_t).astype(np.float32))
        for name, key in PARAMS.items():
            found[f"mhr_{name}"].append(mhr[key].astype(np.float32))
    arrays = {key: np.concatenate(value) if key not in ("frame", "track_id") else np.array(value, np.int32) for key, value in found.items()}
    if not arrays:  # nobody tracked
        arrays = {"frame": np.zeros(0, np.int32), "track_id": np.zeros(0, np.int32), "vertices": np.zeros((0, 0, 3), np.float16), "keypoints": np.zeros((0, 70, 3), np.float32)}
    np.savez_compressed(out / "bodies.npz", faces=estimator.faces.astype(np.int32), **arrays)
    log.info("bodies: %d bodies in %d frames", len(arrays["frame"]), len(by_frame))
    write_debug_video(ctx.work, out, "bodies", draw_bodies(arrays["frame"], arrays["track_id"], arrays["vertices"], K))


def fetch_code(repo: str, commit: str, package: str) -> Path:
    """A folder holding `package` from a pinned commit of a GitHub repo, downloaded into the cache once."""
    dest = CACHE / "code" / f"{repo.split('/')[1]}-{commit[:12]}"
    if not dest.exists():
        log.info("bodies: downloading %s at %s", repo, commit[:12])
        with urllib.request.urlopen(f"https://codeload.github.com/{repo}/zip/{commit}", timeout=300) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        top = archive.namelist()[0].split("/")[0]  # GitHub zips hold one folder, named REPO-COMMIT
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        for name in archive.namelist():
            relative = name[len(top) + 1 :]
            if relative.startswith((package + "/", "LICENSE")):
                archive.extract(name, tmp)
        (tmp / top).rename(dest)
        tmp.rmdir()
    return dest


def load_moge(size: str, device: str):
    """MoGe-2 (vits, vitb or vitl), from its pinned code. Call it inside quiet(): MoGe prints and warns a lot."""
    path = str(fetch_code(*MOGE))
    if path not in sys.path:
        sys.path.insert(0, path)
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained(f"Ruicheng/moge-2-{size}-normal").to(device).eval()


@contextlib.contextmanager
def quiet():
    """Silence a model's prints and warnings."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def estimate_camera(work: Path, meta: dict, size: str, device: str) -> dict:
    """Focal length and image center in pixels: MoGe-2's median over a few frames."""
    import torch

    focals = []
    with quiet():
        model = load_moge(size, device)
        for i in np.linspace(0, meta["num_frames"] - 1, FOV_FRAMES).round().astype(int):
            rgb = cv2.cvtColor(read_frame(work, int(i)), cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
            with torch.inference_mode():
                intrinsics = model.infer(image, use_fp16=device.startswith("cuda"))["intrinsics"]
            focals.append(float(intrinsics[1, 1]) * meta["height"])  # MoGe's intrinsics are relative to image size
    focal = float(np.median(focals))
    return {"fx": focal, "fy": focal, "cx": meta["width"] / 2, "cy": meta["height"] / 2}


def load_body_model(device: str):
    """SAM 3D Body and its crop transforms, loaded quietly: its loaders print a lot."""
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        folder = Path(snapshot_download(BODY_WEIGHTS))
    except GatedRepoError as e:
        raise RuntimeError(
            f"SAM 3D Body's weights are gated: request access at https://huggingface.co/{BODY_WEIGHTS}, "
            "then run `huggingface-cli login`"
        ) from e
    # The checkpoint leaves out the body model's own tensors, which come from mhr_model.pt; SAM 3D Body logs that
    logging.getLogger("sam_3d_body").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sam_3d_body.build_models import load_sam_3d_body
        from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

        model, cfg = load_sam_3d_body(
            str(folder / "model.ckpt"), device=device, mhr_path=str(folder / "assets" / "mhr_model.pt")
        )
        if not (device.startswith("cuda") and torch.cuda.is_bf16_supported()):
            # The backbone comes in bfloat16, which CPUs and older GPUs (such as Kaggle's T4) only emulate.
            # In float32 it ran 12-17x faster on the dev laptop, with the same results.
            model.backbone.float()
            model.backbone_dtype = torch.float32
        return SAM3DBodyEstimator(model, cfg)


def infer(estimator, rgb: np.ndarray, boxes: np.ndarray, K: np.ndarray, device: str) -> dict:
    """SAM 3D Body's body decoder on the boxes of one frame. Returns arrays with one row per box.

    This follows SAM3DBodyEstimator.process_one_image, which always moves its inputs to CUDA, and
    skips the hand decoder: hands are a few pixels across in a wide shot.
    """
    import torch
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    with torch.inference_mode(), cuda_calls_go_to(device):
        batch = recursive_to(prepare_batch(rgb, estimator.transform, boxes, None, None), device)
        estimator.model._initialize_batch(batch)
        batch["cam_int"] = torch.from_numpy(K)[None].to(batch["img"])
        output = estimator.model.run_inference(rgb, batch, inference_type="body")
    return recursive_to(recursive_to(output["mhr"], "cpu"), "numpy")


@contextlib.contextmanager
def cuda_calls_go_to(device: str):
    """SAM 3D Body calls tensor.cuda() in its model; send those calls to our device instead."""
    import torch

    if device == "cuda":
        yield
        return
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def draw_bodies(frames: np.ndarray, track_ids: np.ndarray, vertices: np.ndarray, K: np.ndarray) -> Callable:
    """A draw(frame, index) function that paints each body's mesh vertices in its track's color."""
    by_frame = defaultdict(list)
    for frame, track_id, v in zip(frames, track_ids, vertices):
        by_frame[int(frame)].append((int(track_id), v))

    def draw(image: np.ndarray, i: int) -> None:
        height, width = image.shape[:2]
        for track_id, v in by_frame[i]:
            v = v.astype(np.float32)
            uv = (v @ K.T)[:, :2] / v[:, 2:3]
            inside = (v[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1)
            x, y = uv[inside].astype(int).T
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):  # 2x2 dots, so close-up bodies look solid
                image[y + dy, x + dx] = track_color(track_id)

    return draw
