"""A held object's shape and its pose in every frame, fit together to the tracked masks (plain PyTorch, CPU).

The object is a density grid in its own frame. Each frame's pose is the wrist joint × fit3.py's smoothed grip × a
small learned correction. Rays from the camera through pixels are marched through the grid, and the silhouette they
render (1 - transmittance) is fit to SAM 2's tracked mask (track_paddle.py).

The person comes first (the scene hierarchy): pixels of their hand are ignored, and the rest of their body only
counts where the object is seen, since it may be hidden behind them. A penalty on density keeps the object no bigger
than the views require, one on the corrections keeps poses near the skeleton's and smooth in time.

Every HOLD_EVERY-th frame is held out of the fit. Afterwards only its pose is fit (shape frozen), and its silhouette
IoU says how well the shape explains frames it never saw.

A second, finer pass starts from a coarse one (env INIT=<its dir> CELL=0.003 SAMPLES=96), in a box around its solid.
The mesh is colored from the sharpest frames (the object barely moving in the image), where its surface faces the
camera.

usage: joint.py <work dir> <paddles run dir> <track>   (the lab venv: needs scikit-image and trimesh)
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.measure import marching_cubes

torch.set_num_threads(int(os.environ.get("THREADS", 3)))
torch.manual_seed(0)
WORK, D, track = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
OUT = D / os.environ.get("JOINT_OUT", "joint")
OUT.mkdir(exist_ok=True)
CELL = float(os.environ.get("CELL", 0.006))  # m
BOUNDS = np.array([[-0.16, 0.16], [-0.10, 0.52], [-0.10, 0.10]])  # m, x y z in the object's frame
INIT = os.environ.get("INIT")  # a coarser fit's dir (e.g. joint): start from its shape and poses, in a box around its solid
PAD = 0.03  # m around that solid
STEPS = int(os.environ.get("STEPS", 3000))
BATCH, RAYS = 16, 1024  # frames per step, rays per frame
SAMPLES = int(os.environ.get("SAMPLES", 48))  # per ray: about one per cell along the box
COLOR_FRAMES = 100  # sharpest frames to color the surface from
MAX_SPEED = 0.03  # object lengths per frame in the image: slower counts as sharp
COLOR_STEPS = int(os.environ.get("COLOR_STEPS", 0))  # then: a color per cell, fit with the poses to the video's pixels
COLOR_ROUNDS = 3  # the color phase's rounds; between them, each frame's half turn about the long axis is decided
W_GAIN = 0.1  # keeps each frame's brightness gain near 1
HOLD_EVERY = 10
W_DENSITY = float(os.environ.get("W_DENSITY", 0.05))  # on mean occupancy
W_TV = float(os.environ.get("W_TV", 0.1))  # smoothness of the grid
W_BINARY = float(os.environ.get("W_BINARY", 1.0))  # pushes each cell to empty or solid (else thin sheets stay half-transparent)
POSE_DEG, POSE_M = 10.0, 0.03  # correction prior scale
SMOOTH_DEG, SMOOTH_M = 2.0, 0.005  # frame-to-frame change prior scale
# POSES=curve: how the object sits in the hand is one smooth curve over time (a cubic B-spline with control points every
# KNOT_EVERY frames), so each frame's pose is shared with its neighbors and one bad mask can't pull it far; each frame
# keeps only a small residual (RES_*) for the skeleton's jitter. Frames whose mask disagrees with the fit count less.
POSES = os.environ.get("POSES", "frame")
KNOT_EVERY = int(os.environ.get("KNOT_EVERY", 5))
RES_DEG, RES_M = 3.0, 0.01  # residual prior scale
CURVE_DEG, CURVE_M = 3.0, 0.005  # bend prior per control point (second difference)
REWEIGHT_EVERY = 500  # steps between per-frame weight updates
GOOD_IOU = 0.4  # per-frame silhouette IoU where a frame's weight is one half
# POSES=smooth: each frame keeps its own pose, but the sequence must move smoothly in camera space, as the real object
# does: little angular and linear acceleration (a robust penalty, so real swings can start and stop). Held-out frames are
# in the sequence without data, so their poses are predictions from their neighbors; frames that fit badly start from
# their good neighbors' poses (interpolated) and count less.
W_SMOOTH = float(os.environ.get("W_SMOOTH", 0.1))
ACC_DEG, ACC_M = 5.0, 0.02  # per frame², where the penalty turns from quadratic to linear
LOOSE_DEG, LOOSE_M = 30.0, 0.10  # how far from the starting grip a correction may roam
W_POSE = 1e-3
HAND_RADIUS = 0.12  # m around the holding wrist: the hand
PRIOR = os.environ.get("PRIOR")  # prior.py's occupancy (e.g. prior): fills only the cells the video doesn't see
W_PRIOR = float(os.environ.get("W_PRIOR", 1.0))
SEEN_SCALE = 5.0  # frames: the prior's weight falls as exp(-seen / SEEN_SCALE)

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
hand = json.loads((D / "fit2" / "report.json").read_text())[str(track)]["hand"]
WRIST = {"R": 41, "L": 62}[hand]
with np.load(D / "fit3" / "grips.npz") as f:
    fit = {k.split("_", 1)[1]: f[k] for k in f.files if k.startswith(f"{track}_")}
with np.load(D / os.environ.get("TRACK_DIR", "track") / f"{track}.npz") as f:
    tracked = dict(f)
TSCALE = float(tracked["scale"])
frames_dir = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
coarse = None
if INIT:  # the box shrinks to the coarse solid, so finer cells cost about the same
    with np.load(D / INIT / f"{track}.npz") as f:
        coarse = dict(f)
    c_axes = [coarse["bounds"][i, 0] + float(coarse["cell"]) * (0.5 + np.arange(coarse["occupancy"].shape[i])) for i in range(3)]
    idx = np.nonzero(coarse["occupancy"] > 0.5)
    lo = np.array([c_axes[i][idx[i].min()] for i in range(3)]) - PAD
    hi = np.array([c_axes[i][idx[i].max()] for i in range(3)]) + PAD
    # On the coarse grid's lines: off them, every value is interpolated and a crisp plate turns soft
    base, step_c = coarse["bounds"][:, 0], float(coarse["cell"])
    lo, hi = base + step_c * np.floor((lo - base) / step_c), base + step_c * np.ceil((hi - base) / step_c)
    BOUNDS = np.stack([np.maximum(lo, coarse["bounds"][:, 0]), np.minimum(hi, coarse["bounds"][:, 1])], 1)

# Frames: every one fit3 used (its wrist gate passed), with its smoothed grip as the starting pose
index = {int(x): k for k, x in enumerate(fit["frames"])}
used = [int(x) for x in fit["det_frames"]]
k = np.array([index[x] for x in used])
R_wrist, o_wrist, grips = fit["R"][k], fit["origin"][k], fit["grips"][k]
held = np.arange(len(used)) % HOLD_EVERY == HOLD_EVERY // 2
train, test = np.flatnonzero(~held), np.flatnonzero(held)
triples = torch.tensor((np.diff(used)[1:] == 1) & (np.diff(used)[:-1] == 1), dtype=torch.float32)  # n-1, n, n+1 consecutive


def rotvec_np(v):
    return cv2.Rodrigues(np.asarray(v, np.float64))[0]


R_grip = np.array([rotvec_np(g[:3]) for g in grips])
R0 = torch.tensor(R_wrist @ R_grip, dtype=torch.float32)  # object -> camera, before correction
t_grip = torch.tensor(grips[:, 3:], dtype=torch.float32)
R_w = torch.tensor(R_wrist, dtype=torch.float32)
o_w = torch.tensor(o_wrist, dtype=torch.float32)

# The person's pixels, per frame (half resolution): 1 = body (an occluder), 2 = hand (ignored)
print("person masks", flush=True)
with np.load(WORK / "bodies" / "bodies.npz") as f:
    rows = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(f["frame"], f["track_id"]))}
    mine = [rows.get((x, track)) for x in used]
    sel = [m for m in mine if m is not None]
    verts = dict(zip(sel, f["vertices"][sel]))
    wrists = dict(zip(sel, f["keypoints"][sel, WRIST]))
    faces = f["faces"]


def project(p):
    return np.stack([cam["fx"] * p[..., 0] / p[..., 2] + cam["cx"], cam["fy"] * p[..., 1] / p[..., 2] + cam["cy"]], -1)


person = np.zeros((len(used), 320, 320), np.uint8)
for n, (x, m) in enumerate(zip(used, mine)):
    if m is None:
        continue
    px = np.round((project(verts[m]) - tracked["origin"][x]) * TSCALE / 2).astype(np.int32)
    cv2.fillPoly(person[n], list(px[faces]), 1)
    near = np.linalg.norm(verts[m] - wrists[m], axis=1) < HAND_RADIUS
    cv2.fillPoly(person[n], list(px[faces[near[faces].all(1)]]), 2)
del verts


def mask_of(x):
    return np.unpackbits(tracked["masks"][x], axis=-1)


# Pixel boxes the object's bounds can project into, per frame (from the starting pose; corrections are small)
corners = np.array([[x, y, z] for x in BOUNDS[0] for y in BOUNDS[1] for z in BOUNDS[2]])
boxes = []
for n, x in enumerate(used):
    pts = corners @ R0[n].numpy().T + (R_wrist[n] @ grips[n, 3:] + o_wrist[n])
    px = (project(pts) - tracked["origin"][x]) * TSCALE
    lo, hi = np.clip(px.min(0) - 8, 0, 639), np.clip(px.max(0) + 8, 0, 639)
    boxes.append((lo, hi))

shape = np.round((BOUNDS[:, 1] - BOUNDS[:, 0]) / CELL).astype(int)  # cells along x y z
def to_logit(o):
    return np.log(np.expm1(-np.log(1 - o)))


logits = torch.full((1, 1, shape[2], shape[1], shape[0]), float(to_logit(0.02)), requires_grad=True)
if coarse is not None:  # its occupancy, resampled at the finer cells
    from scipy.ndimage import map_coordinates
    axes = [BOUNDS[i, 0] + CELL * (0.5 + np.arange(shape[i])) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    coords = [(points[..., i] - coarse["bounds"][i, 0]) / float(coarse["cell"]) - 0.5 for i in range(3)]
    start_occ = map_coordinates(coarse["occupancy"], coords, order=1, mode="nearest").clip(1e-4, 1 - 1e-6)  # a floor of
    # even 0.02 per cell fogs a ray crossing ~100 empty cells into looking solid
    with torch.no_grad():
        logits.copy_(torch.tensor(to_logit(start_occ), dtype=torch.float32).permute(2, 1, 0)[None, None])
prior = None
if PRIOR:
    with np.load(D / PRIOR / f"{track}.npz") as f:
        assert np.allclose(f["bounds"], BOUNDS) and float(f["cell"]) == CELL
        # Any of its frames: a part one view shows (say the handle) shouldn't be averaged away. Thick, but it only
        # counts where the video can't say otherwise.
        prior = torch.tensor(f["occupancy"] > 0, dtype=torch.float32).permute(2, 1, 0)[None, None]  # x y z -> the grid's z y x


def seen_counts():
    """Per cell (x y z): in how many training frames it lands on a pixel that is evidence: inside the crop, not the hand,
    and not the body unless the mask is there (else it may be hidden behind them). Starting poses."""
    axes = [BOUNDS[i, 0] + CELL / 2 + CELL * np.arange(shape[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    count = np.zeros(len(points), np.int32)
    for n in train:
        x = used[n]
        cam_points = points @ R0[n].numpy().T + (R_wrist[n] @ grips[n, 3:] + o_wrist[n])
        q = np.round((project(cam_points) - tracked["origin"][x]) * TSCALE).astype(int)
        inside = (q >= 0).all(1) & (q < 640).all(1) & (cam_points[:, 2] > 0.1)
        q = np.clip(q, 0, 639)
        c = person[n][q[:, 1] // 2, q[:, 0] // 2]
        m = mask_of(x)[q[:, 1], q[:, 0]]
        count += inside & (c != 2) & ~((c == 1) & (m == 0))
    return count.reshape(shape)


seen = seen_counts()
prior_weight = torch.tensor(np.exp(-seen / SEEN_SCALE), dtype=torch.float32).permute(2, 1, 0)[None, None] if PRIOR else None
lo_t, hi_t = torch.tensor(BOUNDS[:, 0], dtype=torch.float32), torch.tensor(BOUNDS[:, 1], dtype=torch.float32)
correction = torch.zeros(len(used), 6, requires_grad=True)  # rotvec (rad), translation (m), in the object's frame
if coarse is not None:
    assert list(coarse["frames"]) == used
    with torch.no_grad():
        correction.copy_(torch.tensor(coarse["correction"], dtype=torch.float32))

# The curve: grip(t) = shared grip, turned by a rotvec and moved by a translation that are B-splines in the frame number
curve = False  # pose() reads per-frame grips until the curve is set up
frame_weight = torch.ones(len(used))
if POSES == "curve":
    shared = np.array(json.loads((D / "fit3" / "report.json").read_text())[str(track)]["shared"])
    R_shared = torch.tensor(rotvec_np(shared[:3]), dtype=torch.float32)
    t_shared = torch.tensor(shared[3:], dtype=torch.float32)
    first = used[0] - KNOT_EVERY
    knot_count = (used[-1] - first) // KNOT_EVERY + 4
    basis = np.zeros((len(used), knot_count))
    for n, x in enumerate(used):
        u = (x - first) / KNOT_EVERY
        i, s = int(u) - 1, u - int(u)
        for j, b in enumerate([(1 - s) ** 3 / 6, (3 * s**3 - 6 * s**2 + 4) / 6, (-3 * s**3 + 3 * s**2 + 3 * s + 1) / 6, s**3 / 6]):
            basis[n, i + j] = b
    basis_t = torch.tensor(basis, dtype=torch.float32)
    knots = torch.zeros(knot_count, 6, requires_grad=True)


def rotvec(v):
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = v / theta
    K = torch.zeros(*v.shape[:-1], 3, 3)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3) + s * K + (1 - c) * K @ K


# Frames whose object is turned half a turn about its long axis (y), through flip_center. Its outline barely changes
# (a paddle, a racket, a board), so only the colors can tell; the color phase decides these.
flipped = torch.zeros(len(used), dtype=torch.bool)
FLIP = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
flip_center = torch.zeros(3)


def pose(n):
    """Object -> camera rotation and translation for frames n, with their corrections (and half turns). On the curve,
    the grip comes from it and the correction is the frame's small residual."""
    dR = rotvec(correction[n, :3])
    if curve:
        s = basis_t[n] @ knots
        R_g = R_shared @ rotvec(s[..., :3])
        t_g = t_shared + s[..., 3:]
    else:
        R_g, t_g = torch.tensor(R_grip[n], dtype=torch.float32), t_grip[n]
    R = R_w[n] @ R_g @ dR
    t = (R_w[n] @ (R_g @ correction[n, 3:, None] + t_g[..., None]))[..., 0] + o_w[n]
    f = flipped[n]
    if f.any():  # p -> R (FLIP (p - c) + c) + t
        R_f = R @ FLIP
        t_f = t + (R @ (flip_center - FLIP @ flip_center))
        R, t = torch.where(f[..., None, None], R_f, R), torch.where(f[..., None], t_f, t)
    return R, t


def occupancy():
    return 1 - torch.exp(-F.softplus(logits))  # per cell


def render(n, px, colors=None):
    """Silhouette alpha for frames n (b,) at crop pixels px (b, r, 2). With a color grid, also the color each ray sees
    (cells blend by how much of the ray reaches and stops in them), as (alpha, rgb)."""
    R, t = pose(n)
    image = px / TSCALE + torch.tensor(tracked["origin"][np.asarray(used)[np.asarray(n).reshape(-1)]], dtype=torch.float32)[:, None]  # (a 1-element tensor would index as a scalar)
    d = torch.stack([(image[..., 0] - cam["cx"]) / cam["fx"], (image[..., 1] - cam["cy"]) / cam["fy"], torch.ones(image.shape[:2])], -1)
    o = -(R.transpose(1, 2) @ t[..., None])[..., 0]  # camera center, object frame
    d = d @ R  # ray directions, object frame (row vectors times R = R^T d)
    with torch.no_grad():
        inv = 1 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        a, b = (lo_t - o[:, None]) * inv, (hi_t - o[:, None]) * inv
        near, far = torch.minimum(a, b).amax(-1), torch.maximum(a, b).amin(-1)
        hit = far > near
        near, far = torch.where(hit, near, 0), torch.where(hit, far, 0)
    steps = (torch.arange(SAMPLES) + torch.rand(*near.shape, SAMPLES)) / SAMPLES
    ts = near[..., None] + (far - near)[..., None] * steps
    p = o[:, None, None] + ts[..., None] * d[:, :, None]
    g = 2 * (p - lo_t) / (hi_t - lo_t) - 1
    sigma = F.grid_sample(F.softplus(logits), g.reshape(1, -1, 1, 1, 3), align_corners=False, padding_mode="zeros").reshape(ts.shape)
    sigma = sigma / CELL  # density per meter: a full-occupancy cell's worth along one cell length
    delta = (far - near)[..., None] / SAMPLES * d.norm(dim=-1, keepdim=True)
    alpha = (1 - torch.exp(-(sigma * delta).sum(-1))) * hit
    if colors is None:
        return alpha
    stop = 1 - torch.exp(-sigma * delta)  # per sample
    reach = torch.cumprod(torch.cat([torch.ones_like(stop[..., :1]), 1 - stop[..., :-1] + 1e-7], -1), -1)
    w = reach * stop
    c = torch.sigmoid(F.grid_sample(colors, g.reshape(1, -1, 1, 1, 3), align_corners=False, padding_mode="border").reshape(3, *ts.shape)).permute(1, 2, 3, 0)
    return alpha, (w[..., None] * c).sum(-2) / w.sum(-1, keepdim=True).clamp_min(1e-6)


def batch_pixels(n, rng, count):
    px, target, weight = [], [], []
    for i in n:
        lo, hi = boxes[i]
        p = rng.uniform(lo, hi, (count, 2))
        q = np.clip(p.astype(int), 0, 639)
        m = mask_of(used[i])[q[:, 1], q[:, 0]].astype(np.float32)
        c = person[i][q[:, 1] // 2, q[:, 0] // 2]
        w = np.where(c == 2, 0, np.where((c == 1) & (m == 0), 0, 1)).astype(np.float32) * float(frame_weight[i])
        px.append(p), target.append(m), weight.append(w)
    return [torch.tensor(np.array(a), dtype=torch.float32) for a in (px, target, weight)]


def smooth_prior():
    """Robust penalty on every consecutive triple's angular and linear acceleration in camera space, over all frames."""
    R, t = pose(torch.arange(len(used)))
    v = R[1:] @ R[:-1].transpose(1, 2)  # rotation from each frame to the next
    omega = torch.stack([v[:, 2, 1] - v[:, 1, 2], v[:, 0, 2] - v[:, 2, 0], v[:, 1, 0] - v[:, 0, 1]], -1) / 2  # ~ angle × axis
    acc_r = (omega[1:] - omega[:-1]).norm(dim=-1) / np.radians(ACC_DEG)
    acc_t = (t[2:] - 2 * t[1:-1] + t[:-2]).norm(dim=-1) / ACC_M
    charbonnier = lambda x: torch.sqrt(1 + x**2) - 1
    return ((charbonnier(acc_r) + charbonnier(acc_t)) * triples).mean()


def pose_prior(n_all):
    c = correction[n_all]
    if POSES == "smooth":  # the smoothness is smooth_prior's; this only keeps corrections from roaming
        return ((c[:, :3].norm(dim=-1) / np.radians(LOOSE_DEG)) ** 2 + (c[:, 3:].norm(dim=-1) / LOOSE_M) ** 2).mean()
    if curve:  # small residuals, and a curve that bends gently
        residual = ((c[:, :3].norm(dim=-1) / np.radians(RES_DEG)) ** 2 + (c[:, 3:].norm(dim=-1) / RES_M) ** 2).mean()
        bend = knots[2:] - 2 * knots[1:-1] + knots[:-2]
        return residual + ((bend[:, :3].norm(dim=-1) / np.radians(CURVE_DEG)) ** 2 + (bend[:, 3:].norm(dim=-1) / CURVE_M) ** 2).mean()
    prior = ((c[:, :3].norm(dim=-1) / np.radians(POSE_DEG)) ** 2 + (c[:, 3:].norm(dim=-1) / POSE_M) ** 2).mean()
    consecutive = torch.tensor(np.diff(np.asarray(used)[n_all]) == 1)
    step = c[1:] - c[:-1]
    smooth = (((step[:, :3].norm(dim=-1) / np.radians(SMOOTH_DEG)) ** 2 + (step[:, 3:].norm(dim=-1) / SMOOTH_M) ** 2) * consecutive).mean()
    return prior + smooth


def tv():
    o = occupancy()
    return sum((o.diff(dim=k) ** 2).mean() for k in (2, 3, 4))


def frame_ious(n_set, grid=48):
    """Each frame's silhouette IoU (as iou_of, coarser)."""
    return np.array([iou_of([i], grid) for i in n_set])


def weight_of(ious):
    return torch.tensor(1 / (1 + np.exp(-(ious - GOOD_IOU) / 0.08)), dtype=torch.float32)


def start_curve():
    """Fit the curve to the per-frame grips so far, each frame weighted by how well it fits its mask (bad frames borrow
    from their neighbors), with a little bending penalty; residuals start at zero."""
    global curve
    ious = frame_ious(np.arange(len(used)))
    w = weight_of(ious).numpy()
    with torch.no_grad():
        c = correction.numpy().astype(np.float64)
    targets = []
    for n in range(len(used)):
        R_g = R_grip[n] @ rotvec_np(c[n, :3])
        t_g = R_grip[n] @ c[n, 3:] + grips[n, 3:]
        targets.append(np.concatenate([cv2.Rodrigues(R_shared.numpy().T.astype(np.float64) @ R_g)[0].ravel(), t_g - shared[3:]]))
    targets = np.array(targets)
    bend = np.diff(np.eye(knot_count), 2, axis=0)
    A = basis.T @ (w[:, None] * basis) + 0.5 * bend.T @ bend + 1e-6 * np.eye(knot_count)
    solution = np.linalg.solve(A, basis.T @ (w[:, None] * targets))
    with torch.no_grad():
        knots.copy_(torch.tensor(solution, dtype=torch.float32))
        correction.zero_()
        frame_weight.copy_(weight_of(ious))
    curve = True
    return ious


def start_smooth():
    """Weights from each frame's fit; then frames that fit badly, and all held-out frames (whose own masks mustn't
    leak in), take a grip interpolated (slerp) between their nearest good training neighbors in time."""
    from scipy.spatial.transform import Rotation, Slerp
    ious = frame_ious(np.arange(len(used)))
    frame_weight.copy_(weight_of(ious))
    with torch.no_grad():
        c = correction.numpy().astype(np.float64)
    grip_R = np.array([R_grip[n] @ rotvec_np(c[n, :3]) for n in range(len(used))])
    grip_t = np.array([R_grip[n] @ c[n, 3:] + grips[n, 3:] for n in range(len(used))])
    frame_numbers = np.asarray(used)
    good = np.zeros(len(used), bool)
    good[train] = ious[train] >= 0.3
    anchors = np.flatnonzero(good)
    for n in np.flatnonzero(~good):
        before, after = anchors[anchors < n], anchors[anchors > n]
        if not len(before) and not len(after):
            continue
        a = before[-1] if len(before) else after[0]
        b = after[0] if len(after) else before[-1]
        s = 0.0 if a == b else (frame_numbers[n] - frame_numbers[a]) / (frame_numbers[b] - frame_numbers[a])
        R_new = Slerp([0, 1], Rotation.from_matrix([grip_R[a], grip_R[b]]))([s]).as_matrix()[0]
        t_new = (1 - s) * grip_t[a] + s * grip_t[b]
        c[n, :3] = cv2.Rodrigues(R_grip[n].T @ R_new)[0].ravel()
        c[n, 3:] = R_grip[n].T @ (t_new - grips[n, 3:])
    with torch.no_grad():
        correction.copy_(torch.tensor(c, dtype=torch.float32))
    return ious


def iou_of(n_set, grid=96):
    """Silhouette IoU on a pixel grid over each frame's box, outside the hand; alpha > 0.5 against the mask."""
    out = []
    with torch.no_grad():
        for i in n_set:
            lo, hi = boxes[i]
            u, v = np.meshgrid(np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid))
            p = np.stack([u, v], -1).reshape(1, -1, 2)
            a = render(torch.tensor([i]), torch.tensor(p, dtype=torch.float32))[0].numpy() > 0.5
            q = np.clip(p[0].astype(int), 0, 639)
            m = mask_of(used[i])[q[:, 1], q[:, 0]].astype(bool)
            keep = person[i][q[:, 1] // 2, q[:, 0] // 2] != 2
            a, m = a[keep], m[keep]
            out.append((a & m).sum() / max((a | m).sum(), 1))
    return float(np.mean(out))


rng = np.random.default_rng(0)
report = dict(track=track, frames=len(used), train=len(train), held_out=len(test), cell_m=CELL, steps=STEPS, w_density=W_DENSITY, w_tv=W_TV, w_binary=W_BINARY, prior=PRIOR, w_prior=W_PRIOR if PRIOR else 0)
report["iou_train_start"] = iou_of(train[::10])
pose_params = [correction]
if POSES == "curve":
    ious = start_curve()
    report["poses"] = dict(kind="curve", knot_every=KNOT_EVERY, knots=knot_count, bad_frames_before=round(float(np.mean(ious < 0.3)), 3))
    report["iou_train_curve_start"] = iou_of(train[::10])
    pose_params = [correction, knots]
if POSES == "smooth":
    ious = start_smooth()
    report["poses"] = dict(kind="smooth", w_smooth=W_SMOOTH, bad_frames_before=round(float(np.mean(ious < 0.3)), 3))
    report["iou_held_out_neighbors_start"] = iou_of(test)
print("start", report, flush=True)
opt = torch.optim.Adam([dict(params=[logits], lr=0.05), dict(params=pose_params, lr=2e-3)])
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(STEPS, 1), eta_min=0)
start = time.time()
for step in range(STEPS):
    if (curve or POSES == "smooth") and step and step % REWEIGHT_EVERY == 0:  # frames whose masks disagree count less
        frame_weight[torch.tensor(train)] = weight_of(frame_ious(train))
        print(f"reweighted at step {step}: {int((frame_weight[torch.tensor(train)] < 0.5).sum())} frames below half weight", flush=True)
    n = rng.choice(train, BATCH, replace=False)
    px, target, weight = batch_pixels(n, rng, RAYS)
    alpha = render(torch.tensor(n), px).clamp(1e-5, 1 - 1e-5)
    fit_loss = (F.binary_cross_entropy(alpha, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1)
    o = occupancy()
    ramp = min(1.0, 2 * step / STEPS)  # binarize gradually, once the shape has formed
    loss = fit_loss + W_DENSITY * o.mean() + W_TV * tv() + ramp * W_BINARY * (o * (1 - o)).mean() + W_POSE * pose_prior(torch.tensor(train))
    if POSES == "smooth":
        loss = loss + W_SMOOTH * smooth_prior()
    if prior is not None:
        loss = loss + W_PRIOR * (prior_weight * (o - prior) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    schedule.step()
    if step % 100 == 0 or step == STEPS - 1:
        print(f"step {step}: fit {fit_loss.item():.4f} occupancy {occupancy().mean().item():.4f} {time.time() - start:.0f} s", flush=True)

def frame_speeds(middle):
    """How far the object's middle moves in the image between neighboring frames, in object lengths (inf: no neighbor)."""
    with torch.no_grad():
        Rs, ts = (x.numpy() for x in pose(torch.arange(len(used))))
    centers = np.einsum("nij,j->ni", Rs, middle) + ts
    image_centers, lengths = project(centers), cam["fx"] * 0.4 / centers[:, 2] * TSCALE
    speed = np.full(len(used), np.inf)
    for i in range(len(used)):
        near = [m for m in (i - 1, i + 1) if 0 <= m < len(used) and abs(used[m] - used[i]) == 1]
        if near:
            speed[i] = max(np.linalg.norm(image_centers[m] - image_centers[i]) * TSCALE for m in near) / lengths[i]
    return speed


def silhouette_loss(n):
    px, target, weight = batch_pixels(n, rng, RAYS)
    alpha = render(torch.tensor(n), px).clamp(1e-5, 1 - 1e-5)
    return (F.binary_cross_entropy(alpha, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1)


colors = None
if COLOR_STEPS:
    # Each frame's pixels around the object, kept (uint8) so batches don't reread the video
    patches = []
    for i, x in enumerate(used):
        lo, hi = np.floor(boxes[i][0]).astype(int), np.ceil(boxes[i][1]).astype(int) + 1
        M = np.array([[TSCALE, 0, -tracked["origin"][x][0] * TSCALE], [0, TSCALE, -tracked["origin"][x][1] * TSCALE]])
        image = cv2.warpAffine(cv2.imread(str(frames_dir[x])), M, (640, 640), flags=cv2.INTER_CUBIC)
        patches.append((lo, cv2.cvtColor(image[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2RGB)))
    solid_now = (occupancy().detach()[0, 0] > 0.5).nonzero().float().mean(0)  # z y x cells
    middle = BOUNDS[:, 0] + CELL * (0.5 + solid_now.numpy()[::-1])
    speed = frame_speeds(middle)
    sharpness = 1 / (1 + (speed / MAX_SPEED) ** 2)  # blurred frames count less
    report["color_sharp_frames"] = int((speed < MAX_SPEED).sum())

    def color_batch(n, count):
        """Pixels inside the object's mask (not the hand), with the video's color there and the frame's weight."""
        px, target, weight = [], [], []
        for i in n:
            (x0, y0), patch = patches[i]
            m = mask_of(used[i])
            ys, xs = np.nonzero(m[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]])
            p = np.zeros((count, 2))
            t = np.zeros((count, 3))
            w = np.zeros(count)
            if len(xs):
                k = rng.integers(len(xs), size=count)
                p = np.stack([xs[k] + x0, ys[k] + y0], 1) + rng.uniform(-0.5, 0.5, (count, 2))
                local = (p - [x0, y0]).astype(np.float32)
                t = cv2.remap(patch, local[None, :, 0], local[None, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0] / 255.0
                c = person[i][np.clip(p[:, 1].astype(int), 0, 639) // 2, np.clip(p[:, 0].astype(int), 0, 639) // 2]
                w = (c != 2) * sharpness[i] * float(frame_weight[i])
            px.append(p), target.append(t), weight.append(w)
        return [torch.tensor(np.array(a), dtype=torch.float32) for a in (px, target, weight)]

    def color_loss(n, gain):
        px, target, weight = color_batch(n, RAYS)
        _, rgb = render(torch.tensor(n), px, colors)
        predicted = rgb * torch.exp(gain[n])[:, None]
        error = F.smooth_l1_loss(predicted, target, beta=0.05, reduction="none").sum(-1)
        return (error * weight).sum() / weight.sum().clamp_min(1)

    def decide_flips(n_set, window=5):
        """Each frame keeps or takes the half turn, whichever explains its pixels (colors and outline) better, on the
        same pixels both ways; then a majority over neighboring frames, since an object can't flip within a frame or
        two. Returns how many frames changed."""
        costs = np.zeros((len(n_set), 2))
        before = flipped.clone()
        with torch.no_grad():
            for k, i in enumerate(n_set):
                state = rng.bit_generator.state
                for option in (0, 1):
                    flipped[i] = bool(option)
                    rng.bit_generator.state = state
                    costs[k, option] = color_loss(np.array([i]), gain).item() + silhouette_loss(np.array([i])).item()
                flipped[i] = before[i]
        choice = costs.argmin(1).astype(bool)
        frame_numbers = np.asarray(used)[n_set]
        smoothed = np.array([choice[np.abs(frame_numbers - f) <= window // 2].mean() > 0.5 for f in frame_numbers])
        flipped[torch.tensor(n_set)] = torch.tensor(smoothed)
        return int((before[torch.tensor(n_set)] != flipped[torch.tensor(n_set)]).sum()), int(flipped[torch.tensor(n_set)].sum())

    # The shape stays as fit; colors, each frame's brightness, and the poses (now also aligned by the pattern) move.
    # In rounds: between them, each frame's half turn is decided by the colors fit so far.
    flip_center.copy_(torch.tensor(middle, dtype=torch.float32))
    colors = torch.zeros(1, 3, *logits.shape[2:], requires_grad=True)
    gain = torch.zeros(len(used), 3, requires_grad=True)
    logits.requires_grad_(False)
    opt = torch.optim.Adam([dict(params=[colors], lr=0.05), dict(params=[gain], lr=0.01), dict(params=pose_params, lr=5e-4)])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, COLOR_STEPS, eta_min=0)
    start = time.time()
    report["flips"] = []
    for step in range(COLOR_STEPS):
        if not curve and step and step % (COLOR_STEPS // COLOR_ROUNDS) == 0 and step < COLOR_ROUNDS * (COLOR_STEPS // COLOR_ROUNDS):
            changed, total = decide_flips(train)
            report["flips"].append(dict(step=step, changed=changed, flipped=total))
            print(f"flips at step {step}: {changed} frames changed, {total} of {len(train)} turned", flush=True)
        n = rng.choice(train, BATCH, replace=False)
        c_loss = color_loss(n, gain)
        loss = c_loss + silhouette_loss(n) + W_POSE * pose_prior(torch.tensor(train)) + W_GAIN * (gain[train] ** 2).mean()
        if POSES == "smooth":
            loss = loss + W_SMOOTH * smooth_prior()
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
        if step % 100 == 0 or step == COLOR_STEPS - 1:
            print(f"color step {step}: color {c_loss.item():.4f} {time.time() - start:.0f} s", flush=True)
    colors.requires_grad_(False)

# Held-out frames: fit only their poses (and brightness), shape and colors frozen
report["iou_held_out_start_pose"] = iou_of(test)
logits.requires_grad_(False)
if colors is not None and not curve:
    report["held_out_flipped"] = decide_flips(test, window=1)[1]  # held-out frames aren't neighbors of each other
if POSES == "smooth":  # only held-out frames move now, still in their smooth sequence
    test_rows = torch.zeros(len(used), 1)
    test_rows[torch.tensor(test)] = 1
    correction.register_hook(lambda g: g * test_rows)
opt = torch.optim.Adam([correction] + ([gain] if colors is not None else []), lr=2e-3)
for step in range(300):
    n = test[rng.choice(len(test), min(BATCH, len(test)), replace=False)]
    loss = silhouette_loss(n) + W_POSE * pose_prior(torch.tensor(test))
    if POSES == "smooth":
        loss = loss + W_SMOOTH * smooth_prior()
    if colors is not None:
        loss = loss + color_loss(n, gain) + W_GAIN * (gain[test] ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
if colors is not None:
    # Color error (0-255, mean absolute) on held-out frames' object pixels: the fitted colors, against the best single flat
    # color per frame (what a texture must beat)
    with torch.no_grad():
        fitted, flat = [], []
        for i in test:
            px, target, weight = color_batch([i], 2048)
            if weight.sum() == 0:
                continue
            _, rgb = render(torch.tensor([i]), px, colors)
            keep = weight[0] > 0
            t = target[0][keep]
            fitted.append((rgb[0][keep] * torch.exp(gain[i]) - t).abs().mean().item() * 255)
            flat.append((t - t.median(0).values).abs().mean().item() * 255)
    report["color_error_held_out"] = round(float(np.mean(fitted)), 1)
    report["color_error_flat"] = round(float(np.mean(flat)), 1)
report["iou_train"] = iou_of(train[::5])
report["iou_held_out"] = iou_of(test)
moves = correction.detach().numpy()
report["correction_deg_median"] = round(float(np.degrees(np.median(np.linalg.norm(moves[:, :3], axis=1)))), 1)
report["correction_cm_median"] = round(float(np.median(np.linalg.norm(moves[:, 3:], axis=1)) * 100), 1)

# Shape: solid where occupancy > 0.5; thickness = median solid run across z, over columns that have any
occ = occupancy().detach()[0, 0].permute(2, 1, 0).numpy()  # x y z
solid = occ > 0.5
cols = solid.sum(2)
report["solid_cm3"] = round(float(solid.sum() * CELL**3 * 1e6), 1)
report["extent_cm_xyz"] = [round(float((np.ptp(a) + 1) * CELL * 100), 1) for a in np.nonzero(solid)] if solid.any() else [0, 0, 0]
report["thickness_cm_median"] = round(float(np.median(cols[cols > 0]) * CELL * 100), 1) if solid.any() else 0
report["cells_unseen"] = int((seen < SEEN_SCALE).sum())
final_ious = frame_ious(np.arange(len(used)))
report["bad_frames"] = round(float(np.mean(final_ious < 0.3)), 3)  # silhouette IoU < 0.3, all frames
report["bad_frames_held_out"] = round(float(np.mean(final_ious[test] < 0.3)), 3)
print(report, flush=True)
(OUT / f"report_{track}.json").write_text(json.dumps(report, indent=1))
extra = dict(knots=knots.detach().numpy(), knot_every=KNOT_EVERY, first=first) if curve else {}
np.savez(OUT / f"{track}.npz", occupancy=occ, bounds=BOUNDS, cell=CELL, frames=np.array(used), correction=moves, held_out=held, seen=seen,
         frame_iou=final_ious, frame_weight=frame_weight.numpy(), **extra)

# Slices (face plane z = 0, and a cut across the thickness at x = 0) and a colored mesh
a = (occ[:, :, occ.shape[2] // 2].T[::-1] * 255).astype(np.uint8)
b = (occ[occ.shape[0] // 2][::-1] * 255).astype(np.uint8)
cv2.imwrite(str(OUT / f"slices_{track}.png"), cv2.resize(np.hstack([a, np.zeros((a.shape[0], 4), np.uint8), b]), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
u = (np.exp(-seen / SEEN_SCALE) * 255).astype(np.uint8)  # bright = unseen
cv2.imwrite(str(OUT / f"unseen_{track}.png"), cv2.resize(np.hstack([u[:, :, shape[2] // 2].T[::-1], np.zeros((shape[1], 4), np.uint8), u[shape[0] // 2][::-1]]), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
if solid.any():
    from scipy.ndimage import gaussian_filter
    v, f, normals, _ = marching_cubes(np.pad(gaussian_filter(occ, 0.6), 1), 0.5)  # a little smoothing: no stair steps
    v = (v - 1) * CELL + BOUNDS[:, 0] + CELL / 2
    mesh = trimesh.Trimesh(v, f[:, ::-1])
    if mesh.volume < 0:  # normals must point out, for "facing the camera"
        mesh.invert()
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    v, normals = mesh.vertices, mesh.vertex_normals
    if colors is not None:  # the mesh is colored by recolor.py, from the saved color grid
        np.savez(OUT / f"{track}_colors.npz", colors=colors.numpy(), gain=gain.detach().numpy(), flipped=flipped.numpy(), flip_center=flip_center.numpy())
        # Held-out frames: the video next to the fitted colors rendered with the frame's fitted pose
        tiles = []
        for i in test[:: max(1, len(test) // 6)][:6]:
            (x0, y0), patch = patches[i]
            h, w = patch.shape[:2]
            u, vv = np.meshgrid(np.arange(w) + x0 + 0.5, np.arange(h) + y0 + 0.5)
            px = torch.tensor(np.stack([u, vv], -1).reshape(1, -1, 2), dtype=torch.float32)
            with torch.no_grad():
                alpha, rgb = render(torch.tensor([i]), px, colors)
                rgb = (rgb[0] * torch.exp(gain[i])).clamp(0, 1).numpy().reshape(h, w, 3)
            alpha = alpha[0].numpy().reshape(h, w, 1)
            drawn = (alpha * rgb * 255 + (1 - alpha) * np.array([255, 0, 255])).astype(np.uint8)  # magenta where it isn't
            pair = np.hstack([patch, drawn])
            scale = 200 / max(h, 1)
            tiles.append(cv2.resize(cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), (int(pair.shape[1] * scale), 200), interpolation=cv2.INTER_NEAREST))
        width = max(t.shape[1] for t in tiles)
        cv2.imwrite(str(OUT / f"held_out_{track}.png"), np.vstack([np.pad(t, ((0, 4), (0, width - t.shape[1]), (0, 0))) for t in tiles]))
        (OUT / f"report_{track}.json").write_text(json.dumps(report, indent=1))
        print("done", flush=True)
        sys.exit()
    # Colors from sharp frames: the object barely moves in the image (else it's smeared along its motion)
    with torch.no_grad():
        Rs, ts = (x.numpy() for x in pose(torch.arange(len(used))))
    middle = v.mean(0)
    centers = np.einsum("nij,j->ni", Rs, middle) + ts
    image_centers, lengths = project(centers), cam["fx"] * 0.4 / centers[:, 2] * TSCALE
    speed = np.full(len(used), np.inf)
    for i in range(len(used)):
        near = [m for m in (i - 1, i + 1) if 0 <= m < len(used) and abs(used[m] - used[i]) == 1]
        if near:
            speed[i] = max(np.linalg.norm(image_centers[m] - image_centers[i]) * TSCALE for m in near) / lengths[i]
    sharp = [i for i in np.argsort(speed) if speed[i] < MAX_SPEED][:COLOR_FRAMES] or list(np.argsort(speed)[:COLOR_FRAMES])
    report["color_frames"] = len(sharp)
    colors = []
    for i in sharp:
        R, t = Rs[i], ts[i]
        pts = v @ R.T + t
        px = ((project(pts) - tracked["origin"][used[i]]) * TSCALE).astype(np.float32)
        M = np.array([[TSCALE, 0, -tracked["origin"][used[i]][0] * TSCALE], [0, TSCALE, -tracked["origin"][used[i]][1] * TSCALE]])
        img = cv2.warpAffine(cv2.imread(str(frames_dir[used[i]])), M, (640, 640), flags=cv2.INTER_CUBIC)
        q = np.clip(np.round(px).astype(int), 0, 639)
        facing = np.einsum("ij,ij->i", normals @ R.T, -pts / np.linalg.norm(pts, axis=1, keepdims=True))
        ok = (facing > 0.5) & (mask_of(used[i])[q[:, 1], q[:, 0]] > 0) & (person[i][q[:, 1] // 2, q[:, 0] // 2] != 2)
        sampled = cv2.remap(img, px[None, :, 0], px[None, :, 1], cv2.INTER_LINEAR)[0].astype(np.float32)  # between pixels
        colors.append(np.where(ok[:, None], sampled, np.nan))
    c = np.nanmedian(np.array(colors), 0)
    report["colored_vertices"] = round(float(np.mean(~np.isnan(c[:, 0]))), 3)
    c = np.where(np.isnan(c), 128, c)[:, ::-1]
    mesh.visual.vertex_colors = np.hstack([c, np.full((len(c), 1), 255)]).astype(np.uint8)
    mesh.export(OUT / f"joint_{track}.glb")
    (OUT / f"report_{track}.json").write_text(json.dumps(report, indent=1))
print("done", flush=True)
