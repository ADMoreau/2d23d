"""The ball's path in 3D from one camera: link its detections, size it from the floor and hands, place it by size,
anchor it at hands, and keep its motion consistent over time with gravity.

1. Link: the best path through each frame's candidates (detections.jsonl), as a longest path in a graph where an edge
   from one frame's candidate to a later one costs its missed frames and any speed beyond MAX_SPEED.
2. Size: if a ball sat on the floor, its depth is where its ray meets the plane one radius up; that implies a diameter
   from its pixel size. A ball in the air implies a bigger one (its ray meets the floor farther away), so the true
   diameter is a low percentile of them. Nothing about the sport is assumed.
3. Depth: from pixel size (the box's short side: motion blur stretches the long one). Where the ball is at a player's
   wrist in the image and at a similar depth, the wrist's depth is used instead (the scene hierarchy: people anchor it).
4. Physics, only as temporal consistency: one position per frame, fit at once. Detections pin the image position
   tightly and depth loosely (tightly at a hand); the path should accelerate like gravity, robustly, so contacts break
   through; it stays above the floor. Flights come out as parabolas; missed frames are filled. Frames where the
   acceleration is far from gravity are contacts: bounces at floor height, otherwise hits or catches.

usage: ball3d.py <work dir> <detections dir> <out dir>
Writes ball.json (per frame: world position and kind), debug.mp4 and topdown.png.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORK, DET, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
MIN_SCORE = 0.15  # candidates
MAX_SPEED = 70.0  # px per frame before a jump costs
MISS_COST = 0.4  # per missed frame in a link
MAX_GAP = 15  # frames a link may skip
MIN_TRACKLET_VALUE = 4.0  # a stretch must be worth about three confident detections
STATIC_PX, STATIC_SHARE = 4.0, 0.05  # a candidate this close to the same spot in this share of frames is background
ACC_SCALE = 5.0  # m/s², acceleration beyond gravity where the smoothing penalty turns linear (contacts break through)
CONTACT_ACC = 30.0  # m/s² beyond gravity: a contact (bounce, hit, catch)
PADDLES = Path(sys.argv[4]) if len(sys.argv) > 4 else None  # the paddle experiments' run dir: contacts near a paddle are hits
PADDLE_FIT = "joint_smooth_color"  # whose per-frame poses
PADDLE_REACH = 0.3  # m from a paddle's middle (in the image, at the paddle's depth): a contact there is a hit
PADDLE_DEPTH = 0.75  # m: how far the ball's depth (from its size) may be from the paddle's
SIZE_PERCENTILE = 20  # of the floor-implied diameters
PIXEL_SIGMA = 2.0  # px, the detections' position noise
DEPTH_SIGMA = 0.15  # relative, depth from size
WRIST_RADIUS = 0.15  # m: a ball this close to a wrist (in 3D, from size) is at the hand
HAND_SIGMA = 0.1  # m: depth at a hand (the ball sits in the fingers, a little off the wrist)
W_MOTION = 1.0
G = np.array([0.0, -9.81, 0.0])

cam = json.loads((WORK / "bodies" / "camera.json").read_text())
fps = json.loads((WORK / "ingest" / "meta.json").read_text())["fps"]
M = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
R_cw, t_cw = M[:3, :3], M[:3, 3]
frames_dir = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
n_frames = len(frames_dir)
det = {}
for line in (DET / "detections.jsonl").open():
    d = json.loads(line)
    det[d["frame"]] = [b for b in d["balls"] if b["score"] >= MIN_SCORE]


def center(b):
    return np.array([(b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2])


# 1. Link. First drop static spots: a "ball" at the same place in a large share of the frames is part of the background
# or an overlay (a logo), not something people interact with.
all_centers = np.array([center(b) for f in det for b in det[f]])
static = []
for c in all_centers:
    if sum(np.linalg.norm(np.array([center(b) for b in det[f]]) - c, axis=1).min() < STATIC_PX for f in det if det[f]) > STATIC_SHARE * len(det):
        if all(np.linalg.norm(c - s) > STATIC_PX for s in static):
            static.append(c)
for f in det:
    det[f] = [b for b in det[f] if all(np.linalg.norm(center(b) - s) > STATIC_PX for s in static)]
print(f"static spots dropped: {[tuple(np.round(s).astype(int)) for s in static]}", flush=True)
nodes = [(f, b) for f in sorted(det) for b in det[f]]
def best_path(allowed):
    """The best-scoring linked path through the allowed nodes: (value, node indices)."""
    best = np.full(len(nodes), -np.inf)
    back = np.full(len(nodes), -1)
    for k, (f, b) in enumerate(nodes):
        if not allowed[k]:
            continue
        gain = np.log(b["score"] + 1e-6) + 1.5
        best[k] = gain  # a path may start anywhere
        for j in range(k - 1, -1, -1):
            g, a = nodes[j]
            if g == f or not allowed[j]:
                continue
            if f - g > MAX_GAP:
                break
            speed = np.linalg.norm(center(b) - center(a)) / (f - g)
            value = best[j] + gain - MISS_COST * (f - g - 1) - 0.1 * max(0.0, speed - MAX_SPEED)
            if value > best[k]:
                best[k], back[k] = value, j
    k = int(np.argmax(best))
    path = []
    while k >= 0:
        path.append(k)
        k = back[k]
    return best[path[0]], path[::-1]


# One ball, seen in stretches: take the best path, then the best one in the frames it doesn't cover, and so on
allowed = np.ones(len(nodes), bool)
track, tracklets = {}, []
while allowed.any():
    value, path = best_path(allowed)
    if value < MIN_TRACKLET_VALUE:
        break
    f0, f1 = nodes[path[0]][0], nodes[path[-1]][0]
    tracklets.append((f0, f1, len(path)))
    for k in path:
        track[nodes[k][0]] = nodes[k][1]
    allowed &= ~np.array([f0 <= f <= f1 for f, _ in nodes])
frames = np.array(sorted(track))
print(f"linked: {len(frames)} detections in {len(tracklets)} tracklets over frames {frames[0]}..{frames[-1]}: "
      f"{sorted(tracklets)}", flush=True)

# Rays (camera and world) and pixel sizes
uv = np.array([center(track[f]) for f in frames])
size = np.array([min(track[f]["box"][2] - track[f]["box"][0], track[f]["box"][3] - track[f]["box"][1]) for f in frames])
rays = np.column_stack([(uv[:, 0] - cam["cx"]) / cam["fx"], (uv[:, 1] - cam["cy"]) / cam["fy"], np.ones(len(uv))])  # z = 1
f_px = (cam["fx"] + cam["fy"]) / 2


def depth_on_plane(height):
    """Camera depth (z) where each ray meets the world plane y = height."""
    along = rays @ R_cw[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (height - t_cw[1]) / along
    return np.where(z > 0, z, np.inf)


# Each frame's wrists, in camera coordinates
with np.load(WORK / "check" / "bodies.npz") as b:
    body_frame, wrists_w = b["frame"], b["keypoints"][:, [41, 62]]  # world
wrists = {int(f): (wrists_w[body_frame == f].reshape(-1, 3) - t_cw) @ R_cw for f in frames}


def nearest_wrist(n):
    """The wrist nearest detection n's ray (camera coords), and its distance from the ray, m."""
    w = wrists.get(int(frames[n]))
    if w is None or not len(w):
        return None, np.inf
    unit = rays[n] / np.linalg.norm(rays[n])
    off = np.linalg.norm(np.cross(w, unit), axis=1)
    k = np.argmin(off)
    return w[k], off[k]


# 2. Size. Floor: the fixed point of "the diameter the on-floor frames imply". Hands: where the ray passes a wrist, its
# depth gives the diameter directly. The hands win when enough agree (they're the hierarchy's anchor).
d_floor = 0.1
for _ in range(10):
    implied = size * depth_on_plane(d_floor / 2) / f_px
    d_floor = float(np.percentile(implied[np.isfinite(implied)], SIZE_PERCENTILE))
at_hand = [size[n] * w[2] / f_px for n in range(len(frames)) for w, off in [nearest_wrist(n)] if off < 0.10]
d_hand = float(np.median(at_hand)) if at_hand else np.nan
spread = float(np.subtract(*np.percentile(at_hand, [75, 25])) / d_hand) if len(at_hand) >= 2 else np.inf
diameter = d_hand if len(at_hand) >= 8 and spread < 0.3 else d_floor
print(f"diameter {diameter * 100:.1f} cm (floor: {d_floor * 100:.1f} cm; hands: {d_hand * 100:.1f} cm from {len(at_hand)} frames, "
      f"spread {spread:.2f})", flush=True)

# 3. Depth from size, and wrists
depth = f_px * diameter / size
kind = np.array(["seen"] * len(frames), dtype=object)
for n, f in enumerate(frames):
    w_cam = wrists.get(int(f))
    if w_cam is None or not len(w_cam):
        continue
    p = rays[n] * depth[n]
    near = np.linalg.norm(w_cam - p, axis=1)
    k = np.argmin(near)
    if near[k] < WRIST_RADIUS + DEPTH_SIGMA * depth[n]:  # within reach, given how loose depth from size is
        ray_dist = np.linalg.norm(np.cross(w_cam[k], rays[n] / np.linalg.norm(rays[n])))
        if ray_dist < WRIST_RADIUS:
            depth[n], kind[n] = w_cam[k, 2], "hand"
points_cam = rays * depth[:, None]
points = points_cam @ R_cw.T + t_cw
dirs = (rays / np.linalg.norm(rays, axis=1, keepdims=True)) @ R_cw.T
times = frames / fps


def project_world(p):
    c = (p - t_cw) @ R_cw
    return np.column_stack([cam["fx"] * c[:, 0] / c[:, 2] + cam["cx"], cam["fy"] * c[:, 1] / c[:, 2] + cam["cy"]])


# 4. Physics as temporal consistency: every frame from the first detection to the last gets a position. Detections pin
# the image position tightly and the depth loosely (tightly at a hand); the path should accelerate like gravity, with a
# robust penalty so contacts (bounces, hits, catches) can break through; it can't go below the floor. Flights come out
# as parabolas, and missed frames are filled by the same fit.
import torch  # noqa: E402

span = np.arange(frames[0], frames[-1] + 1)
seen_at = frames - frames[0]
dt = 1 / fps
P = torch.tensor(np.column_stack([np.interp(span, frames, points[:, c]) for c in range(3)]), requires_grad=True)
R_t, t_t, G_t = torch.tensor(R_cw), torch.tensor(t_cw), torch.tensor(G)
uv_t = torch.tensor(uv)
distance = torch.tensor(np.linalg.norm(points - t_cw, axis=1))
depth_sigma = torch.tensor(np.where(kind == "hand", HAND_SIGMA, DEPTH_SIGMA * np.linalg.norm(points - t_cw, axis=1)))
radius = diameter / 2


def charbonnier(x):
    return torch.sqrt(1 + x**2) - 1


def objective():
    c = (P[seen_at] - t_t) @ R_t
    q = torch.stack([cam["fx"] * c[:, 0] / c[:, 2] + cam["cx"], cam["fy"] * c[:, 1] / c[:, 2] + cam["cy"]], -1)
    image = charbonnier((q - uv_t).norm(dim=-1) / PIXEL_SIGMA).sum()
    depth_term = (((P[seen_at] - t_t).norm(dim=-1) - distance) / depth_sigma).pow(2).sum()
    acc = (P[2:] - 2 * P[1:-1] + P[:-2]) / dt**2 - G_t
    motion = charbonnier(acc.norm(dim=-1) / ACC_SCALE).sum()
    floor = (torch.relu(radius - P[:, 1]) / 0.01).pow(2).sum()
    return image + depth_term + W_MOTION * motion + floor


opt = torch.optim.LBFGS([P], max_iter=500, history_size=20, line_search_fn="strong_wolfe")


def closure():
    opt.zero_grad()
    loss = objective()
    loss.backward()
    return loss


for _ in range(4):
    opt.step(closure)
pos = P.detach().numpy()
acc = np.linalg.norm((pos[2:] - 2 * pos[1:-1] + pos[:-2]) / dt**2 - G, axis=1)
kinds = np.array(["unseen"] * len(span), dtype=object)
kinds[seen_at] = np.where(kind == "hand", "hand", "seen")
contacts = np.flatnonzero(acc > CONTACT_ACC) + 1
# A contact is one event even if it spans a few frames: keep each run's strongest frame
events = []
for i in contacts:
    if events and i - events[-1] <= 2:
        if acc[i - 1] > acc[events[-1] - 1]:
            events[-1] = i
    else:
        events.append(i)
def at_floor(i):
    """A bounce happens between frames: the lowest frame near it is within one frame's fall of the floor."""
    near = pos[max(0, i - 3): i + 4, 1]
    fall = abs(pos[i, 1] - pos[max(0, i - 1), 1]) + abs(pos[min(len(pos) - 1, i + 1), 1] - pos[i, 1])
    return near.min() < radius + 0.05 + fall


# The paddles (the scene hierarchy: people's objects anchor the ball): each paddle's middle in world coordinates per frame
paddles = {}
if PADDLES is not None:
    with np.load(PADDLES / "fit3" / "grips.npz") as f:
        grips_all = dict(f)
    for npz in sorted((PADDLES / PADDLE_FIT).glob("[0-9].npz")):
        t = int(npz.stem)
        with np.load(npz) as j:
            occ, b0, cell, used_p, corr = j["occupancy"], j["bounds"], float(j["cell"]), j["frames"], j["correction"]
        middle = b0[:, 0] + cell * (0.5 + np.argwhere(occ > 0.5).mean(0))
        index = {int(x): k for k, x in enumerate(grips_all[f"{t}_frames"])}
        for n, x in enumerate(used_p):
            k = index[int(x)]
            Rw, ow, g = grips_all[f"{t}_R"][k], grips_all[f"{t}_origin"][k], grips_all[f"{t}_grips"][k]
            Rg = cv2.Rodrigues(g[:3].astype(np.float64))[0]
            R = Rw @ Rg @ cv2.Rodrigues(corr[n, :3].astype(np.float64))[0]
            paddles.setdefault(int(x), []).append((t, R @ middle + Rw @ (Rg @ corr[n, 3:] + g[3:]) + ow))  # camera coords


def paddle_near(i):
    """The player whose paddle the ball touches at span index i (within a frame either side), or None. Image position is
    tight, depth from size is loose: the ball must be inside the paddle's reach in the image and roughly at its depth."""
    ball = (pos[i] - t_cw) @ R_cw
    uv_ball = np.array([cam["fx"] * ball[0] / ball[2] + cam["cx"], cam["fy"] * ball[1] / ball[2] + cam["cy"]])
    best = None
    for f in range(span[i] - 1, span[i] + 2):
        for t, c in paddles.get(int(f), []):
            uv_paddle = np.array([cam["fx"] * c[0] / c[2] + cam["cx"], cam["fy"] * c[1] / c[2] + cam["cy"]])
            reach_px = f_px * PADDLE_REACH / c[2]
            d = np.linalg.norm(uv_ball - uv_paddle) / reach_px
            if d < 1 and abs(ball[2] - c[2]) < PADDLE_DEPTH and (best is None or d < best[1]):
                best = (t, d)
    return best


hits = {}
for i in events:
    if at_floor(i):
        kinds[i] = "bounce"
    elif (p := paddle_near(i)) is not None:
        kinds[i], hits[int(span[i])] = "hit", p[0]
    elif kinds[max(0, i - 2): i + 3].tolist().count("hand"):
        kinds[i] = "hand_contact"
    else:
        kinds[i] = "contact"
bounces = [int(span[i]) for i in events if kinds[i] == "bounce"]
others = [int(span[i]) for i in events if kinds[i] in ("contact", "hand_contact")]
px = np.linalg.norm(project_world(pos[seen_at]) - uv, axis=1)
summary = dict(diameter_m=round(diameter, 4), detections=len(frames), first=int(frames[0]), last=int(frames[-1]),
               px_median=round(float(np.median(px)), 2), px_90=round(float(np.percentile(px, 90)), 2),
               bounces=bounces, hits={str(k): v for k, v in hits.items()}, contacts_in_air=others, height_max=round(float(pos[:, 1].max()), 2),
               speed_max=round(float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max() * fps), 1),
               kinds={k: int((kinds == k).sum()) for k in set(kinds)})
(OUT / "ball.json").write_text(json.dumps(dict(summary=summary, frames=span.tolist(), position=np.round(pos, 4).tolist(), kind=kinds.tolist()), indent=0))
print(json.dumps(summary), flush=True)

# Debug video: detection (green), the 3D path reprojected (red, at its size), kind
colors = {"seen": (0, 0, 255), "hand": (255, 0, 255), "unseen": (200, 200, 200), "bounce": (0, 255, 255), "contact": (255, 255, 0)}
writer = cv2.VideoWriter(str(OUT / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
trail = []
for i, file in enumerate(frames_dir):
    image = cv2.imread(str(file))
    if i in track:
        b = track[i]["box"]
        cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    if span[0] <= i <= span[-1]:
        j = i - span[0]
        q = project_world(pos[j:j + 1])[0]
        c_depth = ((pos[j] - t_cw) @ R_cw)[2]
        r = max(2, int(f_px * diameter / 2 / c_depth))
        trail.append(q)
        for a, b2 in zip(trail[-30:-1], trail[-29:]):
            cv2.line(image, tuple(int(v) for v in a), tuple(int(v) for v in b2), (0, 0, 180), 2)
        cv2.circle(image, tuple(int(v) for v in q), r, colors.get(kinds[j], (255, 255, 255)), 2)
        cv2.putText(image, f"{i} {kinds[j]} h={pos[j, 1]:.2f} m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    writer.write(cv2.resize(image, (960, 540)))
writer.release()

# Top-down: the path over the floor (x right, z away from the camera), colored by time; bounces marked
S, scale = 800, 50  # px, px per m
top = np.full((S, S, 3), 255, np.uint8)
for m in range(-8, 9):
    cv2.line(top, (int(S / 2 + m * scale), 0), (int(S / 2 + m * scale), S), (230, 230, 230), 1)
    cv2.line(top, (0, int(S / 2 + m * scale)), (S, int(S / 2 + m * scale)), (230, 230, 230), 1)
xy = np.column_stack([S / 2 + pos[:, 0] * scale, S / 2 + pos[:, 2] * scale]).astype(int)
for j in range(1, len(xy)):
    color = cv2.applyColorMap(np.array([[int(255 * j / len(xy))]], np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
    cv2.line(top, tuple(xy[j - 1]), tuple(xy[j]), color, 2)
for f in bounces:
    cv2.circle(top, tuple(xy[f - span[0]]), 6, (0, 0, 0), 2)
cam_xy = (int(S / 2 + t_cw[0] * scale), int(S / 2 + t_cw[2] * scale))
cv2.drawMarker(top, cam_xy, (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)
cv2.imwrite(str(OUT / "topdown.png"), top)
