"""How the play uses each fixture candidate, from people and ball over the whole video, and whether it's pertinent at
each strictness from 0 (any hint) to 1 (only the unmistakable).

Signals, each against a baseline (the same footprint dropped at random spots spread evenly over the play area, in a
random direction):
1. contacts: ball contacts (bounces, hits) within reach of it, below its top
2. crossed: times the ball's path crosses over its footprint line
3. sides: how many people stay on each side of it, counting only where they're beside its length (the smaller
   side's count, as a share of the people seen there)
4. guarded: the share of the video one person stays near it and in one spot (a keeper, not a crowd passing by, nor a
   player covering the side it runs along)
5. aimed: share of the ball's fast frames whose heading reaches it within AHEAD seconds
6. reached: separate visits of the ball level with it and within reach (a hoop: shots, made or not)

The ball's distance to it counts the camera's line of sight DEPTH_SLACK times looser: depth is what one camera measures
worst.

Strictness s sets how close counts as near (people 3 m down to 1 m, ball 1 m down to 0.3 m), how much of each signal
is needed, and what share of the random spots it must beat (80% up to 99%). A candidate is pertinent at s if any one
signal passes both; its strength is the highest s at which it is.

usage: signals.py <work dir> <candidates.json> <ball.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(sys.argv[1])
cands = json.loads(Path(sys.argv[2]).read_text())
ball = json.loads(Path(sys.argv[3]).read_text())
AHEAD, FAST, STAY, MIN_FRAMES, RANDOM = 1.5, 3.0, 0.95, 30, 300
STRICTNESS = np.linspace(0, 1, 11)
GUARD = 3.0 - 2.0 * STRICTNESS  # m: how close a person must be
REACH = 1.0 - 0.7 * STRICTNESS  # m: how close the ball must be
BEATS = 0.8 + 0.19 * STRICTNESS  # share of random spots a signal must beat
NEEDED = {  # how much of each signal, at each strictness
    "contacts": 1 + 4 * STRICTNESS,
    "crossed": 1 + 9 * STRICTNESS,
    "sides": 0.2 + 0.25 * STRICTNESS,
    "guarded": 0.3 + 0.6 * STRICTNESS,
    "aimed": 0.05 + 0.25 * STRICTNESS,
    "reached": 1 + 1 * STRICTNESS,  # two shots that reach it are plenty if random spots rarely see one
}
DEPTH_SLACK = 3.0
VISIT_GAP = 0.5  # s: ball frames this far apart are separate visits
KEYS = list(NEEDED)
fps = json.loads((WORK / "ingest" / "meta.json").read_text())["fps"]
cam_to_world = np.array(json.loads((WORK / "floor" / "floor.json").read_text())["camera_to_world"])
look = cam_to_world[[0, 2], 2] / np.linalg.norm(cam_to_world[[0, 2], 2])  # the camera's line of sight on the floor

with np.load(WORK / "check" / "bodies.npz") as b:
    frame, track, kp = b["frame"], b["track_id"], b["keypoints"]
people = kp[..., [0, 2]].mean(1)  # each body's place on the floor (x, z)
n_frames = int(frame.max()) + 1
tracks = [t for t in np.unique(track) if (track == t).sum() >= MIN_FRAMES]
pos = np.array(ball["position"])
kind = np.array(ball["kind"])
bxz, by = pos[:, [0, 2]], pos[:, 1]
vel = np.gradient(bxz, axis=0) * fps
contact = np.isin(kind, ["bounce", "contact", "hit", "hand_contact"])
fast = np.flatnonzero(np.linalg.norm(vel, axis=1) > FAST)
ahead = bxz[fast][:, None] + vel[fast][:, None] * np.linspace(0, AHEAD, 30)[None, :, None]  # (fast frames, steps, 2)


def dedupe(cands):
    def iou(a, b):
        w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)
    kept = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(iou(c["box"], k["box"]) < 0.5 for k in kept):
            kept.append(c)
    return kept


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def ball_dist(p, a, b):
    """Like seg_dist, with the offset along the camera's line of sight counted DEPTH_SLACK times looser."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    off = p - (a + t[..., None] * ab)
    along = off @ look
    return np.sqrt(np.sum(off ** 2, -1) - along ** 2 + (along / DEPTH_SLACK) ** 2)


def visits(m):
    f = np.flatnonzero(m)
    return 0 if len(f) == 0 else 1 + int((np.diff(f) > VISIT_GAP * fps).sum())


def side(p, a, b):
    ab = b - a
    return np.sign(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0]))


def signals(a, b, height, bottom=0.0):
    """Each signal at each strictness's distances: {name: array over STRICTNESS}."""
    ab = b - a
    ball_off = ball_dist(bxz, a, b)
    low = (by < height + 0.3) & (by > bottom - 0.3)  # level with it
    contacts = np.array([((ball_off < r) & low & contact).sum() for r in REACH])
    reached = np.array([visits((ball_off < r) & low) for r in REACH])
    s = side(bxz, a, b)
    flips = np.flatnonzero(s[1:] * s[:-1] < 0)
    t = ((bxz[flips] - a) @ ab) / (ab @ ab)  # where along it the ball crossed
    crossed = np.array([(np.abs(t - 0.5) < 0.5 + r / np.linalg.norm(ab)).sum() for r in REACH])
    beside = np.abs(((people - a) @ ab) / (ab @ ab) - 0.5) < 0.75
    stay, counted = {1: 0, -1: 0}, 0
    for tr in tracks:
        m = (track == tr) & beside
        if m.sum() < MIN_FRAMES:
            continue
        counted += 1
        sd = side(people[m], a, b)
        for k in stay:
            stay[k] += (sd == k).mean() >= STAY
    sides = np.full(len(STRICTNESS), min(stay.values()) / max(counted, 1))
    person_off = seg_dist(people, a, b)
    guarded = np.zeros(len(GUARD))
    for i, r in enumerate(GUARD):
        for tr in tracks:
            m = (person_off < r) & (track == tr)
            if m.any():
                spot = np.median(people[m], 0)  # where they stay while near it
                m &= np.linalg.norm(people - spot, axis=1) < r / 2
                guarded[i] = max(guarded[i], len(np.unique(frame[m])) / n_frames)
    reach = seg_dist(ahead, a, b).min(1) if len(fast) else np.zeros(0)
    aimed = np.array([(reach < r).mean() if len(fast) else 0.0 for r in REACH])
    return dict(contacts=contacts, crossed=crossed, sides=sides, guarded=guarded, aimed=aimed, reached=reached)


rng = np.random.default_rng(0)
area = np.percentile(people, [2, 98], axis=0)  # the play area, (x, z) low and high
cands = dedupe(cands)
print(f"{len(tracks)} people, ball over {len(pos)} frames ({contact.sum()} contacts)")
print(f"{'#':>2} {'name':12} {'len m':>5} {'h m':>4}  " + "  ".join(f"{k + ' @0.5':>15}" for k in KEYS) + "  strength  (by)")
for n, c in enumerate(cands):
    a, b = np.array(c["ends"]); length = np.linalg.norm(b - a)
    if length > 30 or np.linalg.norm((a + b) / 2) > 80:
        print(f"{n:>2} {c['name']:12} footprint {length:.0f} m long, skipped"); continue
    real = signals(a, b, c["height"], c.get("bottom", 0.0))
    base = []
    for _ in range(RANDOM):
        mid = rng.uniform(area[0], area[1])
        ang = rng.uniform(0, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)]) * length / 2
        base.append(signals(mid - d, mid + d, c["height"], c.get("bottom", 0.0)))
    beats = {k: np.mean([r[k] < real[k] for r in base], 0) + 0.5 * np.mean([r[k] == real[k] for r in base], 0) for k in KEYS}
    passes = {k: (beats[k] >= BEATS) & (real[k] >= NEEDED[k]) & (real[k] > 0) for k in KEYS}
    ok = np.any([passes[k] for k in KEYS], 0)
    strength = float(STRICTNESS[ok].max()) if ok.any() else None
    why = [k for k in KEYS if strength is not None and passes[k][STRICTNESS == strength][0]]
    mid_s = len(STRICTNESS) // 2
    row = [f"{real[k][mid_s]:.2f}".rstrip("0").rstrip(".") + f" ({beats[k][mid_s]:.0%})" for k in KEYS]
    print(f"{n:>2} {c['name']:12} {length:5.1f} {c['height']:4.1f}  " + "  ".join(f"{r:>15}" for r in row)
          + f"  {'-' if strength is None else f'{strength:.1f}':>8}  {','.join(why)}", flush=True)
