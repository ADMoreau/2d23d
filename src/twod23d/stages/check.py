"""Check: drop bodies that float or sink. Roadmap step 5.

The floor stage snaps every body onto the floor and records how far it had to move it (floor_gap). A body
far off the floor, or suddenly off from the same person's frames around it, is a bad estimate: typically
someone partly hidden, whose unseen legs SAM 3D Body had to guess. Those bodies are dropped, and the
exported animation fills the gaps from the frames around them. Anyone left with under a second of bodies
is dropped entirely.
Output:
- bodies.npz like floor/bodies.npz, without the dropped bodies
- report.json: bodies kept and dropped per track, and why
- debug.mp4: where each body touches the floor, in its track's color if kept, red with its gap if dropped
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from twod23d.pipeline import Context
from twod23d.stages.track import track_color
from twod23d.video import read_meta, write_debug_video

log = logging.getLogger(__name__)

MAX_GAP = 0.25  # meters: the sample's players stayed within 18 cm of the floor, a half-hidden spectator sank 17-56 cm
MAX_JUMP = 0.15  # meters off the same person's median gap around that moment: a one-frame glitch
AROUND_SECONDS = 0.5  # that median is over this long before and after
MIN_TRACK_SECONDS = 1.0  # a person with fewer seconds of good bodies is dropped entirely
SHARED = {"faces"}  # arrays in bodies.npz that aren't one row per body


def run(ctx: Context, out: Path) -> None:
    fps = read_meta(ctx.work)["fps"]
    with np.load(ctx.work / "floor" / "bodies.npz") as f:
        bodies = dict(f)
    far, jump = flag(bodies, fps)
    keep = ~(far | jump)
    good = Counter(bodies["track_id"][keep].tolist())
    frames_per_body = ctx.settings.body_every  # bodies exist for every Nth frame only, as in --tiny
    short = {int(t) for t in np.unique(bodies["track_id"]) if good[int(t)] * frames_per_body < MIN_TRACK_SECONDS * fps}
    keep &= ~np.isin(bodies["track_id"], list(short))
    np.savez_compressed(out / "bodies.npz", **{k: v if k in SHARED else v[keep] for k, v in bodies.items()})

    report = {
        str(t): {
            "kept": int((keep & (bodies["track_id"] == t)).sum()),
            "off_the_floor": int((far & (bodies["track_id"] == t)).sum()),
            "sudden_jumps": int((jump & ~far & (bodies["track_id"] == t)).sum()),
            "dropped_whole_track": int(t) in short,
        }
        for t in np.unique(bodies["track_id"]).tolist()
    }
    (out / "report.json").write_text(json.dumps({"max_gap": MAX_GAP, "max_jump": MAX_JUMP, "tracks": report}, indent=2))
    log.info(
        "check: kept %d of %d bodies; %d off the floor, %d sudden jumps; dropped people: %s",
        keep.sum(), len(keep), far.sum(), (jump & ~far).sum(), sorted(short) or "none",
    )
    camera_to_world = np.array(json.loads((ctx.work / "floor" / "floor.json").read_text())["camera_to_world"])
    camera = json.loads((ctx.work / "bodies" / "camera.json").read_text())
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    write_debug_video(ctx.work, out, "check", draw_verdicts(bodies, keep, np.linalg.inv(camera_to_world), K))


def flag(bodies: dict, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per body: is it off the floor (far), and is it off from the same person around that moment (jump)?"""
    gap = bodies["floor_gap"]
    far = np.abs(gap) > MAX_GAP
    jump = np.zeros_like(far)
    half = round(AROUND_SECONDS * fps)
    for track_id in np.unique(bodies["track_id"]):
        rows = np.flatnonzero(bodies["track_id"] == track_id)
        frames = bodies["frame"][rows]
        for row, frame in zip(rows, frames):
            around = rows[np.abs(frames - frame) <= half]
            jump[row] = abs(gap[row] - np.median(gap[around])) > MAX_JUMP
    return far, jump


def draw_verdicts(bodies: dict, keep: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray):
    """A draw(frame, index) function: each body's floor contact, red and labeled with its gap if dropped."""
    verdicts = defaultdict(list)
    for i, (frame, track_id, vertices) in enumerate(zip(bodies["frame"], bodies["track_id"], bodies["vertices"])):
        contact = vertices[np.argmin(vertices[:, 1])].astype(np.float64) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        if contact[2] > 0.1:
            u, v = (K @ contact)[:2] / contact[2]
            verdicts[int(frame)].append((round(u), round(v), int(track_id), bool(keep[i]), float(bodies["floor_gap"][i])))

    def draw(image: np.ndarray, i: int) -> None:
        for u, v, track_id, kept, gap in verdicts[i]:
            if kept:
                cv2.circle(image, (u, v), 6, track_color(track_id), -1)
            else:
                cv2.circle(image, (u, v), 9, (0, 0, 255), 2)
                cv2.putText(image, f"{100 * gap:+.0f} cm", (u + 12, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return draw
