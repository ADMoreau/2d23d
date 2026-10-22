# Research: objects beyond the people

Experiments toward the layers below the people in the scene hierarchy: objects the players directly interact with (paddles, the ball), pertinent objects (the net), and later the setting. These scripts aren't pipeline stages yet. They read a finished work folder from `2d23d run` and write their results to their own `run/` folders.

They were developed in `work/lab/` (git-ignored), which holds the venvs, the work folder (`work-smooth/`), the results and the logs. This folder is a copy of the code only: no weights, clips or results. To run them, copy them back into a lab folder that has `venv/` (Python 3.10: torch CPU, numpy, scipy, opencv, trimesh, scikit-image) and `work-smooth/`.

All models used are permissively licensed: SAM 2.1 (Apache-2.0), Grounding DINO (Apache-2.0), RF-DETR (Apache-2.0), TripoSR (MIT). Heavy jobs run under `systemd-run` memory caps.

## Paddles: good enough

Each player's paddle is tracked with SAM 2.1 video mode (`paddles/track_paddle.py`). Then `paddles/joint.py` fits its shape (a density grid in the paddle's own frame) together with its pose in every frame, against the tracked masks. That fit is "trained on the video".
- Smooth poses (`POSES=smooth`): per-frame poses with an acceleration prior, robust frame weights, and bad frames refilled from their neighbors. The wrist-anchored curve (`POSES=curve`) failed on wrist jitter.
- Colors come from a per-cell color grid with half-turn flip rounds; `paddles/recolor.py` paints the mesh.
- Chain: `recover.sh` and `step2.sh`, then `next.sh` and `far.sh`.
- Held-out silhouette IoU: 0.65–0.75.
- Known issues: the paddles come out thin, the faces ripple, the texture is blurry, and there's no handle.

Superseded attempts kept for the record: `fit.py`, `fit2.py`, `fit3.py` (rigid grips), `carve.py` (space carving), `shape.py` and `shape2.py` (flat outlines), `prior.py` (TripoSR prior) and the `export_paddles*.py` prototypes.

## Ball: good enough

- `ball/detect.py`: RF-DETR's "sports ball" class on full-resolution tiles of every frame.
- `ball/ball3d.py`:
  - links detections into tracklets and drops static look-alikes
  - sizes the ball from the floor and the hands, and places it in depth by its size
  - fits the whole path at once with gravity, a floor constraint and smoothing
  - labels contacts: bounces, paddle hits, hands
- Result: median reprojection 1.2 px, 13 bounces, 5 hits.
- Known issues: the ball comes out 8.9 cm vs 7.4 cm real, and 48 contacts are unexplained, probably hidden hits and wall hits.

## Net: good enough

- `net/net_mask.py` finds the net by name with Grounding DINO in the background image and cuts it out with SAM 2.1.
- `net/net_build.py` stands it on the fitted floor as a textured, see-through plane, exported as a GLB.
- Known issues:
  - It measures 6.24 m by 0.73 m, against about 6.7 m by 0.86 m in reality, so the scene may be 5–15% small.
  - The court lines behind the net are baked into its texture, along with an inpainting smudge.
