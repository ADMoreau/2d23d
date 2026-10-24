# Research: objects beyond the people

Experiments toward the layers below the people in the scene hierarchy: objects the players directly interact with (paddles, the ball), pertinent objects (the net), and the setting. The pipeline design that came out of them is in `docs/hierarchy.md`. These scripts aren't pipeline stages yet. They read a finished work folder from `2d23d run` and write their results to their own `run/` folders.

They were developed in `work/lab/` (git-ignored), which holds the venvs, the work folder (`work-smooth/`), the results and the logs. This folder is a copy of the code only: no weights, clips or results. To run them, copy them back into a lab folder that has `venv/` (Python 3.10: torch CPU, numpy, scipy, opencv, trimesh, scikit-image) and `work-smooth/`.

All models used are permissively licensed: SAM 2.1 (Apache-2.0), Grounding DINO (Apache-2.0), RF-DETR (Apache-2.0), TripoSR (MIT), LaMa (Apache-2.0; `setting/` expects big-lama.pt in `models/lama/` from the simple-lama-inpainting v0.1.0 release). Heavy jobs run under `systemd-run` memory caps.

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
- Finding the ball in play (`ball3d.py` step 1): tracklets (Hungarian matching, constant velocity, size within 2×), then a chain of them that starts at a ball someone touches. Detections near a body count fully; a moving ball far from everyone counts a little and only directly continues the chain. Each hop must be possible at 40 m/s and costs more the faster it is. `linkplot.py` plots every candidate against the linked path.
- Result: median reprojection 1.2 px, 13 bounces, 5 hits.
- On other sports: soccer's top speed went from 67 m/s (jumping to other games' balls) to 16 m/s. On basketball the diameter comes out 23.9 cm (24 cm real), never assumed.
- Known issues: the ball comes out 8.9 cm vs 7.4 cm real, and 48 contacts are unexplained, probably hidden hits and wall hits.

## Net: good enough

- `net/net_mask.py` finds the net by name with Grounding DINO in the background image and cuts it out with SAM 2.1.
- `net/net_build.py` stands it on the fitted floor as a textured, see-through plane, exported as a GLB.
- Known issues:
  - It measures 6.24 m by 0.73 m, against about 6.7 m by 0.86 m in reality, so the scene may be 5–15% small.
  - The court lines behind the net are baked into its texture, along with an inpainting smudge.

## Fixtures: which objects matter

Tested on three clips (pickleball, pickup soccer, half-court basketball; the last two are CC BY clips kept in the lab, not here).
- **`fixtures/vlm_names.py`**: Qwen3.5-0.8B (Apache-2.0) names the sport and its objects from a frame, on CPU in 1–4 minutes. Its names keep decoys (bags, chairs) out. It sometimes repeats itself.
- **`fixtures/candidates.py`**: every instance of each name (Grounding DINO, then a SAM 2.1 mask). Objects standing on the floor get a footprint from the mask's bottom line (RANSAC) on the floor. Objects hanging in the air (a mask whose lowest scene points are over 0.5 m up) are placed from the scene's MoGe points instead: the basketball rim comes out 2.78–3.19 m high (rim 3.05 m), the backboard 2.99–4.05 m.
- **`fixtures/signals.py`**: six signals from the people and ball over the whole video, each against the same object dropped at 300 random spots in the play area:
  - contacts: ball contacts at it
  - crossed: the ball crossing its line
  - sides: people staying on either side of it (a net)
  - guarded: one person staying in one spot near it (a keeper)
  - aimed: the ball heading for it
  - reached: separate ball visits level with it (a hoop)
- **Strictness** from 0 to 1 sets how near counts, how much of a signal is needed and what share of random spots it must beat. An object's strength is the highest strictness at which any one signal passes.

| Clip | Kept, with strength | Highest of the rest |
|---|---|---|
| Pickleball | net 0.9 (sides) | none |
| Soccer | pop-up goal 0.8 (guarded) | none; with hand-picked decoys, bags beside the keeper 0.7 |
| Basketball | hoop and rim 0.8 (reached) | a tree 0.2 |

Strictness 0.8 keeps exactly the right objects on all three. Known issues: "aimed" never fires for close shots (slow across the floor), and the counts are thin on short clips (two shots on basketball).

`fixtures/peek_boxes.py` draws Grounding DINO's boxes for some names on one image.

## Setting: good enough

- **`setting/topdown.py` and `measure_lines.py`**: the floor from straight above, and its painted lines. The court comes out rectangular but 7–8% small, the same as the net. The bodies set the scale, and the design keeps it (no rescaling from known sizes).
- **`setting/walls.py`**: wall and ceiling planes by RANSAC on the scene mesh. The long wall fits to 4.3 cm and runs within 0.9° of parallel to the court lines.
- **`setting/overlay.py`**: on-screen graphics are sharp edges that never change. That finds the scoreboard, but also quiet painted floor.
- **`setting/fill.py`**: LaMa fills the scoreboard and net, using the scoreboard mask from `net/net_mask.py` with the name "scoreboard".
- **`setting/see_through.py`**: the floor behind the net, recovered through its holes; LaMa fills only its solid parts. `fill_floor.py` (filling in the top-down view) was a minor gain and isn't used.
- **`setting/setting_build.py`**: the scene mesh with the net's stand-in geometry replaced by floor or wall, walls and ceiling snapped to their planes, and painted with the cleaned image.
- **`setting/render.py`**: a quick check render. **`setting/assemble.py`**: the export with the new setting plus the net.
- Known issues: a faint ghost of the net at its left end, a smudge where the wheels were, a blurry far floor, the far wall 6° off square.

## Shadows: dropped for now

- **`shadows/lights.py`**: light positions from people's shadows, by non-negative least squares over candidate lights on the ceiling. On held-out frames it explains 19% of the darkening, against 3–7% for random lights.
- **`shadows/ball_shadow.py`**: the ball's shadow shows only near the floor.
- Indoors, under many soft lights, the shadows are too faint to use. They may matter outdoors.

## All layers together

- `assemble/paddle_meshes.py` (lab venv), then `assemble/assemble_all.py` (repo venv): one GLB with the bodies, paddles, ball, net and cleaned setting.
