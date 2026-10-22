# Scene hierarchy: pipeline design

The pipeline reconstructs a scene the way a person reads one, most important first:

1. **People**
2. **Objects people directly interact with**: what they hold (paddles, rackets, bats) and what they hit (the ball)
3. **Pertinent objects**: things that matter to the play but nobody holds (a net, a goal, a hoop)
4. **The setting**: floor, walls, ceiling, everything else

Each layer is anchored by the ones above it. A layer only explains the pixels the layers above left unexplained, and it reads their results instead of re-estimating them. A last pass checks that the layers agree.

This replaces "Equipment" and "Ball and physics" under "Later" in ROADMAP.md. Every method here was tested on `samples/pickleball.mp4` in `research/`.

## Principles

- **Generic.** Nothing names a sport in code. Object names (such as "paddle", "net") are the only per-sport input, and they come with defaults.
- **No rescaling from known sizes.** The bodies set the scale and every layer follows them. Known sizes (court lines, net length, ball diameter) appear in reports as diagnostics, never as corrections. On the sample, the court measures 7–8% small, and that's accepted.
- **Every frame is used.** Expensive seeding (detection on tiles, image-to-3D) may subsample, and says so.
- **Temporal consistency over per-frame fits.** The same object has one shape across the video and moves smoothly. Per-frame evidence is weighted by how well it fits, and bad frames borrow from their neighbors.
- **Physics only for consistency.** Gravity arcs, contacts and smoothness. No drag or spin.
- **Runs on a laptop CPU**, faster with a gaming GPU. Heavy stages resume where they stopped.
- **When unsure, leave it out.** A layer that fails its checks is dropped from the export, and the report says why.

## Stages

Stages talk through files in the work folder, one subfolder each, as now. New stages are marked *new*.

| Layer | Stage | Reads | Writes |
|---|---|---|---|
| 1 | `ingest`, `track`, `bodies`, `smooth`, `floor`, `colors` | video | as now: tracks, bodies (world), floor frame, camera, body colors |
| 4 (first half) | `background` *new, split from `scene`* | track, ingest | `background.jpg`, the video with people removed |
| 2 | `held` *new* | bodies, smooth, track, ingest | per person: object mesh (own frame), colors, pose per frame (world), confidence per frame, `masks.npz` |
| 2 | `ball` *new* | ingest, bodies, floor, held | path per frame (world), kind per frame (seen, hand, unseen, bounce, hit, contact), diameter, confidence |
| 3 | `fixtures` *new* | background, floor, camera | per object: mesh (an upright plane on the floor for thin objects), RGBA texture, mask, see-through flag |
| 4 | `scene` *changed* | background, fixtures, floor, bodies | graphics removed, fixtures filled from behind, MoGe-2 mesh with walls and ceiling snapped to fitted planes |
| all | `export` *changed* | everything | one GLB: bodies, held objects, ball, fixtures, scene |
| all | `refine` *new, later* | everything | a consistency report; corrections fed back to the stages above |

### Layer 1: people (as now)

Bodies set the world: the camera, the floor frame (y up, floor at y = 0) and the scale. Unchanged.

### Layer 2: objects directly interacted with

**`held`**: one object per person's hand.
1. **Find.** An open-vocabulary detector (OWLv2 or Grounding DINO, Apache-2.0) on person crops, keeping detections at a wrist. Sparse frames are enough.
2. **Track.** SAM 2.1 video mode on a crop that follows the person, seeded by the detections, over every frame. Other people are hidden from the tracker. Memory is pruned and the run resumes.
3. **Fit**, the "train on the video" step. One density grid in the hand's frame plus a pose per frame, fit together to the tracked masks by ray marching:
   - the hand's pixels are ignored, and the body only occludes
   - an acceleration prior keeps poses smooth in camera space
   - frames that fit badly are refilled from their neighbors
   - held-out frames score the shape
   - a second pass at 3 mm cells
4. **Color.** A color grid in the same fit, with per-frame gain and half-turn flips decided by color; then the mesh is painted along its normals.
5. **Output.** Mesh in the object's frame, pose per frame in the world, and silhouette IoU per frame as confidence. Frames below the threshold are hidden in the export.

**`ball`**: one small object in flight.
1. RF-DETR's "sports ball" class on full-resolution tiles, every frame.
2. Link detections into tracklets and drop spots that never move (logos, graphics).
3. Size: self-calibrated from the floor and from frames at a hand, never from a known diameter.
4. One smoothing fit of the whole path: image positions, depth from size, gravity, and the floor as a lower bound.
5. Label contacts: bounces at the floor, hits where a `held` object is within reach, hand contacts.

### Layer 3: pertinent objects

**`fixtures`**:
1. Grounding DINO by name on the background image, then a SAM 2.1 mask.
2. Thin objects become an upright plane on the floor line under the mask, with height from its top edge.
3. Texture by matting against what's behind it.
4. See-through objects (netting, fences) keep their alpha.

A net costs under a minute on CPU.

### Layer 4: the setting

**`background`** is the current median of frames without people's boxes.

**`scene`** builds on it:
1. **Graphics.** On-screen graphics (scoreboards, logos) are found by name and by edges that never change between frames, then removed.
2. **Fill.** Everything removed is filled with LaMa (Apache-2.0, 18 s per 1080p frame on CPU). See-through fixtures are first recovered through their holes: a morphological closing removes the strings, and LaMa fills only the solid parts.
3. **Mesh.** MoGe-2 as now. Walls and ceiling are fit as planes (RANSAC). Points near a plane move onto it along their camera ray, so the texture stays aligned. Where a fixture stood, the mesh continues as floor, or as wall if the wall comes first.

### Export

Every layer goes into one GLB in the world frame:
- bodies, skinned as now
- held objects as nodes animated per frame, hidden during long gaps
- the ball as an animated sphere
- fixtures as unlit textured quads with alpha blending
- the scene and backdrop as now

## How the layers hand off

**Pixels, downward.** Each layer writes the pixels it explains, per frame where they move:
- people: bodies, and hands separately
- held objects: their tracked masks
- the ball
- fixtures: static masks

A lower layer treats them as occluders (they hide it) or as explained (it ignores them). The background leaves out people's boxes, the paddle fit ignores hands, and the setting fill removes fixtures and graphics.

**Anchors, downward.**
- Held objects live in a wrist joint's frame.
- The ball is anchored at hand, held-object and floor contacts.
- Fixtures stand on the fitted floor.
- The setting's floor is the fitted floor.

**Corrections, upward**, handled by `refine`:
- A ball hit with no paddle in reach questions the paddle's pose in that frame.
- A fixture at a size that disagrees with the bodies is reported, not corrected.
- Feet that sink into or float above the floor question the body.
- A held object passing through a body questions both.

## `refine`: the final pass

Runs last. It never re-runs a heavy model and only adjusts what the layers wrote.
- **Render and compare.** Render every layer from the camera and compare with the video, per frame and per layer. Layers that disagree with the video, or with each other, are listed.
- **Physics checks.** Contacts (feet on the floor, object in hand, ball off paddle), no interpenetration, plausible accelerations.
- **Small joint fixes.** Re-fit poses near the problem frames with all layers present. For example, the paddle pose at a hit, given the ball.
- **Report.** What was fixed, what's left, and diagnostics such as known sizes vs. measured ones.

Candidates for later: a vision-language critic (Qwen3-VL, Apache-2.0) to flag frames that look wrong, and video super-resolution for textures. See `research/README.md`.

## Order of work

1. **`background`, `fixtures`, `scene` changes.** Cheap (minutes on CPU), and the biggest visual gain on the sample. Tests can use a synthetic image.
2. **`ball`.** Detection on tiles is the slow part, 13–19 s per frame on CPU, so it needs resume and a GPU path. The fit takes seconds.
3. **`held`.** The heaviest: tracking and fitting take hours per person on CPU. Port `joint.py` with its smooth poses and color phase; drop the superseded attempts.
4. **`export`** with every layer, then **`refine`**.

`--tiny` mode runs each stage on a few seconds of video with coarse settings.

## Known issues to carry

These come from the research runs; `refine` or later work should address them.
- **Paddles:** a little too thin, faces ripple, blurry texture, no handle. Color fitting slightly worsens poses.
- **Ball:** fits about 20% large (detector boxes are loose). About 50 contacts are unexplained: hidden hits and holds, not wall hits.
- **Net:** court lines are baked into its texture. Its base pulls its plane forward, so it measures short.
- **Setting:** a faint ghost of the net at one end, a smudge where the wheels were, a blurry far floor, the far wall 6° off square.
- **Shadows:** too faint indoors to use; dropped for now.
