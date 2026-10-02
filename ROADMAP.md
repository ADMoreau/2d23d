# Roadmap

Turn single-camera video of people playing a sport into an animated 3D scene you can walk around in AR/VR. Processing is offline and fully automatic. We start simple and add pieces only when they're needed.

## Scope for now

- Fixed camera, such as a phone on a tripod. Panning or zooming footage, like TV broadcasts, comes later.
- Offline processing. No live mode.
- People on a floor. Nothing sport-specific yet.
- Anything out of frame is ignored. Nothing is invented off-screen.
- No manual steps. When unsure, the pipeline falls back to something simple instead of guessing.

## Design principles

- **Stages talk through files.** Each stage reads the previous stage's outputs from a work folder and writes its own, so any stage can be rerun or swapped on its own.
- **Every stage writes a debug video** so failures are easy to see.
- **Off-the-shelf first.** Use pretrained models, and train nothing until something clearly needs it.
- **Human priors.** Bodies come from a parametric body model and stay on the floor.
- **Permissive dependencies.** Prefer Apache/MIT code and weights, and document each dependency's license.

## v0.1: Bodies on a floor (about 3–4 weeks)

1. **Scaffold.** Package layout, a `run` command, a stage runner that skips stages whose output already exists, and CI.
2. **Track people.** RF-DETR for detection plus OC-SORT from the `trackers` library for IDs. No training.
3. **3D bodies.** SAM 3D Body on each tracked person, paired with a field-of-view estimator (it supports MoGe-2) to place people relative to the camera. Needs a GPU; Kaggle's free tier works for development.
4. **Floor from the people.** Fit one plane to everyone's feet across the whole video, rotate it flat, snap each body's lowest point onto it, then smooth.
5. **Automatic check.** Drop or smooth frames where a body floats or sinks.
6. **Export and view.** A GLB with animated bodies on a plain floor, plus a static three.js/WebXR viewer with play and scrubbing, VR, and AR tabletop placement. Host it on GitHub Pages.
7. **Tiny mode.** A `--tiny` flag that runs small settings on a few seconds of video, used by the README quickstart.
8. **Scene.** Take the people out of the video with a median of frames that leaves out their boxes, turn the rest into a 3D mesh with MoGe-2, scale it to the bodies where feet touch the floor, lay its floor on the fitted one, and paint it with the video. It replaces the plain floor when it fits the bodies. It only has what the camera saw, so it looks right from near the camera. Single-image depth is the only option on a tripod, because SLAM and splatting need the camera to move.
9. **Colors.** Paint each person's body with their colors from the video. Each vertex gets the median color of the frames where it faced the camera, uncovered, on a pixel that differs from the background without people. Vertices never seen take their neighbors' colors.

**Done when** someone can run one command on a sample fixed-camera clip and walk around the result, and it looks right on a few different clips.

## Later, only as needed (rough order)

- **Bodies on uneven ground.** Stand people on the scene's surfaces, such as ramps, bowls and steps, instead of one flat floor. Fall back to the floor where the surface wasn't seen.
- **Identity across the video.** Link tracks into people automatically using hard constraints (no one in two places at once) and appearance. Split when unsure.
- **Avatars.** Sharper looks than one color per vertex, such as a texture with faces and logos, and plausibility checks such as fixed proportions.
- **Equipment.** What people hold and swing, such as paddles, rackets, bats and sticks: a simple model of the sport's implement, held in the hand and turned to match the video. Leave it out when the video doesn't show it clearly.
- **Moving cameras.** Camera tracking or a court/field fit, packaged as sport-specific "domain packs", so broadcast footage works. Estimate the camera's path with SLAM while masking out people, as TRAM and SLAHMR do, then place bodies along it. A moving camera also lets the scene be reconstructed in full 3D: a mesh for surfaces to stand on, or a Gaussian splat for looks. Splats need their own renderer and get heavy on phones and headsets. Check licenses: the original Gaussian splatting code and DUSt3R/MASt3R are non-commercial, while gsplat (Apache-2.0) and COLMAP (BSD) are permissive.
- **Ball and physics.** Only if the scene looks wrong without them.
- **Formal evaluation.** Benchmarks like WorldPose or SoccerNet-GSR, if accuracy numbers become useful.

## Test footage

- Our own tripod recordings, with the players' permission.
- Public domain DVIDS clips, Pexels and Creative Commons videos. Keep each clip's license and attribution next to it.

## Licensing notes

- Avoid Ultralytics YOLO (AGPL-3.0). RF-DETR is Apache 2.0.
- Body model: [MHR](https://github.com/facebookresearch/MHR) (Apache-2.0) rather than SMPL-X (non-commercial, registration required).
- SAM 3D Body ships under Meta's SAM License. Check the terms before depending on it.

## Compute

- Tracking, floor fitting, export and the viewer run on an ordinary CPU for short clips.
- SAM 3D Body needs an NVIDIA GPU: Kaggle's free tier for development, or a rented 24 GB GPU for longer videos.
- A 30-second clip should take minutes on a GPU.
