# Project brief

Open-source tool that turns single-camera video of people playing a sport into an animated 3D scene viewable in AR/VR. Offline and fully automatic. See ROADMAP.md for the plan. The current goal is v0.1, "Bodies on a floor".

## Decisions so far

- Start simple. Build the smallest thing that works. Items under "Later" in ROADMAP.md are out of scope until asked for.
- v0.1 handles fixed-camera video only, such as a phone on a tripod. No court calibration, ball, physics or benchmarks yet.
- Fully automatic: no manual correction or labeling steps in the pipeline. When unsure, fall back to something simple instead of guessing.
- Ignore anything out of frame. Only the people on the floor matter.
- No polish needed. This is a working open-source repo, not a product.

## Architecture rules

- Python package with one `run` command: video in, GLB out.
- Stages communicate only through files in a work folder, one subfolder per stage. A stage is skipped if its output already exists, unless forced.
- Every stage writes a debug video.
- A `--tiny` mode runs small settings on a few seconds of video on CPU. The README quickstart uses it.
- CI runs unit tests only. Tests that run ML models are marked `ml`, and CI skips them.
- Keep device selection in one place. Code must run on CPU (slowly) and on NVIDIA GPUs. Avoid CUDA-only code where possible.

## v0.1 stack

- Detection: RF-DETR (Apache-2.0). Tracking: OC-SORT from Roboflow's `trackers` library (Apache-2.0). `supervision`'s ByteTrack is deprecated, and OC-SORT kept all four IDs on the sample where ByteTrack split one.
- 3D bodies: SAM 3D Body (MHR body model), with a field-of-view estimator such as MoGe-2.
- Floor: fit one plane to everyone's feet across the whole video, rotate it flat, snap each body's lowest point onto it, then smooth.
- Scene: the video with its people taken out, turned into a mesh by MoGe-2, scaled to the bodies at foot contacts, its floor laid on the fitted one, painted with the video. A plain floor when it doesn't fit the bodies.
- Colors: each body vertex gets the median video color of the frames where it faced the camera, uncovered.
- Output: a GLB with animated bodies, in their colors, in the painted scene.
- Viewer: a static three.js/WebXR page (play and scrubbing, VR, AR tabletop placement), hosted on GitHub Pages.

## Licensing

- Permissive dependencies only (Apache/MIT/BSD). Do not use Ultralytics YOLO (AGPL-3.0).
- Use MHR (Apache-2.0), not SMPL or SMPL-X (non-commercial).
- SAM 3D Body is under Meta's SAM License. Note it in the README.
- Don't commit model weights or video clips without their license alongside them.

## Environment

- Main dev machine: Ubuntu 22.04 laptop, CPU only. Its integrated AMD GPU isn't usable for ML.
- Heavy stages (SAM 3D Body) run on a remote NVIDIA GPU, such as Kaggle's free tier or a rented 24 GB GPU.
