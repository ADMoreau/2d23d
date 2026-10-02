# 2d23d

Turn single-camera video of people playing a sport into an animated 3D scene you can walk around in AR/VR. Offline and fully automatic: one command takes a video and writes a GLB file, and a web viewer plays it in the browser, in VR, or on a table in AR.

**Status:** v0.1, "Bodies on a floor", works end to end on the sample clip. Each person becomes an animated 3D body in their own colors from the video, standing in a 3D copy of the place painted with the video itself. It has only been tried on one clip so far, and v0.1 is done when it looks right on a few. See [ROADMAP.md](ROADMAP.md) for the plan.

## Quickstart

From a clone of this repo, with Python 3.11 or newer:

```bash
# CPU-only machine: get the CPU build of PyTorch first, which skips several GB of CUDA libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
huggingface-cli login  # after requesting access at https://huggingface.co/facebook/sam-3d-body-vith
2d23d run samples/pickleball.mp4 --tiny
```

The sample is 30 seconds of a pickleball doubles final from one fixed camera ([CC BY 3.0](samples/pickleball.LICENSE.txt)). `--tiny` uses the first 3 seconds at 640 px with the smallest models and 3D bodies for every 10th frame, and takes about 4 minutes on a laptop CPU. It writes `pickleball-tiny.glb` and keeps each stage's output in `work/pickleball-tiny/`. To watch it, see [Viewer](#viewer). Tiny mode checks that everything is installed; its bodies and scene look rough, so judge quality from a full run.

SAM 3D Body's weights are gated, so request access on Hugging Face once and log in with a Read token. The first run downloads about 3 GB: the person detector (400 MB, into `~/.roboflow/models`), SAM 3D Body (2.4 GB) and MoGe-2 (140 MB for `--tiny`, 1.3 GB otherwise) into `~/.cache/huggingface`, and SAM 3D Body's and MoGe's code (25 MB, into `~/.cache/2d23d`). If Hugging Face downloads crawl, `HF_HUB_DISABLE_XET=1` switches them to plain HTTP, which was 8 times faster on one connection.

## Full run

```bash
2d23d run samples/pickleball.mp4            # writes pickleball.glb, work in work/pickleball/
2d23d run my-clip.mp4 -o out/my-clip.glb    # any fixed-camera clip
```

| Option | Does |
|---|---|
| `-o OUT` | GLB to write. Default: `NAME.glb`, after the video's name |
| `--work DIR` | work folder. Default: `work/NAME` |
| `--tiny` | first 3 s at 640 px with small models; adds `-tiny` to `NAME` |
| `--force [STAGE]` | rerun `STAGE` and every stage after it, or every stage if none is named |
| `--device DEVICE` | `auto` (the default: CUDA if available, else CPU), `cpu`, `cuda` or `cuda:N` |

The `bodies` stage is the slow one. On an 8-core laptop CPU it takes about 16 seconds per frame with four people: the 30-second sample (900 frames) took 4.4 hours. An NVIDIA GPU, such as Kaggle's free tier or a rented 24 GB GPU, is much faster. The other stages take seconds to minutes on a CPU. Finished stages are kept, so an interrupted run picks up where it stopped.

The full sample makes a 17.5 MB GLB: the four players, in their colors, in a 3D copy of the hall.

## Recording your own clips

- **Keep the camera still.** A phone on a tripod works. Panning, zooming or handheld footage isn't supported yet.
- **Get the whole floor in frame.** Anything out of frame is ignored, and people who leave the frame may come back under a new ID.
- **Let people move around.** The scene is painted from frames with the people taken out, so anyone who stays put the whole time, like a seated spectator, gets painted in.
- **Use the highest resolution you can.** 3D bodies are best when each person is at least about 300 px tall. In the 1080p sample the far players are 115–165 px, so 4K helps them at no extra processing time.

## How it works

`2d23d run` runs these stages in order. Stages talk only through files: each one writes a folder in the work folder, and every folder has a `debug.mp4` showing what the stage did.

| Stage | Does | Writes |
|---|---|---|
| `ingest` | reads the video's frames | `frames/`, `meta.json` |
| `track` | finds people with RF-DETR and follows them with OC-SORT | `tracks.json`: a box and an ID per person per frame |
| `bodies` | estimates each person's 3D body with SAM 3D Body, and the camera's field of view with MoGe-2 | `bodies.npz`: an MHR body per person per frame, `camera.json` |
| `floor` | fits one plane to everyone's feet, turns it flat, stands each body on it and smooths | `bodies.npz` in floor coordinates, `floor.json` with the camera's height and tilt |
| `scene` | takes the people out of the video, turns what's left into a 3D mesh with MoGe-2, scales it to the bodies and lays its floor on the fitted one; skipped if it doesn't fit them | `background.jpg`, `scene.npz`: the mesh, `scene.json` |
| `check` | drops bodies more than 25 cm off the floor or jumping more than 15 cm, and people left with under a second | `bodies.npz`, `report.json` saying what was dropped and why |
| `colors` | gives each spot on each person's body the median color of the frames where it faced the camera, uncovered | `colors.npz`, `colors.json` |
| `export` | builds the scene | `scene.glb`: each person as an animated, skinned body in their colors, in the painted scene (or on a plain floor without one), plus the video's camera |

A stage is skipped if its folder already exists. A stage writes to a temporary folder and only renames it when it succeeds, so a failed stage leaves nothing behind and runs again next time.

## Viewer

`viewer/index.html` is a static three.js page that plays a scene: play, pause and scrub through time, and orbit with the mouse. It opens from the video camera's point of view, where the scene looks just like the video with the people in 3D. The farther you move from there, the more you see what the camera couldn't: the backs of things are missing and gaps are stretched. A painted backdrop that fills the gaps near the camera fades out as you move away. To use it on your computer, serve the repo and open the GLB:

```bash
python -m http.server
# then open http://localhost:8000/viewer/?src=/pickleball-tiny.glb
```

You can also drag any `.glb` from 2d23d onto the page, or use "Open .glb"; the file stays on your computer. Add `&t=12.5` to the address to start 12.5 seconds in.

VR and AR need the page served over HTTPS, so publish it with GitHub Pages: turn Pages on once (Settings > Pages > Source: GitHub Actions), and `.github/workflows/pages.yml` publishes `viewer/` whenever it changes. In VR you stand where the video camera stood, at real scale. In AR, point the phone at a table and tap to place the scene on it, about 80 cm across, without the backdrop. Both need a browser with WebXR, such as Chrome on Android for AR or the Quest browser for VR.

## Limitations

- **Fixed camera only.** Broadcast footage needs camera tracking, which comes later.
- **One clip tested.** Only the pickleball sample has been run in full.
- **IDs don't survive long breaks.** If someone is hidden or leaves the frame, they may come back as a new person. Re-linking them is on the roadmap. A first test found the two teams easy to tell apart and teammates hard.
- **No ball, paddles or faces.** People's colors come from the video, but fine detail like faces and logos blurs, and what they hold isn't modeled.
- **The scene is what the camera saw.** It looks right from near the camera. Anything behind something else, or out of frame, is missing. On-screen graphics, like a broadcast scoreboard, get painted onto the walls.
- **VR and AR are untested on a device.** They follow three.js's standard WebXR setup, and the desktop viewer has been checked in headless Chrome.

## Development

```bash
pip install -e ".[ml,dev]"
pytest              # everything, including the whole pipeline in --tiny mode
pytest -m "not ml"  # unit tests only, as CI runs them
```

CI installs only `.[dev]` and runs the unit tests on every push. It skips tests marked `ml`, which run the pipeline's models.

```
src/twod23d/
  cli.py          the `2d23d run` command
  pipeline.py     settings, --tiny, and the stage runner
  device.py       the one place that picks CPU or GPU
  stages/         one module per stage, in run order in stages/__init__.py
  glb.py          a minimal GLB writer for the export stage
  video.py        reading ingested frames and writing debug videos
tests/            unit tests; ones marked `ml` run models
viewer/           the three.js/WebXR viewer, one static page
samples/          the sample clip and its license
```

To add a stage, write a function that takes the context and an output folder, reads earlier stages' folders, writes its files and a `debug.mp4`, and add it to `STAGES`.

## Licenses

This project is under the [Apache License 2.0](LICENSE). Its dependencies:

| Dependency | Used for | License |
|---|---|---|
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv-python) | video and images | Apache-2.0; its wheels bundle FFmpeg (LGPL-2.1) and Qt (LGPL-3.0) |
| [RF-DETR](https://github.com/roboflow/rf-detr) | finding people (code and the nano to large weights) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) (via RF-DETR) | running models | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) (via RF-DETR) | model building blocks | Apache-2.0 |
| [trackers](https://github.com/roboflow/trackers) | OC-SORT tracking | Apache-2.0 |
| [supervision](https://github.com/roboflow/supervision) (via RF-DETR and trackers) | detection results | MIT |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) | 3D bodies | SAM License, see below |
| [MHR](https://github.com/facebookresearch/MHR) | body model | Apache-2.0 |
| [MoGe-2](https://github.com/microsoft/MoGe) | camera field of view and the scene's shape (code and weights) | MIT |
| PyTorch Lightning, timm, yacs, huggingface_hub; roma, omegaconf, scipy; einops, braceexpand, wandb, utils3d-moge | libraries SAM 3D Body's and MoGe's code needs | Apache-2.0; BSD-3-Clause; MIT |
| [three.js](https://github.com/mrdoob/three.js) | viewer, loaded from jsDelivr | MIT |

**SAM 3D Body** code and weights are under Meta's [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE), which is not a standard open-source license. Read its terms before you use or redistribute it. The weights are gated on Hugging Face, so request access there and download them yourself. This repo never includes SAM 3D Body's code or weights: the pipeline downloads them from Meta on first use.

Bodies use the MHR body model rather than SMPL or SMPL-X, which are non-commercial. Ultralytics YOLO (AGPL-3.0) is avoided.

Model weights and video clips are only committed with their license next to them, like [the sample clip's](samples/pickleball.LICENSE.txt).
