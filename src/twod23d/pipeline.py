"""The stage runner, plus the settings and context every stage gets.

Stages talk only through files. Each stage writes one folder in the work folder,
including a debug.mp4, and reads the folders of earlier stages.
"""

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Every knob a stage may read. The defaults are for real runs."""

    max_seconds: float | None = None  # use only the start of the video
    max_side: int | None = None  # downscale frames so the longer side is at most this many pixels
    detector: str = "medium"  # RF-DETR size: nano, small, medium or large
    fov_model: str = "vitl"  # MoGe-2 size, for the camera's field of view and the scene: vits, vitb or vitl
    body_every: int = 1  # run SAM 3D Body on every Nth frame


# --tiny: a few seconds of small frames and small models, quick on a CPU. The README quickstart uses it.
TINY = Settings(max_seconds=3.0, max_side=640, detector="nano", fov_model="vits", body_every=10)


@dataclass(frozen=True)
class Context:
    video: Path  # the input video; only the ingest stage reads it
    work: Path  # the work folder, one subfolder per stage
    settings: Settings
    device: str  # from device.pick_device()


# A stage writes all of its output, including debug.mp4, into the folder it is given.
Stage = Callable[[Context, Path], None]


def run_stages(stages: dict[str, Stage], ctx: Context, force: str | None = None) -> None:
    """Run the stages in order, skipping each one whose output folder already exists.

    `force` names a stage to rerun anyway. Once a stage runs, every later stage reruns
    too, because its input changed. A stage writes into a temporary folder that is
    renamed when it succeeds, so a failed stage leaves no output and runs again next time.
    """
    if force is not None and force not in stages:
        raise ValueError(f"unknown stage {force!r}")
    ctx.work.mkdir(parents=True, exist_ok=True)
    rerun = False
    for name, stage in stages.items():
        out = ctx.work / name
        rerun = rerun or name == force or not out.exists()
        if not rerun:
            log.info("%s: skipped, output exists", name)
            continue
        tmp = ctx.work / f"{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        log.info("%s: running", name)
        start = time.perf_counter()
        try:
            stage(ctx, tmp)
        except Exception:
            log.error("%s: failed, partial output left in %s", name, tmp)
            raise
        if not (tmp / "debug.mp4").is_file():
            raise RuntimeError(f"stage {name} wrote no debug.mp4")
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        log.info("%s: done in %.1f s", name, time.perf_counter() - start)
