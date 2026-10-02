"""The command line. One command, `2d23d run VIDEO`: video in, GLB out."""

import argparse
import logging
import shutil
from pathlib import Path

from twod23d.device import pick_device
from twod23d.pipeline import TINY, Context, Settings, run_stages
from twod23d.stages import STAGES

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="2d23d", description="Turn fixed-camera video of people playing a sport into an animated 3D scene."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="video in, GLB out")
    run.add_argument("video", type=Path)
    run.add_argument("-o", "--out", type=Path, help="GLB to write (default: NAME.glb, where NAME is the video's name)")
    run.add_argument("--work", type=Path, help="work folder, one subfolder per stage (default: work/NAME)")
    run.add_argument(
        "--tiny",
        action="store_true",
        help=f"first {TINY.max_seconds:g} s at {TINY.max_side} px, quick on a CPU; adds -tiny to NAME",
    )
    run.add_argument(
        "--force",
        nargs="?",
        const=next(iter(STAGES)),
        choices=list(STAGES),
        metavar="STAGE",
        help=f"rerun STAGE and every stage after it, or all stages if no STAGE is given ({', '.join(STAGES)})",
    )
    run.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda or cuda:N")
    args = parser.parse_args(argv)

    # Our own progress at INFO; the libraries we use only warn
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("twod23d").setLevel(logging.INFO)
    name = args.video.stem + ("-tiny" if args.tiny else "")
    ctx = Context(
        video=args.video,
        work=args.work or Path("work") / name,
        settings=TINY if args.tiny else Settings(),
        device=pick_device(args.device),
    )
    log.info("work folder %s, device %s", ctx.work, ctx.device)
    run_stages(STAGES, ctx, force=args.force)
    out = args.out or Path(f"{name}.glb")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ctx.work / "export" / "scene.glb", out)
    log.info("wrote %s", out)
