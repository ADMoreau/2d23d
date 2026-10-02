import shutil
from pathlib import Path

import pytest

from twod23d.pipeline import Context, Settings, run_stages


def fake_stages(calls: list[str]) -> dict:
    def stage(name):
        def run(ctx, out):
            calls.append(name)
            (out / "debug.mp4").touch()

        return run

    return {name: stage(name) for name in ("a", "b", "c")}


def context(work: Path) -> Context:
    return Context(video=Path("clip.mp4"), work=work, settings=Settings(), device="cpu")


def test_skips_existing_output_unless_forced(tmp_path):
    calls = []
    stages, ctx = fake_stages(calls), context(tmp_path)
    run_stages(stages, ctx)
    run_stages(stages, ctx)
    assert calls == ["a", "b", "c"]
    run_stages(stages, ctx, force="b")
    assert calls == ["a", "b", "c", "b", "c"]


def test_missing_output_reruns_later_stages(tmp_path):
    calls = []
    stages, ctx = fake_stages(calls), context(tmp_path)
    run_stages(stages, ctx)
    shutil.rmtree(tmp_path / "b")
    run_stages(stages, ctx)
    assert calls == ["a", "b", "c", "b", "c"]


def test_failed_stage_leaves_no_output(tmp_path):
    def fail(ctx, out):
        (out / "debug.mp4").touch()
        raise ValueError("boom")

    with pytest.raises(ValueError):
        run_stages({"a": fail}, context(tmp_path))
    assert not (tmp_path / "a").exists()


def test_stage_must_write_debug_video(tmp_path):
    with pytest.raises(RuntimeError, match="debug.mp4"):
        run_stages({"a": lambda ctx, out: None}, context(tmp_path))
    assert not (tmp_path / "a").exists()
