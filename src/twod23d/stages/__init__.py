"""The v0.1 pipeline in order. Each stage is a module with run(ctx, out); see its docstring for its files."""

from twod23d.stages import bodies, check, colors, export, floor, ingest, scene, track

STAGES = {
    "ingest": ingest.run,  # video to frames
    "track": track.run,  # people boxes and IDs (roadmap step 2)
    "bodies": bodies.run,  # a 3D body per person per frame (step 3)
    "floor": floor.run,  # one flat floor with everyone standing on it (step 4)
    "scene": scene.run,  # the place without its people, as a mesh painted with the video (step 8)
    "check": check.run,  # drop or smooth floating and sinking frames (step 5)
    "colors": colors.run,  # each person's colors from the video (step 9)
    "export": export.run,  # the GLB (step 6)
}
