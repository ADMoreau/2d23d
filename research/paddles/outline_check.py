"""Tracked mask (red outline) vs the held-out detector mask (green outline), zoomed on the paddle."""
import json, sys
from pathlib import Path
import cv2, numpy as np
WORK, D = Path("../work-smooth"), Path("run")
frames = sorted((WORK / "ingest" / "frames").glob("*.jpg"))
recs = {(r["frame"], r["track_id"]): r for r in json.loads((D / "results.json").read_text())}
picks = json.loads(sys.argv[1])
tiles = []
for t, fr in picks.items():
    t = int(t); p = D / "track" / f"{t}.npz"; f = np.load(p if p.exists() else D / "track" / f"{t}.partial.npz"); scale = float(f["scale"])
    for i in fr:
        o = f["origin"][i]; mask = np.unpackbits(f["masks"][i], axis=-1).astype(bool)
        rec = recs[(i, t)]; s = scale / rec["scale"]
        M = np.array([[s, 0, (rec["origin"][0] - o[0]) * scale], [0, s, (rec["origin"][1] - o[1]) * scale]])
        truth = cv2.warpAffine(cv2.imread(str(D / "crops" / f"{t}_{i:06d}_owlmask.png"), 0), M, (640, 640), flags=cv2.INTER_NEAREST) > 127
        M2 = np.array([[scale, 0, -o[0] * scale], [0, scale, -o[1] * scale]])
        im = cv2.warpAffine(cv2.imread(str(frames[i])), M2, (640, 640), flags=cv2.INTER_CUBIC)
        for m, c in [(truth, (0, 255, 0)), (mask, (0, 0, 255))]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, c, 1)
        ys, xs = np.nonzero(mask | truth)
        cx, cy = (int(xs.mean()), int(ys.mean())) if len(xs) else (320, 320)
        x0, y0 = np.clip([cx - 80, cy - 80], 0, 480)
        tile = cv2.resize(im[y0:y0 + 160, x0:x0 + 160], (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, f"p{t} {i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(D / "track" / "outlines.jpg"), np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]))
