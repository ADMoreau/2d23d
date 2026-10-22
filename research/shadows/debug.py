"""One frame: observed darkening, the people mask, and shadows from a few test lights (lights.py's machinery)."""
import sys
sys.argv = ["lights.py", "../work-smooth", "../setting/run", "../../../samples/pickleball.mp4", "run"]
src = open("lights.py").read().split("def fit(")[0]
src = src.replace("frames = np.arange(STEP // 2, frames_all.max() + 1, STEP)", "frames = np.array([400])")
exec(src)
tests = [np.array([-1.5, ceiling, 5.0]), np.array([0.0, ceiling, -2.0]), np.array([0.0, ceiling, -6.0]), np.array([3.0, ceiling, 0.0])]
to = lambda a: cv2.applyColorMap(np.clip(a / 0.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 400); _, img = cap.read()
small = cv2.resize(img, (W, H))
panels = [small, to(np.where(keep_px[0], obs[0], 0))]
for L in tests:
    sh = shadows(L)[0]
    p = small.copy(); p[sh > 0] = (0.4 * p[sh > 0] + [0, 0, 150]).astype(np.uint8)
    cv2.putText(p, f"x{L[0]:.0f} z{L[2]:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    panels.append(p)
cv2.imwrite("run/debug.jpg", cv2.resize(np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, 6, 2)]), None, fx=2, fy=2))
