"""Thread-realistic render of a PES: each stitch drawn ~0.38mm wide with shading."""
import sys, cv2, numpy as np, pyembroidery as pe
f, out = sys.argv[1], sys.argv[2]
crop = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 and sys.argv[3] else None
PX = float(sys.argv[4]) if len(sys.argv) > 4 else 12  # pixels per mm
p = pe.read(f); x0, y0, x1, y1 = p.extents()
W, H = int((x1-x0)/10*PX)+20, int((y1-y0)/10*PX)+20
img = np.full((H, W, 3), 250, np.uint8)
cols = [t.color for t in p.threadlist]; ci = 0; last = None
tw = max(1, int(round(0.38*PX)))
for x, y, c in p.stitches:
    if c == pe.COLOR_CHANGE: ci += 1; last = None; continue
    if c != pe.STITCH: last = None; continue
    pt = (int((x-x0)/10*PX)+10, int((y-y0)/10*PX)+10)
    if last and last != pt:
        col = cols[ci]; r, g, b = (col >> 16) & 255, (col >> 8) & 255, col & 255
        dark = (int(b*0.6), int(g*0.6), int(r*0.6))
        cv2.line(img, last, pt, dark, tw+1, cv2.LINE_AA)
        cv2.line(img, last, pt, (b, g, r), max(1, tw-1), cv2.LINE_AA)
        hi = (min(255, int(b*1.3)+30), min(255, int(g*1.3)+30), min(255, int(r*1.3)+30))
        cv2.line(img, last, pt, hi, 1, cv2.LINE_AA)
    last = pt
if crop: img = img[crop[1]:crop[3], crop[0]:crop[2]]
cv2.imwrite(out, img)
print(img.shape)
