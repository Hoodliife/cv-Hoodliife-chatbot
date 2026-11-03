"""Digitize the Christmas wreath image into a PES embroidery file.

Every green element (dark, mid and light leaves, pine needles, dots) is
stitched with ONE green thread; the bow and berries use one red thread.

Usage: python3 digitize.py <input-image> <out-dir> [size_mm ...]
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pyembroidery as pe

GREEN = pe.EmbThread()
GREEN.set_color(0x2E, 0x7D, 0x32)
GREEN.description = "Green (all greenery)"
GREEN.catalog_number = "Brother 534 / Christmas Green"
RED = pe.EmbThread()
RED.set_color(0xC6, 0x28, 0x28)
RED.description = "Red (bow + berries)"
RED.catalog_number = "Brother 800 / Red"

ROW_MM = 0.40       # fill row spacing (density)
MAX_ST_MM = 3.5     # longest stitch in a fill row
RUN_MM = 1.8        # running-stitch length
TRIM_MM = 6.0       # travel longer than this becomes a trim + jump
JUMP_MM = 1.6       # shorter travel is stitched; between the two = jump (no trim)


def fill_small_holes(m, max_area):
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = m.copy()
    if hier is not None:
        for c, h in zip(cs, hier[0]):
            if h[3] >= 0 and cv2.contourArea(c) < max_area:
                cv2.drawContours(out, [c], -1, 1, -1)
    return out


def masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    ink = ((S > 45) & (V < 240)) | (V < 110)
    red = (ink & ((H < 12) | (H > 160)) & (S > 70)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    red = fill_small_holes(red, 120)          # berry highlights / dark eyes
    ell = lambda d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    greenish = (H > 25) & (H < 100)
    solid = (ink & ~red.astype(bool)).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, ell(3))
    # drop tiny solid fragments: they are bits of pine needles
    n, lab, st, _ = cv2.connectedComponentsWithStats(solid, 8)
    frag = np.zeros_like(solid)
    for i in range(1, n):
        if st[i, 4] < 45:
            frag[lab == i] = 1
            solid[lab == i] = 0
    solid = fill_small_holes(solid, 25)
    # pine sprigs: faint green texture, merged into regions
    faint = (greenish & (S > 14) & (V < 238)) | (greenish & ink & ~red.astype(bool)) | frag.astype(bool)
    faint &= cv2.dilate(solid, np.ones((3, 3), np.uint8)) == 0
    faint &= cv2.dilate(red, ell(5)) == 0
    sprig = cv2.morphologyEx(faint.astype(np.uint8), cv2.MORPH_CLOSE, ell(9))
    sprig = cv2.morphologyEx(sprig, cv2.MORPH_OPEN, ell(3))
    sprig = fill_small_holes(sprig, 10_000)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sprig, 8)
    for i in range(1, n):
        if st[i, 4] < 350:
            sprig[lab == i] = 0
    shade = np.zeros(V.shape, np.uint8)
    shade[(solid > 0) & (V < 105)] = 1
    shade[(solid > 0) & (V >= 105) & (V < 160)] = 2
    shade[(solid > 0) & (V >= 160)] = 3
    return solid, red, shade, sprig


class Builder:
    def __init__(self, scale):
        self.s = scale            # stitch units (0.1 mm) per pixel
        self.p = pe.EmbPattern()
        self.last = None

    def mm(self, v):
        return v * 10.0 / self.s  # mm -> pixels

    def goto(self, x, y):
        X, Y = x * self.s, y * self.s
        if self.last is not None:
            d = math.dist(self.last, (X, Y))
            if d > JUMP_MM * 10:
                self.tie()
                if d > TRIM_MM * 10:
                    self.p.add_stitch_absolute(pe.TRIM, *self.last)
                self.p.add_stitch_absolute(pe.JUMP, X, Y)
                self.last = (X, Y)
                self.stitch(x, y)
                self.tie()
                return
        self.stitch(x, y)

    def tie(self):
        if self.last is None:
            return
        X, Y = self.last
        for dx in (3, -3, 0):
            self.p.add_stitch_absolute(pe.STITCH, X + dx, Y)

    def stitch(self, x, y):
        X, Y = x * self.s, y * self.s
        if self.last is not None:
            d = math.dist(self.last, (X, Y))
            n = int(d // (MAX_ST_MM * 10))
            for i in range(1, n + 1):
                t = i / (n + 1)
                self.p.add_stitch_absolute(pe.STITCH, self.last[0] + (X - self.last[0]) * t,
                                           self.last[1] + (Y - self.last[1]) * t)
        self.p.add_stitch_absolute(pe.STITCH, X, Y)
        self.last = (X, Y)

    def run(self, pts, closed=False):
        """Running stitch along a polyline (pixels), resampled to RUN_MM."""
        if closed:
            pts = list(pts) + [pts[0]]
        step = self.mm(RUN_MM)
        out = [pts[0]]
        acc = 0.0
        for a, b in zip(pts, pts[1:]):
            acc += math.dist(a, b)
            if acc >= step:
                out.append(b)
                acc = 0.0
        out.append(pts[-1])
        self.goto(*out[0])
        for q in out[1:]:
            self.stitch(*q)

    def color(self, thread):
        if self.p.stitches:
            self.tie()
            self.p.add_stitch_absolute(pe.TRIM, *self.last)
            self.p.add_stitch_absolute(pe.COLOR_BREAK, *self.last)
        self.p.add_thread(thread)


def orientation(comp):
    ys, xs = np.nonzero(comp)
    if len(xs) < 5:
        return 0.0
    c = np.cov(np.vstack([xs, ys]))
    w, v = np.linalg.eigh(c)
    major = v[:, np.argmax(w)]
    return math.atan2(major[1], major[0])


def fill(b, comp, x0, y0, angle, underlay=True):
    """Satin-like tatami fill: rows perpendicular to `angle` (long axis)."""
    if underlay:
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [c for c in cs if len(c) > 12]
        er = cv2.erode(comp, np.ones((3, 3), np.uint8))
        cs2, _ = cv2.findContours(er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs2 if cs2 else cs:
            if len(c) > 12:
                b.run([(p[0][0] + x0, p[0][1] + y0) for p in c], closed=True)
    h, w = comp.shape
    # rotate so the long axis is vertical -> horizontal rows cross the leaf
    rot = math.degrees(angle) - 90
    S = 4  # supersample for smooth edges
    big = cv2.resize(comp * 255, (w * S, h * S), interpolation=cv2.INTER_LINEAR)
    diag = int(math.hypot(w, h) * S) + 4
    M = cv2.getRotationMatrix2D((w * S / 2, h * S / 2), rot, 1.0)
    M[0, 2] += diag / 2 - w * S / 2
    M[1, 2] += diag / 2 - h * S / 2
    r = cv2.warpAffine(big, M, (diag, diag)) > 127
    Mi = cv2.invertAffineTransform(M)
    step = b.mm(ROW_MM) * S
    rows = []
    y = 0.0
    while y < diag:
        row = r[int(y)]
        segs = []
        if row.any():
            d = np.diff(np.concatenate([[0], row.astype(int), [0]]))
            st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
            segs = [(s, e, y) for s, e in zip(st, en) if e - s >= 1]
        rows.append(segs)
        y += step
    used = [[False] * len(rw) for rw in rows]

    def back(px, py):
        X = Mi[0, 0] * px + Mi[0, 1] * py + Mi[0, 2]
        Y = Mi[1, 0] * px + Mi[1, 1] * py + Mi[1, 2]
        return X / S + x0, Y / S + y0

    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if used[i][j]:
                continue
            # build a boustrophedon chain down overlapping rows
            chain, ci, cj = [], i, j
            while True:
                used[ci][cj] = True
                chain.append(rows[ci][cj])
                s, e, _ = rows[ci][cj]
                nxt = None
                if ci + 1 < len(rows):
                    for k, (s2, e2, _) in enumerate(rows[ci + 1]):
                        if not used[ci + 1][k] and s2 <= e and e2 >= s:
                            nxt = k
                            break
                if nxt is None:
                    break
                ci, cj = ci + 1, nxt
            for n, (s, e, yy) in enumerate(chain):
                a, c = (s, e) if n % 2 == 0 else (e, s)
                pa, pc = back(a, yy), back(c, yy)
                if n == 0:
                    b.goto(*pa)
                else:
                    b.stitch(*pa)
                b.stitch(*pc)


def components(mask, min_px=4):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_px:
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        out.append((x, y, comp, cent[i]))
    return out


def order(items):
    """Nearest-neighbour ordering to cut travel."""
    if not items:
        return []
    rest = items[:]
    rest.sort(key=lambda t: (t[3][1], t[3][0]))
    seq = [rest.pop(0)]
    while rest:
        c = seq[-1][3]
        k = min(range(len(rest)), key=lambda i: math.dist(c, rest[i][3]))
        seq.append(rest.pop(k))
    return seq


def concentric(b, comp, x0, y0):
    """Contour (echo) fill for thin curving strokes like the ribbon."""
    cur = comp.copy()
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    it = max(1, int(round(b.mm(ROW_MM))))
    while cur.any():
        cs, _ = cv2.findContours(cur, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) < 6:
                continue
            pts = [(p[0][0] + x0, p[0][1] + y0) for p in c]
            if b.last is not None:  # start at point nearest the needle
                L = (b.last[0] / b.s, b.last[1] / b.s)
                k = min(range(len(pts)), key=lambda i: math.dist(L, pts[i]))
                pts = pts[k:] + pts[:k]
            step = b.mm(0.9)
            out, acc = [pts[0]], 0.0
            for a, q in zip(pts, pts[1:] + [pts[0]]):
                acc += math.dist(a, q)
                if acc >= step:
                    out.append(q)
                    acc = 0.0
            out.append(pts[0])
            b.goto(*out[0])
            for q in out[1:]:
                b.stitch(*q)
        cur = cv2.erode(cur, ker, iterations=it)


def spine_of(region):
    """Longest path through the region skeleton = the sprig's stem."""
    from skimage.morphology import skeletonize
    sk = skeletonize(region > 0)
    pts = set(zip(*np.nonzero(sk)))
    if len(pts) < 5:
        return None
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and (p[0] + dy, p[1] + dx) in pts]

    def bfs(s):
        prev, q, far = {s: None}, [s], s
        for p in q:
            far = p
            for n in nb(p):
                if n not in prev:
                    prev[n] = p
                    q.append(n)
        return far, prev
    a, _ = bfs(next(iter(pts)))
    b, prev = bfs(a)
    path = []
    while b is not None:
        path.append((float(b[1]), float(b[0])))
        b = prev[b]
    # smooth
    arr = np.array(path)
    if len(arr) > 9:
        k = np.ones(9) / 9
        arr = np.column_stack([np.convolve(np.pad(arr[:, i], 4, mode="edge"), k, "valid") for i in (0, 1)])
    return arr


def sprig(b, region, x0, y0):
    """Pine sprig: stem running stitch with needles fanning out (out & back)."""
    sp = spine_of(region)
    if sp is None or len(sp) < 8:
        return
    # orient so needles point toward the tip: tip = thinner end
    dt = cv2.distanceTransform(region, cv2.DIST_L2, 5).astype(float)
    w0 = dt[int(sp[0][1]), int(sp[0][0])]
    w1 = dt[int(sp[-1][1]), int(sp[-1][0])]
    if w0 < w1:
        sp = sp[::-1]
    h, w = region.shape
    step = b.mm(0.9)
    acc, side = step, 1
    ang = math.radians(38)
    b.goto(sp[0][0] + x0, sp[0][1] + y0)
    for i in range(1, len(sp)):
        p, q = sp[i - 1], sp[i]
        acc += math.dist(p, q)
        if acc < step:
            continue
        acc = 0.0
        b.stitch(q[0] + x0, q[1] + y0)
        j = min(len(sp) - 1, i + 3)
        tx, ty = sp[j][0] - sp[max(0, i - 3)][0], sp[j][1] - sp[max(0, i - 3)][1]
        tl = math.hypot(tx, ty) or 1
        tx, ty = tx / tl, ty / tl
        for s in (side, -side):
            ca, sa = math.cos(ang * s), math.sin(ang * s)
            dx, dy = tx * ca - ty * sa, tx * sa + ty * ca
            L = 0.0
            while True:
                nx, ny = q[0] + dx * (L + 1), q[1] + dy * (L + 1)
                if not (0 <= nx < w and 0 <= ny < h) or region[int(ny), int(nx)] == 0:
                    break
                L += 1
            L = min(L * 0.92, b.mm(7.0))
            if L < b.mm(0.8):
                continue
            tip = (q[0] + dx * L + x0, q[1] + dy * L + y0)
            b.stitch(*tip)
            b.stitch(q[0] + x0, q[1] + y0)
        side = -side
    # run back down the stem to lock the needle roots
    for k in range(len(sp) - 1, -1, -max(1, int(b.mm(RUN_MM)))):
        b.stitch(sp[k][0] + x0, sp[k][1] + y0)


def build(img, size_mm):
    green, red, shade, spr = masks(img)
    ys, xs = np.nonzero(green | red | spr)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    scale = size_mm * 10.0 / max(x1 - x0, y1 - y0)
    b = Builder(scale)

    # ---------- GREEN (single thread for every green element) ----------
    b.color(GREEN)
    # pine sprigs first: they sit underneath the leaves
    for x, y, comp, c in order(components(spr, 300)):
        sprig(b, comp, x, y)
    leaves = []
    for sv in (1, 2, 3):   # shades only steer stitch angle; same thread
        leaves += components((shade == sv).astype(np.uint8), 12)
    for x, y, comp, c in order(leaves):
        fill(b, comp, x, y, orientation(comp), underlay=int(comp.sum()) > 80)

    # ---------- RED ----------
    b.color(RED)
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    berries = cv2.morphologyEx(red, cv2.MORPH_OPEN, ell)
    ribbon = red & (cv2.dilate(berries, np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
    ribbon = cv2.morphologyEx(ribbon, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for x, y, comp, c in order(components(berries, 20)):
        fill(b, comp, x, y, orientation(comp) + 0.6, underlay=True)
    for x, y, comp, c in order(components(ribbon, 15)):
        concentric(b, comp, x, y)

    b.tie()
    b.p.add_stitch_absolute(pe.TRIM, *b.last)
    b.p.end()
    # centre design on (0,0) as machines expect
    ext = b.p.extents()
    b.p.translate(-(ext[0] + ext[2]) / 2, -(ext[1] + ext[3]) / 2)
    return b.p


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    sizes = [float(s) for s in sys.argv[3:]] or [100.0]
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(src)
    for size in sizes:
        p = build(img, size)
        name = f"christmas-wreath-{int(size)}mm"
        pe.write_pes(p, str(out / f"{name}.pes"), {"version": 6})
        pe.write_png(p, str(out / f"{name}-preview.png"), {"background": "#ffffff", "linewidth": 2})
        ext = p.extents()
        n = sum(1 for s in p.stitches if s[2] == pe.STITCH)
        print(f"{name}: {(ext[2]-ext[0])/10:.1f} x {(ext[3]-ext[1])/10:.1f} mm, "
              f"{n} stitches, {len(p.threadlist)} threads, {p.count_stitch_commands(pe.TRIM)} trims")


if __name__ == "__main__":
    main()
