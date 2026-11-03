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

SATIN_MM = 0.32     # satin density: distance between penetrations on one side (premium)
TATAMI_MM = 0.38    # tatami row spacing for areas too wide for satin
SATIN_MAX_MM = 7.0  # columns wider than this switch to split tatami
PULL_MM = 0.15      # pull compensation added to each side of a column
UNDER_INSET_MM = 0.35
MAX_ST_MM = 3.5     # longest running / tatami stitch
MIN_ST_MM = 0.25    # drop penetrations closer than this (thread-break safety)
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
    # Leaves are two-toned (split along the midrib). Two smoothed tones give
    # one clean satin column per half-leaf; the thread is the same green.
    sm = cv2.bilateralFilter(img, 7, 40, 5)
    Vs = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)[..., 2].astype(int)
    shade = np.where(solid > 0, np.where(Vs < 128, 1, 2), 0).astype(np.uint8)
    shade = cv2.medianBlur(shade, 5) * (solid > 0)
    shade[(solid > 0) & (shade == 0)] = 2
    for _ in range(3):  # absorb specks into the neighbouring tone
        changed = False
        for sv in (1, 2):
            n, lab, st, _ = cv2.connectedComponentsWithStats((shade == sv).astype(np.uint8), 8)
            for i in range(1, n):
                if st[i, 4] < 70:
                    x, y, w, h = st[i, :4]
                    ring = cv2.dilate((lab[max(0,y-2):y+h+2, max(0,x-2):x+w+2] == i).astype(np.uint8), np.ones((3, 3), np.uint8))
                    sub = shade[max(0,y-2):y+h+2, max(0,x-2):x+w+2]
                    if ((sub == 3 - sv) & (ring > 0)).any():
                        sub[lab[max(0,y-2):y+h+2, max(0,x-2):x+w+2] == i] = 3 - sv
                        changed = True
        if not changed:
            break
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
        for dx, dy in ((4, 2), (-3, -2), (0, 0)):
            self.p.add_stitch_absolute(pe.STITCH, X + dx, Y + dy)

    def stitch(self, x, y, maxlen=MAX_ST_MM, force=False):
        X, Y = x * self.s, y * self.s
        if self.last is not None:
            d = math.dist(self.last, (X, Y))
            if d < MIN_ST_MM * 10 and not force:
                return
            n = int(d // (maxlen * 10))
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


def _rows(b, comp, angle, spacing, S=4):
    """Rasterise a smoothed, rotated copy of comp; return scan rows + back-map."""
    h, w = comp.shape
    pad = 4
    comp = np.pad(comp, pad)
    h, w = comp.shape
    big = cv2.resize(comp.astype(np.float32), (w * S, h * S), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), S * 0.7)          # smooth pixel stair-steps
    diag = int(math.hypot(w, h) * S) + 4
    M = cv2.getRotationMatrix2D((w * S / 2, h * S / 2), math.degrees(angle) - 90, 1.0)
    M[0, 2] += diag / 2 - w * S / 2
    M[1, 2] += diag / 2 - h * S / 2
    r = cv2.warpAffine(big, M, (diag, diag)) > 0.5
    Mi = cv2.invertAffineTransform(M)
    step = b.mm(spacing) * S
    rows, y = [], 0.0
    while y < diag:
        row = r[int(y)]
        segs = []
        if row.any():
            d = np.diff(np.concatenate([[0], row.astype(int), [0]]))
            st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
            segs = [(float(a), float(c), y) for a, c in zip(st, en) if c - a >= 1]
        rows.append(segs)
        y += step

    def back(px, py):
        X = Mi[0, 0] * px + Mi[0, 1] * py + Mi[0, 2]
        Y = Mi[1, 0] * px + Mi[1, 1] * py + Mi[1, 2]
        return X / S - pad, Y / S - pad
    return rows, back, S


def _chains(rows):
    """Group scan segments into continuous columns (overlapping consecutive rows)."""
    used = [[False] * len(rw) for rw in rows]
    out = []
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if used[i][j]:
                continue
            chain, ci, cj = [], i, j
            while True:
                used[ci][cj] = True
                chain.append(rows[ci][cj])
                s0, e0, _ = rows[ci][cj]
                nxt = None
                if ci + 1 < len(rows):
                    for k, (s2, e2, _) in enumerate(rows[ci + 1]):
                        if not used[ci + 1][k] and s2 <= e0 and e2 >= s0:
                            nxt = k
                            break
                if nxt is None:
                    break
                ci, cj = ci + 1, nxt
            out.append(chain)
    return out


def fill(b, comp, x0, y0, angle, tilt=0.0, underlay=True):
    """High-density satin across the shape (tilted by `tilt`), with
    edge-walk + zigzag underlay and pull compensation. Very wide rows fall
    back to staggered tatami so no stitch is longer than SATIN_MAX_MM."""
    ang = angle + tilt
    off = lambda pts: [(p[0] + x0, p[1] + y0) for p in pts]
    if underlay and comp.sum() > b.mm(1.2) ** 2:
        k = max(1, int(round(b.mm(UNDER_INSET_MM))))
        er = cv2.erode(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        cs, _ = cv2.findContours(er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) > 8:
                b.run(off([(p[0][0], p[0][1]) for p in c]), closed=True)
        # zigzag underlay perpendicular to the top stitches' travel
        rows, back, S = _rows(b, er, ang, 1.4)
        for ch in _chains(rows):
            if len(ch) < 2:
                continue
            for n, (sa, ea, yy) in enumerate(ch):
                p = back(sa if n % 2 == 0 else ea, yy)
                (b.goto if n == 0 else b.stitch)(p[0] + x0, p[1] + y0)
    rows, back, S = _rows(b, comp, ang, SATIN_MM / 2)
    pc = b.mm(PULL_MM) * S
    wide = b.mm(SATIN_MAX_MM) * S
    tat = b.mm(TATAMI_MM) * S
    for ch in _chains(rows):
        for n, (sa, ea, yy) in enumerate(ch):
            sa, ea = sa - pc, ea + pc
            if ea - sa > wide:
                # tatami row: split into ~3mm stitches with a 1/3 brick offset
                L = b.mm(3.0) * S
                ph = (int(yy / tat) % 3) * L / 3
                xs = [sa] + list(np.arange(sa + ph + L, ea - L / 3, L)) + [ea]
                if n % 2:
                    xs = xs[::-1]
                for k, xv in enumerate(xs):
                    p = back(xv, yy)
                    (b.goto if (n == 0 and k == 0) else b.stitch)(p[0] + x0, p[1] + y0)
                continue
            p = back(sa if n % 2 == 0 else ea, yy)   # satin zig-zag
            if n == 0:
                b.goto(p[0] + x0, p[1] + y0)
            else:
                b.stitch(p[0] + x0, p[1] + y0, maxlen=SATIN_MAX_MM + 1)


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


def _branches(sk):
    """Split a skeleton into pixel paths between endpoints / junctions."""
    pts = set(zip(*np.nonzero(sk)))
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and (p[0] + dy, p[1] + dx) in pts]
    deg = {p: len(nb(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    seen, paths = set(), []
    for n in nodes:
        for m in nb(n):
            if (n, m) in seen:
                continue
            path, prev, cur = [n], n, m
            while True:
                seen.add((prev, cur)); seen.add((cur, prev))
                path.append(cur)
                if cur in nodes:
                    break
                nx = [q for q in nb(cur) if q != prev and (cur, q) not in seen]
                if not nx:
                    break
                prev, cur = cur, nx[0]
            paths.append(path)
    left = pts - {p for pa in paths for p in pa}
    if left:  # closed loop with no node
        start = next(iter(left))
        path, prev, cur = [start], None, start
        while True:
            nx = [q for q in nb(cur) if q != prev and q in left and q not in path]
            if not nx:
                break
            prev, cur = cur, nx[0]
            path.append(cur)
        path.append(start)
        paths.append(path)
    return paths


def satin_stroke(b, comp, x0, y0):
    """Satin column that follows a curving stroke (the ribbon): centre-run
    underlay out, dense satin back, width taken from the stroke itself."""
    from skimage.morphology import skeletonize
    S = 4
    pad = 3
    comp = np.pad(comp, pad)
    big = cv2.resize(comp.astype(np.float32), (comp.shape[1] * S, comp.shape[0] * S),
                     interpolation=cv2.INTER_CUBIC)
    big = (cv2.GaussianBlur(big, (0, 0), S * 0.8) > 0.5).astype(np.uint8)
    dt = cv2.distanceTransform(big, cv2.DIST_L2, 5)
    sk = skeletonize(big > 0)
    paths = _branches(sk)
    segs = []
    for pa in paths:
        arr = np.array([(q[1], q[0]) for q in pa], float)
        wmax = max(dt[q] for q in pa)
        # drop short spurs created by blobby skeleton ends
        L = np.sum(np.hypot(*np.diff(arr, axis=0).T)) if len(arr) > 1 else 0
        if L < 1.6 * wmax and len(paths) > 1:
            continue
        if len(arr) > 7:
            k = np.ones(7) / 7
            arr = np.column_stack([np.convolve(np.pad(arr[:, i], 3, mode="edge"), k, "valid") for i in (0, 1)])
        segs.append(arr)
    pc = b.mm(PULL_MM) * S
    dpx = b.mm(SATIN_MM / 2) * S
    up = lambda x, y: (x / S - pad + x0, y / S - pad + y0)
    while segs:
        # nearest segment end to the needle
        if b.last is not None:
            L = (b.last[0] / b.s, b.last[1] / b.s)
            best = min([(math.dist(L, up(*sg[0])), i, False) for i, sg in enumerate(segs)] +
                       [(math.dist(L, up(*sg[-1])), i, True) for i, sg in enumerate(segs)])
            _, i, rev = best
        else:
            i, rev = 0, False
        arr = segs.pop(i)
        if rev:
            arr = arr[::-1]
        # resample at satin spacing
        d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(arr, axis=0).T))])
        if d[-1] < 2:
            continue
        t = np.arange(0, d[-1], dpx)
        P = np.column_stack([np.interp(t, d, arr[:, 0]), np.interp(t, d, arr[:, 1])])
        # centre-run underlay forward
        b.goto(*up(*P[0]))
        for q in P[::max(1, int(b.mm(1.5) * S / dpx))]:
            b.stitch(*up(*q))
        b.stitch(*up(*P[-1]))
        # satin back over it
        for n in range(len(P) - 1, -1, -1):
            j0, j1 = max(0, n - 4), min(len(P) - 1, n + 4)
            tx, ty = P[j1] - P[j0]
            tl = math.hypot(tx, ty) or 1
            nx, ny = -ty / tl, tx / tl
            hw = dt[min(big.shape[0] - 1, int(P[n][1])), min(big.shape[1] - 1, int(P[n][0]))] + pc
            hw = max(hw, b.mm(0.35) * S)
            sgn = 1 if n % 2 else -1
            b.stitch(*up(P[n][0] + nx * hw * sgn, P[n][1] + ny * hw * sgn), maxlen=SATIN_MAX_MM + 1)


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
    step = b.mm(0.38)                  # one needle every 0.38 mm, alternating sides
    acc, side = step, 1
    ang = math.radians(36)
    rng = np.random.default_rng(7)
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
        for s in (side,):
            ca, sa = math.cos(ang * s), math.sin(ang * s)
            dx, dy = tx * ca - ty * sa, tx * sa + ty * ca
            L = 0.0
            while True:
                nx, ny = q[0] + dx * (L + 1), q[1] + dy * (L + 1)
                if not (0 <= nx < w and 0 <= ny < h) or region[int(ny), int(nx)] == 0:
                    break
                L += 1
            L = min(L * rng.uniform(0.8, 1.0), b.mm(7.0))
            if L < b.mm(0.9):
                continue
            tip = (q[0] + dx * L + x0, q[1] + dy * L + y0)
            b.stitch(*tip, force=True)
            b.stitch(q[0] + x0, q[1] + y0, force=True)
        side = -side
    # bean-stitch stem: back down and up again (3 passes total)
    st = max(1, int(b.mm(1.2)))
    for k in list(range(len(sp) - 1, -1, -st)) + list(range(0, len(sp), st)) + list(range(len(sp) - 1, -1, -st)):
        b.stitch(sp[k][0] + x0, sp[k][1] + y0)


def split_berries(mask):
    """Separate touching berries (circles) with a distance-transform watershed."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    pk = peak_local_max(dt, min_distance=7, threshold_abs=5, labels=mask)
    mk = np.zeros(mask.shape, np.int32)
    for i, (yy, xx) in enumerate(pk, 1):
        mk[yy, xx] = i
    lab = watershed(-dt, mk, mask=mask > 0)
    items = []
    for i in range(1, lab.max() + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 20:
            continue
        x0, y0 = xs.min(), ys.min()
        comp = (lab[y0:ys.max() + 1, x0:xs.max() + 1] == i).astype(np.uint8)
        # stitch back-to-front: berries lower in a cluster sit on top
        items.append((x0, y0, comp, (xs.mean(), ys.mean())))
    return items


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
    for sv in (1, 2):      # tones only steer stitch angle; same thread
        leaves += components((shade == sv).astype(np.uint8), 12)
    for x, y, comp, c in order(leaves):
        fill(b, comp, x, y, orientation(comp), tilt=math.radians(28))

    # ---------- RED ----------
    b.color(RED)
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    berries = cv2.morphologyEx(red, cv2.MORPH_OPEN, ell)
    ribbon = red & (cv2.dilate(berries, np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
    ribbon = cv2.morphologyEx(ribbon, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for x, y, comp, c in order(split_berries(berries)):
        fill(b, comp, x, y, math.radians(-50), tilt=0.0)
    for x, y, comp, c in order(components(ribbon, 15)):
        satin_stroke(b, comp, x, y)

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
        pe.write_png(p, str(out / f"{name}-preview.png"), {"background": "#ffffff", "linewidth": 1})
        ext = p.extents()
        n = sum(1 for s in p.stitches if s[2] == pe.STITCH)
        print(f"{name}: {(ext[2]-ext[0])/10:.1f} x {(ext[3]-ext[1])/10:.1f} mm, "
              f"{n} stitches, {len(p.threadlist)} threads, {p.count_stitch_commands(pe.TRIM)} trims")


if __name__ == "__main__":
    main()
