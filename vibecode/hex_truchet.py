#!/usr/bin/env python3
"""Hex Truchet loops: generate single-loop Truchet curves on a hexagonal
figure and export them as SVG with exact circular arcs.

Model (see project handoff doc):
  * Axial coords, neighbours RING[k]; edge k of a cell faces RING[k].
  * Tile 0 joins edges (0,1)(2,3)(4,5); tile 1 joins (1,2)(3,4)(5,0).
  * Outside the figure, boundary edges are joined in consecutive pairs
    ("caps") walking around the boundary, with a choosable phase.
  * Every connection (tile or cap) is a circular arc of radius side/2
    about the hex corner shared by the two edges, so the loop is a
    tangent-continuous chain of arcs.

Usage:
  python hex_truchet.py -r 6 --seed 3 -o loop.svg
  python hex_truchet.py -r 4 --count 5 --walls --grid
  python hex_truchet.py -r 6 --no-symmetry --seed 1
  python hex_truchet.py -r 6 --heightmap --width 0.35 --profile round

Search: the loop count never rises. Without symmetry it is planned (merge
three loops where they meet, else a two-flip push that makes them meet);
with symmetry it flips random orbits and restarts at dead ends.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
import math
import os
import random
import sys
from dataclasses import dataclass, field

RING = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]
SQ3 = math.sqrt(3.0)


def rot60(a, b):
    return (a - b, a)


def rot120(a, b):
    return rot60(*rot60(a, b))


# ---------------------------------------------------------------- geometry
def cell_center(a, b, side=1.0):
    """Centre of cell (a,b). RING[0] points along +x, RING[2] at 120 deg."""
    d = SQ3 * side  # centre spacing
    x = d * (a - 0.5 * b)
    y = d * (SQ3 / 2.0) * b
    return (x, y)


def corner(cell, k, side=1.0):
    """Corner of `cell` between edge k and edge k+1 (angle 60k+30)."""
    cx, cy = cell_center(*cell, side)
    ang = math.radians(60 * k + 30)
    return (cx + side * math.cos(ang), cy + side * math.sin(ang))


def edge_mid(cell, k, side=1.0):
    cx, cy = cell_center(*cell, side)
    ang = math.radians(60 * k)
    r = SQ3 * side / 2.0
    return (cx + r * math.cos(ang), cy + r * math.sin(ang))


def hex_corners(cell, side=1.0):
    return [corner(cell, k - 1, side) for k in range(6)]  # angles -30..270


# ------------------------------------------------------------------ figure
class Figure:
    """Hexagon-shaped figure of radius r (3r(r+1)+1 cells)."""

    def __init__(self, r: int, cap_phase: int = 0):
        self.r = r
        self.cap_phase = cap_phase & 1
        cells = []
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if abs(a) <= r and abs(b) <= r and abs(a - b) <= r:
                    cells.append((a, b))
        cells.sort(key=lambda c: (c[1], c[0]))
        self.cells = cells
        self.index = {c: i for i, c in enumerate(cells)}
        self.n = len(cells)
        n6 = 6 * self.n
        # port p = 6*i + k. Cross-edge partner (neighbour or cap).
        self.cross = [-1] * n6
        self.cap_corner = {}  # port -> boundary corner (cell, k) for caps
        for i, c in enumerate(cells):
            for k in range(6):
                nb = (c[0] + RING[k][0], c[1] + RING[k][1])
                j = self.index.get(nb)
                if j is not None:
                    self.cross[6 * i + k] = 6 * j + (k + 3) % 6
        self.boundary = self._walk_boundary()
        m = len(self.boundary)
        assert m % 2 == 0
        ph = self.cap_phase
        for t in range(0, m, 2):
            e1 = self.boundary[(t + ph) % m]
            e2 = self.boundary[(t + 1 + ph) % m]
            p1 = 6 * self.index[e1[0]] + e1[1]
            p2 = 6 * self.index[e2[0]] + e2[1]
            self.cross[p1] = p2
            self.cross[p2] = p1
            # shared corner = corner of e1's cell between e1.k and e1.k+1
            self.cap_corner[p1] = self.cap_corner[p2] = (e1[0], e1[1])
        assert all(x >= 0 for x in self.cross)

    def _is_boundary(self, c, k):
        nb = (c[0] + RING[k][0], c[1] + RING[k][1])
        return nb not in self.index

    def _walk_boundary(self):
        start = None
        for c in self.cells:
            for k in range(6):
                if self._is_boundary(c, k):
                    start = (c, k)
                    break
            if start:
                break
        seq = [start]
        cur = start
        while True:
            c, k = cur
            k1 = (k + 1) % 6
            if self._is_boundary(c, k1):
                nxt = (c, k1)  # convex corner, same cell
            else:
                n = (c[0] + RING[k1][0], c[1] + RING[k1][1])
                nxt = (n, (k - 1) % 6)  # concave corner
                assert self._is_boundary(*nxt)
            if nxt == start:
                break
            seq.append(nxt)
            cur = nxt
        nb = sum(self._is_boundary(c, k) for c in self.cells for k in range(6))
        assert len(seq) == nb, "boundary is not a single cycle"
        return seq

    # tile link: port -> port within the same cell
    @staticmethod
    def tile_partner(port, bit):
        i, k = divmod(port, 6)
        if bit == 0:
            k2 = k ^ 1  # (0,1)(2,3)(4,5)
        else:
            k2 = (k + 1) % 6 if k % 2 == 1 else (k - 1) % 6  # (1,2)(3,4)(5,0)
        return 6 * i + k2

    def count_loops(self, bits):
        n6 = 6 * self.n
        seen = bytearray(n6)
        loops = 0
        tp = self.tile_partner
        cross = self.cross
        for s in range(n6):
            if seen[s]:
                continue
            loops += 1
            p = s
            while not seen[p]:
                seen[p] = 1
                q = tp(p, bits[p // 6])
                seen[q] = 1
                p = cross[q]
        return loops

    def hex_arcs(self, bits):
        """For every hex, its three arcs as (loop, position along loop).

        Positions count tile arcs in order of travel round each loop.
        Returns (arcs, nloops), arcs[i] = [(loop, pos), (loop, pos), (loop, pos)].
        """
        n6 = 6 * self.n
        lab = [-1] * n6
        pos = [0] * n6
        tp = self.tile_partner
        cross = self.cross
        n = 0
        for s in range(n6):
            if lab[s] >= 0:
                continue
            p = s
            k = 0
            while lab[p] < 0:
                q = tp(p, bits[p // 6])
                lab[p] = lab[q] = n
                pos[p] = pos[q] = k
                k += 1
                p = cross[q]
            n += 1
        arcs = []
        for i in range(self.n):
            b = 6 * i + bits[i]  # ports 0,2,4 for tile 0; 1,3,5 for tile 1
            arcs.append([(lab[b], pos[b]), (lab[b + 2], pos[b + 2]),
                         (lab[b + 4], pos[b + 4])])
        return arcs, n

    @staticmethod
    def classify(arcs):
        """Split hexes into merges (3 loops) and doubles (2 loops).

        doubles: list of (hex, P, lo, hi, Q): touches loop P at positions
        lo < hi and loop Q once.
        """
        merges, doubles = [], []
        for i, ((l0, p0), (l1, p1), (l2, p2)) in enumerate(arcs):
            if l0 == l1 == l2:
                continue
            if l0 != l1 and l1 != l2 and l0 != l2:
                merges.append(i)
            elif l0 == l1:
                doubles.append((i, l0, min(p0, p1), max(p0, p1), l2))
            elif l0 == l2:
                doubles.append((i, l0, min(p0, p2), max(p0, p2), l1))
            else:
                doubles.append((i, l1, min(p1, p2), max(p1, p2), l0))
        return merges, doubles

    @staticmethod
    def push_pairs(doubles):
        """Pairs (h, g): flipping h (neutral) makes g touch three loops.

        h = (P, P, Q) and g = (P, P, R), Q != R, whose positions on P
        alternate: exactly one of g's arcs lies between h's two.
        """
        by_loop = {}
        for d in doubles:
            by_loop.setdefault(d[1], []).append(d)
        out = []
        for group in by_loop.values():
            for a in range(len(group)):
                h, _, lo, hi, q = group[a]
                for b in range(a + 1, len(group)):
                    g, _, lo2, hi2, r = group[b]
                    if q != r and (lo < lo2 < hi) != (lo < hi2 < hi):
                        out.append((h, g))
                        out.append((g, h))
        return out

    # ------------------------------------------------- corners and walls
    # Corner index parity is a global invariant (corner k of a cell is
    # corner k+-2 of its neighbours), so the grid's corners split into two
    # classes by k % 2. A tile wraps one whole class with arcs and sends
    # wall spokes to the other: cell i walls to corner k iff k % 2 != bits[i].
    @staticmethod
    def walls_to(bits, i, k):
        return (k % 2) != bits[i]

    @property
    def corner_map(self):
        """Corner point -> list of (cell, corner index) for the cells on it.

        Length 3 is an interior corner, 2 lies on the outline, 1 is fully
        exposed (both flanking cells missing).
        """
        if not hasattr(self, "_cmap"):
            m = defaultdict(list)
            for i, c in enumerate(self.cells):
                for k in range(6):
                    p = corner(c, k)
                    m[(round(p[0], 6), round(p[1], 6))].append((i, k))
            self._cmap = dict(m)
        return self._cmap

    @property
    def boundary_cells(self):
        """Cells with at least one missing neighbour."""
        if not hasattr(self, "_bcells"):
            self._bcells = {i for i, c in enumerate(self.cells)
                            if any((c[0] + a, c[1] + b) not in self.index
                                   for a, b in RING)}
        return self._bcells

    def wall_degree(self, bits, spot):
        return sum(1 for i, k in spot if self.walls_to(bits, i, k))

    def stubs(self, bits, interior=False):
        """Corners where one wall spoke ends and nothing continues.

        Interior ones (three cells present) are the tips of the maze's dead
        ends and are excluded by default; the rest are the ragged spokes on
        the outer edge.
        """
        out = []
        for p, spot in self.corner_map.items():
            if len(spot) == 3 and not interior:
                continue
            if self.wall_degree(bits, spot) == 1:
                out.append(p)
        return out

    def orbits(self):
        """Orbits of the 120-degree rotation about the centre cell."""
        done = set()
        out = []
        for c in self.cells:
            if c in done:
                continue
            orb = [c]
            x = rot120(*c)
            while x != c:
                orb.append(x)
                x = rot120(*x)
            done.update(orb)
            out.append([self.index[o] for o in orb])
        return out

    # ----------------------------------------------------- loop geometry
    def trace_loops(self, bits):
        """Return list of loops; each loop is a list of arcs.

        Arc = dict(center, start, end, kind) with kind 'tile' or 'cap',
        plus 'inside' = a point the arc should bend towards (to choose
        the sweep direction).
        """
        n6 = 6 * self.n
        seen = bytearray(n6)
        loops = []
        for s in range(n6):
            if seen[s]:
                continue
            arcs = []
            p = s
            while not seen[p]:
                seen[p] = 1
                q = self.tile_partner(p, bits[p // 6])
                seen[q] = 1
                arcs.append(self._tile_arc(p, q))
                r = self.cross[q]
                if q in self.cap_corner:
                    arcs.append(self._cap_arc(q, r))
                p = r
            loops.append(arcs)
        return loops

    def _tile_arc(self, p, q):
        i, k1 = divmod(p, 6)
        _, k2 = divmod(q, 6)
        c = self.cells[i]
        # shared corner index: the lower of the two edges (mod 6 wrap)
        kc = k1 if (k1 + 1) % 6 == k2 else k2
        return dict(kind="tile", center=corner(c, kc), start=edge_mid(c, k1),
                    end=edge_mid(c, k2), toward=cell_center(*c), away=False)

    def _cap_arc(self, q, r):
        cq, kq = self.cells[q // 6], q % 6
        cr, kr = self.cells[r // 6], r % 6
        cc, kc = self.cap_corner[q]
        Q = corner(cc, kc)
        if cq == cr:
            # convex corner: 240-degree arc bulging away from the cell
            return dict(kind="cap", center=Q, start=edge_mid(cq, kq),
                        end=edge_mid(cr, kr), toward=cell_center(*cq), away=True)
        # concave corner: 120-degree arc inside the outside cell
        out = (cc[0] + RING[kc][0], cc[1] + RING[kc][1])
        return dict(kind="cap", center=Q, start=edge_mid(cq, kq),
                    end=edge_mid(cr, kr), toward=cell_center(*out), away=False)


def arc_params(arc, side=1.0):
    """Resolve an arc to (cx, cy, radius, a0, sweep) in math coords.

    sweep is signed radians (positive = counter-clockwise)."""
    (cx, cy), (sx, sy), (ex, ey) = arc["center"], arc["start"], arc["end"]
    rad = math.hypot(sx - cx, sy - cy)
    a0 = math.atan2(sy - cy, sx - cx)
    a1 = math.atan2(ey - cy, ex - cx)
    tx, ty = arc["toward"]
    am = math.atan2(ty - cy, tx - cx)
    if arc["away"]:
        am += math.pi
    ccw = (a1 - a0) % (2 * math.pi)
    if (am - a0) % (2 * math.pi) < ccw:
        sweep = ccw
    else:
        sweep = ccw - 2 * math.pi
    return cx, cy, rad, a0, sweep


def loop_polyline(arcs, seg_deg=5.0):
    """Sample a loop's arcs into a closed polyline (list of (x,y))."""
    pts = []
    for arc in arcs:
        cx, cy, rad, a0, sw = arc_params(arc)
        n = max(2, int(math.ceil(abs(math.degrees(sw)) / seg_deg)))
        for t in range(n):  # omit last point; next arc starts there
            a = a0 + sw * t / n
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return pts


# ------------------------------------------------------------------ search
def loop_parity(fig):
    """Parity of the loop count, which no tiling can change.

    A flip moves the count by 0 or +-2, and any tiling can be reached from
    any other by flips, so one count settles it for the whole figure.
    """
    return fig.count_loops([0] * fig.n) % 2


def reachable_target(fig, target):
    """The lowest count >= target that this figure's parity allows."""
    return target if target % 2 == loop_parity(fig) else target + 1


def pin_boundary(fig, bit=None):
    """Fix every boundary cell so the outer walls close into a band.

    All boundary cells take the same tile: two that share an outline corner
    have to agree there or leave a stub, and those constraints chain all the
    way round. So there are exactly two border modes -- every Y one way up,
    or every Y the other.

    The two are mirror images by every measure of the walls alone: same
    number of stubs, same links and gaps, each a connected band enclosing
    one region, and the stubs fall in alternating 60-degree sectors (1, 3, 5
    for one, 0, 2, 4 for the other). What separates them is the caps, which
    are fixed independently: with the wrong mode the border arcs fight the
    caps and the loop count explodes (62 instead of 2 at radius 5). The
    border has to agree with the caps, and bit == cap_phase is that
    agreement (checked at radii 3 to 11, both phases).
    """
    return {i: fig.cap_phase if bit is None else bit for i in fig.boundary_cells}


def default_restarts(symmetric):
    """Dead-end restarts are on by default with symmetry, off without."""
    return bool(symmetric)


def find_single_loop(fig: Figure, seed: int, symmetric=True, guided=True,
                     max_steps=200000, progress=None, interval=2.0,
                     uphill=0.0, restarts=None, fixed=None, target=1):
    """Search for a tiling with exactly one loop. Returns (bits, flips),
    or (None, flips) on failure. The loop count never rises (unless
    `uphill` is set).

    Without symmetry (and guided=True) the search is planned, not random:
      1. If some hex touches three loops, flip it (merges them, -2).
      2. Else look for a pushing pair (Figure.push_pairs): flip h, which
         re-splices two loops so that g then touches three; flip g.
      3. Else flip a random hex that touches two loops (neutral) and retry.

    With symmetry, or guided=False, it flips a random orbit / hex and keeps
    the flip only if the loop count does not rise.

    fixed: {cell: bit} held constant (the pinned border). target: stop
    at this many loops -- 1 normally, 2 for an enclosed maze, where the
    enclosing band is a wall cycle and one loop is therefore impossible.

    uphill: chance per step of making a flip regardless of its effect.
    restarts: start again from a new random tiling at a dead end (no move
    that avoids an increase); None = on with symmetry, off without.

    progress: optional callable, given a dict (step, max_steps, loops,
    best_loops, since_best, restarts, elapsed, rate, done) every
    `interval` seconds, and once more at the end if it was called at all.
    """
    import time
    t0 = time.monotonic()
    next_report = t0 + interval
    reported = False
    best = None
    best_step = 0
    nrestarts = 0
    if restarts is None:
        restarts = default_restarts(symmetric)
    # Asking for a count the figure's parity forbids can never succeed, so
    # settle for the lowest reachable one rather than burning max_steps.
    target = reachable_target(fig, target)
    rng = random.Random(seed)
    fixed = dict(fixed or {})
    groups = [g for g in (fig.orbits() if symmetric else
                          [[i] for i in range(fig.n)])
              if not any(i in fixed for i in g)]
    if not groups:
        raise ValueError("every cell is fixed")
    free_cells = [g[0] for g in groups]

    def report(step, loops, done=False):
        nonlocal reported
        el = time.monotonic() - t0
        progress(dict(step=step, max_steps=max_steps, loops=loops,
                      best_loops=best, since_best=step - best_step,
                      restarts=nrestarts, elapsed=el,
                      rate=step / el if el > 0 else 0.0, done=done))
        reported = True

    def tick(step, loops):
        nonlocal best, best_step, next_report
        if best is None or loops < best:
            best, best_step = loops, step
        if progress and time.monotonic() >= next_report:
            report(step, loops)
            next_report = time.monotonic() + interval

    def finish(bits, step, loops):
        if progress and reported:
            report(step, loops, done=True)
        return (bits if loops <= target else None), step

    def random_start():
        b = [0] * fig.n
        for g in groups:
            v = rng.randint(0, 1)
            for i in g:
                b[i] = v
        for i, v in fixed.items():
            b[i] = v
        return b

    bits = random_start()

    if guided and not symmetric:
        step = 0
        loops = None
        while step < max_steps:
            arcs, loops = fig.hex_arcs(bits)
            tick(step, loops)
            if loops <= target:
                break
            if uphill and rng.random() < uphill:
                bits[rng.choice(free_cells)] ^= 1
                step += 1
                continue
            merges, doubles = fig.classify(arcs)
            merges = [i for i in merges if i not in fixed]
            doubles = [d for d in doubles if d[0] not in fixed]
            if merges:
                bits[rng.choice(merges)] ^= 1
                step += 1
                continue
            pairs = fig.push_pairs(doubles)
            if pairs:
                h, g = rng.choice(pairs)
                bits[h] ^= 1
                bits[g] ^= 1
                step += 2
                continue
            if doubles:
                bits[rng.choice(doubles)[0]] ^= 1
                step += 1
                continue
            if not restarts:  # dead end: every flip would add loops
                break
            bits = random_start()
            nrestarts += 1
        else:
            loops = fig.count_loops(bits)
        return finish(bits, step, loops)

    def dead_end(bits, loops):
        for g in groups:
            for i in g:
                bits[i] ^= 1
            n = fig.count_loops(bits)
            for i in g:
                bits[i] ^= 1
            if n <= loops:
                return False
        return True

    cur = fig.count_loops(bits)
    rejected = 0
    for step in range(max_steps):
        if step & 31 == 0 or cur <= target:
            tick(step, cur)
        if cur <= target:
            return finish(bits, step, cur)
        g = groups[rng.randrange(len(groups))]
        for i in g:
            bits[i] ^= 1
        new = fig.count_loops(bits)
        if new <= cur or (uphill and rng.random() < uphill):
            cur = new
            rejected = 0
        else:
            for i in g:
                bits[i] ^= 1
            rejected += 1
            if restarts and rejected >= 2 * len(groups):
                rejected = 0
                if dead_end(bits, cur):
                    bits = random_start()
                    cur = fig.count_loops(bits)
                    nrestarts += 1
    return finish(bits, max_steps, cur)


def print_progress(info, label="", stream=None):
    """Default progress printer: one status line per report, to stderr."""
    stream = stream or sys.stderr
    state = "done" if info["done"] else "searching"
    stream.write(
        f"{label}{state}: step {info['step']:,}/{info['max_steps']:,}, "
        f"{info['loops']} loop{'s' if info['loops'] != 1 else ''} (best {info['best_loops']}, "
        f"{info['since_best']:,} steps ago), "
        + (f"{info['restarts']} restarts, " if info.get('restarts') else "")
        + f"{info['elapsed']:.1f}s, "
        f"{info['rate']:,.0f} steps/s\n")
    stream.flush()


def generate(r, seed=0, symmetric=True, cap_phase=0, max_steps=200000,
             guided=True, progress=None, interval=2.0, uphill=0.0,
             restarts=None, fixed=None, target=1):
    """Return (figure, bits, steps) for the target loop count, or raise."""
    fig = Figure(r, cap_phase)
    bits, steps = find_single_loop(fig, seed, symmetric, guided, max_steps,
                                   progress=progress, interval=interval,
                                   uphill=uphill, restarts=restarts,
                                   fixed=fixed, target=target)
    if bits is None:
        raise RuntimeError(f"no tiling with {target} loop(s) (r={r}, seed={seed})")
    return fig, bits, steps


# --------------------------------------------------------------------- SVG
def _fmt(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def svg_path_d(arcs, scale):
    """Exact SVG path (A commands). Math y-up is flipped to SVG y-down."""
    out = []
    for j, arc in enumerate(arcs):
        cx, cy, rad, a0, sw = arc_params(arc)
        if j == 0:
            sx, sy = arc["start"]
            out.append(f"M{_fmt(sx * scale)},{_fmt(-sy * scale)}")
        ex, ey = arc["end"]
        large = 1 if abs(sw) > math.pi else 0
        # SVG sweep=1 means increasing angle in y-down coords; flipping y
        # turns a math-CCW (sw > 0) arc into a decreasing SVG angle.
        sweep = 0 if sw > 0 else 1
        R = _fmt(rad * scale)
        out.append(f"A{R},{R} 0 {large} {sweep} {_fmt(ex * scale)},{_fmt(-ey * scale)}")
    out.append("Z")
    return " ".join(out)


def to_svg(fig, bits, scale=20.0, stroke=None, walls=False, grid=False,
           caps=True, meta=None, loop_color="#1b4965", bg="#ffffff",
           second_color=None, grid_color="#c9d6df", stroke_color=None):
    stroke = stroke if stroke is not None else scale * 0.18
    loops = fig.trace_loops(bits)
    # bounds (include cap bulge: corners + side/2)
    xs, ys = [], []
    for c in fig.cells:
        for (x, y) in hex_corners(c):
            xs.append(x)
            ys.append(y)
    pad = 0.5 + 0.3
    x0, x1 = (min(xs) - pad) * scale, (max(xs) + pad) * scale
    y0, y1 = (-max(ys) - pad) * scale, (-min(ys) + pad) * scale
    w, h = x1 - x0, y1 - y0
    L = []
    L.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="{_fmt(x0)} {_fmt(y0)} {_fmt(w)} {_fmt(h)}" '
             f'width="{_fmt(w)}" height="{_fmt(h)}">')
    if meta is not None:
        L.append("<metadata>" + json.dumps(meta).replace("<", "&lt;") + "</metadata>")
    L.append(f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(w)}" '
             f'height="{_fmt(h)}" fill="{bg}"/>')
    if grid:
        L.append(f'<g fill="none" stroke="{grid_color}" stroke-width="'
                 f'{_fmt(scale * 0.03)}">')
        for c in fig.cells:
            pts = " ".join(f"{_fmt(x * scale)},{_fmt(-y * scale)}"
                           for x, y in hex_corners(c))
            L.append(f'<polygon points="{pts}"/>')
        L.append("</g>")
    if walls:
        L.append(f'<g stroke="#222" stroke-width="{_fmt(scale * 0.08)}" '
                 'stroke-linecap="round">')
        for i, c in enumerate(fig.cells):
            cx, cy = cell_center(*c)
            # walls go to the corners NOT wrapped by an arc
            ks = (1, 3, 5) if bits[i] == 0 else (0, 2, 4)
            for k in ks:
                x, y = corner(c, k)
                L.append(f'<line x1="{_fmt(cx * scale)}" y1="{_fmt(-cy * scale)}" '
                         f'x2="{_fmt(x * scale)}" y2="{_fmt(-y * scale)}"/>')
        L.append("</g>")
    L.append(f'<g fill="none" stroke="{loop_color}" stroke-width="{_fmt(stroke)}" '
             'stroke-linejoin="round" stroke-linecap="round">')
    order = sorted(range(len(loops)), key=lambda j: -len(loops[j]))
    rank = {j: a for a, j in enumerate(order)}
    for n, arcs in enumerate(loops):
        col = ""
        if second_color and rank[n] > 0:
            col = f' stroke="{second_color}"'
        if not caps:
            # draw interior arcs only, as separate segments
            for arc in arcs:
                if arc["kind"] == "tile":
                    d = svg_path_d([arc], scale)[:-2]
                    L.append(f'<path d="{d}"/>')
        else:
            L.append(f'<path id="loop{n}"{col} d="{svg_path_d(arcs, scale)}"/>')
    L.append("</g>")
    L.append("</svg>")
    return "\n".join(L) + "\n"


def wall_topology(fig, bits, trim=False):
    """(components, cycle rank) of the wall graph.

    The cycle rank is the number of corridor regions the walls enclose, so
    rank 1 means one corridor. A single loop forces rank 0 (the walls are
    all trees); an enclosing band is itself a cycle, hence rank 1 and two
    loops.
    """
    drop = set(fig.stubs(bits)) if trim else set()
    E = set()
    for i, c in enumerate(fig.cells):
        ctr = cell_center(*c)
        ck = (round(ctr[0], 6), round(ctr[1], 6))
        for k in range(6):
            if not fig.walls_to(bits, i, k):
                continue
            p = corner(c, k)
            pk = (round(p[0], 6), round(p[1], 6))
            if pk in drop:
                continue
            E.add(frozenset((ck, pk)))
    V = {v for e in E for v in e}
    adj = {v: set() for v in V}
    for e in E:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    comps = 0
    for v in V:
        if v in seen:
            continue
        comps += 1
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x])
    return comps, len(E) - len(V) + comps


def wall_trails(fig, bits, trim=True):
    """The wall segments merged into as few continuous polylines as possible.

    One path per spoke would mean a blade plunge at every segment, so each
    wall component is walked as long trails instead: start at an odd-degree
    vertex where possible (a tree needs about one trail per two leaves),
    then pick up any leftover closed circuits.
    """
    drop = set(fig.stubs(bits)) if trim else set()
    adj = defaultdict(list)
    n = 0
    for i, c in enumerate(fig.cells):
        ctr = cell_center(*c)
        a = (round(ctr[0], 6), round(ctr[1], 6))
        for k in range(6):
            if not fig.walls_to(bits, i, k):
                continue
            p = corner(c, k)
            b = (round(p[0], 6), round(p[1], 6))
            if b in drop:
                continue
            adj[a].append((b, n))
            adj[b].append((a, n))
            n += 1
    used = [False] * n
    deg = {v: len(e) for v, e in adj.items()}

    def walk(v):
        trail = [v]
        while True:
            step = next(((w, e) for w, e in adj[v] if not used[e]), None)
            if step is None:
                return trail
            w, e = step
            used[e] = True
            deg[v] -= 1
            deg[w] -= 1
            v = w
            trail.append(v)

    trails = []
    left = n
    while left:
        start = next((v for v in adj if deg[v] % 2 and deg[v]), None)
        if start is None:
            start = next((v for v in adj if deg[v]), None)
        t = walk(start)
        left -= len(t) - 1
        trails.append(t)
    return trails


def to_walls_svg(fig, bits, scale=30.0, trim=True, meta=None,
                 fill=None, grid=None, wall="#000000",
                 bg=None, wall_width=0.09, grid_width=0.025):
    """The walls as cut lines: continuous stroked paths, nothing else.

    By default nothing is filled and the hex grid is not drawn, since a
    cutter would follow those too. Pass fill/grid/bg for a shaded preview.

    Colours and widths are presentation only. A cutter follows the centre
    line of a path, so none of them change what gets cut.

    trim drops the spokes that end at a stub. It is a single pass on
    purpose: repeating it would peel the wall trees away entirely.
    """
    pts = [p for c in fig.cells for p in hex_corners(c)]
    pad = 0.25
    x0 = min(p[0] for p in pts) - pad
    x1 = max(p[0] for p in pts) + pad
    y0 = min(p[1] for p in pts) - pad
    y1 = max(p[1] for p in pts) + pad
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'viewBox="{_fmt(x0 * scale)} {_fmt(-y1 * scale)} {_fmt(w)} {_fmt(h)}" '
         f'width="{_fmt(w)}" height="{_fmt(h)}">']
    if meta is not None:
        L.append("<metadata>" + json.dumps(meta).replace("<", "&lt;") + "</metadata>")
    if bg:
        L.append(f'<rect x="{_fmt(x0 * scale)}" y="{_fmt(-y1 * scale)}" '
                 f'width="{_fmt(w)}" height="{_fmt(h)}" fill="{bg}"/>')
    if fill or grid:
        L.append(f'<g fill="{fill or "none"}" stroke="{grid or "none"}" '
                 f'stroke-width="{_fmt(scale * grid_width)}">')
        for c in fig.cells:
            pp = " ".join(f"{_fmt(x * scale)},{_fmt(-y * scale)}"
                          for x, y in hex_corners(c))
            L.append(f'<polygon points="{pp}"/>')
        L.append("</g>")
    L.append(f'<g fill="none" stroke="{wall}" '
             f'stroke-width="{_fmt(scale * wall_width)}" '
             'stroke-linecap="round" stroke-linejoin="round">')
    for t in wall_trails(fig, bits, trim):
        d = "M" + " L".join(f"{_fmt(x * scale)},{_fmt(-y * scale)}" for x, y in t)
        if t[0] == t[-1]:
            d += " Z"
        L.append(f'<path d="{d}"/>')
    L.append("</g></svg>")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------- height map
def distance_field(arcs, px_per_side, pad=0.8, bounds=None):
    """Distance (in hex-side units) from each pixel centre to the loop.

    Returns (dist, (x0, y_top)) with dist a float32 array, row 0 at the
    top. Each arc only touches pixels near its circle, so the work is
    done in a small window per arc.
    """
    import numpy as np
    xs = [a["center"][0] for a in arcs]
    ys = [a["center"][1] for a in arcs]
    if bounds is None:
        bounds = (min(xs) - pad - 0.5, max(xs) + pad + 0.5,
                  min(ys) - pad - 0.5, max(ys) + pad + 0.5)
    x0, x1, y0, y1 = bounds
    W = int(math.ceil((x1 - x0) * px_per_side))
    H = int(math.ceil((y1 - y0) * px_per_side))
    big = np.float32(1e9)
    dist = np.full((H, W), big, dtype=np.float32)
    reach = pad + 0.5  # beyond this the value only matters as "far"
    for arc in arcs:
        cx, cy, rad, a0, sw = arc_params(arc)
        s = a0 if sw > 0 else a0 + sw
        span = abs(sw)
        c0 = max(0, int((cx - rad - reach - x0) * px_per_side))
        c1 = min(W, int((cx + rad + reach - x0) * px_per_side) + 2)
        r0 = max(0, int((y1 - (cy + rad + reach)) * px_per_side))
        r1 = min(H, int((y1 - (cy - rad - reach)) * px_per_side) + 2)
        if c0 >= c1 or r0 >= r1:
            continue
        px = x0 + (np.arange(c0, c1) + 0.5) / px_per_side
        py = y1 - (np.arange(r0, r1) + 0.5) / px_per_side
        X, Y = np.meshgrid(px - cx, py - cy)
        rho = np.hypot(X, Y)
        th = np.mod(np.arctan2(Y, X) - s, 2 * math.pi)
        on = np.abs(rho - rad)
        ex1 = (rad * math.cos(s), rad * math.sin(s))
        ex2 = (rad * math.cos(s + span), rad * math.sin(s + span))
        off = np.minimum(np.hypot(X - ex1[0], Y - ex1[1]),
                         np.hypot(X - ex2[0], Y - ex2[1]))
        d = np.where(th <= span, on, off).astype(np.float32)
        win = dist[r0:r1, c0:c1]
        np.minimum(win, d, out=win)
    return dist, bounds


def height_map(dist, width=0.35, profile="round", invert=False, fillet=0.1):
    """Map distance-to-centreline onto heights in [0, 1].

    width   ribbon width in hex-side units (the corridor between walls is
            about 0.87 wide, so up to ~0.8 keeps strands apart).
    profile 'round' = semicircular cross-section; 'flat' = flat top with
            rounded shoulders of radius `fillet`.
    invert  raise everything except the ribbon (a groove / walls model).
    """
    import numpy as np
    hw = width / 2.0
    if profile == "round":
        t = np.clip(dist / hw, 0, 1)
        h = np.sqrt(1 - t * t)
    elif profile == "flat":
        f = min(fillet, hw)
        u = np.clip((dist - (hw - f)) / f, 0, 1)  # 0 on top, 1 at the edge
        h = np.sqrt(np.clip(1 - u * u, 0, 1))
        h = np.where(dist <= hw - f, 1.0, h)
    else:
        raise ValueError(profile)
    h = np.where(dist >= hw, 0.0, h)
    return 1.0 - h if invert else h


def write_height_png(path, h, meta=None, bits16=True):
    """Save heights in [0,1] as a 16-bit (or 8-bit) grayscale PNG."""
    import numpy as np
    from PIL import Image, PngImagePlugin
    info = PngImagePlugin.PngInfo()
    for k, v in (meta or {}).items():
        info.add_text(str(k), str(v))
    if bits16:
        img = Image.fromarray((np.clip(h, 0, 1) * 65535 + 0.5).astype(np.uint16))
    else:
        img = Image.fromarray((np.clip(h, 0, 1) * 255 + 0.5).astype(np.uint8))
    img.save(path, pnginfo=info)


def bits_grid(fig, bits):
    r = fig.r
    rows = []
    for b in range(r, -r - 1, -1):
        row = []
        for a in range(-r, r + 1):
            i = fig.index.get((a, b))
            row.append("." if i is None else str(bits[i]))
        rows.append(" ".join(row))
    return "\n".join(rows) + "\n"


def _unique(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--radius", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1, help="seeds seed..seed+count-1")
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--cap-phase", type=int, default=0, choices=(0, 1))
    ap.add_argument("--scale", type=float, default=20.0, help="px per hex side")
    ap.add_argument("--stroke", type=float, default=None, help="loop stroke px")
    ap.add_argument("--walls", action="store_true", help="also draw Y walls")
    ap.add_argument("--grid", action="store_true",
                    help="also draw the hex grid (a preview aid: these lines "
                         "are not merged into the cut paths, and a cutter "
                         "would follow them too)")
    ap.add_argument("--no-caps", action="store_true",
                    help="draw only arcs inside the figure")
    ap.add_argument("--no-guide", action="store_true",
                    help="random flips instead of the planned search (no symmetry only)")
    mz = ap.add_argument_group("maze (walls only, enclosed by a border of tiles)")
    mz.add_argument("--maze", action="store_true",
                    help="walls-only maze: border tiles pinned into a closed "
                         "band, cells filled, stubs trimmed, no curve drawn")
    mz.add_argument("--free-border", action="store_true",
                    help="with --maze, leave the border tiles to the search")
    mz.add_argument("--curves", action="store_true",
                    help="draw the enclosed maze as its curves instead of "
                         "its walls; the enclosed corridor is drawn in the "
                         "wall colour, anything outside it fainter")
    mz.add_argument("--fill", action="store_true",
                    help="shaded preview: fill the cells, and draw the grid "
                         "unless --grid already did (off by default, since a "
                         "cutter would follow those)")
    mz.add_argument("--wall-width", type=float, default=0.09, metavar="W",
                    help="wall stroke in hex-side units (default 0.09)")
    mz.add_argument("--wall-color", default=None, metavar="C",
                    help="wall colour (default black)")
    mz.add_argument("--grid-width", type=float, default=0.025, metavar="W",
                    help="grid stroke in hex-side units (default 0.025)")
    mz.add_argument("--grid-color", default=None, metavar="C",
                    help="grid colour (default a graph-paper blue). Colours "
                         "and widths are display only: a cutter follows the "
                         "centre line, so none of them change the cut")
    mz.add_argument("--keep-stubs", action="store_true",
                    help="with --maze, draw the stub spokes instead of trimming")
    mz.add_argument("--loops", type=int, default=None, metavar="N",
                    help="target loop count (default 1, or 2 with --maze)")
    ap.add_argument("--uphill", type=float, default=0.0, metavar="P",
                    help="chance of keeping a flip that raises the loop count "
                         "(default 0)")
    ap.add_argument("--restarts", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="restart from a new random tiling at a dead end "
                         "(default: on with symmetry, off without)")
    ap.add_argument("--max-steps", type=int, default=200000,
                    help="give up on a seed after this many steps")
    ap.add_argument("--progress", type=float, default=2.0, metavar="SECONDS",
                    help="print search status to stderr every SECONDS "
                         "(default 2; runs shorter than that print nothing; 0 = off)")
    hm = ap.add_argument_group("height map (PNG, needs numpy + Pillow)")
    hm.add_argument("--heightmap", action="store_true",
                    help="also write NAME_height.png")
    hm.add_argument("--hm-px", type=float, default=40.0,
                    help="height-map pixels per hex side (default 40)")
    hm.add_argument("--width", type=float, default=0.35,
                    help="ribbon width in hex-side units (default 0.35)")
    hm.add_argument("--profile", choices=("round", "flat"), default="round")
    hm.add_argument("--fillet", type=float, default=0.1,
                    help="shoulder radius for --profile flat")
    hm.add_argument("--invert", action="store_true",
                    help="raise everything but the ribbon (groove/walls)")
    hm.add_argument("--hm-8bit", action="store_true", help="8-bit instead of 16-bit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-o", "--out", default=None, help="output .svg")
    args = ap.parse_args(argv)
    target = args.loops if args.loops is not None else (2 if args.maze else 1)
    reach = reachable_target(Figure(args.radius, args.cap_phase), target)
    if reach != target:
        why = ("an even radius gives two corridor regions; use an odd radius "
               "for one" if args.maze else
               "odd radii never give a single loop")
        print(f"note: radius {args.radius} cannot reach {target} loop(s) -- {why}. "
              f"Stopping at {reach} instead.", file=sys.stderr)
        target = reach
    fixed = None
    if args.maze and not args.free_border:
        fixed = pin_boundary(Figure(args.radius, args.cap_phase))
    cmd = "python hex_truchet.py " + " ".join(argv if argv is not None else sys.argv[1:])
    failed = 0
    for seed in range(args.seed, args.seed + args.count):
        label = f"[r={args.radius} seed={seed}] "
        prog = None
        if args.progress > 0:
            prog = lambda info, label=label: print_progress(info, label)
        try:
            fig, bits, steps = generate(args.radius, seed, not args.no_symmetry,
                                        args.cap_phase, args.max_steps,
                                        guided=not args.no_guide, progress=prog,
                                        interval=args.progress or 2.0,
                                        uphill=args.uphill,
                                        restarts=args.restarts,
                                        fixed=fixed, target=target)
        except RuntimeError:
            print(f"{label}no tiling with {target} loop(s) within "
                  f"{args.max_steps:,} steps; skipped", file=sys.stderr)
            failed += 1
            continue
        assert fig.count_loops(bits) <= target
        meta = dict(generator="hex_truchet.py", command=cmd, radius=args.radius,
                    seed=seed, symmetric=not args.no_symmetry,
                    guided=not args.no_guide,
                    uphill=args.uphill,
                    restarts=(args.restarts if args.restarts is not None
                              else default_restarts(not args.no_symmetry)),
                    cap_phase=args.cap_phase, steps=steps,
                    bits="".join(map(str, bits)))
        if args.maze:
            trim = not args.keep_stubs
            meta["maze"] = True
            meta["border"] = "free" if args.free_border else "pinned"
            meta["trimmed"] = trim
            meta["filled"] = args.fill
            meta["grid"] = args.grid or args.fill
            gcol = args.grid_color or ("#cfe4de" if args.fill else "#5b9bd5")
            if args.curves:
                meta["curves"] = True
                svg = to_svg(fig, bits, args.scale, args.stroke, False,
                             args.grid or args.fill, not args.no_caps, meta,
                             loop_color=args.wall_color or "#111111",
                             bg="#ffffff" if args.fill else "#ffffff",
                             second_color="#b9c9d2", grid_color=gcol)
                out = args.out or f"hex_r{args.radius}_s{seed}.svg"
                if args.count > 1 and args.out:
                    b_, e_ = os.path.splitext(args.out)
                    out = f"{b_}_s{seed}{e_}"
                if not args.overwrite:
                    out = _unique(out)
                with open(out, "w") as f:
                    f.write(svg)
                print(f"{out}: {fig.count_loops(bits)} loops, {fig.n} tiles, "
                      f"{steps} steps (curves)")
                continue
            svg = to_walls_svg(
                fig, bits, args.scale, trim, meta,
                fill="#a8cec7" if args.fill else None,
                grid=gcol if (args.grid or args.fill) else None,
                bg="#ffffff" if args.fill else None,
                wall=args.wall_color or "#000000",
                wall_width=args.wall_width, grid_width=args.grid_width)
        else:
            svg = to_svg(fig, bits, args.scale, args.stroke, args.walls, args.grid,
                         not args.no_caps, meta)
        out = args.out or f"hex_r{args.radius}_s{seed}.svg"
        if args.count > 1 and args.out:
            b, e = os.path.splitext(args.out)
            out = f"{b}_s{seed}{e}"
        if not args.overwrite:
            out = _unique(out)
        with open(out, "w") as f:
            f.write(svg)
        with open(os.path.splitext(out)[0] + ".txt", "w") as f:
            f.write(f"# {cmd}\n# radius={args.radius} seed={seed}\n")
            f.write(bits_grid(fig, bits))
        loops = fig.count_loops(bits)
        msg = f"{out}: {loops} loop{'s' if loops != 1 else ''}, {fig.n} tiles, {steps} steps"
        if args.maze:
            comps, rank = wall_topology(fig, bits, trim=not args.keep_stubs)
            msg += (f"; walls {comps} pieces, {rank} corridor region"
                    f"{'s' if rank != 1 else ''}, "
                    f"{len(fig.stubs(bits))} stub{'s' if len(fig.stubs(bits)) != 1 else ''}"
                    + (" trimmed" if not args.keep_stubs else "")
                    + f"; {len(wall_trails(fig, bits, not args.keep_stubs))} cut paths")
        if args.heightmap:
            (arcs,) = fig.trace_loops(bits)
            dist, _ = distance_field(arcs, args.hm_px)
            h = height_map(dist, args.width, args.profile, args.invert, args.fillet)
            hpath = os.path.splitext(out)[0] + "_height.png"
            if not args.overwrite:
                hpath = _unique(hpath)
            hmeta = dict(meta, px_per_side=args.hm_px, width=args.width,
                         profile=args.profile, invert=args.invert,
                         fillet=args.fillet)
            write_height_png(hpath, h, hmeta, bits16=not args.hm_8bit)
            msg += f"; {hpath} {h.shape[1]}x{h.shape[0]}"
        print(msg)
    if failed:
        print(f"{failed} of {args.count} seeds failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
