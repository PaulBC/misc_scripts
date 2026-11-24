#!/usr/bin/env python3
"""
hextable.py -- the table of every single-loop hex Truchet tiling.

WHAT THIS IS
------------
A hex Truchet tile cuts its hexagon with three arcs.  Each arc joins the
midpoints of two adjacent edges, wrapping one corner.  There are two ways to
do it, and they are mirror images:

    tile 0 joins edge pairs (0,1) (2,3) (4,5)
    tile 1 joins edge pairs (1,2) (3,4) (5,0)

so one bit per hexagon describes a whole board.  The arcs of neighbouring
tiles meet at the shared edge midpoint, so the arcs link up into closed
curves.  Along the outside of the board the free edge ends are paired off
into "caps", which closes the curves there too.

The question the board poses is how to set the bits so that the arcs form a
*single* loop.  At radius 4 (61 hexagons, so 2^61 boards) there are exactly
1,848,868 such tilings, and this script enumerates all of them.

Parity note: a flip changes the loop count by -2, 0 or +2, so the count's
parity is fixed by the board.  At odd radius the minimum is two loops and the
table is empty; radius 2 and 4 are the interesting small cases.

BIT LAYOUT
----------
Hexagons are numbered 0..n-1 in reading order: by row first (axial b, which
runs up the screen, taken from the bottom), then by column (axial a) within
the row.  Bit i of the word is the tile bit of hexagon i.  Run

    python3 hextable.py --layout

to print the numbering as a grid.  The hexagons of radius R sit in a
(2R+1)x(2R+1) array with two opposite corners missing -- a Moore neighbourhood
of radius R minus the two cells where |a-b| > R.

Packing: bit i of a 64-bit word is hexagon i, so board = sum(bit_i << i).
At radius 4 that is 61 bits in one word, with the top three unused.  (Note the
browser version of this table splits the word as 31 low bits plus 30 high bits
across two uint32s, because JavaScript bitwise operators are 32-bit; this
script writes true little-endian uint64 and does not use that split.)

INVARIANT
---------
Every single-loop tiling at radius 4 has exactly 24 ones, and at radius 2
exactly 6: that is 3R^2/2 in both cases.  The skeleton of the curve has to
reach the same set of boundary corners whatever route it takes, so it uses the
same number of each tile orientation.  --verify checks this, and independently
re-traces the curve of sampled entries by following ports, rather than
trusting the union-find that generated them.

USAGE
-----
    python3 hextable.py --layout
    python3 hextable.py --radius 2 --format text
    python3 hextable.py --build --jobs 2 --cache r4.u64
    python3 hextable.py --build --method filter --k 10 --cache r4.u64
    python3 hextable.py --cache r4.u64 --format u64 --out table.bin
    python3 hextable.py --cache r4.u64 --stats
    python3 hextable.py --cache r4.u64 --sample 3 --art
    python3 hextable.py --cache r4.u64 --verify 200

Building radius 4 here takes about 2 minutes 40 seconds with --jobs 2, or six
minutes on one core; the 256 prefixes are independent, so more cores help
almost linearly.  The cache file is plain little-endian uint64, one per
solution, sorted, so it is easy to read from anything.

On compressibility: the raw table is 14.8 MB and lzma takes it to 181 KB, but
sorting first and storing varint deltas gets it to 19 KB, which is 0.08 bits
per solution.  An arbitrary set of this size drawn from 2^61 would need about
42 bits each, so the table is enormously structured -- neighbouring solutions
in sorted order differ in very few bits, and the delta stream repeats.  That is
the structure a ZDD would capture directly.
"""

import argparse
import os
import random
import struct
import sys
import time

# ---------------------------------------------------------------- geometry

# Neighbour k of cell (a,b), for k = 0..5 going counterclockwise from east.
RING = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]


def make_cells(R):
    """Hexagons of the radius-R board, in reading order (row b, then column a)."""
    cells = [(a, b)
             for a in range(-R, R + 1)
             for b in range(-R, R + 1)
             if abs(a) <= R and abs(b) <= R and abs(a - b) <= R]
    cells.sort(key=lambda c: (c[1], c[0]))
    return cells


def tile_partner(port, bit):
    """The port that `port` is joined to inside its own hexagon, given the tile."""
    i, k = divmod(port, 6)
    if bit == 0:
        k2 = k ^ 1                      # (0,1) (2,3) (4,5)
    else:
        k2 = (k + 1) % 6 if k % 2 else (k + 5) % 6   # (1,2) (3,4) (5,0)
    return 6 * i + k2


def make_figure(R, cap_phase=0):
    """Cells, the port-to-port crossing map, and which cells touch the border.

    Port 6*i+k is the midpoint of edge k of hexagon i.  cross[] sends it to the
    matching port of the neighbour across that edge; on the border it sends it
    to the other end of a cap.  Caps pair consecutive border edges two at a
    time around the outline, so the choice of where to start pairing is the
    cap phase: the two phases give different (mirror) boards.
    """
    cells = make_cells(R)
    index = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    cross = [-1] * (6 * n)
    for i, c in enumerate(cells):
        for k in range(6):
            j = index.get((c[0] + RING[k][0], c[1] + RING[k][1]))
            if j is not None:
                cross[6 * i + k] = 6 * j + (k + 3) % 6

    def is_border(c, k):
        return (c[0] + RING[k][0], c[1] + RING[k][1]) not in index

    # walk the outline as a cyclic sequence of (cell, edge) pairs
    start = None
    for c in cells:
        for k in range(6):
            if is_border(c, k):
                start = (c, k)
                break
        if start:
            break
    seq = [start]
    cur = start
    while True:
        c, k = cur
        k1 = (k + 1) % 6
        if is_border(c, k1):
            nxt = (c, k1)
        else:
            nxt = ((c[0] + RING[k1][0], c[1] + RING[k1][1]), (k + 5) % 6)
        if nxt == start:
            break
        seq.append(nxt)
        cur = nxt

    m = len(seq)
    for t in range(0, m, 2):
        e1 = seq[(t + cap_phase) % m]
        e2 = seq[(t + 1 + cap_phase) % m]
        p1 = 6 * index[e1[0]] + e1[1]
        p2 = 6 * index[e2[0]] + e2[1]
        cross[p1] = p2
        cross[p2] = p1

    return cells, index, cross


def node_map(cross, n):
    """Each crossing pair of ports is one node of the connectivity graph.

    The cross map is a fixed-point-free involution on the 6n ports, so there
    are exactly 3n nodes: one per edge of the board, counting a cap as an edge.
    """
    node = [-1] * (6 * n)
    nn = 0
    for p in range(6 * n):
        if node[p] < 0:
            node[p] = node[cross[p]] = nn
            nn += 1
    return node, nn


def arc_table(cells, cross, node):
    """arcs[bit][i] = the node pairs that tile `bit` joins inside hexagon i."""
    n = len(cells)
    arcs = [[], []]
    for bit in (0, 1):
        for i in range(n):
            seen, lst = set(), []
            for k in range(6):
                p = 6 * i + k
                q = tile_partner(p, bit)
                key = (min(p, q), max(p, q))
                if key not in seen:
                    seen.add(key)
                    lst.append((node[p], node[q]))
            arcs[bit].append(tuple(lst))
    return arcs

# ------------------------------------------------------------ enumeration


def enumerate_prefixes(R, cap_phase, lo, hi):
    """All single-loop boards whose first few tiles are fixed, as 64-bit ints.

    The scan order is the cell order itself, which is row by row, so the set of
    nodes that can still change -- the frontier -- stays narrow.  Two prunes do
    the work: a component that closes before the last hexagon can never be
    joined to anything again, and at the end every node must be in one
    component.

    Hexagons 0..D-1 are fixed by the prefix index, which is what lets the work
    be split across processes.  Each prefix is independent.
    """
    cells, index, cross = make_figure(R, cap_phase)
    n = len(cells)
    node, nn = node_map(cross, n)
    arcs = arc_table(cells, cross, node)
    D = min(PREFIX_BITS, n - 1)
    par = list(range(nn))
    out = []

    def find(x):
        # No path compression: the search undoes unions by resetting the nodes
        # recorded on the trail, and compression would rewrite parents that the
        # trail does not know about.  The trees stay shallow anyway, because the
        # frontier is narrow.
        while par[x] != x:
            x = par[x]
        return x

    def rec(ix, word, trail):
        if ix == n:
            r = find(0)
            for v in range(1, nn):
                if find(v) != r:
                    return
            out.append(word)
            return
        for bit in (0, 1):
            mark = len(trail)
            closed = False
            for u, v in arcs[bit][ix]:
                ru, rv = find(u), find(v)
                if ru == rv:
                    closed = True
                    break
                par[ru] = rv
                trail.append(ru)
            if not (closed and ix < n - 1):
                rec(ix + 1, word | (1 << ix) if bit else word, trail)
            while len(trail) > mark:
                x = trail.pop()
                par[x] = x
        return

    for pfx in range(lo, hi):
        for i in range(nn):
            par[i] = i
        word = 0
        dead = False
        for ix in range(D):
            bit = (pfx >> ix) & 1
            for u, v in arcs[bit][ix]:
                ru, rv = find(u), find(v)
                if ru == rv:
                    dead = True            # closed early: this prefix is dead
                    break
                par[ru] = rv
            if dead:
                break
            if bit:
                word |= 1 << ix
        if not dead:
            rec(D, word, [])
    return out


PREFIX_BITS = 8          # 256 independent prefixes, enough to spread over cores


def _worker(args):
    return enumerate_prefixes(*args)


def build(R, cap_phase=0, jobs=1, progress=True):
    n = len(make_cells(R))
    D = min(PREFIX_BITS, n - 1)
    total = 1 << D
    t0 = time.time()
    if jobs > 1:
        import multiprocessing as mp
        chunk = max(1, total // (jobs * 8))
        spans = [(R, cap_phase, s, min(s + chunk, total))
                 for s in range(0, total, chunk)]
        out = []
        with mp.Pool(jobs) as pool:
            for k, part in enumerate(pool.imap_unordered(_worker, spans), 1):
                out.extend(part)
                if progress:
                    print(f"\r  {k}/{len(spans)} chunks, {len(out)} so far, "
                          f"{time.time() - t0:.0f}s", end="", file=sys.stderr)
    else:
        out = []
        step = max(1, total // 32)
        for s in range(0, total, step):
            out.extend(enumerate_prefixes(R, cap_phase, s, min(s + step, total)))
            if progress:
                print(f"\r  {min(s + step, total)}/{total} prefixes, {len(out)} so far, "
                      f"{time.time() - t0:.0f}s", end="", file=sys.stderr)
    if progress:
        print(f"\r  done: {len(out)} solutions in {time.time() - t0:.1f}s"
              + " " * 20, file=sys.stderr)
    out.sort()
    return out

# ------------------------------------------------- filter-and-check build
#
# A second way to build the same table, without backtracking.
#
#  1. Forbidden patterns.  Trace every closed curve that touches at most k
#     tiles; each one fixes the bits of the tiles it passes through, and that
#     assignment can never appear in a single-loop board.  Away from the border
#     these are islands of one colour ringed by the other (k=3: three equal
#     tiles around a vertex; k=7: one hex inside its six neighbours; k=10: an
#     adjacent pair inside its eight); near the border the caps add variants.
#     The search runs on the actual board, so the border needs no special case.
#
#  2. Trimmed DP.  Patterns of <= 3 tiles are short in cell order, so a window
#     of the last L bits plus the ones count is enough state.  Trimming that DP
#     backward to states that can still reach the end with 3R^2/2 ones means a
#     forward walk through live states never hits a dead end.  The larger
#     patterns are applied during the walk as (word & mask) == value tests.
#
#  3. Check.  Trace from port 0; the board is one loop iff the trace visits all
#     3n nodes.  Most failures close early and are dropped as soon as they do.
#
# Patterns do not help the backtracker in enumerate_prefixes(): it already
# rejects a closed curve the moment its last arc is placed, which is exactly
# when the pattern would fire.  They pay off here, as lookahead and as a cheap
# reject before the trace.
#
# The ones count 3R^2/2 is used as a constraint.  It is verified exhaustively
# at radius 2 and 4 (cap phase 0) but not proved, so this method is limited to
# cap phase 0, and at other radii its output rests on that invariant.  Needs
# numpy.  Radius 4 with k=10 takes about 9 s here, against 16 s for k=3.


def loop_patterns(cells, cross, k):
    """Minimal {tile: bit} assignments that close a curve on <= k tiles,
    as a list of sorted ((tile, bit), ...) tuples."""
    NP = 6 * len(cells)
    found = set()
    asg = {}

    def walk(s, p):
        t = p // 6
        if t in asg:
            choices = (asg[t],)
        elif len(asg) < k:
            choices = (0, 1)
        else:
            return
        for b in choices:
            new = t not in asg
            if new:
                asg[t] = b
            q = tile_partner(p, b)
            nxt = cross[q]
            if q >= s and nxt >= s:          # s is the curve's lowest port: each curve once
                if nxt == s:
                    found.add(tuple(sorted(asg.items())))
                else:
                    walk(s, nxt)
            if new:
                del asg[t]

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 6 * k + 100))
    try:
        for s in range(NP):
            walk(s, s)
    finally:
        sys.setrecursionlimit(old)
    minimal, sets = [], []
    for p in sorted(found, key=len):
        ps = set(p)
        if not any(m <= ps for m in sets):
            sets.append(ps)
            minimal.append(p)
    return minimal


def build_filter(R, cap_phase=0, k=10, progress=True):
    try:
        import numpy as np
    except ImportError:
        sys.exit("--method filter needs numpy")
    if cap_phase != 0:
        sys.exit("--method filter assumes the 3R^2/2 ones count, verified only for cap phase 0")
    t0 = time.time()
    cells, index, cross = make_figure(R, cap_phase)
    n = len(cells)
    T = 3 * R * R // 2
    if (3 * R * R) % 2:
        return []                            # odd radius: no single loop
    pats = loop_patterns(cells, cross, max(k, 3))
    small = [p for p in pats if len(p) <= 3]
    big = [p for p in pats if len(p) > 3]

    # DP over cells: state = last L bits (bit t = cell i-t) and ones so far
    L = max(max(i for i, _ in p) - min(i for i, _ in p) for p in small) + 1
    W = 1 << L
    ws = np.arange(W)
    allowed = np.ones((n, W), bool)
    for p in small:
        i = max(j for j, _ in p)
        m = np.ones(W, bool)
        for j, b in p:
            m &= ((ws >> (i - j)) & 1) == b
        allowed[i] &= ~m
    shift = [((ws << 1) | b) & (W - 1) for b in (0, 1)]
    fw = np.zeros((n + 1, W, T + 1), bool)
    fw[0, 0, 0] = True
    for i in range(n):
        for b in (0, 1):
            w2 = shift[b]
            src = fw[i] & allowed[i][w2][:, None]
            for w in np.nonzero(src.any(1))[0]:
                if b:
                    fw[i + 1, w2[w], 1:] |= src[w, :T]
                else:
                    fw[i + 1, w2[w]] |= src[w]
    alive = np.zeros_like(fw)
    alive[n, :, T] = fw[n, :, T]
    for i in range(n - 1, -1, -1):
        for b in (0, 1):
            w2 = shift[b]
            nxt = alive[i + 1][w2]
            ok = allowed[i][w2][:, None]
            if b:
                alive[i, :, :T] |= ok & nxt[:, 1:]
            else:
                alive[i] |= ok & nxt
        alive[i] &= fw[i]

    # larger patterns, filed by last cell and grouped by mask
    by_last = {}
    for p in big:
        last = max(i for i, _ in p)
        m = sum(1 << i for i, _ in p)
        v = sum(b << i for i, b in p)
        by_last.setdefault(last, {}).setdefault(m, []).append(v)
    for d in by_last.values():
        for m in d:
            d[m] = np.array(sorted(d[m]), np.uint64)

    # forward walk through live states
    word = np.zeros(1, np.uint64)
    win = np.zeros(1, np.int64)
    ones = np.zeros(1, np.int64)
    for i in range(n):
        parts = []
        for b in (0, 1):
            w2 = ((win << 1) | b) & (W - 1)
            o2 = ones + b
            keep = o2 <= T
            keep[keep] = alive[i + 1, w2[keep], o2[keep]]
            wd = word[keep] | np.uint64(1 << i) if b else word[keep]
            parts.append((wd, w2[keep], o2[keep]))
        word = np.concatenate([q[0] for q in parts])
        win = np.concatenate([q[1] for q in parts])
        ones = np.concatenate([q[2] for q in parts])
        if i in by_last:
            ok = np.ones(len(word), bool)
            for m, vals in by_last[i].items():
                x = word & np.uint64(m)
                pos = np.searchsorted(vals, x)
                pos[pos == len(vals)] = 0
                ok &= vals[pos] != x
            word, win, ones = word[ok], win[ok], ones[ok]
    del win, ones, fw, alive
    t1 = time.time()
    if progress:
        print(f"  k={k}: {len(small)} + {len(big)} patterns, {len(word)} candidates "
              f"in {t1 - t0:.1f}s", file=sys.stderr)

    # check: trace from port 0, drop a board as soon as its curve closes early
    cr = np.array(cross, np.int64)
    part = np.array([[tile_partner(p, b) for p in range(6 * n)] for b in (0, 1)], np.int64)
    single = np.zeros(len(word), bool)
    B = 1 << 21
    for s in range(0, len(word), B):
        wd = word[s:s + B]
        idx = np.arange(len(wd))
        p = np.zeros(len(wd), np.int64)
        for step in range(1, 3 * n + 1):
            bit = ((wd >> (p // 6).astype(np.uint64)) & np.uint64(1)).astype(np.int64)
            p = cr[part[bit, p]]
            hit = p == 0
            if step == 3 * n:
                single[s + idx[hit]] = True
            elif hit.any():
                keep = ~hit
                wd, p, idx = wd[keep], p[keep], idx[keep]
                if len(p) == 0:
                    break
    out = np.sort(word[single]).tolist()
    if progress:
        print(f"  done: {len(out)} solutions in {time.time() - t0:.1f}s", file=sys.stderr)
    return out

# ----------------------------------------------------------------- loading


def read_cache(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % 8:
        sys.exit(f"{path}: not a whole number of 64-bit words")
    return list(struct.unpack(f"<{len(raw) // 8}Q", raw))


def write_cache(path, table):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(table)}Q", *table))

# ------------------------------------------------------------- the layout


def layout_grid(R, label=None):
    """The hexagons in their (2R+1)^2 array, two opposite corners missing.

    Row b runs up the screen, so the printed grid is drawn with b = R on top.
    Each row is indented by half a column relative to the row below, which is
    how the hexagons actually sit.
    """
    cells = make_cells(R)
    index = {c: i for i, c in enumerate(cells)}
    if label is None:
        label = lambda i: f"{i:2d}"                               # noqa: E731
    w = max(len(label(i)) for i in range(len(cells)))
    rows = []
    for b in range(R, -R - 1, -1):
        row = []
        for a in range(-R, R + 1):
            i = index.get((a, b))
            row.append(label(i).rjust(w) if i is not None else " " * w)
        # b descending: each lower row shifts right by half a cell
        pad = " " * ((R - b) * (w + 1) // 2)
        rows.append(pad + " ".join(row).rstrip())
    return "\n".join(rows)

# --------------------------------------------------------------- ascii art

# A hexagon is drawn pointy-top on a character grid where the centre of cell
# (a,b) sits at column 12a - 6b and row -6b.  That puts every edge midpoint on
# an integer position:
#
#     k=0 east (+6, 0)   k=1 north-east (+3,-3)   k=2 north-west (-3,-3)
#     k=3 west (-6, 0)   k=4 south-west (-3,+3)   k=5 south-east (+3,+3)
#
# An arc joins two adjacent midpoints, so it is a short diagonal (3 columns by
# 3 rows, drawn with two interior characters) or a horizontal run of five.
# Tile 0 draws an arch over a floor, tile 1 a ceiling over a funnel.
#
# The character at a midpoint itself depends on both tiles that meet there.
# If the two arcs leave in opposite directions the curve runs straight through
# and the character continues the line; otherwise both arcs leave the same way
# and the midpoint is a turning point, drawn v ^ < or > pointing away from them.
#
# Border midpoints get a letter instead.  The two ends of each cap share a
# letter: the curve leaves the board at one and comes back at the other.  Caps
# are what close the curves, so a letter is a jump, not a loose end.

EDGE_OFFSET = {0: (6, 0), 1: (3, -3), 2: (-3, -3),
               3: (-6, 0), 4: (-3, 3), 5: (3, 3)}
TILE_ARCS = {0: [(0, 1), (2, 3), (4, 5)],
             1: [(1, 2), (3, 4), (5, 0)]}


def _other_end(bit, k):
    """The edge the arc through edge k leads to, in tile `bit`."""
    for k1, k2 in TILE_ARCS[bit]:
        if k == k1:
            return k2
        if k == k2:
            return k1
    raise AssertionError


def _line_char(dc, dr):
    if dr == 0:
        return "_"
    return "/" if (dc > 0) == (dr < 0) else "\\"


def art(R, word, cap_phase=0):
    """One board drawn as hexagons, arcs only."""
    cells, index, cross = make_figure(R, cap_phase)
    n = len(cells)
    bits = [(word >> i) & 1 for i in range(n)]
    centre = {i: (12 * a - 6 * b, -6 * b) for i, (a, b) in enumerate(cells)}
    canvas = {}

    def mid(i, k):
        c, r = centre[i]
        dc, dr = EDGE_OFFSET[k]
        return (c + dc, r + dr)

    # the arcs themselves, interior points only
    for i in range(n):
        for k1, k2 in TILE_ARCS[bits[i]]:
            (c1, r1), (c2, r2) = mid(i, k1), mid(i, k2)
            dc, dr = c2 - c1, r2 - r1
            steps = max(abs(dc), abs(dr))
            ch = _line_char(dc, dr)
            for t in range(1, steps):
                canvas[(r1 + dr * t // steps, c1 + dc * t // steps)] = ch

    # the midpoints: a junction inside the board, a cap letter on the border
    caps, letters = {}, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(n):
        for k in range(6):
            p = 6 * i + k
            q = cross[p]
            j, k2 = divmod(q, 6)
            here = mid(i, k)
            if j != i and mid(j, k2) == here:
                # a shared edge: both tiles have an arc leaving this point
                d1 = EDGE_OFFSET[_other_end(bits[i], k)]
                d1 = (d1[0] - EDGE_OFFSET[k][0], d1[1] - EDGE_OFFSET[k][1])
                d2 = EDGE_OFFSET[_other_end(bits[j], k2)]
                d2 = (d2[0] - EDGE_OFFSET[k2][0], d2[1] - EDGE_OFFSET[k2][1])
                if (d1[0] + d2[0], d1[1] + d2[1]) == (0, 0):
                    ch = _line_char(*d1)
                else:
                    sc, sr = d1[0] + d2[0], d1[1] + d2[1]
                    ch = ("v" if sr < 0 else "^") if sr else ("<" if sc < 0 else ">")
                canvas[(here[1], here[0])] = ch
            else:
                key = (min(p, q), max(p, q))
                if key not in caps:
                    caps[key] = letters[len(caps) % len(letters)]
                canvas[(here[1], here[0])] = caps[key]

    rows = [r for (r, _) in canvas]
    cols = [c for (_, c) in canvas]
    out = []
    for r in range(min(rows), max(rows) + 1):
        out.append("".join(canvas.get((r, c), " ")
                           for c in range(min(cols), max(cols) + 1)).rstrip())
    return "\n".join(out)


# ------------------------------------------------------------ verification


def trace_loops(R, word, cap_phase=0):
    """Count the closed curves of a board by walking ports.

    This deliberately does not reuse the union-find that built the table: it
    alternates a step inside a hexagon (tile_partner) with a step across an
    edge or cap (cross), which is what the curve actually does.
    """
    cells, index, cross = make_figure(R, cap_phase)
    n = len(cells)
    bits = [(word >> i) & 1 for i in range(n)]
    seen = [False] * (6 * n)
    loops = 0
    for p0 in range(6 * n):
        if seen[p0]:
            continue
        loops += 1
        p = p0
        while not seen[p]:
            seen[p] = True
            q = tile_partner(p, bits[p // 6])
            seen[q] = True
            p = cross[q]
    return loops

# ------------------------------------------------------------------- stats


def stats(table, R):
    import bz2
    import gzip
    import lzma
    n = len(table)
    raw = struct.pack(f"<{n}Q", *table)
    txt = "".join(f"{''.join(str((w >> i) & 1) for i in range(3 * R * (R + 1) + 1))}\n"
                  for w in table).encode()
    # sorted neighbours differ in their low bits most of the time, so the
    # delta stream is a fairer test of what the structure is worth
    deltas = bytearray()
    prev = 0
    for w in table:
        d = w - prev
        prev = w
        while True:                      # varint
            b = d & 0x7F
            d >>= 7
            deltas.append(b | (0x80 if d else 0))
            if not d:
                break
    rows = []
    for name, blob in (("raw u64", raw), ("text 0/1 lines", txt),
                       ("varint deltas", bytes(deltas))):
        rows.append((name, len(blob),
                     len(gzip.compress(blob, 9)),
                     len(bz2.compress(blob, 9)),
                     len(lzma.compress(blob, preset=6))))
    w = max(len(r[0]) for r in rows)
    print(f"{n} solutions, {3 * R * (R + 1) + 1} hexagons, "
          f"log2(count) = {n.bit_length() - 1}..{n.bit_length()}")
    print(f"{'':{w}}  {'plain':>12} {'gzip':>12} {'bz2':>12} {'lzma':>12}")
    for name, a, b, c, d in rows:
        print(f"{name:{w}}  {a:12,} {b:12,} {c:12,} {d:12,}")
    print()
    import math
    best = min(min(r[2:]) for r in rows)
    cells = 3 * R * (R + 1) + 1
    arbitrary = (cells - math.log2(n) + math.log2(math.e))
    print(f"best: {best:,} bytes, {best * 8 / n:.3f} bits per solution.")
    print(f"An arbitrary set of {n:,} boards out of 2^{cells} would need about "
          f"{arbitrary:.0f} bits each, so this set is nothing like arbitrary: "
          f"the sorted order puts near-identical boards next to each other and "
          f"the delta stream repeats itself. That is the same structure a ZDD "
          f"would exploit, and it suggests the table is worth storing as a "
          f"decision diagram rather than a list.")

# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, default=4,
                    help="board radius (default 4; odd radii have no solutions)")
    ap.add_argument("--cap-phase", type=int, default=0, choices=(0, 1),
                    help="which pairing of border edges into caps (default 0)")
    ap.add_argument("--cache", metavar="FILE",
                    help="read the table from FILE, or write it there after --build")
    ap.add_argument("--build", action="store_true",
                    help="enumerate even if the cache exists")
    ap.add_argument("--jobs", type=int, default=1, help="processes for --build")
    ap.add_argument("--method", choices=("backtrack", "filter"), default="backtrack",
                    help="backtrack: the original search; filter: short-loop "
                         "patterns + trimmed DP + trace (needs numpy, cap phase 0)")
    ap.add_argument("--k", type=int, default=10,
                    help="for --method filter: forbid loops touching <= k tiles (default 10)")
    ap.add_argument("--format", choices=("text", "u64", "none"), default="none",
                    help="text: one 0/1 line per solution; u64: little-endian words")
    ap.add_argument("--out", metavar="FILE", help="where to write --format")
    ap.add_argument("--layout", action="store_true",
                    help="print the hexagon numbering and stop")
    ap.add_argument("--sample", type=int, metavar="N", help="show N random entries")
    ap.add_argument("--art", action="store_true", help="draw --sample as hexagons")
    ap.add_argument("--seed", type=int, help="seed for --sample")
    ap.add_argument("--verify", type=int, metavar="N", nargs="?", const=100,
                    help="check the ones-count invariant, and re-trace N entries")
    ap.add_argument("--stats", action="store_true", help="compressed sizes")
    args = ap.parse_args()

    R = args.radius
    n_cells = 3 * R * (R + 1) + 1

    if args.layout:
        print(f"radius {R}: {n_cells} hexagons, bit i of the word is hexagon i\n")
        print(layout_grid(R))
        print("\n(row b runs up the screen; within a row, column a runs right)")
        return

    table = None
    if args.cache and os.path.exists(args.cache) and not args.build:
        table = read_cache(args.cache)
        print(f"{len(table)} solutions from {args.cache}", file=sys.stderr)
    else:
        if args.method == "filter":
            table = build_filter(R, args.cap_phase, args.k)
        else:
            if R >= 4:
                print(f"enumerating radius {R} in Python; this is the slow part "
                      f"(radius 4 takes minutes, --jobs helps)", file=sys.stderr)
            table = build(R, args.cap_phase, max(1, args.jobs))
        if args.cache:
            write_cache(args.cache, table)
            print(f"wrote {args.cache}", file=sys.stderr)

    if args.verify is not None:
        want = 3 * R * R // 2
        bad = [w for w in table if bin(w).count("1") != want]
        print(f"ones-count {want} (3R^2/2): "
              f"{'all ' + str(len(table)) + ' entries agree' if not bad else str(len(bad)) + ' DISAGREE'}")
        rnd = random.Random(args.seed)
        picks = rnd.sample(table, min(args.verify, len(table)))
        bad2 = [w for w in picks if trace_loops(R, w, args.cap_phase) != 1]
        print(f"re-traced {len(picks)} entries by following ports: "
              f"{'every one is a single loop' if not bad2 else str(len(bad2)) + ' are NOT'}")
        dups = len(table) - len(set(table))
        print(f"duplicates: {dups}")

    if args.stats:
        stats(table, R)

    if args.sample:
        rnd = random.Random(args.seed)
        for w in rnd.sample(table, min(args.sample, len(table))):
            print(f"\n{'=' * 60}\nword 0x{w:016x}   ones {bin(w).count('1')}")
            if args.art:
                print()
                print(art(R, w, args.cap_phase))
            else:
                print(layout_grid(R, lambda i, w=w: str((w >> i) & 1)))

    if args.format != "none":
        if args.format == "u64":
            blob = struct.pack(f"<{len(table)}Q", *table)
            if args.out:
                with open(args.out, "wb") as f:
                    f.write(blob)
            else:
                sys.stdout.buffer.write(blob)
        else:
            lines = "".join("".join(str((w >> i) & 1) for i in range(n_cells)) + "\n"
                            for w in table)
            if args.out:
                with open(args.out, "w") as f:
                    f.write(lines)
            else:
                sys.stdout.write(lines)
        if args.out:
            print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
