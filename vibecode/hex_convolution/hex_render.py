"""
Standalone renderer for the hex non-branching-path grid format.

Deliberately decoupled from hex_walksat.py: this program's only input is
the plain text grid file (first line n, then n lines of n 0/1 characters -
see write_grid_txt() in hex_walksat.py). It doesn't import or depend on
the solver at all, just the shared geometry needed to place and connect
hexagon cells.

Supported formats (--mode):
  fill      Filled hexagons, live/dead colored differently. Matches
            hex_walksat.py's render_png output.
  skeleton  A skeleton/graph view: instead of filling cells, draw a line
            between the centers of every pair of ring-adjacent live
            cells. Since Rule 1 guarantees every live cell's live-neighbor
            count IS its path connections (2 for a pass-through, 3 for a
            branch point under --allow-branch), this reconstructs the
            path network exactly as a set of edges - no cell fill needed.
            This is a skeleton/starting format; the intent is more
            rendering styles get added here later.
  heightmap Grayscale distance field from the skeleton edges, for
            physical casting/printing - see render_heightmap and
            --profile for the available cross-section shapes.

All modes support --shape {rhombus,hexagon} and --radius: rhombus is just
the base tile; hexagon reconstructs a hexagon made of 3*radius^2 rotated
copies of the base rhombus via the same equivalence() lattice quotient
used by the solver (radius=1 is the single hexagon; radius=2 extends it
to twice the linear size for more room for paths to close into loops
before hitting the outer boundary), all without needing any manual
alignment - the rotational structure is already inherent in
equivalence(), so a larger radius is purely a wider boundary test.

--invert (any mode) relabels live and dead before rendering (see
value_at): a pure swap of the same solved grid, not a different solve.
The channel network stays exactly the same simple non-branching rings
either way, since that's unconditional under Rule 1; the inverted ridge
network can genuinely branch, since it only has to satisfy the weaker
Rule 2 - so "non-branching-path" above describes the default, not a
guarantee that survives --invert.

--color-cycles (fill and skeleton modes) colors each distinct cycle of live
cells differently (see find_cycles). Cycles are found on the base tile
through equivalence(), so a path that crosses a rotated seam is one cycle,
and every rotated copy of a cycle in a hexagon render shares its color -
the number of distinct colors is the number of genuinely distinct cycles
in the pattern, not the number of loops drawn. Not available for
heightmap (grayscale).

If the grid file carries provenance records (hex_walksat.py and hex_connect.py
append them as "# meta" lines), the PNG's hex_pipeline / hex_pipeline_commands
text chunks hold the whole chain - every command that led to the image, oldest
first, ending with this render - and flag a grid that was edited since.

Every PNG this writes also embeds the exact invocation that produced it,
as PNG text metadata (see build_pnginfo) - so a render can be reproduced
or tweaked later without having to remember which flags were used.
"""

import argparse
import colorsys
import datetime
import json
import math
import shlex
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from hex_common import (UsageError, args_for_command, canonical_argv, grid_digest,
                        namespace_from, read_meta, unique_path)


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------

def read_grid_txt(path):
    """Read the format written by hex_walksat.py's write_grid_txt(): first
    line n, then n lines of n '0'/'1' characters (row i, then column j).
    Returns (grid, n) where grid[i][j] is a bool."""
    with open(path) as f:
        n = int(f.readline().strip())
        grid = []
        for i in range(n):
            row = f.readline().rstrip("\n")
            if len(row) != n:
                raise ValueError(f"row {i} has length {len(row)}, expected {n}")
            grid.append([c == "1" for c in row])
    return grid, n


# ---------------------------------------------------------------------------
# Shared geometry (kept independent of hex_walksat.py on purpose)
# ---------------------------------------------------------------------------

# Ring-neighbor offsets in cyclic order - see hex_walksat.py's derivation
# notes for why this specific order is required.
RING = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]


def equivalence(i, j, n):
    """Rotated-rhombus periodic identification - maps any (i, j) back onto
    the base n x n tile via the 120-degree rotation."""
    while True:
        if ((i // n) + (j // n)) % 3 == 0:
            return i % n, j % n
        i, j = -j - 1, i - j


def is_extra_cell(i, j, n):
    """True if (i, j), within the bounding square [-n, n-1] x [-n, n-1] of
    the three rotated rhombus copies, falls in one of the two leftover
    corner triangles rather than in the hexagon itself."""
    return (j - i > n) or (i - j >= n)


def hexagon_growth_shifts(n, radius):
    """Shift vectors (in units of n) placing copies of the single hexagon
    so their union forms a properly 3-fold-symmetric, rule-consistent
    hexagon of the given radius, anchored so the shape grows AWAY from
    the origin corner rather than expanding symmetrically around R0 (see
    the derivation: is_extra_cell(i,j,radius*n) alone would recenter on
    the rotation's fixed point for any radius, which is a different,
    center-anchored shape, not this one).

    radius=1: just the original hexagon (3 rhombi).
    radius=2: the original plus 3 more full hexagon-copies at shifts
    (n,0), (n,n), (0,n) - one full rotational triple of n*RING - verified
    to give exactly 12n^2 cells, zero rule violations at every seam, and
    genuine 3-fold symmetry around the union's own (displaced) center.
    """
    if radius == 1:
        return [(0, 0)]
    if radius == 2:
        return [(0, 0), (n, 0), (n, n), (0, n)]
    raise ValueError(f"radius {radius} not yet supported - only 1 and 2 have "
                      f"been derived and verified so far")


def in_shape(i, j, n, shape, radius=1):
    """True if (i, j) should be rendered, for the given shape:
    'rhombus'   - the base n x n tile only (radius unused)
    'hexagon'   - a hexagon at the given radius, made of 3*radius^2 rotated
                  rhombus copies, anchored to grow from the origin corner
                  (see hexagon_growth_shifts). radius=1 is the single
                  hexagon (3 rhombi); radius=2 is 12 rhombi (a central
                  6-rhombus star plus 6 outer rhombi), matching the
                  reference picture. Value lookup always uses the
                  original n (see value_at) - only the boundary/extent
                  grows.
    """
    if shape == "rhombus":
        return 0 <= i <= n - 1 and 0 <= j <= n - 1
    if shape == "hexagon":
        for dx, dy in hexagon_growth_shifts(n, radius):
            if not is_extra_cell(i - dx, j - dy, n):
                return True
        return False
    raise ValueError(f"unknown shape {shape!r}")


def center_px(i, j, cell_px):
    # x = 0.5*j - i, y = 0.866*j (unit spacing, 120 degrees between axes) -
    # matches hex_walksat.py's render_png exactly, including orientation.
    x = 0.5 * j - i
    y = 0.866 * j
    return x * cell_px, -y * cell_px  # negate y: PIL's y-axis points down


def hexagon_points(cx, cy, r):
    # pointy-top hexagon (vertex pointing up)
    pts = []
    for k in range(6):
        angle = math.radians(90 + 60 * k)
        pts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    return pts


def value_at(grid, n, i, j, shape, invert=False):
    """Live/dead state at (i, j), folding through equivalence() - always
    using the original n, regardless of radius - for any shape other than
    the base rhombus.

    invert: flip every cell's value before returning it. This is the one
    place every other function in the file reads grid state through, so
    setting it upstream (render_fill/render_skeleton/render_heightmap)
    swaps live and dead everywhere downstream for free - a pure
    relabeling of the same solved grid, not a different solve. Rule 1
    (exactly 2 live ring-neighbors) is unconditional, so former-live
    cells keep forming the same simple non-branching rings after
    inverting, just drawn as channels instead of ridges. Former-dead
    cells only had to satisfy the weaker Rule 2 (no 3 consecutive dead
    neighbors, no minimum), so the inverted ridge network can genuinely
    branch - see compute_outer_wall_cycle and compute_pruned_positions
    for why they deliberately don't take this flag.
    """
    if shape == "rhombus":
        ci, cj = i, j
    else:
        ci, cj = equivalence(i, j, n)
    value = grid[ci][cj]
    return (not value) if invert else value


def index_range(n, shape, radius=1):
    if shape == "rhombus":
        return (0, n - 1)
    if shape == "hexagon":
        shifts = hexagon_growth_shifts(n, radius)
        dxs = [s[0] for s in shifts]
        dys = [s[1] for s in shifts]
        lo = min(min(dxs), min(dys)) - n
        hi = max(max(dxs), max(dys)) + n - 1
        return (lo, hi)
    raise ValueError(f"unknown shape {shape!r}")


def find_ray_entry_cell(grid, n, shape, radius=1):
    """Cast a deterministic ray in from outside the shape - a vertical
    line at a fixed i (the shape's own horizontal center) - and return
    the first live cell it hits, or None if that particular line never
    crosses a live cell.

    The direction/entry point doesn't actually matter for a
    hex_walksat.py --wall-boundary solve: the cycle enclosing the whole
    pattern is, by construction, the first live cell ANY incoming ray
    will hit, from any direction - a closed encircling curve is a real
    geometric barrier, not a heuristic. Fixed here purely for
    determinism, not because the choice is load-bearing.
    """
    lo, hi = index_range(n, shape, radius)
    center_i = (lo + hi) // 2
    for j in range(lo, hi + 1):
        if not in_shape(center_i, j, n, shape, radius):
            continue
        if value_at(grid, n, center_i, j, shape):
            return (center_i, j)
    return None


def compute_outer_wall_cycle(grid, n, shape, radius=1):
    """Find the single cycle of live cells that encloses the whole rest
    of the pattern, by casting a fixed deterministic ray in from outside
    the shape (find_ray_entry_cell) and flood-filling from the first
    live cell it hits, along live-live ring adjacency.

    Well-defined for a hex_walksat.py --wall-boundary solve: under Rule
    1 every live cell has exactly 2 live neighbors, so live cells always
    decompose into disjoint simple cycles - none ever touch or merge via
    a live-live edge. The dead wall forces the cells running just inside
    it to curve along its entire length rather than escape inward, so
    they typically close into one continuous cycle girding the whole
    pattern - and since that cycle fully encloses the interior, any ray
    from outside must cross it before reaching anything deeper.

    Without a wall (or in other cases), this is genuinely ambiguous -
    there's no guarantee of a single enclosing cycle at all. This will
    still return SOME cycle (whichever one the fixed ray happens to hit
    first), just not necessarily "the" outer one.

    Deliberately has no invert parameter: the "disjoint simple cycles"
    guarantee above comes from Rule 1, which only governs live cells
    (see value_at) - it doesn't carry over to the dead-cell complement,
    so this always walks the original live-cell structure regardless of
    whether the caller is about to invert for display. The cycle it
    returns is a set of original-live cells; effective_live treats them
    as deleted from the ORIGINAL grid and only then applies invert, so
    the removed ring becomes background in either polarity (dead
    normally, ridge-side under invert) rather than staying visible as a
    groove.
    """
    lo, hi = index_range(n, shape, radius)
    start = find_ray_entry_cell(grid, n, shape, radius)
    if start is None:
        return set()

    def in_view(i, j):
        return lo <= i <= hi and lo <= j <= hi and in_shape(i, j, n, shape, radius)

    component = {start}
    stack = [start]
    while stack:
        ci, cj = stack.pop()
        for di, dj in RING:
            ni, nj = ci + di, cj + dj
            if (ni, nj) in component:
                continue
            if not in_view(ni, nj):
                continue
            if not value_at(grid, n, ni, nj, shape):
                continue
            component.add((ni, nj))
            stack.append((ni, nj))
    return component


def compute_pruned_positions(grid, n, shape, radius=1):
    """Find every rendered POSITION (i, j) - not canonical grid cell, since
    different rotated copies of the same cell can have different boundary
    status - that belongs to a path/loop missing at least one edge because
    it runs off the rendered region, then return the full set of positions
    in that same connected component (the whole cut path, not just the
    cell where it happens to be missing an edge).

    Doesn't touch the grid at all - this only decides which on-screen
    instances to hide. Most meaningful with shape='hexagon', where the
    boundary being crossed is the shape's true edge; in 'rhombus' mode
    nearly every border cell is "cut" in this sense (it's simply where
    the single tile stops), so pruning there
    would remove most of the image.

    Deliberately has no invert parameter, for the same reason as
    compute_outer_wall_cycle: this relies on Rule 1's guarantee that
    live cells decompose into disjoint simple cycles, which doesn't hold
    for the dead-cell complement, so it always operates on the original
    live cells regardless of what the caller will invert for display.
    What it returns is treated as deleted from the original grid before
    any invert is applied (see effective_live).
    """
    lo, hi = index_range(n, shape, radius)

    def in_view(i, j):
        return lo <= i <= hi and lo <= j <= hi and in_shape(i, j, n, shape, radius)

    live_positions = []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if not in_shape(i, j, n, shape, radius):
                continue
            if value_at(grid, n, i, j, shape):
                live_positions.append((i, j))

    adjacency = {}
    visible_degree = {}
    true_degree = {}
    for (i, j) in live_positions:
        nbrs_in_view = []
        true_count = 0
        for di, dj in RING:
            ni, nj = i + di, j + dj
            if value_at(grid, n, ni, nj, shape):
                true_count += 1
                if in_view(ni, nj):
                    nbrs_in_view.append((ni, nj))
        adjacency[(i, j)] = nbrs_in_view
        visible_degree[(i, j)] = len(nbrs_in_view)
        true_degree[(i, j)] = true_count

    cut_positions = {p for p in live_positions if visible_degree[p] < true_degree[p]}

    # connected components over the visible adjacency graph (BFS)
    visited = set()
    pruned = set()
    for start in live_positions:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = [start]
        while stack:
            cur = stack.pop()
            for nb in adjacency[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
                    component.append(nb)
        if any(p in cut_positions for p in component):
            pruned.update(component)

    return pruned


def effective_live(grid, n, i, j, shape, pruned=None, invert=False):
    """Like value_at, but positions in `pruned` are treated as dead in the
    ORIGINAL grid - and invert is then applied on top of that, like any
    other cell. So pruning means "delete this from the pattern" in either
    polarity: without invert a pruned cell is dead (background); with
    invert it flips along with every other original-dead cell and becomes
    live (ridge-side), dissolving into the surrounding field.

    pruned is computed on the original grid (see compute_outer_wall_cycle)
    - the cells it names are original-live, which invert already makes
    dead. Forcing them to False AFTER inverting (as an earlier version of
    this did) is therefore a no-op under invert: they were never going to
    be ridge-side anyway, so --remove-outer-wall silently did nothing.
    The pruning has to happen before the flip to mean anything there."""
    if pruned is not None and (i, j) in pruned:
        return invert
    return value_at(grid, n, i, j, shape, invert=invert)


def compute_canvas(n, shape, cell_px, margin_px, r, radius=1):
    lo, hi = index_range(n, shape, radius)
    if shape != "rhombus":
        # The naive bounding box of the SQUARE's 4 corners is noticeably
        # looser than the shape's actual extent (that's exactly the
        # leftover corner cells from the is_extra_cell derivation) - so
        # scan the actually-included cells for a tight fit instead.
        xs, ys = [], []
        for i in range(lo, hi + 1):
            for j in range(lo, hi + 1):
                if not in_shape(i, j, n, shape, radius):
                    continue
                x, y = center_px(i, j, cell_px)
                xs.append(x)
                ys.append(y)
    else:
        corners = [center_px(i, j, cell_px) for i in (lo, hi) for j in (lo, hi)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
    x_min, x_max = min(xs) - r, max(xs) + r
    y_min, y_max = min(ys) - r, max(ys) + r
    width = int(x_max - x_min) + 2 * margin_px
    height = int(y_max - y_min) + 2 * margin_px
    ox = margin_px - x_min
    oy = margin_px - y_min
    return width, height, ox, oy


# ---------------------------------------------------------------------------
# Distinct cycles
# ---------------------------------------------------------------------------

def find_cycles(grid, n):
    """Distinct cycles of the pattern: connected components of live cells
    on the base n x n tile, with adjacency taken through equivalence() so a
    path leaving one edge of the tile and re-entering through a rotated
    seam is ONE cycle rather than two broken pieces. Under Rule 1 every
    component is a simple closed loop; with branching a component can be a
    network instead.

    Always works on the original live cells (never inverted or pruned):
    these are the loops themselves, independent of how they're displayed.

    Returns (cycle_id, sizes, branching): cycle_id maps every live
    canonical cell to an id (largest cycle first, ties broken by position,
    so ids are stable for a given grid); sizes[id] is its cell count;
    branching is True if any live cell has more than 2 live neighbors.
    """
    def live_nbrs(i, j):
        out = []
        for di, dj in RING:
            a, b = equivalence(i + di, j + dj, n)
            if grid[a][b]:
                out.append((a, b))
        return out

    seen = set()
    comps = []
    branching = False
    for i in range(n):
        for j in range(n):
            if not grid[i][j] or (i, j) in seen:
                continue
            seen.add((i, j))
            stack = [(i, j)]
            comp = []
            while stack:
                ci, cj = stack.pop()
                comp.append((ci, cj))
                nbrs = live_nbrs(ci, cj)
                if len(nbrs) > 2:
                    branching = True
                for d in nbrs:
                    if d not in seen:
                        seen.add(d)
                        stack.append(d)
            comps.append(comp)
    comps.sort(key=lambda c: (-len(c), min(c)))
    cycle_id = {}
    for k, comp in enumerate(comps):
        for cell in comp:
            cycle_id[cell] = k
    return cycle_id, [len(c) for c in comps], branching


def cycle_palette(k, dark=False):
    """k visually distinct RGB colors (golden-ratio hue steps, saturation
    and brightness cycling so neighboring ids differ by more than hue).
    dark=True gives deeper colors that read against a light background
    (skeleton lines); the default suits the dark dead cells of fill mode."""
    out = []
    for idx in range(k):
        h = (idx * 0.6180339887) % 1.0
        sat, val = ((0.70, 0.95), (0.90, 0.78), (0.55, 0.90))[idx % 3]
        if dark:
            sat, val = min(1.0, sat + 0.15), val * 0.72
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def describe_cycles(sizes, branching, show=12):
    kind = "components (branching network)" if branching else "cycles"
    head = ", ".join(str(sz) for sz in sizes[:show])
    more = f", ... (+{len(sizes) - show} more)" if len(sizes) > show else ""
    return f"{len(sizes)} {kind}; cell counts, largest first: {head}{more}"


def cycle_of(cycle_id, n, i, j, shape):
    """Cycle id of the rendered position (i, j), or None if it isn't part
    of any cycle (dead). Positions fold to their canonical cell through
    equivalence() for any shape but the base rhombus, so every rotated
    copy of a cycle reports the same id."""
    cell = (i, j) if shape == "rhombus" else equivalence(i, j, n)
    return cycle_id.get(cell)


# ---------------------------------------------------------------------------
# Render modes
# ---------------------------------------------------------------------------

def render_fill(grid, n, path, cell_px=32, live_color=(222, 184, 135),
                 dead_color=(15, 15, 15), edge_color=(106, 90, 205),
                 background=(255, 255, 255), margin_px=24, shape="rhombus",
                 radius=1, prune_boundary=False, remove_outer_wall=False,
                 invert=False, color_cycles=False, pnginfo=None):
    """Filled hexagons - live/dead colored differently.

    color_cycles: paint every cell that belongs to a cycle (see
    find_cycles) in that cycle's own color. Cycles are the ORIGINAL live
    cells, so under invert it's the channels that get colored, in place of
    dead_color; pruned positions are skipped so they still read as
    background.

    invert: swap which cells render as live_color vs dead_color (see
    value_at). prune_boundary/remove_outer_wall are computed on the
    original, non-inverted grid and applied before the swap (see
    effective_live), so a pruned/removed position always renders as the
    surrounding background - dead_color normally, live_color under invert
    - rather than standing out as a ring.
    """
    r = cell_px / math.sqrt(3)
    width, height, ox, oy = compute_canvas(n, shape, cell_px, margin_px, r, radius)

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    pruned = set()
    if prune_boundary:
        pruned |= compute_pruned_positions(grid, n, shape, radius)
    if remove_outer_wall:
        pruned |= compute_outer_wall_cycle(grid, n, shape, radius)
    pruned = pruned or None

    cycle_id = palette = None
    if color_cycles:
        cycle_id, sizes, _ = find_cycles(grid, n)
        palette = cycle_palette(len(sizes))

    lo, hi = index_range(n, shape, radius)
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if not in_shape(i, j, n, shape, radius):
                continue
            cx, cy = center_px(i, j, cell_px)
            cx, cy = cx + ox, cy + oy
            color = live_color if effective_live(grid, n, i, j, shape, pruned, invert) else dead_color
            if cycle_id is not None and not (pruned and (i, j) in pruned):
                cid = cycle_of(cycle_id, n, i, j, shape)
                if cid is not None:
                    color = palette[cid]
            draw.polygon(hexagon_points(cx, cy, r), fill=color, outline=edge_color, width=1)

    img.save(path, pnginfo=pnginfo)
    return path


def collect_edges(grid, n, shape, pruned=None, radius=1, invert=False):
    """Returns the list of live-live ring-adjacent cell pairs as grid-space
    segments [((i1,j1), (i2,j2)), ...], each edge appearing once. Shared by
    render_skeleton and render_heightmap.

    For shape='hexagon', values wrap through equivalence() but
    positions don't, so segments are in true (unwrapped) space and cross
    the rhombus seams cleanly. For shape='rhombus', a live cell's neighbor
    is only included if it also falls inside the visible n x n tile.

    pruned: optional set of positions (from compute_pruned_positions or
    compute_outer_wall_cycle) to treat as dead - used to drop boundary-cut
    paths, or the outer wall cycle, entirely. Always computed on the
    original, non-inverted grid (see compute_outer_wall_cycle) and
    applied before invert flips anything (see effective_live): under
    invert the pruned cells become ridge-side, so they gain edges to
    their neighbors rather than losing them.

    invert: edges connect former-DEAD cells instead (see value_at) - the
    ridge network rather than the channel network, which can include
    genuine branch points (3+ edges at one cell) since it's only governed
    by Rule 2. A line-graph consumer that assumed degree <= 2 per cell
    would need to account for that; render_skeleton's point-drawing loop
    and render_heightmap's distance transform both already handle any
    degree correctly, since neither assumes a maximum.
    """
    lo, hi = index_range(n, shape, radius)

    def in_view(i, j):
        return lo <= i <= hi and lo <= j <= hi and in_shape(i, j, n, shape, radius)

    seen_edges = set()
    edges = []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if not in_shape(i, j, n, shape, radius):
                continue
            if not effective_live(grid, n, i, j, shape, pruned, invert):
                continue
            for di, dj in RING:
                ni, nj = i + di, j + dj
                if not in_view(ni, nj):
                    continue
                if not effective_live(grid, n, ni, nj, shape, pruned, invert):
                    continue
                edge = frozenset({(i, j), (ni, nj)})
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edges.append(((i, j), (ni, nj)))
    return edges


def render_skeleton(grid, n, path, cell_px=32, line_color=(15, 15, 15),
                     point_color=(180, 60, 60), background=(255, 255, 255),
                     margin_px=24, shape="rhombus", radius=1, line_width=3,
                     point_radius=4, draw_points=True, prune_boundary=False,
                     remove_outer_wall=False, invert=False, color_cycles=False,
                     pnginfo=None):
    """Skeleton/graph view: a line between the centers of every pair of
    ring-adjacent live cells, no cell fill.

    color_cycles: draw each cycle's lines and points in its own color (see
    find_cycles). Ignored under invert, since the inverted skeleton is the
    ridge network, which isn't made of the cycles.

    invert: draws the former-dead ridge network instead of the channel
    network (see value_at/collect_edges) - markers at a branch point just
    get more than 2 lines converging on them, drawn the same as any other
    point.
    """
    r = cell_px / math.sqrt(3)
    width, height, ox, oy = compute_canvas(n, shape, cell_px, margin_px, r, radius)

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    pruned = set()
    if prune_boundary:
        pruned |= compute_pruned_positions(grid, n, shape, radius)
    if remove_outer_wall:
        pruned |= compute_outer_wall_cycle(grid, n, shape, radius)
    pruned = pruned or None

    cycle_id = palette = None
    if color_cycles and not invert:
        cycle_id, sizes, _ = find_cycles(grid, n)
        palette = cycle_palette(len(sizes), dark=True)

    def color_at(i, j, default):
        if cycle_id is None:
            return default
        cid = cycle_of(cycle_id, n, i, j, shape)
        return default if cid is None else palette[cid]

    edges = collect_edges(grid, n, shape, pruned, radius, invert)
    live_points = set()
    for (i, j), (ni, nj) in edges:
        live_points.add((i, j))
        live_points.add((ni, nj))
        x1, y1 = center_px(i, j, cell_px)
        x2, y2 = center_px(ni, nj, cell_px)
        draw.line([(x1 + ox, y1 + oy), (x2 + ox, y2 + oy)],
                  fill=color_at(i, j, line_color), width=line_width)

    # Also include live cells with zero in-view edges (e.g. a border cell
    # in rhombus mode whose rule-satisfying neighbors fall just outside
    # the visible tile) - collect_edges alone would silently drop these
    # since they never appear as an edge endpoint. Pruned positions are
    # deliberately excluded here too, or pruning would just turn a cut
    # path into a lone dot instead of removing it.
    lo, hi = index_range(n, shape, radius)
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if not in_shape(i, j, n, shape, radius):
                continue
            if effective_live(grid, n, i, j, shape, pruned, invert):
                live_points.add((i, j))

    if draw_points:
        for (i, j) in live_points:
            cx, cy = center_px(i, j, cell_px)
            cx, cy = cx + ox, cy + oy
            draw.ellipse([cx - point_radius, cy - point_radius,
                          cx + point_radius, cy + point_radius],
                         fill=color_at(i, j, point_color))

    img.save(path, pnginfo=pnginfo)
    return path


def sigmoid_height_profile(dist_grid, d0, k):
    """Height (in [0, 1]) as a smooth logistic function of dist_grid, the
    distance to the nearest skeleton edge in grid units. Factored out of
    render_heightmap so it can be selected via --profile alongside
    rounded_height_profile.

    h(d) = 1 / (1 + exp(k * (d - d0)))

    Produces a flat plateau at 1 right at the skeleton, a flat plateau at
    0 far from it, and one smooth S-curve transition between - a good
    stand-in for a rounded ridge, but not a literal geometric fillet: k
    and d0 just tune a logistic curve's steepness and midpoint, with no
    actual radius anywhere in the formula (see rounded_height_profile for
    that). d0, k are in grid units (1 grid unit = the distance between
    two ring-adjacent cell centers), not pixels, so they stay meaningful
    across different cell_px/resolution choices.
    """
    return 1.0 / (1.0 + np.exp(k * (dist_grid - d0)))


def rounded_height_profile(dist_grid, thickness, corner_radius):
    """Height (in [0, 1]) as a literal rounded cross-section: a flat
    plateau at 1 out to thickness/2 - corner_radius from the skeleton,
    then a genuine quarter-circle fillet of radius corner_radius (same
    grid units as thickness) curving down to exactly 0 at
    d = thickness/2 - tangent to the plateau at the inner edge (zero
    slope, matching the flat top) and tangent to the base plane at the
    outer edge (vertical slope, meeting it edge-on rather than at a
    corner). Unlike sigmoid_height_profile, corner_radius really is the
    fillet's radius of curvature, not just a tuning knob for a
    curve-shaped-like-a-fillet.

    Two degenerate cases fall out for free at the ends of the valid
    range (0 <= corner_radius <= thickness/2):
      corner_radius = 0             flat-top box profile: 1 for
                                     d < thickness/2, 0 beyond it, with a
                                     hard vertical edge and no rounding
                                     at all.
      corner_radius = thickness/2   fully round profile: the flat
                                     plateau shrinks to nothing and the
                                     fillet alone runs from d=0, giving a
                                     plain round cable/pipe cross-section
                                     rather than a flat top with rounded
                                     shoulders.

    thickness, corner_radius are in grid units, same convention as
    sigmoid_height_profile's d0/k. Caller is responsible for validating
    0 <= corner_radius <= thickness/2 (render_heightmap does this before
    calling in).
    """
    half_width = thickness / 2.0
    flat_edge = half_width - corner_radius
    h = np.where(dist_grid < flat_edge, 1.0, 0.0)
    if corner_radius > 0:
        in_fillet = (dist_grid >= flat_edge) & (dist_grid < half_width)
        u = dist_grid - flat_edge
        fillet_h = np.sqrt(np.clip(corner_radius**2 - u**2, 0, None)) / corner_radius
        h = np.where(in_fillet, fillet_h, h)
    return h


def render_heightmap(grid, n, path, cell_px=32, margin_px=24, shape="rhombus",
                      radius=1, profile="sigmoid", d0=0.4, k=6.0,
                      thickness=0.8, corner_radius=0.0, floor=0.0,
                      line_width_px=2, supersample=2, prune_boundary=False,
                      remove_outer_wall=False, invert=False, pnginfo=None):
    """
    Grayscale heightmap: intensity at each pixel is a function of the
    distance to the nearest skeleton edge, producing a flat plateau right
    at each edge and a flat plateau at the midpoint between parallel
    edges, with a smooth transition between them (see the discussion this
    was designed from - no sharp V-shaped valleys or sharp ridge peaks
    like a naive distance-based height would give).

    profile picks which function makes that transition, from dist_grid
    (distance to the nearest skeleton edge, in grid units) to height in
    [0, 1]:
      sigmoid  (default) sigmoid_height_profile, tuned by d0/k - a smooth
               S-curve that looks like a rounded ridge but isn't a
               literal geometric radius anywhere.
      rounded  rounded_height_profile, tuned by thickness/corner_radius -
               an actual rounded cross-section: flat plateau, then a real
               corner_radius quarter-circle fillet down to the base at
               d = thickness/2.
    d0/k are ignored under "rounded"; thickness/corner_radius are ignored
    under "sigmoid". Both are in grid units (1 grid unit = the distance
    between two ring-adjacent cell centers), not pixels, so they stay
    meaningful across different cell_px/resolution choices - and neither
    adapts to local gap width, both are tuned to the hex lattice's
    typical spacing rather than measured per-pixel.

    floor: rescales h from [0, 1] to [floor, 1], so the valley floor
    doesn't go all the way to zero - useful when this is meant to become
    a physical print (e.g. a coaster base under the path texture), where
    the floor still needs some minimum thickness rather than dropping to
    nothing between paths. 0.0 (default) leaves the valley at true zero;
    e.g. 0.4 keeps the valley at 40% of the ridge height. Applied
    identically regardless of profile.

    When prune_boundary or remove_outer_wall is also on, floor only
    applies near the surviving pattern: a dead cell counts as "part of
    the pattern" (and gets floor treatment) only if it directly touches a
    surviving live cell. Everything else - including areas that used to
    be near a since-removed path - is forced to true zero regardless of
    floor, rather than being uniformly raised across the whole shape.
    With neither on, floor applies uniformly across the whole interior.

    invert: swap which cells count as live before any of the above runs
    (see value_at) - a relabeling of the same solved grid, not a
    different solve. The channel network (former-live cells) stays
    exactly the same simple non-branching rings either way, since that's
    unconditional under Rule 1; the ridge network (former-dead cells) can
    genuinely branch, since it only has to satisfy the weaker Rule 2 -
    collect_edges and the distance transform both handle any degree
    correctly regardless, so branching itself isn't a problem here.

    prune_boundary/remove_outer_wall are still computed on the ORIGINAL,
    non-inverted live cells (see compute_outer_wall_cycle), then applied
    BEFORE the flip: the removed cells are deleted from the original
    pattern and invert treats them like any other original-dead cell. So
    without invert, removing the outer wall lowers that ring to
    background (the near-pattern/true-zero behavior above); with invert,
    the same ring - which is a groove there, since it was original-live -
    fills in and becomes part of the surrounding ridge, so the outermost
    groove is simply the next one in. (An earlier version applied the
    pruning after the flip, which made --remove-outer-wall a silent no-op
    under --invert: those cells were already channel-side, so forcing
    them "dead" changed nothing.)

    The near-pattern test above (what gets floor versus true zero) always
    asks about the surviving ORIGINAL live cells, in either polarity. So
    the pattern's footprint is the same with or without invert: under
    invert the removed wall and everything else outside that footprint
    goes to true zero (dark, like the area outside the shape), while the
    pattern keeps its raised surface and grooves. The edge of the
    footprint is a hard step from the surface (or groove floor) down to
    zero, following the hexagon cell outlines, exactly as it does
    without invert.

    supersample: the line mask is rasterized at this multiple of the final
    resolution, then the distance transform and profile are computed at
    that resolution and downsampled - this avoids the heightmap's
    plateaus/transitions looking pixelated or jagged along the edges.

    Every effectively-live cell also gets a small round dot at its own
    center in that mask, the same diameter as the lines, in addition to
    whatever edges collect_edges finds for it - without this, a live cell
    with zero live neighbors (possible for the inverted ridge network
    under the weaker Rule 2, impossible for the channel network under
    Rule 1 - see value_at) never appears in collect_edges' output and is
    invisible to the distance transform, rendering as a flat, edgeless
    patch rather than a small rounded ridge of its own.
    """
    from scipy.ndimage import distance_transform_edt

    if profile == "rounded":
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        if not (0 <= corner_radius <= thickness / 2):
            raise ValueError(f"corner_radius must be between 0 and "
                              f"thickness/2 ({thickness / 2}), got "
                              f"{corner_radius}")
    elif profile != "sigmoid":
        raise ValueError(f"unknown profile {profile!r}")

    r = cell_px / math.sqrt(3)
    width, height, ox, oy = compute_canvas(n, shape, cell_px, margin_px, r, radius)

    ss = supersample
    ss_w, ss_h = width * ss, height * ss

    # Rasterize the skeleton edges into a binary line mask at supersampled
    # resolution.
    mask_img = Image.new("L", (ss_w, ss_h), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    pruned = set()
    if prune_boundary:
        pruned |= compute_pruned_positions(grid, n, shape, radius)
    if remove_outer_wall:
        pruned |= compute_outer_wall_cycle(grid, n, shape, radius)
    pruned = pruned or None
    edges = collect_edges(grid, n, shape, pruned, radius, invert)
    for (i, j), (ni, nj) in edges:
        x1, y1 = center_px(i, j, cell_px)
        x2, y2 = center_px(ni, nj, cell_px)
        mask_draw.line([((x1 + ox) * ss, (y1 + oy) * ss),
                         ((x2 + ox) * ss, (y2 + oy) * ss)],
                        fill=255, width=line_width_px * ss)

    # Also seed every effectively-live cell's own center with a round dot,
    # the same diameter as the lines - not just the edges collect_edges
    # found for it. collect_edges only emits an edge between two live
    # ring-neighbors, so a live cell with zero live neighbors (legal
    # under the weaker Rule 2 that governs the inverted ridge network;
    # impossible under Rule 1's exactly-2 for the channel network - see
    # value_at) never appears as an edge endpoint and would otherwise be
    # invisible to the distance transform entirely - its hexagon gets its
    # height from whatever unrelated skeleton happens to be nearby rather
    # than a ridge of its own, rendering as a flat, edgeless patch. The
    # same dot also rounds off what would otherwise be PIL's flat-cut
    # line end at any degree-1 (dead-end) cell; for a cell with 2+ edges
    # it's fully redundant with pixels the lines already cover, so this
    # only changes anything for the cells that actually needed it.
    r_dot = (line_width_px * ss) / 2
    lo, hi = index_range(n, shape, radius)
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if not in_shape(i, j, n, shape, radius):
                continue
            if not effective_live(grid, n, i, j, shape, pruned, invert):
                continue
            cx, cy = center_px(i, j, cell_px)
            cx, cy = (cx + ox) * ss, (cy + oy) * ss
            mask_draw.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot],
                               fill=255)

    line_mask = np.array(mask_img) > 0

    # Shape mask: filled hexagon polygons for every included cell, at the
    # same supersampled resolution. Used to hard-clip the output to the
    # actual shape boundary - outside it, the height is exactly 0
    # regardless of floor (a printable coaster has a real physical edge
    # there, not a fade-out).
    shape_img = Image.new("L", (ss_w, ss_h), 0)
    shape_draw = ImageDraw.Draw(shape_img)
    lo, hi = index_range(n, shape, radius)

    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if not in_shape(i, j, n, shape, radius):
                continue
            cx, cy = center_px(i, j, cell_px)
            cx, cy = (cx + ox) * ss, (cy + oy) * ss
            shape_draw.polygon(hexagon_points(cx, cy, r * ss), fill=255)
    shape_mask = np.array(shape_img) > 0

    if prune_boundary or remove_outer_wall:
        # Redefine "background" locally rather than as everything inside
        # the outer shape: a dead cell only counts as part of the pattern
        # (and gets floor treatment) if it's directly touching a
        # surviving live cell. Everything else - including areas that
        # used to be near a now-removed path - is forced to true zero,
        # not raised by floor. Without this, floor would still uniformly
        # raise regions that no longer have any nearby pattern after
        # pruning/removal took out what used to be there.
        #
        # "Pattern" here always means the surviving ORIGINAL live cells,
        # whichever polarity is being drawn (hence invert=False below):
        # the footprint of the pattern is a fact about the pattern, not
        # about whether its cells are displayed as ridges or as grooves.
        # Testing it with invert on would count the dense ridge network
        # as "the pattern", which touches nearly every cell - so the
        # true-zero clip would never fire, and the removed wall would
        # stay a bright band instead of going dark.
        active_img = Image.new("L", (ss_w, ss_h), 0)
        active_draw = ImageDraw.Draw(active_img)

        def in_view(i, j):
            return lo <= i <= hi and lo <= j <= hi and in_shape(i, j, n, shape, radius)

        for i in range(lo, hi + 1):
            for j in range(lo, hi + 1):
                if not in_shape(i, j, n, shape, radius):
                    continue
                is_live = effective_live(grid, n, i, j, shape, pruned, False)
                near_pattern = is_live
                if not near_pattern:
                    for di, dj in RING:
                        ni, nj = i + di, j + dj
                        if in_view(ni, nj) and effective_live(grid, n, ni, nj, shape, pruned, False):
                            near_pattern = True
                            break
                if near_pattern:
                    cx, cy = center_px(i, j, cell_px)
                    cx, cy = (cx + ox) * ss, (cy + oy) * ss
                    active_draw.polygon(hexagon_points(cx, cy, r * ss), fill=255)
        active_mask = np.array(active_img) > 0
    else:
        active_mask = shape_mask


    # Distance (in supersampled pixels) from every pixel to the nearest
    # line pixel: distance_transform_edt gives distance to the nearest
    # *zero* pixel, so invert the mask first (line=0/background,
    # elsewhere=1/foreground) to get distance-to-nearest-line instead.
    dist_px = distance_transform_edt(~line_mask)

    # Convert to grid units: the (i,j) -> pixel mapping is a similarity
    # transform (equal-length, 120-degree-separated basis vectors), so
    # Euclidean pixel distance is just cell_px times Euclidean grid-unit
    # distance, uniformly in every direction - no distortion to correct.
    dist_grid = dist_px / (cell_px * ss)

    if profile == "sigmoid":
        h = sigmoid_height_profile(dist_grid, d0, k)
    else:
        h = rounded_height_profile(dist_grid, thickness, corner_radius)
    h = np.clip(h, 0.0, 1.0)
    h = floor + (1.0 - floor) * h
    h = np.where(active_mask, h, 0.0)
    gray_ss = (h * 255).astype(np.uint8)

    img_ss = Image.fromarray(gray_ss, mode="L")
    img = img_ss.resize((width, height), Image.LANCZOS)
    img.save(path, pnginfo=pnginfo)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_pnginfo(args, grid=None, n=None, upstream=None, argv=None):
    """PNG metadata (tEXt chunks) recording exactly how this image was
    produced, so a rendered PNG is self-documenting even once it's been
    separated from the terminal/script that made it.

    Two chunks are written:
      hex_render_command  sys.argv, shell-quoted (shlex.join) so it can
                           be pasted straight back into a shell to
                           reproduce the image exactly.
      hex_render_args      the parsed argparse Namespace as a JSON object,
                           keyed by flag name - every argument, including
                           ones left at their default, since a default is
                           still part of what produced this exact image.
                           Easier to consume programmatically later than
                           re-parsing the command string.

    Read back later via Image.open(path).text (Pillow loads text chunks
    lazily, so this works fine without an explicit .load() call first).
    """
    info = PngImagePlugin.PngInfo()
    command = shlex.join(argv if argv is not None else sys.argv)
    info.add_text("hex_render_command", command)
    info.add_text("hex_render_args", json.dumps(vars(args), sort_keys=True))

    # The whole pipeline: every record the grid file carried (solver,
    # connect, ...) followed by this render. The grid's hash is compared with
    # the last upstream record's, so a grid edited by hand since is flagged.
    upstream = list(upstream or [])
    digest = grid_digest(grid, n) if grid is not None else None
    matches = (None if not upstream or digest is None
               else upstream[-1].get("grid_sha256") == digest)
    chain = upstream + [{
        "program": "hex_render.py",
        "command": command,
        "reproduce": command,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "args": vars(args),
        "input_file": getattr(args, "input", None),
        "input_grid_sha256": digest,
        "input_matches_upstream": matches,
    }]
    info.add_text("hex_pipeline", json.dumps(chain, sort_keys=True, default=str))
    info.add_text("hex_pipeline_commands",
                  "\n".join(r.get("reproduce") or r.get("command", "") for r in chain))
    return info


def build_parser():
    """The command-line options. Their names (with - turned into _) are also
    the keyword options of render_file()."""
    parser = argparse.ArgumentParser(
        description="Render a hex path grid (from the .txt format) to a PNG."
    )
    parser.add_argument("input", type=str,
                         help="grid text file (first line n, then n lines of 0/1)")
    parser.add_argument("--mode", choices=["fill", "skeleton", "heightmap"], default="fill",
                         help="fill: colored hexagons. skeleton: line graph between "
                              "adjacent live cells. heightmap: grayscale distance "
                              "field from the skeleton edges, see --profile "
                              "(default: fill)")
    parser.add_argument("--shape", choices=["rhombus", "hexagon"], default="rhombus",
                         help="rhombus: just the base tile. hexagon: a hexagon made of "
                              "3*radius^2 rotated rhombus copies, auto-aligned via "
                              "equivalence() (default: rhombus)")
    parser.add_argument("--radius", type=int, default=1,
                         help="hexagon shape only: size multiplier. 1 = the single "
                              "hexagon (3 rhombi). 2 = the same construction extended "
                              "to twice the linear size (12 rhombi: a central 6-rhombus "
                              "star plus 6 outer rhombi) - more room for paths to close "
                              "into loops before hitting the outer boundary. The "
                              "rotational structure is already inherent in "
                              "equivalence(), so this only widens the boundary test - "
                              "no new shift/tiling logic needed (default: 1)")
    parser.add_argument("--prune-boundary", action="store_true",
                         help="drop any path/loop that's cut off by the rendered "
                              "boundary (missing an edge because a neighbor falls "
                              "outside view), leaving only fully self-contained loops. "
                              "Only meaningful with --shape hexagon - in rhombus mode "
                              "nearly every border cell counts as 'cut' (default: off)")
    parser.add_argument("--remove-outer-wall", action="store_true",
                         help="remove the single cycle of live cells that encloses the "
                              "whole rest of the pattern - found by casting a fixed ray "
                              "in from outside and flood-filling from the first live "
                              "cell it hits. Well-defined on a hex_walksat.py "
                              "--wall-boundary solve (that cycle is, by construction, the "
                              "first thing any incoming ray must cross); ambiguous "
                              "otherwise - it will still remove some cycle, just not "
                              "necessarily 'the' outer one (default: off)")
    parser.add_argument("--invert", action="store_true",
                         help="swap live and dead before rendering (see "
                              "value_at) - a relabeling of the same solved "
                              "grid, not a different solve. The channel "
                              "network (former-live cells) stays the same "
                              "simple non-branching rings either way, since "
                              "that's unconditional under Rule 1; the ridge "
                              "network (former-dead cells) can genuinely "
                              "branch, since it only has to satisfy the "
                              "weaker Rule 2. --prune-boundary/"
                              "--remove-outer-wall are computed on the "
                              "original, non-inverted live cells and applied "
                              "before the flip, so under --invert the "
                              "removed ring fills in and merges with the "
                              "surrounding ridge instead of staying a "
                              "groove (default: off)")
    parser.add_argument("--color-cycles", action="store_true",
                         help="fill/skeleton modes: color each distinct cycle of live cells "
                              "differently. Cycles are found through the periodic boundary, so "
                              "a path crossing a seam is one cycle, and every rotated copy of "
                              "a cycle shares its color. Prints the cycle count and sizes.")
    parser.add_argument("--cell-px", type=int, default=32,
                         help="hexagon size in pixels (default: 32)")
    parser.add_argument("--line-width", type=int, default=3,
                         help="skeleton mode: edge line width in pixels (default: 3)")
    parser.add_argument("--point-radius", type=int, default=4,
                         help="skeleton mode: live-cell marker radius in pixels, 0 to "
                              "disable markers (default: 4)")
    parser.add_argument("--d0", type=float, default=0.4,
                         help="heightmap mode, --profile sigmoid only: sigmoid "
                              "transition center, in grid units (default: 0.4)")
    parser.add_argument("--k", type=float, default=6.0,
                         help="heightmap mode, --profile sigmoid only: sigmoid "
                              "steepness, in grid units (default: 6.0)")
    parser.add_argument("--profile", choices=["sigmoid", "rounded"], default="sigmoid",
                         help="heightmap mode only: distance-to-height function. "
                              "sigmoid: smooth logistic curve tuned by --d0/--k - "
                              "looks like a rounded ridge but isn't a literal "
                              "radius. rounded: an actual rounded cross-section - "
                              "flat plateau, then a real --corner-radius "
                              "quarter-circle fillet down to the base at "
                              "d = --thickness/2 (default: sigmoid)")
    parser.add_argument("--thickness", type=float, default=0.8,
                         help="heightmap mode, --profile rounded only: total ridge "
                              "width in grid units - the profile reaches the base "
                              "(before --floor) at distance thickness/2 from the "
                              "skeleton (default: 0.8)")
    parser.add_argument("--corner-radius", type=float, default=0.0,
                         help="heightmap mode, --profile rounded only: fillet "
                              "radius in grid units, must be between 0 and "
                              "thickness/2. 0 (default) gives a flat-top box "
                              "profile with a hard vertical edge; thickness/2 "
                              "gives a fully round profile with no flat top at "
                              "all (default: 0.0)")
    parser.add_argument("--floor", type=float, default=0.0,
                         help="heightmap mode: rescales intensity from [0,1] to "
                              "[floor,1], so the valley doesn't drop to zero - useful "
                              "for a printable coaster base under the path texture "
                              "(default: 0.0, valley at true zero)")
    parser.add_argument("-o", "--output", type=str, default="hex_render.png",
                         help="output PNG path (default: hex_render.png)")
    return parser


def run(args, argv=None, echo=print):
    """Render for parsed arguments exactly as the command line does: read the
    grid, draw, and save the PNG with the provenance chain from the grid file
    plus this render. Returns a namespace with image_path, grid and n. Raises
    UsageError for an option combination that can't be rendered."""
    if argv is None:
        argv = list(sys.argv)

    if args.color_cycles and args.mode == "heightmap":
        raise UsageError("--color-cycles isn't available for --mode heightmap (grayscale)")
    grid, n = read_grid_txt(args.input)
    if args.color_cycles:
        _, cycle_sizes, branching = find_cycles(grid, n)
        echo(describe_cycles(cycle_sizes, branching))
        if args.invert and args.mode == "skeleton":
            echo("note: --color-cycles has no effect on --mode skeleton with --invert "
                  "(the inverted skeleton is the ridge network, not the cycles)")
    out_path = unique_path(args.output)
    pnginfo = build_pnginfo(args, grid, n, read_meta(args.input), argv)

    if args.mode == "fill":
        render_fill(grid, n, out_path, cell_px=args.cell_px, shape=args.shape,
                    radius=args.radius, prune_boundary=args.prune_boundary,
                    remove_outer_wall=args.remove_outer_wall,
                    invert=args.invert, color_cycles=args.color_cycles,
                    pnginfo=pnginfo)
    elif args.mode == "skeleton":
        render_skeleton(grid, n, out_path, cell_px=args.cell_px, shape=args.shape,
                         radius=args.radius, line_width=args.line_width,
                         draw_points=args.point_radius > 0,
                         point_radius=args.point_radius,
                         prune_boundary=args.prune_boundary,
                         remove_outer_wall=args.remove_outer_wall,
                         invert=args.invert, color_cycles=args.color_cycles,
                         pnginfo=pnginfo)
    else:
        render_heightmap(grid, n, out_path, cell_px=args.cell_px, shape=args.shape,
                          radius=args.radius, profile=args.profile, d0=args.d0,
                          k=args.k, thickness=args.thickness,
                          corner_radius=args.corner_radius, floor=args.floor,
                          prune_boundary=args.prune_boundary,
                          remove_outer_wall=args.remove_outer_wall,
                          invert=args.invert, pnginfo=pnginfo)

    echo("read", n, "x", n, "grid from", args.input)
    echo("saved image to", out_path)
    return SimpleNamespace(image_path=out_path, grid=grid, n=n)


def render_file(input, echo=print, **options):
    """Render a grid file from Python, e.g.

        r = render_file("hex_grid_connected_19.txt", mode="heightmap",
                        shape="hexagon", profile="rounded", thickness=1.1,
                        corner_radius=0.4, floor=0, remove_outer_wall=True)
        r.image_path

    Keyword options are the command-line options with - written as _ ; any
    not given take the command line's defaults."""
    parser = build_parser()
    argv = canonical_argv(parser, namespace_from(parser, dict(options, input=input)),
                          "hex_render.py")
    return run(args_for_command(parser, argv), argv=argv, echo=echo)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args, argv=(list(sys.argv) if argv is None else ["hex_render.py"] + list(argv)))
    except UsageError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
