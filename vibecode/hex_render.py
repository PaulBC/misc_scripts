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

Both modes support --full-hex, reconstructing the full hexagon (3 rotated
copies of the base rhombus) via the same equivalence() lattice quotient
used by the solver, rather than needing manually rotated/aligned copies.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw


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


def value_at(grid, n, i, j, full_hex):
    """Live/dead state at (i, j), folding through equivalence() when
    outside the base tile and full_hex is enabled."""
    if full_hex:
        ci, cj = equivalence(i, j, n)
    else:
        ci, cj = i, j
    return grid[ci][cj]


def index_range(n, full_hex):
    return (-n, n - 1) if full_hex else (0, n - 1)


def compute_pruned_positions(grid, n, full_hex):
    """Find every rendered POSITION (i, j) - not canonical grid cell, since
    different rotated copies of the same cell can have different boundary
    status - that belongs to a path/loop missing at least one edge because
    it runs off the rendered region, then return the full set of positions
    in that same connected component (the whole cut path, not just the
    cell where it happens to be missing an edge).

    Doesn't touch the grid at all - this only decides which on-screen
    instances to hide. Most meaningful with full_hex=True, where the
    boundary being crossed is the hexagon's true edge; in base-rhombus
    mode nearly every border cell is "cut" in this sense (it's simply
    where the single tile stops), so pruning there would remove most of
    the image.
    """
    lo, hi = index_range(n, full_hex)

    def in_view(i, j):
        if full_hex:
            return lo <= i <= hi and lo <= j <= hi and not is_extra_cell(i, j, n)
        return lo <= i <= hi and lo <= j <= hi

    live_positions = []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if full_hex and is_extra_cell(i, j, n):
                continue
            if value_at(grid, n, i, j, full_hex):
                live_positions.append((i, j))

    adjacency = {}
    visible_degree = {}
    true_degree = {}
    for (i, j) in live_positions:
        nbrs_in_view = []
        true_count = 0
        for di, dj in RING:
            ni, nj = i + di, j + dj
            if value_at(grid, n, ni, nj, full_hex):
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


def effective_live(grid, n, i, j, full_hex, pruned=None):
    """Like value_at, but positions in `pruned` are treated as dead."""
    if pruned is not None and (i, j) in pruned:
        return False
    return value_at(grid, n, i, j, full_hex)


def compute_canvas(n, full_hex, cell_px, margin_px, r):
    lo, hi = index_range(n, full_hex)
    if full_hex:
        # The naive bounding box of the SQUARE's 4 corners is noticeably
        # looser than the hexagon's actual extent (that's exactly the
        # n^2 leftover corner cells from the is_extra_cell derivation) -
        # so scan the actually-included cells for a tight fit instead.
        xs, ys = [], []
        for i in range(lo, hi + 1):
            for j in range(lo, hi + 1):
                if is_extra_cell(i, j, n):
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
# Render modes
# ---------------------------------------------------------------------------

def render_fill(grid, n, path, cell_px=32, live_color=(222, 184, 135),
                 dead_color=(15, 15, 15), edge_color=(106, 90, 205),
                 background=(255, 255, 255), margin_px=24, full_hex=False,
                 prune_boundary=False):
    """Filled hexagons - live/dead colored differently."""
    r = cell_px / math.sqrt(3)
    width, height, ox, oy = compute_canvas(n, full_hex, cell_px, margin_px, r)

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    pruned = compute_pruned_positions(grid, n, full_hex) if prune_boundary else None

    lo, hi = index_range(n, full_hex)
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if full_hex and is_extra_cell(i, j, n):
                continue
            cx, cy = center_px(i, j, cell_px)
            cx, cy = cx + ox, cy + oy
            color = live_color if effective_live(grid, n, i, j, full_hex, pruned) else dead_color
            draw.polygon(hexagon_points(cx, cy, r), fill=color, outline=edge_color, width=1)

    img.save(path)
    return path


def collect_edges(grid, n, full_hex, pruned=None):
    """Returns the list of live-live ring-adjacent cell pairs as grid-space
    segments [((i1,j1), (i2,j2)), ...], each edge appearing once. Shared by
    render_skeleton and render_heightmap.

    In full_hex mode, values wrap through equivalence() but positions
    don't, so segments are in true (unwrapped) space and cross the rhombus
    seams cleanly. In base-rhombus mode, a live cell's neighbor is only
    included if it also falls inside the visible n x n tile.

    pruned: optional set of positions (from compute_pruned_positions) to
    treat as dead - used to drop boundary-cut paths entirely.
    """
    lo, hi = index_range(n, full_hex)

    def in_view(i, j):
        if full_hex:
            return lo <= i <= hi and lo <= j <= hi and not is_extra_cell(i, j, n)
        return lo <= i <= hi and lo <= j <= hi

    seen_edges = set()
    edges = []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if full_hex and is_extra_cell(i, j, n):
                continue
            if not effective_live(grid, n, i, j, full_hex, pruned):
                continue
            for di, dj in RING:
                ni, nj = i + di, j + dj
                if not in_view(ni, nj):
                    continue
                if not effective_live(grid, n, ni, nj, full_hex, pruned):
                    continue
                edge = frozenset({(i, j), (ni, nj)})
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edges.append(((i, j), (ni, nj)))
    return edges


def render_skeleton(grid, n, path, cell_px=32, line_color=(15, 15, 15),
                     point_color=(180, 60, 60), background=(255, 255, 255),
                     margin_px=24, full_hex=False, line_width=3,
                     point_radius=4, draw_points=True, prune_boundary=False):
    """Skeleton/graph view: a line between the centers of every pair of
    ring-adjacent live cells, no cell fill."""
    r = cell_px / math.sqrt(3)
    width, height, ox, oy = compute_canvas(n, full_hex, cell_px, margin_px, r)

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    pruned = compute_pruned_positions(grid, n, full_hex) if prune_boundary else None

    edges = collect_edges(grid, n, full_hex, pruned)
    live_points = set()
    for (i, j), (ni, nj) in edges:
        live_points.add((i, j))
        live_points.add((ni, nj))
        x1, y1 = center_px(i, j, cell_px)
        x2, y2 = center_px(ni, nj, cell_px)
        draw.line([(x1 + ox, y1 + oy), (x2 + ox, y2 + oy)],
                  fill=line_color, width=line_width)

    # Also include live cells with zero in-view edges (e.g. a border cell
    # in base-rhombus mode whose rule-satisfying neighbors fall just
    # outside the visible tile) - collect_edges alone would silently drop
    # these since they never appear as an edge endpoint. Pruned positions
    # are deliberately excluded here too, or pruning would just turn a cut
    # path into a lone dot instead of removing it.
    lo, hi = index_range(n, full_hex)
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if full_hex and is_extra_cell(i, j, n):
                continue
            if effective_live(grid, n, i, j, full_hex, pruned):
                live_points.add((i, j))

    if draw_points:
        for (i, j) in live_points:
            cx, cy = center_px(i, j, cell_px)
            cx, cy = cx + ox, cy + oy
            draw.ellipse([cx - point_radius, cy - point_radius,
                          cx + point_radius, cy + point_radius], fill=point_color)

    img.save(path)
    return path


def render_heightmap(grid, n, path, cell_px=32, margin_px=24, full_hex=False,
                      d0=0.4, k=6.0, floor=0.0, line_width_px=2, supersample=2,
                      prune_boundary=False):
    """
    Grayscale heightmap: intensity at each pixel is a sigmoid function of
    the distance to the nearest skeleton edge, producing a flat plateau
    right at each edge and a flat plateau at the midpoint between parallel
    edges, with a smooth S-curve transition between them (see the
    discussion this was designed from - no sharp V-shaped valleys or
    sharp ridge peaks like a naive distance-based height would give).

    h(d) = 1 / (1 + exp(k * (d - d0)))

    d0, k are in *grid units* (1 grid unit = the distance between two
    ring-adjacent cell centers), not pixels, so they stay meaningful
    across different cell_px/resolution choices. d0 is fixed rather than
    adapted to local gap width - tuned to the hex lattice's typical
    spacing rather than measured per-pixel.

    floor: rescales h from [0, 1] to [floor, 1], so the valley floor
    doesn't go all the way to zero - useful when this is meant to become
    a physical print (e.g. a coaster base under the path texture), where
    the floor still needs some minimum thickness rather than dropping to
    nothing between paths. 0.0 (default) leaves the valley at true zero;
    e.g. 0.4 keeps the valley at 40% of the ridge height.

    When prune_boundary is also on, floor only applies near the surviving
    pattern: a dead cell counts as "part of the pattern" (and gets floor
    treatment) only if it directly touches a surviving live cell.
    Everything else - including areas that used to be near a since-pruned
    path - is forced to true zero regardless of floor, rather than being
    uniformly raised across the whole shape. Without pruning, floor
    applies uniformly across the whole interior as before.

    supersample: the line mask is rasterized at this multiple of the final
    resolution, then the distance transform and sigmoid are computed at
    that resolution and downsampled - this avoids the heightmap's
    plateaus/transitions looking pixelated or jagged along the edges.
    """
    from scipy.ndimage import distance_transform_edt

    r = cell_px / math.sqrt(3)
    width, height, ox, oy = compute_canvas(n, full_hex, cell_px, margin_px, r)

    ss = supersample
    ss_w, ss_h = width * ss, height * ss

    # Rasterize the skeleton edges into a binary line mask at supersampled
    # resolution.
    mask_img = Image.new("L", (ss_w, ss_h), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    pruned = compute_pruned_positions(grid, n, full_hex) if prune_boundary else None
    edges = collect_edges(grid, n, full_hex, pruned)
    for (i, j), (ni, nj) in edges:
        x1, y1 = center_px(i, j, cell_px)
        x2, y2 = center_px(ni, nj, cell_px)
        mask_draw.line([((x1 + ox) * ss, (y1 + oy) * ss),
                         ((x2 + ox) * ss, (y2 + oy) * ss)],
                        fill=255, width=line_width_px * ss)

    line_mask = np.array(mask_img) > 0

    # Shape mask: filled hexagon polygons for every included cell, at the
    # same supersampled resolution. Used to hard-clip the output to the
    # actual rhombus/hexagon boundary - outside it, the height is exactly
    # 0 regardless of floor (a printable coaster has a real physical edge
    # there, not a fade-out).
    shape_img = Image.new("L", (ss_w, ss_h), 0)
    shape_draw = ImageDraw.Draw(shape_img)
    lo, hi = index_range(n, full_hex)
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if full_hex and is_extra_cell(i, j, n):
                continue
            cx, cy = center_px(i, j, cell_px)
            cx, cy = (cx + ox) * ss, (cy + oy) * ss
            shape_draw.polygon(hexagon_points(cx, cy, r * ss), fill=255)
    shape_mask = np.array(shape_img) > 0

    if prune_boundary:
        # Redefine "background" locally rather than as everything inside
        # the outer shape: a dead cell only counts as part of the pattern
        # (and gets floor treatment) if it's directly touching a
        # surviving live cell. Everything else - including areas that
        # used to be near a now-pruned path - is forced to true zero, not
        # raised by floor. Without this, floor would still uniformly
        # raise regions that no longer have any nearby pattern after
        # pruning removed what used to be there.
        active_img = Image.new("L", (ss_w, ss_h), 0)
        active_draw = ImageDraw.Draw(active_img)

        def in_view(i, j):
            if full_hex:
                return lo <= i <= hi and lo <= j <= hi and not is_extra_cell(i, j, n)
            return lo <= i <= hi and lo <= j <= hi

        for i in range(lo, hi + 1):
            for j in range(lo, hi + 1):
                if full_hex and is_extra_cell(i, j, n):
                    continue
                is_live = effective_live(grid, n, i, j, full_hex, pruned)
                near_pattern = is_live
                if not near_pattern:
                    for di, dj in RING:
                        ni, nj = i + di, j + dj
                        if in_view(ni, nj) and effective_live(grid, n, ni, nj, full_hex, pruned):
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

    h = 1.0 / (1.0 + np.exp(k * (dist_grid - d0)))
    h = floor + (1.0 - floor) * h
    h = np.where(active_mask, h, 0.0)
    gray_ss = (h * 255).astype(np.uint8)

    img_ss = Image.fromarray(gray_ss, mode="L")
    img = img_ss.resize((width, height), Image.LANCZOS)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def unique_path(path):
    base, ext = os.path.splitext(path)
    n = 1
    candidate = f"{base}_{n}{ext}"
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}_{n}{ext}"
    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a hex path grid (from the .txt format) to a PNG."
    )
    parser.add_argument("input", type=str,
                         help="grid text file (first line n, then n lines of 0/1)")
    parser.add_argument("--mode", choices=["fill", "skeleton", "heightmap"], default="fill",
                         help="fill: colored hexagons. skeleton: line graph between "
                              "adjacent live cells. heightmap: grayscale sigmoid "
                              "distance field from the skeleton edges (default: fill)")
    parser.add_argument("--full-hex", action="store_true",
                         help="render the full hexagon (3 rotated copies of the base "
                              "rhombus) instead of just the base rhombus")
    parser.add_argument("--prune-boundary", action="store_true",
                         help="drop any path/loop that's cut off by the rendered "
                              "boundary (missing an edge because a neighbor falls "
                              "outside view), leaving only fully self-contained loops. "
                              "Only meaningful with --full-hex - in base-rhombus mode "
                              "nearly every border cell counts as 'cut' (default: off)")
    parser.add_argument("--cell-px", type=int, default=32,
                         help="hexagon size in pixels (default: 32)")
    parser.add_argument("--line-width", type=int, default=3,
                         help="skeleton mode: edge line width in pixels (default: 3)")
    parser.add_argument("--point-radius", type=int, default=4,
                         help="skeleton mode: live-cell marker radius in pixels, 0 to "
                              "disable markers (default: 4)")
    parser.add_argument("--d0", type=float, default=0.4,
                         help="heightmap mode: sigmoid transition center, in grid units "
                              "(default: 0.4)")
    parser.add_argument("--k", type=float, default=6.0,
                         help="heightmap mode: sigmoid steepness, in grid units "
                              "(default: 6.0)")
    parser.add_argument("--floor", type=float, default=0.0,
                         help="heightmap mode: rescales intensity from [0,1] to "
                              "[floor,1], so the valley doesn't drop to zero - useful "
                              "for a printable coaster base under the path texture "
                              "(default: 0.0, valley at true zero)")
    parser.add_argument("-o", "--output", type=str, default="hex_render.png",
                         help="output PNG path (default: hex_render.png)")
    args = parser.parse_args()

    grid, n = read_grid_txt(args.input)
    out_path = unique_path(args.output)

    if args.mode == "fill":
        render_fill(grid, n, out_path, cell_px=args.cell_px, full_hex=args.full_hex,
                    prune_boundary=args.prune_boundary)
    elif args.mode == "skeleton":
        render_skeleton(grid, n, out_path, cell_px=args.cell_px, full_hex=args.full_hex,
                         line_width=args.line_width,
                         draw_points=args.point_radius > 0,
                         point_radius=args.point_radius,
                         prune_boundary=args.prune_boundary)
    else:
        render_heightmap(grid, n, out_path, cell_px=args.cell_px, full_hex=args.full_hex,
                          d0=args.d0, k=args.k, floor=args.floor,
                          prune_boundary=args.prune_boundary)

    print("read", n, "x", n, "grid from", args.input)
    print("saved image to", out_path)
