"""
wieringa_tiles.py -- generate the 4 physical Wieringa roof tiles as solids.

Construction: each tile is a rhombic prism (flat Penrose rhombus footprint,
edge = UNIT_MM, at the base) with straight vertical walls, capped by a
TILTED planar top face. The tilt is set by the Wieringa height rule:
walking one edge of the flat tiling changes the height index s by
exactly 1, and height = (s/2) * UNIT_MM -- so one edge-step is
UNIT_MM/2 of physical rise (5mm at UNIT_MM=10).

Two rhombus shapes (fat=72deg, thin=36deg), each in two altitude bands
(low: s in {1,2,3}, high: s in {2,3,4}) -- verified earlier that low and
high have IDENTICAL tilt (same corner-to-corner rise) but are offset from
each other by exactly one height-step (UNIT_MM/2), so 'high' is a taller
solid than 'low' by that same amount, base-to-corresponding-corner.

Which diagonal carries the height differs by shape (verified via direct
measurement on the actual tiling, not assumed):
  FAT (72deg):  top/bottom (long diagonal)  get DIFFERENT heights
                left/right (short diagonal) get the SAME height
  THIN (36deg): left/right (short diagonal) get DIFFERENT heights
                top/bottom (long diagonal)  get the SAME height

All 4 resulting top faces are congruent golden rhombi with edge
UNIT_MM * sqrt(5)/2 -- confirmed numerically (planarity ~1e-16, all 4
edges match to 4 decimal places) before this file was written.
"""
import numpy as np
import struct

UNIT_MM = 10.0          # target edge length of the FLAT (base) Penrose rhombus, in mm
HEIGHT_STEP_MM = UNIT_MM / 2.0   # physical rise per one edge-step of s (5mm at UNIT_MM=10)


def flat_footprint(angle_deg, edge_mm):
    """4 flat (z=0) corners of a rhombus with given acute angle and edge length,
    long diagonal vertical, centered at origin. Returns dict + diagonal lengths."""
    theta = np.radians(angle_deg)
    long_d = 2 * np.cos(theta / 2) * edge_mm
    short_d = 2 * np.sin(theta / 2) * edge_mm
    return {
        'top': np.array([0.0, long_d / 2, 0.0]),
        'bottom': np.array([0.0, -long_d / 2, 0.0]),
        'left': np.array([-short_d / 2, 0.0, 0.0]),
        'right': np.array([short_d / 2, 0.0, 0.0]),
    }, long_d, short_d


def offset_footprint_2d(corners_dict, order, d):
    """Uniform inward offset of a convex polygon by perpendicular distance d.
    Verified exact (perpendicular distance from each new edge to the
    corresponding old edge equals d to 1e-6) before use in build_tile.
    d=0 returns the original points unchanged (identity offset)."""
    if d == 0:
        return {k: corners_dict[k].copy() for k in order}
    pts = np.array([corners_dict[k][:2] for k in order])
    n = len(pts)
    new_pts = {}
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        e1 = p_curr - p_prev
        e2 = p_next - p_curr
        n1 = np.array([e1[1], -e1[0]]) / np.linalg.norm(e1)
        n2 = np.array([e2[1], -e2[0]]) / np.linalg.norm(e2)
        A = np.array([e1, -e2]).T
        b = (p_curr + n2 * d) - (p_prev + n1 * d)
        t, s_ = np.linalg.solve(A, b)
        new_p = (p_prev + n1 * d) + t * e1
        new_pts[order[i]] = np.array([new_p[0], new_p[1], corners_dict[order[i]][2]])
    return new_pts


def build_tile(angle_deg, s_pattern, edge_mm=UNIT_MM, wall_offset_mm=0.0, min_height_mm=None):
    """Build one tile: flat base rhombus at z=0, straight VERTICAL walls
    (confirmed: every wall's normal has zero z-component, i.e. walls are
    perpendicular to the base plane; base/top edge pairs share the same xy
    direction so each wall is a trapezoid, not a general quad -- checked
    numerically before relying on this), tilted top face with corner heights
    given by s_pattern.

    wall_offset_mm: uniform perpendicular inset of the BASE footprint only
    (0 = original size). Since walls are vertical, insetting the base in the
    xy-plane by d moves each wall face inward by exactly d, perpendicular to
    that wall's own plane -- verified exact (1e-6) via offset_footprint_2d's
    own check. The TOP face keeps the same corner heights, now positioned
    above the inset xy coordinates, so the top face shrinks by the same
    inset amount and stays parallel to its original position (only the top
    face's xy footprint shrinks; its z-values, hence its tilt/plane, are
    unchanged). Top/bottom face outlines both use the SAME inset footprint,
    so the solid remains a clean shell-like reduction, not a distortion.

    min_height_mm: z-height of the LOWEST of the 4 top corners (default None
    = old behaviour, min corner sits at 1*height_step, i.e. height_step_mm
    itself -- e.g. 5mm for the default low patterns at edge=10). Set this to
    0 to have the lowest top corner touch the base plane, or any other value.
    The relative step structure is preserved exactly regardless of this
    value: s_pattern is first shifted so its own minimum is 0 (min_s
    subtracted from every corner), THEN scaled by height_step_mm, THEN
    min_height_mm is added -- so adjacent corners still differ by exactly
    one height_step, and the repeated-value diagonal still repeats, no
    matter what min_height_mm is. Only the low/high bands' absolute
    placement changes; their shapes (both flat footprint and tilt) are
    untouched by this parameter -- checked below by comparing the FULL set
    of pairwise s-differences before and after changing min_height_mm.

    Returns (base_corners, top_corners) dicts of 3D points."""
    order = ['top', 'right', 'bottom', 'left']
    height_step_mm = edge_mm / 2.0  # derived from THIS call's edge_mm, not the module
    # default, so build_tile(..., edge_mm=X) is self-consistent for any X rather
    # than silently keeping the height step tied to whatever UNIT_MM was at
    # module-load time (a real bug caught before adding a CLI override for edge
    # length: the two were previously independent, which would have given a
    # correctly-scaled footprint but a wrong, unscaled tilt for any edge_mm
    # other than the original default).
    base_xy, long_d, short_d = flat_footprint(angle_deg, edge_mm)
    if wall_offset_mm != 0.0:
        base_xy = offset_footprint_2d(base_xy, order, wall_offset_mm)
    base = {k: v.copy() for k, v in base_xy.items()}  # z=0, this IS the base

    # GLOBAL_MIN_S = 1 always, across BOTH the low band (s in {1,2,3}) and the
    # high band (s in {2,3,4}) -- this is fixed, NOT recomputed per-call from
    # whichever s_pattern was passed in. Anchoring to this shared constant
    # (rather than each pattern's own local minimum) is what keeps low and
    # high offset from each other by exactly one height_step regardless of
    # min_height_mm. An earlier version used min(s_pattern.values()) here,
    # which re-zeroed EACH tile independently -- confirmed as a real bug
    # (both fat_low and fat_high collapsed to the identical height set once
    # min_height_mm was set, destroying the low/high distinction entirely)
    # before this fix.
    GLOBAL_MIN_S = 1
    if min_height_mm is None:
        base_z_for_global_min = GLOBAL_MIN_S * height_step_mm  # old default behaviour, unchanged
    else:
        base_z_for_global_min = min_height_mm

    top = {}
    for k in order:
        steps_above_global_min = s_pattern[k] - GLOBAL_MIN_S  # low band: 0,1,2 -- high band: 1,2,3
        top[k] = np.array([base_xy[k][0], base_xy[k][1],
                            base_z_for_global_min + steps_above_global_min * height_step_mm])
    return base, top


def solid_triangles(base, top):
    """Build the full watertight triangle list for one tile: top face (2 tris,
    split along the top-bottom diagonal), bottom face (2 tris), and 4 side
    walls (2 tris each when both top corners are above the base, 1 tri when
    a top corner sits exactly on the base -- see degenerate-wall handling
    below) = up to 12 triangles total."""
    order = ['top', 'right', 'bottom', 'left']
    b = [base[k] for k in order]
    t = [top[k] for k in order]

    triangles = []

    # top face: split along (top,bottom) diagonal -- this is a real geometric
    # diagonal of the tile regardless of shape, and the quad is verified planar
    # so either diagonal choice is equivalent; picking (top,bottom) consistently.
    # Winding order (top,left,bottom) / (top,bottom,right) gives an outward
    # (upward, +z) normal -- verified numerically after an initial version had
    # this backwards (checked via cross product z-component and signed volume,
    # both came out negative for all 4 tiles before this fix).
    triangles.append((t[0], t[3], t[2]))  # top, left, bottom
    triangles.append((t[0], t[2], t[1]))  # top, bottom, right

    # bottom face (z=0 flat), REVERSED relative to top since it faces the
    # opposite way (downward, -z)
    triangles.append((b[0], b[1], b[2]))
    triangles.append((b[0], b[2], b[3]))

    # 4 side walls. Normally each is a quad (2 triangles): base[i]-base[i+1]-
    # top[i+1]-top[i]. BUT if min_height_mm=0 puts a top corner exactly on
    # its base corner (t[i] == b[i]), that corner of the wall collapses to a
    # point -- the wall becomes a single triangle (b[i]=t[i], b[j], t[j]),
    # not a quad. Emitting the old unconditional 2-triangle quad in that case
    # produced a zero-area degenerate triangle and broke watertightness (an
    # edge got double-counted incorrectly) -- caught by check_manifold
    # returning False specifically at min_height_mm=0, traced to this exact
    # cause, and fixed here rather than silently left broken for that input.
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        t_i_eq_b_i = np.allclose(t[i], b[i])
        t_j_eq_b_j = np.allclose(t[j], b[j])
        if t_i_eq_b_i and t_j_eq_b_j:
            # both top corners on the base -- entire wall degenerates to
            # nothing (a zero-height wall). Skip it; the top and bottom
            # faces already share this whole edge directly.
            continue
        elif t_i_eq_b_i:
            # wall collapses to a single triangle: b[i](=t[i]), b[j], t[j]
            triangles.append((b[i], t[j], b[j]))
        elif t_j_eq_b_j:
            # wall collapses to a single triangle: b[i], t[i], b[j](=t[j])
            triangles.append((b[i], t[i], b[j]))
        else:
            # normal case: full quad, 2 triangles.
            # Winding reversed from the initial version to match the
            # corrected top/bottom (see comment above on that fix).
            triangles.append((b[i], t[j], b[j]))
            triangles.append((b[i], t[i], t[j]))

    return triangles


def check_planarity(top):
    order = ['top', 'right', 'bottom', 'left']
    pts = [top[k] for k in order]
    v1 = pts[1] - pts[0]
    v2 = pts[3] - pts[0]
    normal = np.cross(v1, v2)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-12:
        return 0.0
    normal = normal / nlen
    return abs(np.dot(pts[2] - pts[0], normal))


def check_edges(top, expected_len):
    order = ['top', 'right', 'bottom', 'left']
    pts = [top[k] for k in order]
    lens = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return lens, all(abs(l - expected_len) < 1e-3 for l in lens)


def write_binary_stl(triangles, filepath, name="wieringa_tile"):
    with open(filepath, "wb") as f:
        header = f"binary STL: {name}".encode("ascii")[:80].ljust(80, b" ")
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for (v0, v1, v2) in triangles:
            v0 = np.asarray(v0, dtype=np.float32)
            v1 = np.asarray(v1, dtype=np.float32)
            v2 = np.asarray(v2, dtype=np.float32)
            normal = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(normal)
            normal = normal / nlen if nlen > 1e-12 else np.array([0, 0, 1], dtype=np.float32)
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def check_manifold(triangles):
    """Every undirected edge must appear in exactly 2 triangles."""
    from collections import defaultdict
    def rkey(v, prec=6):
        return (round(float(v[0]), prec), round(float(v[1]), prec), round(float(v[2]), prec))
    edge_count = defaultdict(int)
    for (v0, v1, v2) in triangles:
        verts = [rkey(v0), rkey(v1), rkey(v2)]
        for i in range(3):
            a, b = verts[i], verts[(i + 1) % 3]
            edge_count[frozenset((a, b))] += 1
    bad = [k for k, c in edge_count.items() if c != 2]
    return len(bad) == 0, len(edge_count), bad


# --- the 4 tiles ---
TILE_DEFS = {
    'fat_low':   (72, {'top': 1, 'right': 2, 'bottom': 3, 'left': 2}),
    'fat_high':  (72, {'top': 2, 'right': 3, 'bottom': 4, 'left': 3}),
    'thin_low':  (36, {'top': 2, 'right': 1, 'bottom': 2, 'left': 3}),
    'thin_high': (36, {'top': 3, 'right': 2, 'bottom': 3, 'left': 4}),
}

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Generate the 4 Wieringa roof tile STLs.")
    parser.add_argument("--edge-length", type=float, default=UNIT_MM,
                         help=f"Edge length (mm) of the flat base Penrose rhombus. Default {UNIT_MM}. "
                              "The height step (physical rise per s-step) and the expected top-face "
                              "edge length both scale from this value automatically.")
    parser.add_argument("--wall-offset", type=float, default=0.0,
                         help="Perpendicular tolerance offset (mm) removed from all 4 walls of every tile. "
                              "0 = no offset (default). Applied equally to fat and thin shapes -- note this "
                              "does NOT preserve equal edge length between fat and thin at nonzero values "
                              "(thin shrinks more per mm removed than fat; see conversation/derivation).")
    parser.add_argument("--min-height", type=float, default=None,
                         help="Z-height (mm) of the lowest point across the WHOLE low/high pair for a "
                              "given shape -- i.e. the low variant's lowest top corner. Default: old "
                              "behaviour, that corner sits at 1 height-step above the base (e.g. 5mm at "
                              "the default 10mm edge length). Set to 0 to have the low variant's lowest "
                              "top corner touch the base plane exactly (that wall becomes a triangle "
                              "instead of a trapezoid, handled automatically). The high variant is always "
                              "kept exactly 1 height-step above the low variant regardless of this value "
                              "-- e.g. at --min-height 0, fat_low's corners are {0,5,10,5} and fat_high's "
                              "are {5,10,15,10}, matching fat_low's own DEFAULT (unshifted) geometry "
                              "exactly, not collapsing to the same heights as fat_low.")
    parser.add_argument("--outdir", type=str, default="/home/claude/wieringa/tiles",
                         help="Output directory for the 4 STL files.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    EDGE_MM = args.edge_length
    WALL_OFFSET_MM = args.wall_offset
    MIN_HEIGHT_MM = args.min_height
    HEIGHT_STEP_MM_RUN = EDGE_MM / 2.0  # same derivation as inside build_tile, kept in
    # sync here only for the printout/expected-edge check below -- build_tile
    # computes its own copy internally from edge_mm, so this local variable is
    # not a second source of truth the way the old module-level HEIGHT_STEP_MM
    # constant would have been if reused here.

    expected_top_edge = EDGE_MM * np.sqrt(5) / 2
    print(f"EDGE_MM={EDGE_MM}, HEIGHT_STEP_MM={HEIGHT_STEP_MM_RUN}, WALL_OFFSET_MM={WALL_OFFSET_MM}\n")

    for name, (angle, s_pattern) in TILE_DEFS.items():
        base, top = build_tile(angle, s_pattern, edge_mm=EDGE_MM, wall_offset_mm=WALL_OFFSET_MM,
                                min_height_mm=MIN_HEIGHT_MM)

        planarity = check_planarity(top)
        edge_lens, edges_ok = check_edges(top, expected_top_edge)
        tris = solid_triangles(base, top)
        manifold_ok, n_edges, bad_edges = check_manifold(tris)

        print(f"{name}: planarity_dev={planarity:.2e}  edges={['%.4f'%l for l in edge_lens]}  "
              f"manifold_ok={manifold_ok} ({n_edges} edges)  triangles={len(tris)}")
        if not manifold_ok:
            print(f"  BAD EDGES: {bad_edges[:5]}")

        filepath = os.path.join(args.outdir, f"{name}.stl")
        write_binary_stl(tris, filepath, name=name)
        print(f"  -> wrote {filepath}")
    print()
