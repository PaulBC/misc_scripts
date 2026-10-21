"""
Domed Voronoi tiles from the upper envelope of equal spheres.

Put a sphere of radius S at every point (all centers at the same height).
The upper envelope is, at each (x, y), the sphere of the NEAREST site, so
its creases project onto the ordinary Voronoi edges. A horizontal plane
below the interior creases trims away the (huge) lower half of the union of
balls. The result is the slab between the plane and the envelope:

    height(x, y) = max(0, sqrt(S^2 - d^2) - h)

where d is the distance to the nearest site and h is the plane's height
above the sphere centers. Cells with no neighbor to cut them off (outer
rim, gaps between clusters) keep curving down until they hit the plane,
which gives a circular arc of radius R = sqrt(S^2 - h^2) around the site.

Sphere radius:  S = --sphere-factor x L_ref, where L_ref is a low
                percentile of the Delaunay edge lengths.
Plane default:  just below the lowest INTERIOR Voronoi vertex, where
                interior means every cell meeting at that vertex is a
                finite cell, and the vertex is not a "bridge" between
                clusters (circumradius > --bridge-factor x the median).
                Override with --plane-height.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from PIL import Image, PngImagePlugin
from scipy.spatial import ConvexHull, Voronoi, cKDTree


def read_points(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                points.append([x, y])
    return np.array(points)


DEFAULT_POINTS = [
    (0.000000, 0.000000), (1.000000, 0.000000), (0.500000, -0.363271),
    (0.309017, -0.951057), (-0.500000, -0.363271), (-0.690983, -0.951057),
    (-0.500000, -1.538842), (0.000000, -1.902113), (0.500000, -1.538842),
    (1.000000, -1.902113), (0.190983, -2.489898), (0.690983, -2.853170),
    (1.309017, -2.853170), (1.809017, -2.489898), (1.618034, -1.902113),
    (2.118034, -1.538842), (2.427051, -2.489898), (2.927051, -2.126627),
    (3.118034, -1.538842), (2.927051, -0.951057), (2.309017, -0.951057),
    (2.118034, -0.363271), (3.118034, -0.363271), (2.927051, 0.224514),
    (2.427051, 0.587785), (1.809017, 0.587785), (1.618034, -0.000000),
    (1.309017, 0.951057), (0.690983, 0.951057), (0.190983, 0.587785),
    (-0.809017, 0.587785), (-0.000000, 1.175571), (0.500000, 1.538842),
    (1.118034, 1.538842), (2.118034, 1.538842), (2.927051, 0.951057),
    (3.427051, 0.587785), (3.618034, -0.000000), (3.927051, -0.951057),
    (3.618034, -1.902113), (3.427051, -2.489898), (2.927051, -2.853170),
    (2.118034, -3.440955), (1.118034, -3.440955), (0.500000, -3.440955),
    (0.000000, -3.077684), (-0.809017, -2.489898), (-1.118034, -1.538842),
    (-1.309017, -0.951057), (-1.118034, -0.363271), (1.309017, -0.951057),
    (-0.190983, -0.587785), (-0.190983, -1.314328), (0.500000, -2.265384),
    (1.190983, -2.489898), (2.309017, -2.126627), (2.736068, -1.538842),
    (2.736068, -0.363271), (2.309017, 0.224514), (1.190983, 0.587785),
    (0.500000, 0.363271), (-0.190983, 0.587785), (1.618034, 1.175571),
    (2.309017, 0.951057), (3.427051, -0.587785), (3.427051, -1.314328),
    (2.309017, -2.853170), (1.618034, -3.077684), (-0.190983, -2.489898),
    (-0.618034, -1.902113), (-0.618034, -0.000000), (1.118034, -0.363271),
    (0.690983, -0.951057), (1.118034, -1.538842), (1.809017, -1.314328),
    (1.809017, -0.587785),
]


def delaunay_edge_lengths(vor):
    # vor.ridge_points is exactly the Delaunay edge list (the dual graph)
    p0 = vor.points[vor.ridge_points[:, 0]]
    p1 = vor.points[vor.ridge_points[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1)


def certified_cells(vor):
    """Boolean mask of sites whose Voronoi cell is provably exact.

    Assumes the points are ALL the points of a larger tiling that lie in some
    convex window, so everything that was removed is outside the convex hull.
    A Voronoi vertex is the center of an empty circle through three sites; if
    that whole circle lies inside the hull, no removed point can be in it, so
    the vertex is real. A cell is exact when all of its vertices are real
    (the true cell is convex, contains every vertex, and can only shrink).
    This drops infinite cells AND the finite-but-distorted layer(s) next to
    them, without needing to know which points were removed.
    """
    pts = vor.points
    eq = ConvexHull(pts).equations  # n.x + b <= 0 inside
    rho = np.full(len(vor.vertices), np.inf)
    seen = np.zeros(len(vor.vertices), bool)
    for i, r in enumerate(vor.point_region):
        for v in vor.regions[r]:
            if v >= 0 and not seen[v]:
                seen[v] = True
                rho[v] = np.linalg.norm(vor.vertices[v] - pts[i])
    depth = -(vor.vertices @ eq[:, :2].T + eq[:, 2]).max(axis=1)  # dist to hull
    real = rho <= depth
    ok = np.zeros(len(pts), bool)
    for i, r in enumerate(vor.point_region):
        reg = vor.regions[r]
        ok[i] = len(reg) > 0 and -1 not in reg and real[reg].all()
    return ok


def shape_classes(vor, max_radius, tol=1e-3):
    """Group finite cells into congruence classes (rotation/reflection),
    comparing, for each vertex, the edge length and the distance to the site.
    Returns an int class id per site, -1 for infinite or numerically
    degenerate cells (any vertex farther than max_radius from its site)."""
    pts = vor.points
    ids = np.full(len(pts), -1)
    reps = []  # (n_vertices, flattened canonical sequence)

    def candidates(V, site):
        out = []
        for W in (V, V[::-1]):
            edges = np.linalg.norm(np.roll(W, -1, 0) - W, axis=1)
            rad = np.linalg.norm(W - site, axis=1)
            seq = np.column_stack([edges, rad])
            for k in range(len(W)):
                out.append(np.roll(seq, -k, 0).ravel())
        return out

    for i, r in enumerate(vor.point_region):
        reg = vor.regions[r]
        if len(reg) == 0 or -1 in reg:
            continue
        V = vor.vertices[reg]
        if np.linalg.norm(V - pts[i], axis=1).max() > max_radius:
            continue
        ang = np.arctan2(V[:, 1] - pts[i][1], V[:, 0] - pts[i][0])
        V = V[np.argsort(ang)]
        cands = candidates(V, pts[i])
        for cid, (n, rep) in enumerate(reps):
            if n == len(V) and any(np.abs(c - rep).max() <= tol for c in cands):
                ids[i] = cid
                break
        else:
            reps.append((len(V), cands[0]))
            ids[i] = len(reps) - 1
    return ids


def extend_by_shape(vor, certified, max_radius, min_count):
    """Also keep finite cells whose shape class is recognized: the class
    contains a certified-exact cell, or has at least min_count members.
    (Truncated cells are idiosyncratic; real tile shapes repeat.)"""
    ids = shape_classes(vor, max_radius)
    keep = certified.copy()
    print(f"Shape classes among finite cells: {ids.max() + 1}")
    for cid in range(ids.max() + 1):
        members = np.where(ids == cid)[0]
        n_cert = int(certified[members].sum())
        accepted = n_cert > 0 or len(members) >= min_count
        if accepted:
            keep[members] = True
        reg = vor.regions[vor.point_region[members[0]]]
        area = ConvexHull(vor.vertices[reg]).volume
        print(f"  class {cid}: {len(reg)}-gon area={area:.3f}  "
              f"members={len(members):2d} certified={n_cert:2d}  "
              f"{'kept' if accepted else 'rejected'}")
    return keep


def cells_inside_circle(vor, center, radius):
    """Mask of sites whose Voronoi cell is finite and lies entirely inside
    the circle (all vertices within `radius` of `center`) -- the same
    "complete cells inside the circle" selection as voronoi_surface4.py."""
    inside = ((vor.vertices - np.asarray(center)) ** 2).sum(axis=1) <= radius ** 2
    ok = np.zeros(len(vor.points), bool)
    for i, r in enumerate(vor.point_region):
        reg = vor.regions[r]
        ok[i] = len(reg) > 0 and -1 not in reg and inside[reg].all()
    return ok


def interior_vertex_circumradii(vor, cell_ok=None, require_all=True):
    """Circumradii of Voronoi vertices where EVERY meeting cell is finite
    (and, if cell_ok is given, also marked ok).

    Stricter than "finite vertex": hull cells can have finite vertices that
    are numerical blowups (near-collinear triples), but those cells are
    unbounded, so they are excluded here.
    """
    good = np.array([
        len(vor.regions[r]) > 0 and -1 not in vor.regions[r]
        for r in vor.point_region
    ])
    if cell_ok is not None:
        good &= cell_ok
    owners = defaultdict(list)
    for i, r in enumerate(vor.point_region):
        for v in vor.regions[r]:
            if v >= 0:
                owners[v].append(i)
    radii = []
    for v, cells in owners.items():
        if (all if require_all else any)(good[c] for c in cells):
            radii.append(np.linalg.norm(vor.vertices[v] - vor.points[cells[0]]))
    return np.array(radii)


META_KEY = "VoronoiSpheresArguments"


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_png_text(path):
    with Image.open(path) as im:
        return dict(getattr(im, "text", {}) or im.info)


def expand_from_png(argv):
    """Replace '--from-png FILE' by the arguments stored in that PNG; any other
    arguments given on the command line come after them and override them."""
    for i, a in enumerate(argv):
        if a == "--from-png" or a.startswith("--from-png="):
            if a == "--from-png":
                path, rest = argv[i + 1], argv[:i] + argv[i + 2:]
            else:
                path, rest = a.split("=", 1)[1], argv[:i] + argv[i + 1:]
            text = read_png_text(path)
            if META_KEY not in text:
                raise SystemExit(f"{path} has no stored voronoi_spheres arguments.")
            return json.loads(text[META_KEY]) + rest
    return argv


def main():
    parser = argparse.ArgumentParser(
        description="Domed Voronoi tiles: upper envelope of equal spheres "
                    "cut by a horizontal plane")
    parser.add_argument("--pts", metavar="FILE", default=None,
                        help="text file with 'x y' per line (default: built-in set)")
    parser.add_argument("--size", type=int, default=800, help="image size in pixels")
    parser.add_argument("--percentile", type=float, default=25.0,
                        help="percentile of Delaunay edge lengths used as the "
                             "reference length L_ref (default 25)")
    parser.add_argument("--sphere-factor", type=float, default=5.0,
                        help="sphere radius S = factor x L_ref (default 5)")
    parser.add_argument("--sphere-mm", type=float, default=None,
                        help="sphere radius in mm (overrides --sphere-factor; "
                             "requires --width-mm, which fixes the scale)")
    parser.add_argument("--plane-offset", type=float, default=0.03,
                        help="default plane position: this many L_ref below the "
                             "lowest interior vertex (default 0.03)")
    parser.add_argument("--bridge-factor", type=float, default=2.0,
                        help="interior vertices with circumradius above this "
                             "multiple of the median are treated as bridging "
                             "between clusters and do not set the plane "
                             "(default 2.0)")
    parser.add_argument("--plane-height", type=float, default=None,
                        help="override: absolute plane height above the sphere "
                             "centers (same units as the points)")
    parser.add_argument("--clip-center", nargs=2, type=float, metavar=("X", "Y"),
                        default=None,
                        help="center of the circular cutoff region "
                             "(default: center of the points' bounding box)")
    parser.add_argument("--clip-radius", type=float, default=None,
                        help="cutoff radius: keep only sites whose complete "
                             "Voronoi cell lies inside the circle; the rest "
                             "are removed before the spheres are built")
    parser.add_argument("--prune", action="store_true",
                        help="prune incomplete cells: build the surface from ALL "
                             "sites so every kept cell keeps its true Voronoi "
                             "shape, and remove the pixels owned by cells that "
                             "were not kept (no arcs; needs a selection such as "
                             "--clip-radius)")
    parser.add_argument("--interior-only", action="store_true",
                        help="keep only sites whose Voronoi cell is provably "
                             "exact (drops infinite cells and the distorted "
                             "layer next to them); the rest are removed "
                             "from the point set before building the spheres")
    parser.add_argument("--extend-by-shape", action="store_true",
                        help="with --interior-only: also keep finite cells "
                             "whose shape is a recognized repeated tile shape "
                             "(implies --interior-only)")
    parser.add_argument("--min-class-count", type=int, default=5,
                        help="a shape class with at least this many cells is "
                             "recognized even with no certified member "
                             "(default 5)")
    parser.add_argument("--width-mm", type=float, default=None,
                        help="scale: the footprint width in mm; all dimensions are "
                             "also printed in mm")
    parser.add_argument("--shade", action="store_true",
                        help="render lit shading instead of a plain heightmap "
                             "(makes shallow creases easier to see)")
    parser.add_argument("--z-scale", type=float, default=10.0,
                        help="height exaggeration used by --shade (default 10)")
    parser.add_argument("--output", default=None,
                        help="output PNG (default: voronoi_spheres_height.png, or "
                             "voronoi_spheres_shaded.png with --shade)")
    parser.add_argument("--from-png", metavar="FILE", default=None,
                        help="repeat a run: use the arguments stored in the "
                             "metadata of a PNG made by this script (extra "
                             "arguments on the command line override them)")
    parser.add_argument("--show-meta", metavar="FILE", default=None,
                        help="print the metadata stored in a PNG and exit")
    argv = expand_from_png(sys.argv[1:])
    args = parser.parse_args(argv)
    if args.show_meta:
        for k, v in read_png_text(args.show_meta).items():
            print(f"{k}:\n{v}\n")
        return
    rendering = "shaded" if args.shade else "height map"
    if args.output is None:
        args.output = "voronoi_spheres_shaded.png" if args.shade else "voronoi_spheres_height.png"
    name = os.path.basename(args.output).lower()
    if "shade" in name and not args.shade:
        print(f"Note: '{os.path.basename(args.output)}' says 'shade' but this run writes a "
              f"plain height map (add --shade for a lit rendering).")
    if "height" in name and args.shade:
        print(f"Note: '{os.path.basename(args.output)}' says 'height' but this run writes a "
              f"shaded rendering (drop --shade for a height map).")

    points = read_points(args.pts) if args.pts else np.array(DEFAULT_POINTS)
    print(f"{len(points)} points")

    vor = Voronoi(points)
    L_ref = np.percentile(delaunay_edge_lengths(vor), args.percentile)
    print(f"L_ref ({args.percentile:.0f}th pct Delaunay edge) = {L_ref:.4f}")

    keep = None
    if args.interior_only or args.extend_by_shape:
        keep = certified_cells(vor)
        print(f"Provably exact cells: {keep.sum()} of {len(points)}")
        if args.extend_by_shape:
            keep = extend_by_shape(vor, keep, 10.0 * L_ref, args.min_class_count)
    if args.clip_radius is not None:
        if args.clip_center:
            ccx, ccy = args.clip_center
        else:
            ccx = (points[:, 0].min() + points[:, 0].max()) / 2
            ccy = (points[:, 1].min() + points[:, 1].max()) / 2
        inside = cells_inside_circle(vor, (ccx, ccy), args.clip_radius)
        print(f"Circular cutoff: center=({ccx:.4f}, {ccy:.4f}) "
              f"radius={args.clip_radius:.4f}: {inside.sum()} complete cells inside")
        keep = inside if keep is None else (keep & inside)
    sites = points
    if keep is not None:
        print(f"Cells kept: {keep.sum()} of {len(points)} "
              f"({(~keep).sum()} removed)")
        if keep.sum() == 0:
            raise SystemExit("No cells kept; enlarge the cutoff radius or "
                             "relax the selection.")
        sites = points[keep]

    prune = args.prune and keep is not None
    if args.prune and keep is None:
        print("Note: --prune needs a selection (--clip-radius, --interior-only, "
              "--extend-by-shape); ignoring it.")
    # When pruning, every vertex of a kept (complete) cell is a real vertex,
    # so all of them may set the plane; otherwise only vertices surrounded
    # entirely by kept cells do.
    rho = interior_vertex_circumradii(vor, keep, require_all=not prune)
    rho_max = None
    if len(rho):
        # Vertices between clusters can still have only finite cells around
        # them (the facing cells are bounded by the other cluster), but their
        # circumradii are many times the typical one. They must fall BELOW
        # the plane so the clusters separate, so they are excluded here.
        median = np.median(rho)
        not_bridge = rho <= args.bridge_factor * median
        print(f"Interior vertices: {len(rho)} (median circumradius {median:.4f}); "
              f"{(~not_bridge).sum()} excluded as bridging (> {args.bridge_factor}x median)")
        rho = rho[not_bridge]
        rho_max = rho.max()
        print(f"Lowest interior crease uses circumradius = {rho_max:.4f}")

    if prune:
        V = np.vstack([vor.vertices[vor.regions[vor.point_region[i]]]
                       for i in np.where(keep)[0]])
    size = args.size
    tree_all = cKDTree(points)
    tree_sites = cKDTree(sites)

    def build(S):
        """Everything that depends on the sphere radius S (point units)."""
        if rho_max is not None and rho_max >= S:
            raise SystemExit("Largest interior circumradius >= sphere radius; "
                             "increase --sphere-factor / --sphere-mm.")
        z_lowest = np.sqrt(S**2 - rho_max**2) if rho_max is not None else None
        if args.plane_height is not None:
            h = args.plane_height
        elif z_lowest is not None:
            h = z_lowest - args.plane_offset * L_ref
        else:
            raise SystemExit("No interior vertices found; pass --plane-height.")
        if not (0 <= h < S):
            raise SystemExit(f"Plane height {h:.4f} must satisfy 0 <= h < S ({S:.4f}).")
        R = np.sqrt(S**2 - h**2)
        top = S - h

        # Render extent
        if prune:
            pad = 0.25 * L_ref
            x_lo, x_hi = V[:, 0].min() - pad, V[:, 0].max() + pad
            y_lo, y_hi = V[:, 1].min() - pad, V[:, 1].max() + pad
        else:
            pad = R + 0.25 * L_ref
            x_lo, x_hi = sites[:, 0].min() - pad, sites[:, 0].max() + pad
            y_lo, y_hi = sites[:, 1].min() - pad, sites[:, 1].max() + pad
        span = max(x_hi - x_lo, y_hi - y_lo)
        cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        x_lo, x_hi, y_lo, y_hi = cx - span/2, cx + span/2, cy - span/2, cy + span/2
        xs = np.linspace(x_lo, x_hi, size)
        ys = np.linspace(y_lo, y_hi, size)
        X, Y = np.meshgrid(xs, ys)

        # Upper envelope of equal spheres = sphere of the nearest site
        grid = np.column_stack([X.ravel(), Y.ravel()])
        if prune:
            # Explicit Voronoi ownership over ALL sites; pixels owned by a cell
            # that was not kept (incomplete) are removed.
            d, owner = tree_all.query(grid)
            owned_by_kept = keep[owner].reshape(size, size)
        else:
            d, _ = tree_sites.query(grid)
            owned_by_kept = True
        d = d.reshape(size, size)
        Z = np.where(d < S, np.sqrt(np.clip(S**2 - d**2, 0, None)), 0.0)
        H = np.where(owned_by_kept, np.clip(Z - h, 0.0, None), 0.0)

        # Footprint of the solid (flat bottom = the cutting plane)
        mask = H > 0
        pitch = xs[1] - xs[0]
        cols = np.where(mask.any(axis=0))[0]
        rows = np.where(mask.any(axis=1))[0]
        fx0, fx1 = xs[cols[0]] - pitch / 2, xs[cols[-1]] + pitch / 2
        fy0, fy1 = ys[rows[0]] - pitch / 2, ys[rows[-1]] + pitch / 2
        return SimpleNamespace(S=S, h=h, R=R, top=top, z_lowest=z_lowest,
                               xs=xs, ys=ys, H=H, pitch=pitch,
                               fx0=fx0, fx1=fx1, fy0=fy0, fy1=fy1,
                               fw=fx1 - fx0, fd=fy1 - fy0)

    S = args.sphere_factor * L_ref
    if args.sphere_mm is not None:
        if not args.width_mm:
            raise SystemExit("--sphere-mm needs --width-mm to fix the scale.")
        # The scale (mm per unit) is width_mm / footprint width, and the
        # footprint can depend on S, so solve S = sphere_mm / scale by
        # iteration. With --prune the footprint is independent of S.
        for _ in range(12):
            S_new = args.sphere_mm * build(S).fw / args.width_mm
            converged = abs(S_new - S) <= 1e-3 * S
            S = S_new
            if converged:
                break
    r = build(S)
    S, h, R, top, z_lowest = r.S, r.h, r.R, r.top, r.z_lowest
    xs, ys, H, pitch = r.xs, r.ys, r.H, r.pitch
    fx0, fx1, fy0, fy1, fw, fd = r.fx0, r.fx1, r.fy0, r.fy1, r.fw, r.fd

    print(f"Sphere radius S = {S:.4f}")
    print(f"Plane height h = {h:.4f} above sphere centers")
    print(f"Cut radius R = sqrt(S^2 - h^2) = {R:.4f}  ({R / L_ref:.3f} x L_ref)")
    print(f"Slab thickness at a site = {top:.4f}")

    crease = (z_lowest - h) if z_lowest is not None else None
    scale = (args.width_mm / fw) if args.width_mm else None

    def line(label, value, unit_note=""):
        txt = f"  {label:<38s}{value:10.4f}"
        if scale is not None:
            txt += f"   = {value * scale:9.3f} mm"
        print(txt + unit_note)

    print("\nDimensions of the solid (point units; z = 0 is the flat bottom):")
    print(f"  Footprint x range: {fx0:.4f} .. {fx1:.4f}")
    print(f"  Footprint y range: {fy0:.4f} .. {fy1:.4f}")
    line("Footprint width  (x)", fw)
    line("Footprint depth  (y)", fd)
    line("Thickness at a site (dome top)", top)
    if crease is not None:
        line("Thickness at lowest crease/rim", crease)
    line("Sphere radius S (true curvature)", S)
    line("Sphere centers below the bottom by", h)
    line("Cut radius at the plane (R)", R)
    line("  xy size of one pixel", pitch)
    if args.shade:
        print(f"  Rendering: shaded (gray levels are lighting, not height; "
              f"heights exaggerated {args.z_scale:g}x for lighting only).\n")
    else:
        print("  Rendering: height map (gray level is proportional to height):")
        line("  z height of one gray level", top / 255.0)
        line("  gray 255 corresponds to z =", top)
        print(f"  Treating 1 gray level as 1 pixel of z would exaggerate the height by "
              f"{(255.0 * pitch) / top:.2f}x; use the z values above for a true sphere.\n")

    if args.shade:
        dx = xs[1] - xs[0]
        gy, gx = np.gradient(H * args.z_scale, dx)
        norm = np.sqrt(gx**2 + gy**2 + 1.0)
        nx, ny, nz = -gx / norm, -gy / norm, 1.0 / norm
        light = np.array([-0.5, 0.5, 0.7])
        light /= np.linalg.norm(light)
        lit = np.clip(nx*light[0] + ny*light[1] + nz*light[2], 0, 1)
        img_arr = np.where(H > 0, lit, 0.0)
    else:
        img_arr = H / top

    # ---- Metadata so the run can be repeated (the point data itself is not
    # embedded; its file name, point count and SHA-256 are, to check it later)
    def f(v):
        return None if v is None else float(v)

    computed = {
        "points_file": os.path.basename(args.pts) if args.pts else "built-in default points",
        "points_count": int(len(points)),
        "points_sha256": file_sha256(args.pts) if args.pts else None,
        "cells_kept": int(keep.sum()) if keep is not None else None,
        "L_ref": f(L_ref),
        "sphere_radius_units": f(S),
        "plane_height_above_sphere_centers_units": f(h),
        "cut_radius_units": f(R),
        "thickness_at_site_units": f(top),
        "thickness_at_lowest_crease_units": f(crease),
        "footprint_x_range_units": [f(fx0), f(fx1)],
        "footprint_y_range_units": [f(fy0), f(fy1)],
        "footprint_width_units": f(fw),
        "footprint_depth_units": f(fd),
        "pixel_pitch_units": f(pitch),
        "mm_per_unit": f(scale),
    }
    if args.clip_radius is not None:
        computed["clip_center_used"] = [f(ccx), f(ccy)]
    if scale is not None:
        computed["footprint_mm"] = [f(fw * scale), f(fd * scale)]
        computed["sphere_radius_mm"] = f(S * scale)
        computed["thickness_at_site_mm"] = f(top * scale)
        computed["thickness_at_lowest_crease_mm"] = f(crease * scale) if crease is not None else None
    computed["rendering"] = rendering
    if args.shade:
        computed["shading_z_exaggeration"] = float(args.z_scale)
    else:
        computed["gray_255_equals_z_units"] = f(top)
        computed["z_per_gray_level_units"] = f(top / 255.0)
        if scale is not None:
            computed["gray_255_equals_z_mm"] = f(top * scale)
    script = os.path.basename(sys.argv[0]) or "voronoi_spheres.py"
    try:
        computed["script_sha256"] = file_sha256(os.path.abspath(sys.argv[0]))
    except OSError:
        pass
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Software", "voronoi_spheres.py")
    meta.add_text("Rendering", rendering)
    meta.add_text("Command", script + " " + " ".join(
        f'"{a}"' if " " in a else a for a in argv))
    meta.add_text(META_KEY, json.dumps(argv))
    meta.add_text("Computed", json.dumps(computed, indent=1))

    Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8), 'L').save(
        args.output, pnginfo=meta)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
