import numpy as np
from scipy.spatial import Voronoi, cKDTree
from PIL import Image, ImageDraw
import argparse
import time


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
    p0 = vor.points[vor.ridge_points[:, 0]]
    p1 = vor.points[vor.ridge_points[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Per-site-clipped Voronoi surface with quarter-circle fillets")
    parser.add_argument("--pts", metavar="FILE", default=None)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--percentile", type=float, default=25.0,
                        help="percentile of Delaunay edge lengths defining typical spacing (default 25)")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="margin above typical spacing for the per-site clip radius R (default 0.5 = 50%%)")
    parser.add_argument("--radius", type=float, default=None,
                        help="override: use this absolute clip radius R directly instead of percentile/factor")
    parser.add_argument("--max-coeff", type=float, default=0.4,
                        help="fillet plateau distance as a fraction of the percentile edge length (default 0.4)")
    parser.add_argument("--output", default="voronoi_cluster_surface.png")
    parser.add_argument("--dots", action="store_true")
    args = parser.parse_args()

    if args.pts:
        points = read_points(args.pts)
    else:
        points = np.array(DEFAULT_POINTS)
    print(f"{len(points)} points")

    vor = Voronoi(points)

    lengths = delaunay_edge_lengths(vor)
    if args.radius is not None:
        R = args.radius
        L_p = np.percentile(lengths, args.percentile)  # still reported/used for max_dist
    else:
        L_p = np.percentile(lengths, args.percentile)
        R = (L_p / 2.0) * (1.0 + args.factor)
    max_dist = args.max_coeff * L_p
    print(f"L_p ({args.percentile:.0f}th pct Delaunay edge) = {L_p:.4f}")
    print(f"R (per-site clip radius) = {R:.4f}")
    print(f"max_dist (fillet plateau) = {max_dist:.4f}")

    # Straight Voronoi edges per owning cell (finite ridges only — an
    # unbounded/open side of a cell simply has no straight-edge candidate,
    # which is correct: there IS no real edge there, only the arc)
    # Build a map from Voronoi vertex index -> set of point indices whose
    # cell touches that vertex. A finite vertex is generically where
    # exactly 3 cells meet, which lets us find the "third" point for any
    # ray (used below to get the correct outward direction locally,
    # without relying on any single global center).
    vertex_points = {}
    for (p0, p1), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        for v in ridge:
            if v >= 0:
                vertex_points.setdefault(v, set()).update([p0, p1])

    far_length = R * 10.0  # safely exceeds R so the ray always clears the disk
    # Guard against near-collinear points: scipy sometimes reports a
    # "finite" Voronoi vertex that's actually a numerical blowup (circumcenter
    # of an almost-degenerate triangle), far outside any sane coordinate
    # range. Treat those the same as a real ray: anchor at the bisector
    # midpoint instead of trusting the bogus coordinate.
    spread = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    sane_limit = max(spread, R) * 50.0

    point_edges = {i: [] for i in range(len(points))}
    for (p0, p1), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if ridge[0] >= 0 and ridge[1] >= 0:
            v0 = vor.vertices[ridge[0]]
            v1 = vor.vertices[ridge[1]]
            if (np.linalg.norm(v0 - points[p0]) < sane_limit and
                    np.linalg.norm(v1 - points[p0]) < sane_limit):
                point_edges[p0].append((v0, v1))
                point_edges[p1].append((v0, v1))
                continue
            # else: fall through to ray-style reconstruction below

        # Ray (or degenerate "finite" ridge with a blown-up vertex):
        # reconstruct as a real long segment instead of dropping or
        # trusting a bad coordinate.
        known = None
        for v in ridge:
            if v >= 0 and np.linalg.norm(vor.vertices[v] - points[p0]) < sane_limit:
                known = v
                break
        t = points[p1] - points[p0]
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        midpoint = (points[p0] + points[p1]) / 2
        v_known = vor.vertices[known] if known is not None else midpoint
        others = vertex_points.get(known, set()) - {p0, p1} if known is not None else set()
        if others:
            p3 = points[next(iter(others))]
            direction = np.sign(np.dot(midpoint - p3, n)) * n
        else:
            direction = n
        v_far = v_known + direction * far_length
        point_edges[p0].append((v_known, v_far))
        point_edges[p1].append((v_known, v_far))

    # Render extent: bounding box of points, padded by R + margin
    margin = 0.5
    pad = R + margin
    x_lo, x_hi = points[:, 0].min() - pad, points[:, 0].max() + pad
    y_lo, y_hi = points[:, 1].min() - pad, points[:, 1].max() + pad
    span = max(x_hi - x_lo, y_hi - y_lo)
    cxm, cym = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    x_lo, x_hi = cxm - span / 2, cxm + span / 2
    y_lo, y_hi = cym - span / 2, cym + span / 2

    size = args.size
    xs = np.linspace(x_lo, x_hi, size)
    ys = np.linspace(y_lo, y_hi, size)
    X, Y = np.meshgrid(xs, ys)

    tree = cKDTree(points)
    pixel_coords = np.column_stack([X.ravel(), Y.ravel()])
    dist_to_owner, owner = tree.query(pixel_coords)
    owner = owner.reshape(size, size)
    dist_to_owner = dist_to_owner.reshape(size, size)

    # The key simplification: since each disk is centered ON its own site,
    # ownership (nearest-site) plus "dist to that site <= R" exactly
    # reconstructs membership in the clipped cell. No polygon math needed.
    is_background = dist_to_owner > R

    print("Computing distance field...")
    t0 = time.time()
    D = np.full(size * size, np.inf)
    owner_flat = owner.ravel()
    dist_owner_flat = dist_to_owner.ravel()
    col_idx = np.arange(size * size) % size
    row_idx = np.arange(size * size) // size
    PX = xs[col_idx]
    PY = ys[row_idx]
    cell_pixels = {i: [] for i in range(len(points))}
    for flat_idx, cell_idx in enumerate(owner_flat):
        cell_pixels[cell_idx].append(flat_idx)

    for cell_idx in range(len(points)):
        idxs = np.array(cell_pixels[cell_idx])
        if len(idxs) == 0:
            continue
        px = PX[idxs]
        py = PY[idxs]
        best = np.full(len(idxs), np.inf)
        for (v0, v1) in point_edges[cell_idx]:
            dx, dy = v1[0]-v0[0], v1[1]-v0[1]
            len2 = dx*dx + dy*dy
            if len2 == 0:
                d2 = (px-v0[0])**2 + (py-v0[1])**2
            else:
                t = np.clip(((px-v0[0])*dx + (py-v0[1])*dy) / len2, 0.0, 1.0)
                cx = v0[0] + t*dx
                cy = v0[1] + t*dy
                d2 = (px-cx)**2 + (py-cy)**2
            np.minimum(best, d2, out=best)
        # Arc candidate: distance to this site's own clip circle. Exact,
        # since the disk is centered on the site itself.
        d_arc = np.abs(R - dist_owner_flat[idxs])
        np.minimum(best, d_arc**2, out=best)
        D[idxs] = np.sqrt(best)
    D = D.reshape(size, size)
    print(f"Distance field computed in {time.time()-t0:.2f}s")

    H = np.where(D < max_dist,
                 np.sqrt(np.clip(D * (2*max_dist - D), 0, None)),
                 max_dist)
    H[is_background] = 0.0
    H_norm = H / max_dist

    grey = (H_norm * 255).astype(np.uint8)
    img = Image.fromarray(grey, 'L').convert('RGB')
    draw = ImageDraw.Draw(img)

    def to_pixel(px, py):
        ix = int((px - x_lo) / (x_hi - x_lo) * size)
        iy = int((py - y_lo) / (y_hi - y_lo) * size)
        return ix, iy

    if args.dots:
        r_dot = max(2, size // 300)
        for (px, py) in points:
            cx, cy = to_pixel(px, py)
            draw.ellipse([cx-r_dot, cy-r_dot, cx+r_dot, cy+r_dot],
                         fill=(30, 30, 200), outline=(0, 0, 0))

    img.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
