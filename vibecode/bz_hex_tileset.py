"""Generate a SET of BZ hexagon tiles that share the same near-boundary
data (so any combination of them tiles seamlessly) but differ in their
interior.

Standalone from bz_hex.py by design, the same way bz_hex.py is standalone
from zb3_padded.py: the core simulation machinery (PaddedGrid,
fill_border, the reaction equations, FILTER, normalize/map_hex/
build_hex_border_indices) is copied over unchanged, since it's already
verified correct there. The tileset logic below is the new part.

Approach:
  1. Warm up ONE master tile from random noise, using its own ordinary
     self-consistent periodic wraparound, until it reaches a stable,
     low topological defect count -- i.e. a few clear, well-separated
     spiral cores, rather than many small interacting fragments.
     Verified via winding-number defect counting (see defect count in
     the phase field of A, B, C): defect count drops from the 70s at
     iteration 10 down to a steady ~10 by iteration 300, then stays flat
     for 900+ more iterations. This is NOT the same as reaching a stable
     typical domain SIZE, which happens much earlier (~15-30 iterations,
     measured via domain-wall density) but doesn't mean the pattern has
     resolved into clearly-defined spirals -- an earlier version of this
     script used domain-wall density to justify warmup=50, which left
     ~19 defects (roughly double the steady-state count) still
     interacting and annihilating, looking unconverged. Default
     warmup=300 for margin. This step is unconstrained -- a tile
     matching only itself has no propagation-budget concern at all.
  2. Branch the master's warmed-up state into N independent copies.
  3. Perturb each copy's center with fresh, independently-seeded random
     noise, in a (2*perturb_radius) x (2*perturb_radius) box centered on
     the hexagon's true center (m, m) -- the one point equidistant (= m)
     from all six edges (four outer, two internal cut lines).
  4. Run each copy for post_iters further iterations. The convolution
     filter has radius 3, which is a HARD bound (not a typical case) on
     how many cells information can cross per iteration. So as long as

         perturb_radius + 3*post_iters + safety_margin <= m

     every cell within safety_margin of any edge is provably untouched,
     and therefore bit-for-bit identical across every tile generated
     this way, no matter how differently their centers evolved.
     post_iters is automatically clamped to the largest value satisfying
     this -- see safe_post_iters().
  5. Verify it. Don't just trust the arithmetic above: after generating
     all tiles, directly check that cells within safety_margin of any
     edge are exactly identical across every pair. This is a real, live
     check the script performs every run, not just documentation of the
     intended guarantee. (An earlier check of mine here, comparing
     against a plain radius-from-center threshold rather than the actual
     distance-to-any-edge, mistakenly flagged this as broken -- the
     center perturbation's influence is SUPPOSED to reach almost the
     entire tile when post_iters uses the full safe budget; only the
     margin right at the edge is supposed to stay untouched. Checking
     the wrong region looked like a failure that wasn't one.)

Why warm up before branching rather than perturbing from iteration 0:
measured via domain-wall density AND re-checked with the defect metric
above (since the former missed the true global warmup time by 6x, it
needed independent confirmation here too) -- a perturbed patch surrounded
by already-mature structure settles back to its local pre-perturbation
defect count within about 5 iterations, and stays there. That's a much
smaller cost than the ~300 iterations a whole tile needs to organize from
scratch, so warming up the master first and perturbing after leaves
nearly the entire post_iters budget free for actually showing divergence,
rather than spending most of it just reaching a mature state.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve
from PIL import Image

# ---------------------------------------------------------------------------
# Hex-lattice coordinate reduction and core BZ machinery: unchanged from
# bz_hex.py. See that file for the derivation and verification notes.
# ---------------------------------------------------------------------------

def normalize(x, y, m):
    a1, a2 = 2 * m + 1, m + 1
    b1, b2 = m, 2 * m + 1
    D = a1 * b2 - a2 * b1
    i = (b2 * x - b1 * y) // D
    j = (-a2 * x + a1 * y) // D
    return x - a1 * i - b1 * j, y - a2 * i - b2 * j


def map_hex(m):
    to_hex = {}
    for i in range(2 * m + 1):
        for j in range(2 * m + 1):
            if abs(i - j) <= m:
                to_hex[normalize(i, j, m)] = (i, j)
    return to_hex


FILTER_STR = '''
1111000
1111100
1111110
1111111
0111111
0011111
0001111
'''
FILTER_INT = [[int(c) for c in row] for row in FILTER_STR.split()]
FILTER = np.array(FILTER_INT) / sum(sum(row) for row in FILTER_INT)
PAD = 3


class PaddedGrid:
    def __init__(self, padded_arr, pad):
        self._arr = padded_arr
        self._pad = pad

    def __getitem__(self, idx):
        p, k, i, j = idx
        return self._arr[p, k, i + self._pad, j + self._pad]

    def __setitem__(self, idx, value):
        p, k, i, j = idx
        self._arr[p, k, i + self._pad, j + self._pad] = value

    @property
    def raw(self):
        return self._arr


def build_hex_border_indices(n, pad, m):
    """See bz_hex.py for the full derivation. Fixed version: cells inside
    the (2m+1) x (2m+1) box but outside the true hexagon (|i-j| > m) get
    the same wraparound mirroring as the outer padding ring -- both are
    "not a true hex cell", which is the condition that actually matters,
    not "inside vs outside the box"."""
    to_hex = map_hex(m)
    lo, hi = -pad, n - 1 + pad
    border_i, border_j, src_i, src_j = [], [], [], []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if 0 <= i < n and 0 <= j < n and abs(i - j) <= m:
                continue  # true hex cell
            ti, tj = to_hex[normalize(i, j, m)]
            border_i.append(i)
            border_j.append(j)
            src_i.append(ti)
            src_j.append(tj)
    return (np.array(border_i), np.array(border_j),
            np.array(src_i), np.array(src_j))


def fill_border(g, border_i, border_j, src_i, src_j, p):
    g[p, :, border_i, border_j] = g[p, :, src_i, src_j]


def update(p, g, border_i, border_j, src_i, src_j, alpha, beta, gamma):
    q = (p + 1) % 2
    fill_border(g, border_i, border_j, src_i, src_j, p)
    size = g.raw.shape[2]
    s = np.zeros((3, size, size))
    for k in range(3):
        s[k] = convolve(g.raw[p, k], FILTER, mode='constant', cval=0.0)
    g.raw[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
    g.raw[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
    g.raw[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])
    np.clip(g.raw[q], 0, 1, g.raw[q])


def unique_path(path):
    p = Path(path)
    if not p.exists():
        return str(p)
    k = 1
    while True:
        candidate = p.with_name(f"{p.stem}_{k}{p.suffix}")
        if not candidate.exists():
            return str(candidate)
        k += 1


# ---------------------------------------------------------------------------
# Tileset-specific logic (the new part).
# ---------------------------------------------------------------------------

def safe_post_iters(m, perturb_radius, safety_margin, requested):
    """The guard against perturbing the boundary. Returns (post_iters, K):
    K is the largest number of post-perturbation iterations for which
    perturb_radius + 3*post_iters + safety_margin <= m is guaranteed to
    hold (filter radius 3 is a hard per-iteration propagation bound, not
    a typical case). `requested` (possibly None) is clamped down to K if
    it would exceed it; None means "use the full safe budget"."""
    if perturb_radius >= m:
        raise ValueError(f"perturb_radius ({perturb_radius}) must be < m ({m})")
    K = (m - perturb_radius - safety_margin) // 3
    if K < 1:
        raise ValueError(
            f"perturb_radius={perturb_radius} and safety_margin={safety_margin} "
            f"leave no safe iteration budget for m={m} (K={K}); reduce one of "
            f"them or increase m.")
    if requested is None or requested > K:
        if requested is not None:
            print(f"warning: requested --post-iters {requested} exceeds the safe "
                  f"budget K={K}; clamping to {K}.")
        return K, K
    return requested, K


def perturb_center(g, pad, m, radius, rng):
    """Overwrite a (2*radius) x (2*radius) box centered on the hexagon's
    true center (m, m) with fresh random values, in buffer slot 0."""
    lo, hi = m - radius, m + radius
    g.raw[0, :, pad + lo:pad + hi, pad + lo:pad + hi] = rng.random((3, 2 * radius, 2 * radius))


def verify_boundary_match(tiles, n, m, safety_margin):
    """Directly check (not just trust the arithmetic) that every cell
    within safety_margin of any edge -- four outer sides plus the two
    internal |i-j|=m cut lines -- is exactly identical across every tile.
    Returns (max_diff, n_cells_checked)."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    hex_mask = np.abs(ii - jj) <= m
    dist_to_boundary = np.minimum.reduce([ii, (n - 1) - ii, jj, (n - 1) - jj, m - np.abs(ii - jj)])
    check_mask = hex_mask & (dist_to_boundary <= safety_margin)
    max_diff = 0.0
    for t in range(1, len(tiles)):
        d = np.abs(tiles[t] - tiles[0])[:, check_mask].max()
        max_diff = max(max_diff, d)
    return max_diff, int(check_mask.sum())


def render_rgba(interior, n, m):
    """Masked hexagon RGBA array, pre-affine-transform (same recipe as
    bz_hex.py)."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    hex_mask = (np.abs(ii - jj) <= m).astype(np.uint8)[:, :, None]
    disp = np.concatenate((interior.transpose(2, 1, 0), hex_mask), axis=2)
    return Image.fromarray((disp * 255).astype(np.uint8), mode='RGBA')


def affine_transform(img):
    w, h = img.size
    return img.transform((int(w * 3), int(h * math.sqrt(3))), Image.Transform.AFFINE,
                          (0.5, 0.5 / math.sqrt(3), -0.5 * w, 0, 1 / math.sqrt(3), 0),
                          resample=Image.Resampling.BICUBIC)


def render_montage(imgs, cols=2, pad=20, bg=(255, 255, 255, 0)):
    """Simple grid of already-affine-transformed tile images, for a quick
    side-by-side look. The individual per-tile PNGs are the real output;
    this is just a preview."""
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    canvas = Image.new('RGBA', (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(img, (x, y), img)
    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Generate a set of BZ hexagon tiles with matching boundaries '
                 'but differing interiors, by warming up one master tile, '
                 'branching it, and perturbing each copy\'s center with a '
                 'provable safety margin against reaching the boundary.')
parser.add_argument('-m', '--order', type=int, default=100,
                     help='Hexagon order (default: 100).')
parser.add_argument('-n', '--num-tiles', type=int, default=4,
                     help='Number of distinct tiles to generate (default: 4).')
parser.add_argument('--warmup', type=int, default=300,
                     help='Iterations to mature the master tile before '
                          'branching (default: 300). NOT the same as '
                          'reaching a stable typical domain SIZE (which '
                          'happens much earlier, ~15-30 iterations) -- '
                          'that\'s a different thing from reaching a '
                          'stable, low topological defect count (i.e. a '
                          'few clear spiral cores rather than many small '
                          'interacting fragments), which is what actually '
                          'looks like "converged into clear spirals". '
                          'Measured directly via winding-number defect '
                          'counting: defect count is still ~19 (vs. a '
                          'steady-state ~10) at iteration 50, and doesn\'t '
                          'settle until ~300, after which it\'s flat for '
                          '900+ more iterations.')
parser.add_argument('--perturb-radius', type=int, default=None,
                     help='Half-width of the perturbed center box, in cells '
                          '(default: max(3, m // 10)).')
parser.add_argument('--safety-margin', type=int, default=None,
                     help='Cells of guaranteed-untouched margin to keep '
                          'beyond the bare minimum at the boundary (default: '
                          'max(2*PAD, m // 20)). Larger = more conservative, '
                          'fewer post_iters available.')
parser.add_argument('--post-iters', type=int, default=None,
                     help='Iterations to run each tile after perturbation '
                          '(default: the full safe budget K; see '
                          'safe_post_iters -- always clamped to K regardless '
                          'of what\'s requested).')
parser.add_argument('-p', '--param', type=float, default=1.0,
                     help='Value for alpha, beta, gamma (default: 1.0).')
parser.add_argument('--seed', type=int, default=0,
                     help='Base random seed; the master uses this seed, each '
                          'tile\'s perturbation uses seed + 1000 + tile index '
                          '(default: 0).')
parser.add_argument('--grayscale', action='store_true',
                     help='Render as grayscale height maps instead of RGB.')
args = parser.parse_args()

m = args.order
n = 2 * m + 1
alpha = beta = gamma = args.param
perturb_radius = args.perturb_radius if args.perturb_radius is not None else max(3, m // 10)
safety_margin = args.safety_margin if args.safety_margin is not None else max(2 * PAD, m // 20)

post_iters, K = safe_post_iters(m, perturb_radius, safety_margin, args.post_iters)
print(f"m={m}  perturb_radius={perturb_radius}  safety_margin={safety_margin}  "
      f"-> K={K}, using post_iters={post_iters}")

# ---------------------------------------------------------------------------
# 1-2: warm up the master, then branch
# ---------------------------------------------------------------------------

np.random.seed(args.seed)
master_data = np.random.random(size=(2, 3, n, n))
size = n + 2 * PAD
master_arr = np.zeros((2, 3, size, size))
master = PaddedGrid(master_arr, PAD)
master.raw[:, :, PAD:PAD + n, PAD:PAD + n] = master_data

border_i, border_j, src_i, src_j = build_hex_border_indices(n, PAD, m)

for it in range(args.warmup):
    update(it % 2, master, border_i, border_j, src_i, src_j, alpha, beta, gamma)
master_p = args.warmup % 2
master_state = master.raw[master_p].copy()

# ---------------------------------------------------------------------------
# 3-4: perturb each branch, run for post_iters
# ---------------------------------------------------------------------------

tiles = []
for t in range(args.num_tiles):
    arr_t = np.zeros((2, 3, size, size))
    arr_t[0] = master_state.copy()
    g_t = PaddedGrid(arr_t, PAD)
    rng = np.random.default_rng(args.seed + 1000 + t)
    perturb_center(g_t, PAD, m, perturb_radius, rng)
    for it in range(post_iters):
        update(it % 2, g_t, border_i, border_j, src_i, src_j, alpha, beta, gamma)
    final_p = post_iters % 2
    fill_border(g_t, border_i, border_j, src_i, src_j, final_p)
    tiles.append(g_t.raw[final_p, :, PAD:PAD + n, PAD:PAD + n].copy())

# ---------------------------------------------------------------------------
# 5: verify, then render
# ---------------------------------------------------------------------------

max_diff, n_checked = verify_boundary_match(tiles, n, m, safety_margin)
status = "OK" if max_diff < 1e-9 else "FAILED"
print(f"boundary-match check ({status}): max|diff| = {max_diff:.2e} across "
      f"{n_checked} cells within {safety_margin} of any edge, over "
      f"{args.num_tiles - 1} tile pairs.")
if status == "FAILED":
    print("Not saving output -- something is wrong; the guarantee this script "
          "exists to provide did not hold. Please report this.")
    raise SystemExit(1)

rendered = []
paths = []
for t, interior in enumerate(tiles):
    if args.grayscale:
        sorted_s = np.sort(interior, axis=0)
        margin_field = sorted_s[2] - sorted_s[1]
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        hex_mask = np.abs(ii - jj) <= m
        margin_field = np.where(hex_mask, margin_field, 0.0)
        img = Image.fromarray((margin_field * 255).astype(np.uint8), mode='L')
    else:
        img = render_rgba(interior, n, m)
    img = affine_transform(img)
    rendered.append(img)
    path = unique_path(f'bz_hex_tile_{t}.png')
    img.save(path)
    paths.append(path)
    print(f"saved {path}")

if not args.grayscale:
    montage_path = unique_path('bz_hex_tileset_montage.png')
    render_montage(rendered).save(montage_path)
    print(f"saved {montage_path}")
