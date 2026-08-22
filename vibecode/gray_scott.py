import argparse
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import convolve, zoom, gaussian_filter

# Gray-Scott reaction-diffusion model:
#   dU/dt = Du*Laplacian(U) - U*V^2 + f*(1 - U)
#   dV/dt = Dv*Laplacian(V) + U*V^2 - (f + k)*V
# U is the substrate, V is the autocatalyst; f (feed) replenishes U, k (kill)
# removes V. Unlike the BZ script's 3-species cyclic-dominance system, this is
# a 2-species feed/kill system: it can reach a genuine standstill (spots stop
# growing, branches stop extending) rather than sustaining perpetual chaos,
# which is what gives it a "grew, then stopped" organic read.
#
# The named regions of (f, k) space below (mitosis, coral, solitons, maze)
# come from J. E. Pearson, "Complex Patterns in a Simple System," Science
# 261 (1993), 189-192 -- the paper that first systematically classified
# 2D Gray-Scott patterns. Values here are commonly-used points within each
# of his named regions, not read directly off the paper's figures.
PRESETS = {
    'mitosis':  (0.0367, 0.0649),
    'coral':    (0.0545, 0.0620),
    'solitons': (0.0780, 0.0610),
    'maze':     (0.0290, 0.0570),
}

parser = argparse.ArgumentParser(
    description='Simulate the Gray-Scott reaction-diffusion system and render it '
                 'as an image, or sweep a grid of (feed, kill) values into a '
                 'contact sheet for picking a target before a full render.')
parser.add_argument('-n', '--size', type=int, default=256,
                     help='Side length of the n x n grid (default: 256).')
parser.add_argument('-i', '--iterations', type=int, default=4000,
                     help='Number of iterations to run (default: 4000). Gray-Scott '
                          'needs far more steps than the BZ script to settle -- '
                          'patterns are often still forming at iteration 500.')
parser.add_argument('--preset', choices=sorted(PRESETS), default='coral',
                     help="Named (feed, kill) region to start from (default: coral). "
                          "Overridden by -f/-k if either is given explicitly.")
parser.add_argument('-f', '--feed', type=float, default=None,
                     help='Feed rate. Overrides --preset if given.')
parser.add_argument('-k', '--kill', type=float, default=None,
                     help='Kill rate. Overrides --preset if given.')
parser.add_argument('--du', type=float, default=0.16,
                     help='Diffusion rate for U (default: 0.16).')
parser.add_argument('--dv', type=float, default=0.08,
                     help='Diffusion rate for V (default: 0.08).')
parser.add_argument('--dt', type=float, default=1.0,
                     help='Timestep (default: 1.0). Kept stable at the default '
                          'du/dv above; raising du/dv without lowering dt can blow up.')
parser.add_argument('--num-seeds', type=int, default=3,
                     help='Number of random nucleation blobs to seed the grid with '
                          '(default: 3). More seeds means more competing growth '
                          'fronts, closer in spirit to the multi-center spiral '
                          'competition in the BZ piece.')
parser.add_argument('--cmap', type=str, default='gray',
                     help="Matplotlib colormap for output (default: 'gray', since "
                          "a plain grayscale render doubles as a usable heightmap). "
                          "Try 'bone' or 'pink' for a preview with more tonal range.")
parser.add_argument('--seed', type=int, default=None,
                     help='Random seed, for reproducible runs (default: unseeded).')
parser.add_argument('--flat', action='store_true',
                     help='Skip the hex display skew and save the raw n x n array '
                          "as-is. Useful for confirming the hex neighborhood/kernel "
                          "is doing what's expected before trusting the sheared "
                          'render. Only affects the single-render path, not --sweep.')
parser.add_argument('--hex-tile', action='store_true',
                     help='Simulate on a genuine hexagon-shaped, edge-to-edge domain '
                          '(order -m/--hex-order, 3m^2+3m+1 cells) with a '
                          "translation-only periodic boundary, ported from bz_hex.py, "
                          'instead of the default periodic-rhombus torus. Ignores '
                          '-n/--size. Only affects the single-render path, not --sweep.')
parser.add_argument('-m', '--hex-order', type=int, default=60,
                     help='Hexagon order for --hex-tile: the tile has 3m^2+3m+1 cells, '
                          'stored in a (2m+1) x (2m+1) grid (default: 60).')
parser.add_argument('--rotate-tile', action='store_true',
                     help='Simulate on an n x n rhombus (-n/--size) whose border is '
                          'filled from a 120-degree-ROTATED copy of itself, rather than '
                          "a plain translated copy (the default periodic-rhombus torus' "
                          'wraparound) or a genuine hexagon (--hex-tile). Ported from '
                          "zb3_padded.py's default 'rotate' --symmetry: 3 tiles meet "
                          'around each shared vertex, related by successive 120-degree '
                          'rotations, which is what makes spirals/patterns line up '
                          'across tile copies instead of just repeating. Mutually '
                          'exclusive with --hex-tile. Only affects the single-render '
                          'path, not --sweep.')
parser.add_argument('--assemble-hex', action='store_true',
                     help='Requires --rotate-tile. Instead of rendering just the one '
                          'simulated rhombus, extend the mesh to its own 120- and '
                          '240-degree rotated copies about the shared origin vertex and '
                          'render all 3 together as a hexagon -- the construction from '
                          'the reference picture, done directly rather than assembled '
                          'by hand from 3 separate renders.')
parser.add_argument('--contrast-stretch', action='store_true',
                     help='Apply a levels-style contrast stretch to V before coloring: '
                          'clip to the [--stretch-low-pct, --stretch-high-pct] '
                          'percentile range (excluding outliers), then rescale that '
                          'range to [--stretch-out-min, --stretch-out-max]. Defaults '
                          '(1st/99th percentile -> 0.4-1.0) turn a mostly-black field '
                          'with muted ridges into a solid mid-tone base with the '
                          'pattern standing out against it -- useful for fabric/'
                          'decorative renders ("coaster" look).')
parser.add_argument('--stretch-low-pct', type=float, default=1.0,
                     help='Low percentile for --contrast-stretch (default: 1.0).')
parser.add_argument('--stretch-high-pct', type=float, default=99.0,
                     help='High percentile for --contrast-stretch (default: 99.0).')
parser.add_argument('--stretch-out-min', type=float, default=0.4,
                     help='Output range minimum for --contrast-stretch (default: 0.4).')
parser.add_argument('--stretch-out-max', type=float, default=1.0,
                     help='Output range maximum for --contrast-stretch (default: 1.0).')
parser.add_argument('--output-size', type=int, default=None,
                     help='Upscale the simulated field to roughly this many cells per '
                          'side before coloring/shearing (default: no upscaling, i.e. '
                          "the simulation's own -n/-m resolution is used directly). "
                          'Useful for image-to-STL pipelines: the raw simulation grid '
                          'is often too coarse to extrude cleanly, producing jagged '
                          'stair-step cliffs between cells instead of smooth ridges. '
                          'Cubic-interpolation upscaling (optionally followed by '
                          '--upscale-blur-sigma) fixes this without needing to increase '
                          '-n itself, which would slow the simulation down for the same '
                          'visual effect. Try 2048 as a starting point.')
parser.add_argument('--upscale-blur-sigma', type=float, default=1.5,
                     help='Gaussian blur sigma, in upscaled-grid cells, applied after '
                          '--output-size upscaling to further smooth jagged transitions '
                          '(default: 1.5). Set to 0 to keep the sharp cubic-upscaled '
                          'result with no additional blur. Has no effect unless '
                          '--output-size is also given.')
parser.add_argument('--transparent-background', action='store_true',
                     help='Keep the background outside the hexagon/rhombus shape fully '
                          'transparent (default: composited onto opaque black instead). '
                          'A transparent background can leave faint semi-transparent '
                          'edge pixels with nonzero RGB brightness -- from antialiasing '
                          'in the hex shear or --output-size upscaling -- that an '
                          'image-to-STL tool ignoring the alpha channel would read as '
                          'real height, producing thin spikes at the boundary. Opaque '
                          "black is the safer default for that reason; use this flag "
                          'if you specifically want transparency, e.g. for compositing '
                          'elsewhere.')
parser.add_argument('--sweep', action='store_true',
                     help='Instead of a single render, run a grid of small, fast '
                          'simulations across a range of feed/kill values and save '
                          'them as one contact-sheet image. Ignores --preset/-f/-k.')
parser.add_argument('--sweep-f-range', type=float, nargs=2, default=[0.030, 0.070],
                     metavar=('MIN', 'MAX'), help='Feed rate range for --sweep (default: 0.030 0.070).')
parser.add_argument('--sweep-k-range', type=float, nargs=2, default=[0.055, 0.065],
                     metavar=('MIN', 'MAX'), help='Kill rate range for --sweep (default: 0.055 0.065).')
parser.add_argument('--sweep-steps', type=int, default=5,
                     help='Grid resolution per axis for --sweep (default: 5, giving a 5x5 = 25-cell sheet).')
parser.add_argument('--sweep-size', type=int, default=120,
                     help='Grid size for each --sweep cell (default: 120, smaller than -n for speed).')
parser.add_argument('--sweep-iterations', type=int, default=2500,
                     help='Iterations for each --sweep cell (default: 2500, enough to reveal '
                          'character without paying for full convergence 25 times over).')
args = parser.parse_args()

if args.size < 10:
    parser.error(f"--size must be at least 10, got {args.size}")
if args.iterations < 1:
    parser.error(f"--iterations must be >= 1, got {args.iterations}")
if args.sweep_f_range[0] >= args.sweep_f_range[1]:
    parser.error(f"--sweep-f-range must have MIN < MAX, got {args.sweep_f_range}")
if args.sweep_k_range[0] >= args.sweep_k_range[1]:
    parser.error(f"--sweep-k-range must have MIN < MAX, got {args.sweep_k_range}")
if args.sweep_steps < 2:
    parser.error(f"--sweep-steps must be at least 2, got {args.sweep_steps}")
if args.sweep_steps > 8:
    print(f"Note: --sweep-steps {args.sweep_steps} means {args.sweep_steps**2} "
          f"separate simulations -- this may take a while.")
if args.hex_tile and args.hex_order < 5:
    parser.error(f"--hex-order must be at least 5, got {args.hex_order}")
if args.hex_tile and any(a.startswith('-n') or a.startswith('--size') for a in sys.argv[1:]):
    print("Note: --hex-tile ignores -n/--size; use -m/--hex-order to size the tile.")
if args.hex_tile and args.rotate_tile:
    parser.error("--hex-tile and --rotate-tile are mutually exclusive; pick one domain.")
if args.assemble_hex and not args.rotate_tile:
    parser.error("--assemble-hex requires --rotate-tile.")
if args.contrast_stretch and not (0 <= args.stretch_low_pct < args.stretch_high_pct <= 100):
    parser.error(f"--stretch-low-pct/--stretch-high-pct must satisfy "
                 f"0 <= low < high <= 100, got {args.stretch_low_pct}, {args.stretch_high_pct}")
if args.contrast_stretch and not (args.stretch_out_min < args.stretch_out_max):
    parser.error(f"--stretch-out-min must be < --stretch-out-max, got "
                 f"{args.stretch_out_min}, {args.stretch_out_max}")
if args.output_size is not None and args.output_size < 10:
    parser.error(f"--output-size must be at least 10, got {args.output_size}")
if args.upscale_blur_sigma < 0:
    parser.error(f"--upscale-blur-sigma must be >= 0, got {args.upscale_blur_sigma}")

rng = np.random.default_rng(args.seed)

# Hex-neighbor discrete Laplacian: uniform 1/6 weight on the 6 neighbors of
# a hexagonal lattice embedded diagonally in a square array, center -1.
#
# A hex lattice stored in a square array keeps all 4 orthogonal neighbors
# but only ONE of the two diagonals -- the other diagonal direction has no
# counterpart in a hexagonal tiling (each cell has 6 neighbors, not 8).
# This uses the (+1,+1)/(-1,-1) diagonal (NW-SE), matching bz_hex.py's
# FILTER orientation (same diagonal, same handedness) so that this grid's
# coordinate convention lines up with a future true-hexagon domain built
# the way bz_hex.py's order-m centered hexagon is: dropping in that
# reduction later should require no change to this kernel or to how the
# grid is indexed, only to the boundary condition.
#
# All 6 neighbors are equidistant on a true hex lattice, so unlike the old
# corner/edge-weighted Moore kernel, no split weighting is needed here.
LAPLACIAN_KERNEL = np.array([
    [1/6, 1/6, 0.0],
    [1/6, -1.0, 1/6],
    [0.0, 1/6, 1/6],
])


def laplacian(Z):
    """Discrete Laplacian on a hex-neighbor stencil, with periodic wrap on
    both storage axes.

    Storing a hex lattice in a plain n x n array with wraparound on both
    axes makes the domain a periodic rhombus (parallelogram) tile of the
    hex lattice -- this is the default domain. It's not the genuine
    hexagon-shaped, edge-to-edge tile bz_hex.py builds for the BZ system
    (order-m centered hexagon, coset reduction via normalize()/to_hex(),
    translation-only boundary); that domain is available here too via
    --hex-tile / run_hex_tile(), which reuses this same kernel and
    coordinate handedness, just with bz_hex.py's boundary condition
    swapped in instead of plain wrap.
    """
    return convolve(Z, LAPLACIAN_KERNEL, mode='wrap')


def stretch_contrast(field, low_pct, high_pct, out_min, out_max, mask=None):
    """Levels-style contrast stretch, matching a manual GIMP Levels workflow:

    1. Find the low_pct/high_pct percentiles of the field (excluding
       outliers at both ends, rather than using the raw min/max).
    2. Clip the field to that [lo, hi] range -- values below lo become
       exactly lo, values above hi become exactly hi (a floor/ceiling
       clip, not a clip to 0/1).
    3. Linearly rescale [lo, hi] to [out_min, out_max].

    With the default 1st/99th percentiles and an out_min/out_max of
    0.4/1.0, this compresses V's dynamic range into the upper 60% of
    [0, 1] -- what was a mostly-black field with thin bright ridges
    becomes a solid mid-tone base with the same ridges standing out
    against it, instead of muted/low-contrast throughout.

    mask, if given, restricts which cells the percentiles are COMPUTED
    from (e.g. a hexagon or --hex-tile mask) -- important because
    assemble_rotate_hex()/run_hex_tile() fill area outside the true shape
    with 0 or leftover simulation noise; including those in the percentile
    calculation would skew lo/hi incorrectly. The rescale itself is still
    applied to the whole array; masked-out cells are invisible anyway
    (alpha=0) so their exact value doesn't matter.
    """
    sample = field[mask] if mask is not None else field
    lo = np.percentile(sample, low_pct)
    hi = np.percentile(sample, high_pct)
    clipped = np.clip(field, lo, hi)
    if hi > lo:
        scaled = (clipped - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(field)  # degenerate case: a flat field
    return np.clip(scaled * (out_max - out_min) + out_min, 0.0, 1.0)


def upscale_field(V, mask, target_size, blur_sigma):
    """Upscale the simulated field (and its mask, if any) to roughly
    target_size cells per side, using cubic-spline interpolation, then
    optionally apply a Gaussian blur.

    This is for image-to-STL pipelines: the simulation grid is often too
    coarse to extrude cleanly, since each simulated cell becomes a jagged
    stair-step in the final geometry rather than a smooth ridge wall.
    Cubic interpolation already fits a smooth curve through the samples in
    a single pass, so unlike a manual "scale in several steps" workflow
    (which is mainly a workaround for lower-order/nearest-neighbor
    scaling), one direct zoom to the target resolution is enough; the
    optional Gaussian blur afterward is for smoothing out any remaining
    jaggedness beyond what interpolation alone removes.

    mask is upscaled separately, as its 0/1 representation through linear
    (not cubic) interpolation and re-thresholded at 0.5, to keep a clean
    boolean mask with a reasonably smooth edge rather than either a jagged
    nearest-neighbor edge or a semi-transparent blurred one. Note: cubic
    interpolation can overshoot slightly near the sharp value/zero
    discontinuity at a mask's edge (e.g. --hex-tile's corner cells, or
    assemble_rotate_hex's exterior fill) -- any overshoot outside the mask
    is invisible (alpha=0 there regardless), but a thin sliver just INSIDE
    the edge can pick up a small amount of it. This is a minor, localized
    edge effect, not a full-field artifact.
    """
    n = V.shape[0]
    zoom_factor = target_size / n
    V_up = np.clip(zoom(V, zoom_factor, order=3), 0.0, 1.0)
    if blur_sigma > 0:
        V_up = np.clip(gaussian_filter(V_up, sigma=blur_sigma), 0.0, 1.0)
    mask_up = None
    if mask is not None:
        mask_up = zoom(mask.astype(float), zoom_factor, order=1) >= 0.5
    return V_up, mask_up


def colorize(field, cmap_name, mask=None):
    """Map a 2D field (values in [0, 1]) through a colormap to uint8 RGBA.

    mask, if given, is a boolean array the same shape as field: True cells
    keep their normal alpha, False cells get alpha=0 (used to hide the
    corner cells that fall outside a true hexagon tile -- see
    run_hex_tile()). mask=None (the periodic-rhombus domain, where every
    cell is real) leaves alpha untouched.
    """
    rgba = np.clip(matplotlib.colormaps[cmap_name](field), 0, 1)
    if mask is not None:
        rgba[..., 3] = np.where(mask, rgba[..., 3], 0.0)
    return (rgba * 255).astype(np.uint8)


def hex_skew_image(rgba):
    """Shear a uint8 RGBA array (n, n, 4) so the hex-neighbor lattice
    displays as true hexagonal packing instead of a plain square grid.

    The array's two axes (row i, column j) are stored as an ordinary
    square grid, but laplacian()'s kernel treats them as a hex lattice
    where the (Delta_i, Delta_j) = (+1, +1) / (-1, -1) diagonal is a
    nearest neighbor and (+1, -1) / (-1, +1) is not (hex distance, via
    max(|Delta_i|, |Delta_j|, |Delta_i - Delta_j|), is 1 for the former,
    2 for the latter). For that to look isotropic on screen, row i and
    column j need to be mapped to Cartesian axes 120 degrees apart (not
    90), with the same handedness as bz_hex.py's FILTER:

        X = j - 0.5*i,   Y = i * (sqrt(3)/2)

    Under this map, unit steps in (Delta_i, Delta_j) = (1,0), (0,1), and
    (1,1) all have Cartesian length 1 and land 60 degrees apart, while
    (1,-1) lands at length sqrt(3) -- i.e. exactly the neighbors the
    kernel treats as distance-1 come out equal-length and 60 degrees
    apart, and the excluded diagonal comes out farther away, as it should.

    Used for both domains this script supports: the default periodic
    rhombus (whole array real, mask=None going in) and --hex-tile's
    genuine hexagon (corner cells already made transparent by colorize's
    mask) -- this function only cares about geometry, not which cells are
    authoritative.
    """
    n = rgba.shape[0]
    src = Image.fromarray(rgba, mode='RGBA')

    x_min = -0.5 * (n - 1)
    out_w = max(1, int(round(1.5 * n)))
    out_h = max(1, int(round(n * math.sqrt(3) / 2)))

    # PIL's AFFINE gives, for each output pixel (x, y), the source pixel to
    # sample: source = (a*x + b*y + c, d*x + e*y + f). Inverting
    # X = out_x + x_min, Y = out_y  against  X = j - 0.5*i, Y = i*sqrt(3)/2
    # gives source column j = X + Y/sqrt(3), source row i = Y * 2/sqrt(3).
    coeffs = (1.0, 1.0 / math.sqrt(3), x_min,
              0.0, 2.0 / math.sqrt(3), 0.0)
    return src.transform((out_w, out_h), Image.Transform.AFFINE, coeffs,
                          resample=Image.Resampling.BICUBIC)


def crop_to_content(img):
    """Crop a PIL RGBA image to the bounding box of its non-transparent
    content, so the saved file has no blank margin around the hexagon /
    rhombus / parallelogram shape.

    Cropped strictly from the ALPHA channel, not PIL's default getbbox()
    (which looks at all channels together): a masked-out cell's RGB isn't
    guaranteed to be black for every --cmap choice (only the default
    'gray' maps value 0 to black), so an all-channel bbox could include
    the transparent margin for other colormaps. Alpha alone is always 0
    there regardless of --cmap, so it's the reliable signal.
    """
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    return img.crop(bbox) if bbox else img


def flatten_onto_black(img):
    """Composite a cropped RGBA image onto a solid opaque black
    background, returning a plain RGB image with no alpha channel at all.

    This is the fix for a real reported issue: many image-to-STL tools
    ignore the alpha channel entirely and read RGB brightness directly for
    height. A transparent background can leave faint semi-transparent
    edge pixels with nonzero RGB brightness -- introduced by antialiasing
    during the hex shear's resampling, or by --output-size's cubic
    upscaling -- that alpha alone would have hidden from view but that an
    alpha-blind tool reads as real (if faint) height, producing thin
    spikes right at the shape's boundary. Confirmed independently: filling
    the background with opaque black in GIMP made the spikes disappear.

    Compositing onto black bakes the alpha into the RGB values directly
    (a pixel at 20% alpha becomes 20%-brightness-toward-black, not
    "invisible but secretly still bright"), and dropping the alpha channel
    entirely afterward means there's no separate channel left for a
    downstream tool to disregard.
    """
    background = Image.new('RGBA', img.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, img).convert('RGB')


def build_png_metadata(args, f, k):
    """Build a PngInfo of tEXt chunks recording the parameters behind a
    render, so the exact settings survive in the file itself -- no need
    to remember or separately log which f/k/seed/etc. produced a given
    image. Includes the full command line verbatim (the most complete,
    unambiguous record) plus individual keys for the parameters people are
    most likely to want to check or search on without re-parsing it.
    """
    info = PngInfo()
    info.add_text('gray_scott_command', ' '.join(sys.argv))
    info.add_text('feed_f', f'{f:.6f}')
    info.add_text('kill_k', f'{k:.6f}')
    info.add_text('du', str(args.du))
    info.add_text('dv', str(args.dv))
    info.add_text('dt', str(args.dt))
    info.add_text('iterations', str(args.iterations))
    info.add_text('seed', str(args.seed))
    info.add_text('num_seeds', str(args.num_seeds))
    info.add_text('cmap', args.cmap)
    if args.feed is None and args.kill is None:
        info.add_text('preset', args.preset)
    if args.hex_tile:
        info.add_text('domain', f'hex-tile order={args.hex_order}')
    elif args.rotate_tile:
        domain = f'rotate-tile n={args.size}'
        if args.assemble_hex:
            domain += ' assembled-hexagon'
        info.add_text('domain', domain)
    else:
        info.add_text('domain', f'periodic-rhombus n={args.size}')
    if args.contrast_stretch:
        info.add_text('contrast_stretch',
                       f'{args.stretch_low_pct}-{args.stretch_high_pct}pct '
                       f'-> {args.stretch_out_min}-{args.stretch_out_max}')
    if args.output_size is not None:
        info.add_text('output_size', str(args.output_size))
        info.add_text('upscale_blur_sigma', str(args.upscale_blur_sigma))
    info.add_text('transparent_background', str(args.transparent_background))
    return info


def seed_grid(n, num_seeds, rng, dense=False):
    """Initialize U=1, V=0 nearly everywhere, with perturbed blobs to
    nucleate pattern growth and a little noise throughout to avoid
    perfectly symmetric (and therefore artificial-looking) decay.

    A few isolated blobs (dense=False) is the right initial condition for
    self-replicating-spot regimes like mitosis: each blob grows and splits
    independently. But coral/solitons/maze are Turing instabilities of the
    *whole field* -- a handful of islands in an otherwise near-uniform grid
    just relax into static stable spots and never destabilize outward,
    which is the "3 squares" failure mode. dense=True scatters many more,
    smaller blobs across the grid instead, which is enough perturbation to
    actually trigger space-filling branching growth.

    Each blob is a smooth 2D Gaussian bump, not a filled square or a
    filled circle. A flat-value region has a hard edge -- even a
    rasterized disk is jagged and locally flat-then-sharp at small radius
    -- and the reaction term amplifies whatever boundary it's given faster
    than diffusion rounds it off, so growth inherits and enlarges the
    seed's edge shape instead of curving. A Gaussian has no edge to
    inherit in the first place, which is what actually produces curved
    growth instead of squares; more iterations or softer/circular hard
    edges don't fix this on their own (verified empirically, not assumed).
    """
    U = np.ones((n, n))
    V = np.zeros((n, n))
    U += 0.02 * rng.random((n, n))
    V += 0.02 * rng.random((n, n))
    if dense:
        count = max(num_seeds, n // 4)
        sigma = max(2, n // 60)
    else:
        count = num_seeds
        sigma = max(3, n // 20)
    yy, xx = np.ogrid[:n, :n]
    for _ in range(count):
        cy = rng.integers(0, n)
        cx = rng.integers(0, n)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        bump = 0.25 * np.exp(-d2 / (2 * sigma ** 2))
        V += bump
        U -= bump
    np.clip(U, 0, 1, out=U)
    np.clip(V, 0, 1, out=V)
    return U, V


def step(U, V, Du, Dv, f, k, dt):
    """Advance one Euler timestep."""
    Lu = laplacian(U)
    Lv = laplacian(V)
    reaction = U * V * V
    U_new = U + dt * (Du * Lu - reaction + f * (1 - U))
    V_new = V + dt * (Dv * Lv + reaction - (f + k) * V)
    np.clip(U_new, 0, 1, out=U_new)
    np.clip(V_new, 0, 1, out=V_new)
    return U_new, V_new


def run(n, iterations, Du, Dv, f, k, dt, num_seeds, rng, dense=False):
    U, V = seed_grid(n, num_seeds, rng, dense=dense)
    for _ in range(iterations):
        U, V = step(U, V, Du, Dv, f, k, dt)
    return U, V


# ---------------------------------------------------------------------------
# --hex-tile: a genuine hexagon-shaped, edge-to-edge domain, ported from
# bz_hex.py. normalize(), map_hex(), build_hex_border_indices(), PaddedGrid,
# and fill_border() are copied over unchanged -- they're generic lattice
# machinery that doesn't know or care what reaction runs on top of it (the
# species count only shows up via PaddedGrid's ':' slicing, which works for
# 2 species here same as it worked for BZ's 3). Only the per-step update
# rule differs, since Gray-Scott's diffusion-plus-reaction equations aren't
# BZ's neighborhood-average-plus-reaction rule, and the kernel here is a
# radius-1 Laplacian rather than bz_hex.py's radius-3 averaging filter, so
# the padding width differs too (1, not 3).
# ---------------------------------------------------------------------------

def normalize(x, y, m):
    """Reduce (x, y) to its coset's representative in the parallelogram
    spanned by the oblique lattice generators a=(2m+1, m+1), b=(m, 2m+1).
    See bz_hex.py for the full derivation; unchanged here.
    """
    a1, a2 = 2 * m + 1, m + 1
    b1, b2 = m, 2 * m + 1
    D = a1 * b2 - a2 * b1          # = 3m^2 + 3m + 1
    i = (b2 * x - b1 * y) // D
    j = (-a2 * x + a1 * y) // D
    return x - a1 * i - b1 * j, y - a2 * i - b2 * j


def map_hex(m):
    """The D = 3m^2+3m+1 cells of the order-m hexagon, as a dict from each
    cell's parallelogram-normalized coset representative to the cell's own
    (i, j) inside the hexagon. Unchanged from bz_hex.py."""
    to_hex = {}
    for i in range(2 * m + 1):
        for j in range(2 * m + 1):
            if abs(i - j) <= m:
                to_hex[normalize(i, j, m)] = (i, j)
    return to_hex


def build_hex_border_indices(n, pad, m):
    """Precompute the border-ring coordinates and their hex-lattice source
    coordinates, once. Unchanged from bz_hex.py except that pad is a
    parameter here (1, to match this script's radius-1 kernel) rather than
    always 3.
    """
    to_hex = map_hex(m)
    lo, hi = -pad, n - 1 + pad
    border_i, border_j, src_i, src_j = [], [], [], []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if 0 <= i < n and 0 <= j < n and abs(i - j) <= m:
                continue  # true hex cell -- authoritative, never overwritten
            ti, tj = to_hex[normalize(i, j, m)]
            border_i.append(i)
            border_j.append(j)
            src_i.append(ti)
            src_j.append(tj)
    return (np.array(border_i), np.array(border_j),
            np.array(src_i), np.array(src_j))


class PaddedGrid:
    """Adapter so the simulation can be indexed with conceptual (signed)
    (i, j) coordinates, e.g. g[p, k, -1, -1], while the real storage is an
    ordinary non-negative-indexed numpy array shifted by `pad`. Unchanged
    from bz_hex.py; k here ranges over 2 species (U, V) instead of 3.
    """
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


def fill_border(g, border_i, border_j, src_i, src_j, p):
    """Vectorized border fill for one buffer slot p, all species at once.
    Unchanged from bz_hex.py."""
    g[p, :, border_i, border_j] = g[p, :, src_i, src_j]


HEX_TILE_PAD = 1  # matches LAPLACIAN_KERNEL's radius (3x3 -> radius 1);
                   # bz_hex.py's PAD=3 matches ITS radius-3 filter instead.


def hex_tile_step(p, g, border_i, border_j, src_i, src_j, Du, Dv, f, k, dt):
    """One Euler step on the padded hex-tile array: refresh buffer p's
    border from the true hex-lattice periodic image first (same pattern as
    bz_hex.py's update()), then compute the ordinary Gray-Scott Laplacian
    via non-wrapping convolution -- correct in the interior because the
    border was just filled out to exactly the kernel's radius.
    """
    q = (p + 1) % 2
    fill_border(g, border_i, border_j, src_i, src_j, p)

    U, V = g.raw[p, 0], g.raw[p, 1]
    Lu = convolve(U, LAPLACIAN_KERNEL, mode='constant', cval=0.0)
    Lv = convolve(V, LAPLACIAN_KERNEL, mode='constant', cval=0.0)
    reaction = U * V * V
    U_new = U + dt * (Du * Lu - reaction + f * (1 - U))
    V_new = V + dt * (Dv * Lv + reaction - (f + k) * V)
    np.clip(U_new, 0, 1, out=U_new)
    np.clip(V_new, 0, 1, out=V_new)
    g.raw[q, 0], g.raw[q, 1] = U_new, V_new


def run_hex_tile(m, iterations, Du, Dv, f, k, dt, num_seeds, rng, dense=False):
    """Run Gray-Scott on a genuine order-m hexagon tile (3m^2+3m+1 cells,
    translation-only periodic boundary) instead of the periodic-rhombus
    torus laplacian()/run() use. Mirrors bz_hex.py's structure: simulate
    the whole (2m+1) x (2m+1) storage box (including the corner cells
    outside the true hexagon, which run the same equations but are never
    kept consistent with anything -- see build_hex_border_indices), refresh
    a 1-cell border each step, and return the full interior. The corner
    cells should be masked out at render time (see colorize()'s mask).
    """
    n = 2 * m + 1
    U0, V0 = seed_grid(n, num_seeds, rng, dense=dense)

    size = n + 2 * HEX_TILE_PAD
    arr = np.zeros((2, 2, size, size))  # (parity, species[U,V], row, col)
    arr[0, 0, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n] = U0
    arr[0, 1, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n] = V0
    g = PaddedGrid(arr, HEX_TILE_PAD)

    border_i, border_j, src_i, src_j = build_hex_border_indices(n, HEX_TILE_PAD, m)

    for i in range(iterations):
        hex_tile_step(i % 2, g, border_i, border_j, src_i, src_j, Du, Dv, f, k, dt)

    # Only buffer p's border is refreshed at the *start* of the step that
    # reads it; the buffer just written was never itself refreshed. Do one
    # more pass so the returned border (used by the hex_mask'd interior's
    # own edge cells) actually reflects the boundary condition.
    final_p = iterations % 2
    fill_border(g, border_i, border_j, src_i, src_j, final_p)
    U = g.raw[final_p, 0, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n]
    V = g.raw[final_p, 1, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n]
    return U, V


# ---------------------------------------------------------------------------
# --rotate-tile: an n x n rhombus whose border is filled from a
# 120-degree-ROTATED copy of itself rather than a plain translated copy.
# Ported from zb3_padded.py's default 'rotate' --symmetry (that file also
# has 'reflect'/'blend' variants and a --border-symmetry post-pass; only
# 'rotate' is ported here, since that's the construction that produces the
# 3-fold-symmetric tiling in the reference picture).
#
# Structurally simpler than --hex-tile: there are no "corner cells outside
# the true domain" to mask, since a rotate-tile's n x n storage box IS the
# whole rhombus -- every cell is authoritative. Only the border-fill
# relation differs from the default torus.
# ---------------------------------------------------------------------------

def rotate_equivalence(i, j, n):
    """Find the equivalent coordinates in the base rhombus by rotating
    through symmetries. i, j may be any integers, including negative or
    >= n. Unchanged from zb3_padded.py's equivalence(): the rotation
    ((i // n) + (j // n)) % 3 identifies which of the 3 rhombic tiles
    around the base rhombus (i, j) falls into, and the affine map
    (i, j) -> (-j - 1, i - j) is a 120-degree rotation that walks toward
    that base tile.
    """
    while True:
        if ((i // n) + (j // n)) % 3 == 0:
            return i % n, j % n
        i, j = -j - 1, i - j


def build_rotate_border_indices(n, pad):
    """Precompute the border-ring coordinates and their rotate-equivalent
    source coordinates in the base n x n rhombus, once. Same structure as
    build_hex_border_indices, using rotate_equivalence instead of the
    hex-tile's normalize()/to_hex() coset reduction.
    """
    lo, hi = -pad, n - 1 + pad
    border_i, border_j, src_i, src_j = [], [], [], []
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            if 0 <= i < n and 0 <= j < n:
                continue  # interior, not border
            ti, tj = rotate_equivalence(i, j, n)
            border_i.append(i)
            border_j.append(j)
            src_i.append(ti)
            src_j.append(tj)
    return (np.array(border_i), np.array(border_j),
            np.array(src_i), np.array(src_j))


def rotate_tile_step(p, g, border_i, border_j, src_i, src_j, Du, Dv, f, k, dt):
    """One Euler step on the padded rotate-tile array. Identical structure
    to hex_tile_step() -- only which border_i/border_j/src_i/src_j indices
    get passed in differs (built by build_rotate_border_indices instead of
    build_hex_border_indices)."""
    q = (p + 1) % 2
    fill_border(g, border_i, border_j, src_i, src_j, p)

    U, V = g.raw[p, 0], g.raw[p, 1]
    Lu = convolve(U, LAPLACIAN_KERNEL, mode='constant', cval=0.0)
    Lv = convolve(V, LAPLACIAN_KERNEL, mode='constant', cval=0.0)
    reaction = U * V * V
    U_new = U + dt * (Du * Lu - reaction + f * (1 - U))
    V_new = V + dt * (Dv * Lv + reaction - (f + k) * V)
    np.clip(U_new, 0, 1, out=U_new)
    np.clip(V_new, 0, 1, out=V_new)
    g.raw[q, 0], g.raw[q, 1] = U_new, V_new


def run_rotate_tile(n, iterations, Du, Dv, f, k, dt, num_seeds, rng, dense=False):
    """Run Gray-Scott on an n x n rhombus with 120-degree-rotation border
    filling. Same structure as run_hex_tile(), simpler in one respect: no
    masking needed on return, since every cell in the n x n box is real.
    """
    U0, V0 = seed_grid(n, num_seeds, rng, dense=dense)

    size = n + 2 * HEX_TILE_PAD
    arr = np.zeros((2, 2, size, size))
    arr[0, 0, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n] = U0
    arr[0, 1, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n] = V0
    g = PaddedGrid(arr, HEX_TILE_PAD)

    border_i, border_j, src_i, src_j = build_rotate_border_indices(n, HEX_TILE_PAD)

    for i in range(iterations):
        rotate_tile_step(i % 2, g, border_i, border_j, src_i, src_j, Du, Dv, f, k, dt)

    final_p = iterations % 2
    fill_border(g, border_i, border_j, src_i, src_j, final_p)
    U = g.raw[final_p, 0, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n]
    V = g.raw[final_p, 1, HEX_TILE_PAD:HEX_TILE_PAD + n, HEX_TILE_PAD:HEX_TILE_PAD + n]
    return U, V


def _point_in_hex(i, j, hexagon):
    """Even-odd point-in-polygon test for a hexagon given as a list of
    (i, j) tuples in order around its boundary (works for any simple
    polygon, hexagon included)."""
    inside = False
    x1, y1 = hexagon[-1]
    for x2, y2 in hexagon:
        if (y1 > j) != (y2 > j):
            x_at_j = (x2 - x1) * (j - y1) / (y2 - y1) + x1
            if i < x_at_j:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def assemble_rotate_hex(V, n):
    """Plot the values of the simulated n x n rotate-tile inside a single
    hexagon-shaped region of the plane, using rotate_equivalence() as a
    pure pointwise lookup -- no boundaries should be visible, since it's
    all one continuous field, just displayed over a differently-shaped
    window than the plain n x n tile.

    Two earlier versions of this function were wrong, for different
    reasons worth recording:

    1. Naive axis-aligned quadrants of the (i//n, j//n) tile-index grid.
       Broken because the rotation's fixed point is (-2/3, -1/3), not a
       lattice cell, so an axis-aligned block isn't rigid under it --
       different points in the SAME block could take a different number
       of rotation steps to reduce home, fracturing it into a patchwork.

    2. Three separate parallelograms, each found by applying the rotation
       map to the base tile's own corners (once for 120 degrees, twice for
       240). Each one, checked individually, WAS a genuine rigid rotated
       copy (verified by de-warping it through its own corner-derived
       local axes and reproducing the base tile's content exactly). But
       the three of them never actually touched each other -- checked by
       counting adjacent cell-pairs across the whole assembled mask, which
       came back zero for every pair. Matching values at cells that reduce
       to the same source (which rotate_equivalence guarantees trivially,
       everywhere) isn't the same claim as those cells being geometrically
       adjacent -- conflating the two was the mistake.

    The actual fix needs no per-copy bookkeeping at all: the field
    val(i, j) = V[rotate_equivalence(i, j, n)] is already smooth over the
    entire plane (this was confirmed independently, early on, by a
    wide-area gradient check spanning several tile-widths). So ANY compact
    region filled this way renders seamlessly -- the region only needs to
    be hexagon-shaped, not built from any particular number of "copies."

    The hexagon is centered on the rotation's true fixed point (found by
    solving (i,j) = (-j-1, i-j), giving (-2/3, -1/3) -- not a lattice
    point, which is exactly why approach 1 above didn't work) and sized,
    via the same shear used for display, to have screen-space area equal
    to 3 tile-areas (matching "3 copies" in scale without needing 3
    separate pieces).
    """
    T = np.array([[-0.5, 1.0], [math.sqrt(3) / 2, 0.0]])
    T_inv = np.linalg.inv(T)
    pivot_ij = np.array([-2 / 3, -1 / 3])
    pivot_xy = T @ pivot_ij

    tile_xy = [T @ np.array(p) for p in [(0, 0), (n, 0), (n, n), (0, n)]]
    tile_area = abs(sum(tile_xy[k][0] * tile_xy[(k + 1) % 4][1]
                        - tile_xy[(k + 1) % 4][0] * tile_xy[k][1]
                        for k in range(4))) / 2
    R = math.sqrt(3 * tile_area / (3 * math.sqrt(3) / 2))  # circumradius

    angles = np.radians(np.arange(0, 360, 60))
    hexagon = [tuple(T_inv @ (pivot_xy + R * np.array([np.cos(a), np.sin(a)])))
               for a in angles]

    lo = -n - 2
    size = 2 * n + 4  # generous bound around the hexagon
    big = np.zeros((size, size))
    mask = np.zeros((size, size), dtype=bool)
    for a in range(size):
        i = a + lo
        for b in range(size):
            j = b + lo
            if _point_in_hex(i, j, hexagon):
                si, sj = rotate_equivalence(i, j, n)
                big[a, b] = V[si, sj]
                mask[a, b] = True
    return big, mask


def unique_path(path):
    """Return `path` unchanged if it doesn't exist yet, otherwise append
    _1, _2, etc. before the extension until a non-colliding name is found.
    """
    p = Path(path)
    if not p.exists():
        return str(p)
    i = 1
    while True:
        candidate = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not candidate.exists():
            return str(candidate)
        i += 1


if args.sweep:
    # Contact-sheet cells stay flat (unsheared) even though the underlying
    # kernel is hex-neighbor now -- this view is for fast comparison across
    # (f, k), not for judging final geometry, so the shear is skipped here.
    f_values = np.linspace(args.sweep_f_range[0], args.sweep_f_range[1], args.sweep_steps)
    k_values = np.linspace(args.sweep_k_range[0], args.sweep_k_range[1], args.sweep_steps)
    fig, axes = plt.subplots(args.sweep_steps, args.sweep_steps,
                              figsize=(args.sweep_steps * 2, args.sweep_steps * 2))
    total = args.sweep_steps ** 2
    done = 0
    for row, k_val in enumerate(k_values):
        for col, f_val in enumerate(f_values):
            U, V = run(args.sweep_size, args.sweep_iterations, args.du, args.dv,
                       f_val, k_val, args.dt, args.num_seeds, rng, dense=True)
            ax = axes[row, col]
            ax.imshow(V, cmap=args.cmap, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == args.sweep_steps - 1:
                ax.set_xlabel(f'f={f_val:.4f}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'k={k_val:.4f}', fontsize=8)
            done += 1
            print(f'  sweep cell {done}/{total} done (f={f_val:.4f}, k={k_val:.4f})')
    fig.suptitle('Gray-Scott sweep -- rows: kill rate k, columns: feed rate f')
    fig.tight_layout()
    out_path = unique_path('gray_scott_sweep.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved sweep contact sheet to {out_path}')
else:
    f = args.feed if args.feed is not None else PRESETS[args.preset][0]
    k = args.kill if args.kill is not None else PRESETS[args.preset][1]
    # Mitosis is a self-replicating-spot regime: sparse isolated blobs are
    # the correct seed, since each one grows and splits on its own. The
    # other presets are Turing instabilities of the whole field and need
    # denser scattered perturbation to destabilize -- see seed_grid().
    dense = args.preset != 'mitosis' if args.feed is None and args.kill is None else True

    if args.hex_tile:
        m = args.hex_order
        U, V = run_hex_tile(m, args.iterations, args.du, args.dv, f, k, args.dt,
                             args.num_seeds, rng, dense=dense)
        n = 2 * m + 1
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        hex_mask = np.abs(ii - jj) <= m  # True inside the true hexagon
    elif args.rotate_tile:
        U, V = run_rotate_tile(args.size, args.iterations, args.du, args.dv, f, k,
                                args.dt, args.num_seeds, rng, dense=dense)
        if args.assemble_hex:
            V, hex_mask = assemble_rotate_hex(V, args.size)
        else:
            hex_mask = None  # every cell in a rotate-tile rhombus is real
    else:
        U, V = run(args.size, args.iterations, args.du, args.dv, f, k, args.dt,
                   args.num_seeds, rng, dense=dense)
        hex_mask = None  # periodic-rhombus domain: every cell is real

    if args.contrast_stretch:
        V = stretch_contrast(V, args.stretch_low_pct, args.stretch_high_pct,
                              args.stretch_out_min, args.stretch_out_max, mask=hex_mask)

    if args.output_size is not None:
        V, hex_mask = upscale_field(V, hex_mask, args.output_size, args.upscale_blur_sigma)

    out_path = unique_path('gray_scott_out.png')
    rgba = colorize(V, args.cmap, mask=hex_mask)
    if args.flat:
        # Raw array, no shear -- the hex-neighbor kernel (and, with
        # --hex-tile, the true hexagon mask) is still in effect, but this
        # is what it looks like without correcting the display for hex
        # geometry. Mostly useful for sanity-checking the domain itself.
        img = crop_to_content(Image.fromarray(rgba, mode='RGBA'))
    else:
        img = crop_to_content(hex_skew_image(rgba))
    if not args.transparent_background:
        img = flatten_onto_black(img)
    img.save(out_path, pnginfo=build_png_metadata(args, f, k))
    preset_note = f" (preset: {args.preset})" if args.feed is None and args.kill is None else ""
    if args.hex_tile:
        domain_note = f", hex-tile order {args.hex_order}"
    elif args.rotate_tile and args.assemble_hex:
        domain_note = f", rotate-tile n={args.size} assembled into hexagon"
    elif args.rotate_tile:
        domain_note = f", rotate-tile n={args.size}"
    else:
        domain_note = ""
    upscale_note = f", upscaled to ~{args.output_size}px" if args.output_size is not None else ""
    print(f'Rendered with f={f:.4f}, k={k:.4f}{preset_note}{domain_note}{upscale_note}')
    print(f'Saved to {out_path}')
