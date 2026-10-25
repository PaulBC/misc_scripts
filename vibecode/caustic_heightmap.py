"""Caustic of an arbitrary height-map piece (dome side up, flat bottom), exact refraction.

    python caustic_heightmap.py heightmap.png --width-mm 100 --relief-mm 0.6 --n 1.55 [--equalize]

Top-surface height above the flat bottom, in mm:  z = base + relief * (pixel / max pixel).
One ray per pixel (jittered inside the pixel) travelling straight down (sun). At the dome the
ray refracts air->resin (Snell, exact), crosses the resin to the flat bottom, refracts
resin->air (rays past the critical angle are dropped and counted), and lands on a card D mm
below the piece. Brightness = density of landing points. Orientation = the PNG's.

--equalize: measure every complete cell's own best focus distance (smallest half-light radius of its
bright core) and rescale that cell's relief so all cells focus at the median distance. Valid because
the groove network sits at height 0, so cells can be scaled independently and stay continuous.
Writes <prefix>_equalized.png (16-bit) and reports the relief to use with it.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt, label

ap = argparse.ArgumentParser()
ap.add_argument("png")
ap.add_argument("--width-mm", type=float, default=100.0, help="physical width of the whole image")
ap.add_argument("--relief-mm", type=float, default=0.6, help="height of the brightest pixel above the darkest")
ap.add_argument("--base-mm", type=float, default=1.0, help="resin thickness under the darkest pixel")
ap.add_argument("--n", type=float, default=1.55)
ap.add_argument("--blur-px", type=float, default=2.0, help="Gaussian blur of the height before taking slopes")
ap.add_argument("--dmax", type=float, default=160.0, help="largest card distance scanned (mm)")
ap.add_argument("--groove", type=float, default=0.06, help="pixels below this fraction of max height are 'groove'")
ap.add_argument("--equalize", action="store_true")
ap.add_argument("--prefix", default="caustic_hm")
args = ap.parse_args()

im = np.array(Image.open(args.png))
if im.ndim == 3:
    im = im[..., 0]
h0 = im.astype(np.float32) / float(im.max())
Ny, Nx = h0.shape
pitch = args.width_mm / Nx
print(f"image {Nx}x{Ny}px, {pitch*1000:.1f} um/px, footprint {args.width_mm:.0f} x {Ny*pitch:.0f} mm")
n = args.n
rng = np.random.default_rng(0)
cols, rows = np.meshgrid(np.arange(Nx), np.arange(Ny))
jx, jy = rng.random(cols.shape, dtype=np.float32), rng.random(cols.shape, dtype=np.float32)

cell_lab, ncell = label(h0 > args.groove)
sizes = np.bincount(cell_lab.ravel(), minlength=ncell + 1)
border = set(np.unique(np.r_[cell_lab[0], cell_lab[-1], cell_lab[:, 0], cell_lab[:, -1]]))
ids = np.array([i for i in range(1, ncell + 1) if sizes[i] > 3000 and i not in border])
area_mm2 = sizes[ids] * pitch**2


def build_rays(h, relief):
    z = args.base_mm + relief * h
    gy, gx = np.gradient(gaussian_filter(z, args.blur_px), pitch)
    gn = np.sqrt(1 + gx**2 + gy**2)
    Nx_, Ny_, Nz_ = -gx / gn, -gy / gn, 1 / gn
    eta = 1.0 / n
    coef = eta * Nz_ - np.sqrt(np.clip(1 - eta**2 * (1 - Nz_**2), 0, None))
    tx, ty, tz = coef * Nx_, coef * Ny_, -eta + coef * Nz_
    inx, iny = tx / np.abs(tz) * z, ty / np.abs(tz) * z
    ox, oy = n * tx, n * ty
    tir = (ox**2 + oy**2) >= 0.999
    oz = np.sqrt(np.clip(1 - ox**2 - oy**2, 1e-6, None))
    keep = ~tir
    return dict(
        px=((cols + jx) * pitch + inx)[keep].astype(np.float32),
        py=((rows + jy) * pitch + iny)[keep].astype(np.float32),
        sx=(ox / oz)[keep].astype(np.float32), sy=(oy / oz)[keep].astype(np.float32),
        lab=cell_lab[keep], tir=float(tir.mean()), tilt=float(np.degrees(np.arctan(np.hypot(gx, gy).mean()))))


pad = 10.0
ext = [-pad, args.width_mm + pad, -pad, Ny * pitch + pad]


def hist(r, D, bins, step=1):
    H, _, _ = np.histogram2d((r["py"] + r["sy"] * D)[::step], (r["px"] + r["sx"] * D)[::step],
                             bins=bins, range=[[ext[2], ext[3]], [ext[0], ext[1]]])
    return H


def sharpest(r):
    Ds = np.linspace(4, args.dmax, 64)
    s = []
    for D in Ds:
        H = hist(r, D, 500, step=3)
        s.append((H**2).sum() / H.sum()**2)
    return float(Ds[int(np.argmax(s))])


def per_cell(r):
    """Per complete cell: the card distance at which the radius holding 50% of that cell's light is
    smallest (the bright 'core'), plus that radius; and the least-RMS distance (dominated by the halo)."""
    lab = r["lab"]
    order = np.argsort(lab, kind="stable")
    lo_, hi_ = np.searchsorted(lab[order], ids), np.searchsorted(lab[order], ids, side="right")
    Dscan = np.linspace(15, args.dmax, 117)
    Dcore, rcore, Drms = [], [], []
    for a_, b_ in zip(lo_, hi_):
        sel = order[a_:b_][::7]
        px, py, sx, sy = r["px"][sel], r["py"][sel], r["sx"][sel], r["sy"][sel]
        best, brms = (1e9, 0.0), (1e9, 0.0)
        for D in Dscan:
            x, y = px + sx * D, py + sy * D
            r50 = np.median(np.hypot(x - np.median(x), y - np.median(y)))
            rms = np.sqrt(x.var() + y.var())
            if r50 < best[0]: best = (r50, D)
            if rms < brms[0]: brms = (rms, D)
        Dcore.append(best[1]); rcore.append(best[0]); Drms.append(brms[1])
    return np.array(Dcore), np.array(rcore), np.array(Drms)


def report(tag, r, Dc, rc, Dr, D_best):
    print(f"[{tag}] TIR loss {r['tir']*100:.2f}%; mean surface tilt {r['tilt']:.1f} deg")
    print(f"[{tag}] most concentrated light overall at D = {D_best:.1f} mm")
    print(f"[{tag}] {len(ids)} complete cells (area {area_mm2.min():.0f}-{area_mm2.max():.0f} mm^2)")
    print(f"[{tag}] bright-core focus per cell: median {np.median(Dc):.0f} mm, 10-90% {np.percentile(Dc,10):.0f}-{np.percentile(Dc,90):.0f} mm, "
          f"range {Dc.min():.0f}-{Dc.max():.0f} mm; corr with cell area {np.corrcoef(Dc, area_mm2)[0,1]:+.2f}")
    print(f"[{tag}] half-light radius at each cell's best core focus: median {np.median(rc):.2f} mm (cell radius ~{np.sqrt(area_mm2.mean()/np.pi):.1f} mm)")
    print(f"[{tag}] (least-RMS focus, which includes the halo: median {np.median(Dr):.0f} mm, range {Dr.min():.0f}-{Dr.max():.0f} mm)")


def picture(r, D, B=1100):
    H = gaussian_filter(hist(r, D, B), 1.2)
    return np.clip(H / np.percentile(H, 99.6), 0, 1) ** 0.5


def show(ax, r, D, title, overlay=False):
    ax.imshow(picture(r, D), extent=[ext[0], ext[1], ext[3], ext[2]], cmap="gray", vmin=0, vmax=1)
    if overlay:
        ey, ex = np.where((h0 < args.groove)[::4, ::4])
        ax.plot(ex * 4 * pitch, ey * 4 * pitch, ".", ms=0.8, color="#ff3030", alpha=0.6)
    x0, y0 = 4, Ny * pitch - 4
    ax.plot([x0, x0 + 10], [y0, y0], color="white", lw=3)
    ax.text(x0 + 5, y0 - 2.5, "10 mm", color="white", ha="center", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, pad=10)
    ax.set_xlim(0, args.width_mm); ax.set_ylim(Ny * pitch, 0)
    ax.axis("off")


# ---------------- pass 1: the height map as given ----------------
r1 = build_rays(h0, args.relief_mm)
D1 = sharpest(r1)
Dc1, rc1, Dr1 = per_cell(r1)
report("as given", r1, Dc1, rc1, Dr1, D1)

fig, axs = plt.subplots(1, 4, figsize=(25, 6.6), facecolor="black")
for ax, D in zip(axs[:3], [0.6 * D1, D1, 1.5 * D1]):
    show(ax, r1, D, f"D = {D:.0f} mm" + ("  (most concentrated)" if D == D1 else ""))
show(axs[3], r1, D1, f"D = {D1:.0f} mm  + groove network (red)", overlay=True)
plt.tight_layout(pad=2.0)
plt.savefig(f"{args.prefix}_panels.png", dpi=100, facecolor="black")
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 10.3), facecolor="black")
show(ax, r1, D1, f"Simulated caustic, D = {D1:.0f} mm  (n = {n}, relief {args.relief_mm} mm, width {args.width_mm:.0f} mm)")
plt.tight_layout(pad=1.5)
plt.savefig(f"{args.prefix}_focus.png", dpi=150, facecolor="black")
plt.close(fig)

# ---------------- pass 2: equalize every cell to a common focal distance ----------------
if args.equalize:
    Dt = float(np.median(Dc1))
    s = np.ones(ncell + 1)
    s[ids] = Dc1 / Dt                 # focus distance scales as 1/relief, so relief_i *= D_i / D_target
    _, (iy, ix) = distance_transform_edt(cell_lab == 0, return_indices=True)
    h2 = h0 * s[cell_lab[iy, ix]].astype(np.float32)   # grooves take the scale of the nearest cell
    relief2 = args.relief_mm * float(h2.max())
    print(f"[equalized] per-cell relief multipliers {s[ids].min():.2f} .. {s[ids].max():.2f}; target D = {Dt:.0f} mm")
    print(f"[equalized] peak relief becomes {relief2:.3f} mm -> use --relief-mm {relief2:.3f} with {args.prefix}_equalized.png")
    Image.fromarray(np.round(h2 / h2.max() * 65535).astype(np.uint16)).save(f"{args.prefix}_equalized.png")
    r2 = build_rays(h2, args.relief_mm)
    D2 = sharpest(r2)
    Dc2, rc2, Dr2 = per_cell(r2)
    report("equalized", r2, Dc2, rc2, Dr2, D2)
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.6), facecolor="black")
    show(axs[0], r1, Dt, f"as given, D = {Dt:.0f} mm")
    show(axs[1], r2, Dt, f"each cell's relief rescaled, D = {Dt:.0f} mm")
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{args.prefix}_compare.png", dpi=110, facecolor="black")
    plt.close(fig)
print("done")
