#!/usr/bin/env python3
"""Build a labeled thumbnail grid from Gray-Scott PNGs.

Reads f, k, Du, Dv directly from each file's embedded PNG metadata
(the tEXt chunks written by gray_scott.py), so no parameters need to
be typed in by hand.

Usage:
    python3 gray_scott_examples.py file1.png file2.png ...
    python3 gray_scott_examples.py *.png -o out.png
    python3 gray_scott_examples.py            # defaults to *.png in cwd
"""
import argparse
import glob
import math
import os
import re

from PIL import Image, ImageDraw, ImageFont

THUMB = 380
PAD = 28
MARGIN = 45
LABEL_H = 78
TITLE_H = 64
GAP_TITLE = 20
TITLE_TEXT = "Gray-Scott Reaction-Diffusion — Examples"


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{name}"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def centered_text(draw, cx, y, text, font, fill="black"):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((cx - w / 2, y), text, fill=fill, font=font)


def make_tag(path):
    """'gray_scott_out_95.png' -> '#95'; anything without trailing digits -> its stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", stem)
    return f"#{m.group(1)}" if m else stem


def get_params(path):
    """Read f/k/du/dv from the PNG's embedded text metadata, if present."""
    with Image.open(path) as im:
        info = dict(im.info)

    def fnum(key):
        try:
            return float(info[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {"f": fnum("feed_f"), "k": fnum("kill_k"), "du": fnum("du"), "dv": fnum("dv")}


def grid_dims(n, default_cols=3):
    """Use `default_cols` columns once there are enough files to fill a row;
    for 1-2 files, use fewer columns so the grid isn't mostly empty space."""
    cols = min(default_cols, n)
    rows = math.ceil(n / cols)
    return cols, rows


def build_grid(files, output, default_cols=3):
    font_title = load_font(28, bold=True)
    font_tag = load_font(19, bold=True)
    font_label = load_font(16)

    cols, rows = grid_dims(len(files), default_cols)
    cell_w, cell_h = THUMB, THUMB + LABEL_H

    grid_w = MARGIN * 2 + cols * cell_w + (cols - 1) * PAD
    canvas_h = TITLE_H + GAP_TITLE + rows * cell_h + (rows - 1) * PAD + MARGIN

    # Widen the canvas if the title is wider than the grid itself (small file counts).
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title_bbox = tmp_draw.textbbox((0, 0), TITLE_TEXT, font=font_title)
    title_w = (title_bbox[2] - title_bbox[0]) + MARGIN * 2
    canvas_w = max(grid_w, title_w)
    x_offset = (canvas_w - grid_w) / 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    centered_text(draw, canvas_w / 2, 22, TITLE_TEXT, font_title)

    start_y = TITLE_H + GAP_TITLE

    for idx, path in enumerate(files):
        row, col = divmod(idx, cols)
        x = x_offset + MARGIN + col * (cell_w + PAD)
        y = start_y + row * (cell_h + PAD)

        im = Image.open(path).convert("RGB")
        im.thumbnail((THUMB, THUMB), Image.LANCZOS)
        tx, ty = x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2
        canvas.paste(im, (int(tx), int(ty)))
        draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(200, 200, 200), width=1)

        cx, ly = x + THUMB / 2, y + THUMB + 8
        centered_text(draw, cx, ly, make_tag(path), font_tag)

        params = get_params(path)
        fk = [f"{k} = {params[k]:.4f}" for k in ("f", "k") if params[k] is not None]
        duv = [f"{'Du' if k == 'du' else 'Dv'} = {params[k]}" for k in ("du", "dv") if params[k] is not None]
        for i, line in enumerate(l for l in ("   ".join(fk), "   ".join(duv)) if l):
            centered_text(draw, cx, ly + 24 + i * 22, line, font_label)

    canvas.save(output)
    print(f"saved {output} ({canvas_w}x{canvas_h}), {len(files)} image(s), grid {cols}x{rows}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="PNG files to include")
    parser.add_argument("-o", "--output", default="gray_scott_examples.png", help="output file path")
    parser.add_argument("--cols", type=int, default=3, help="grid columns once there are enough files (default: 3)")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob("*.png"))
    if not files:
        parser.error("no PNG files given, and none found in the current directory")

    build_grid(files, args.output, default_cols=args.cols)


if __name__ == "__main__":
    main()
