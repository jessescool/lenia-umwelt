# figure_generation/make_creature_strip.py — Jesse Cool (jessescool)
"""Render a horizontal strip of all creature thumbnails."""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from matplotlib import colormaps
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CREATURE_COLORS

SPECIES_NAMES = {
    "O2u": "Orbium unicaudatus",
    "S1s": "Scutium solidus",
    "K4s": "Kronium solidus",
    "K6s": "Ferrokronium solidus",
    "O2v": "Orbium orbulalis",
    "P4al": "Paraptera arcus labens",
}


def main():
    parser = argparse.ArgumentParser(description="Render labeled creature strip")
    parser.add_argument("--codes", nargs="+",
                        default=["O2u", "S1s", "K4s", "K6s"])
    parser.add_argument("--ori", type=int, default=3, help="Orientation index (o-file)")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--gap", type=int, default=3, help="Gap between cells (px)")
    parser.add_argument("--panel-border", type=int, default=1,
                        help="Border around each creature panel (px, pre-upscale)")
    parser.add_argument("--border", type=int, default=1, help="Border around image (px)")
    parser.add_argument("--upscale", type=int, default=8, help="Upscale factor for 600dpi")
    parser.add_argument("--colormap", default="magma")
    parser.add_argument("--invert", action="store_true", default=True,
                        help="Invert colormap and use white for empty space")
    parser.add_argument("--output", default="figure_generation/creature_strip.png")
    args = parser.parse_args()

    codes = args.codes
    gap = args.gap
    border = args.border
    n = len(codes)
    upscale = args.upscale

    cmap_name = args.colormap + ("_r" if args.invert else "")
    cmap_lut = (colormaps[cmap_name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    if args.invert:
        # near-zero activations → white so empty space matches background
        cmap_lut[:3] = 255

    # load and crop creatures at native resolution
    crops, names = [], []
    max_dim = 0
    for code in codes:
        path = f"initializations/{code}/s{args.scale}/{code}_s{args.scale}_o{args.ori}.pt"
        data = torch.load(path, weights_only=False)
        t = data["tensor"].numpy()
        names.append(SPECIES_NAMES.get(code, data.get("name", code)))
        active = t > 0.01
        rows, cols = np.where(active)
        pad = 2
        r0, r1 = max(0, rows.min() - pad), min(t.shape[0], rows.max() + pad + 1)
        c0, c1 = max(0, cols.min() - pad), min(t.shape[1], cols.max() + pad + 1)
        crops.append(t[r0:r1, c0:c1])
        max_dim = max(max_dim, r1 - r0, c1 - c0)

    # cell = largest creature bbox, no rescaling
    cell = max_dim
    inner_w = n * cell + (n - 1) * gap
    img_w = (inner_w + 2 * border) * upscale
    img_h = (cell + 2 * border) * upscale
    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    for i, (cropped, code) in enumerate(zip(crops, codes)):
        h, w = cropped.shape
        box = np.zeros((cell, cell), dtype=np.float32)
        box[(cell - h) // 2:(cell - h) // 2 + h,
            (cell - w) // 2:(cell - w) // 2 + w] = cropped
        rgb = cmap_lut[(np.clip(box, 0, 1) * 255).astype(np.uint8)]
        rgb_up = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
        x0 = (border + i * (cell + gap)) * upscale
        y0 = border * upscale
        cw = cell * upscale  # cell width/height in upscaled px
        canvas[y0:y0 + cw, x0:x0 + cw, :] = rgb_up
        # panel border
        pb = args.panel_border * upscale
        if pb > 0:
            canvas[y0:y0 + pb, x0:x0 + cw] = 0            # top
            canvas[y0 + cw - pb:y0 + cw, x0:x0 + cw] = 0  # bottom
            canvas[y0:y0 + cw, x0:x0 + pb] = 0             # left
            canvas[y0:y0 + cw, x0 + cw - pb:x0 + cw] = 0   # right

    # text strip (font sizes scale with upscale)
    text_h_px = 30 * upscale
    fig = plt.figure(figsize=(img_w / 600, text_h_px / 600), dpi=600, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img_w)
    ax.set_ylim(text_h_px, 0)
    ax.axis("off")
    for i, (code, name) in enumerate(zip(codes, names)):
        x_center = (border + i * (cell + gap) + cell / 2) * upscale
        ax.text(x_center, 2.5 * upscale, name, ha="center", va="top",
                fontsize=2.35 * upscale, color="black", fontstyle="italic")
        color = CREATURE_COLORS.get(code, 'black')
        # Render code text, then place dot to its left
        txt = ax.text(x_center + 1.5 * upscale, 15 * upscale, code, ha="center", va="top",
                      fontsize=3.5 * upscale, color="black", fontweight="bold")
        # Get text extent to position dot
        fig.canvas.draw()
        bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        # Transform to data coords
        inv = ax.transData.inverted()
        x_left = inv.transform((bbox.x0, 0))[0]
        y_mid = inv.transform((0, (bbox.y0 + bbox.y1) / 2))[1]
        dot_x = x_left - 8 * upscale
        ax.plot(dot_x, y_mid - 1.3 * upscale, 'o', color=color, markersize=2.2 * upscale,
                clip_on=False, zorder=10)
    fig.canvas.draw()
    text_arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    text_arr = text_arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close()
    text_img = PILImage.fromarray(text_arr).resize(
        (img_w, text_h_px), PILImage.Resampling.LANCZOS)

    # assemble: canvas + text + bottom border
    bottom_border = np.full((border * upscale, img_w, 3), 255, dtype=np.uint8)
    final = np.vstack([canvas, np.array(text_img), bottom_border])

    out = args.output
    PILImage.fromarray(final).save(out, dpi=(600, 600))
    print(f"Saved: {out}  ({final.shape[1]}x{final.shape[0]})")
    print(f"Print size: {final.shape[1]/600:.1f} x {final.shape[0]/600:.1f} inches at 600dpi")


if __name__ == "__main__":
    main()
