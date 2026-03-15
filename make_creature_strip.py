"""Render a labeled strip of Lenia creatures at native scale 4 resolution.

Usage:
    python make_creature_strip.py
    python make_creature_strip.py --codes O2u K4s S1s --ori 0
    python make_creature_strip.py --output results/my_strip.png --upscale 6
"""

import argparse
import torch
import numpy as np
from matplotlib import colormaps
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Render labeled creature strip")
    parser.add_argument("--codes", nargs="+",
                        default=["O2u", "O2v", "K4s", "K6s", "S1s", "P4al"])
    parser.add_argument("--ori", type=int, default=3, help="Orientation index (o-file)")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--gap", type=int, default=2, help="Gap between cells (px)")
    parser.add_argument("--border", type=int, default=3, help="Border around image (px)")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor for 300dpi")
    parser.add_argument("--colormap", default="magma")
    parser.add_argument("--output", default="results/env_competency/creatures_6x1_ori3_labeled.png")
    args = parser.parse_args()

    codes = args.codes
    gap = args.gap
    border = args.border
    n = len(codes)
    upscale = args.upscale

    cmap_lut = (colormaps[args.colormap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # load and crop creatures at native resolution
    crops, names = [], []
    max_dim = 0
    for code in codes:
        path = f"initializations/{code}/s{args.scale}/{code}_s{args.scale}_o{args.ori}.pt"
        data = torch.load(path, weights_only=False)
        t = data["tensor"].numpy()
        names.append(data["name"])
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
        canvas[border * upscale:border * upscale + cell * upscale,
               x0:x0 + cell * upscale, :] = rgb_up

    # text strip (font sizes scale with upscale)
    text_h_px = 24 * upscale
    fig = plt.figure(figsize=(img_w / 300, text_h_px / 300), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img_w)
    ax.set_ylim(text_h_px, 0)
    ax.axis("off")
    for i, (code, name) in enumerate(zip(codes, names)):
        x_center = (border + i * (cell + gap) + cell / 2) * upscale
        ax.text(x_center, 2.5 * upscale, name, ha="center", va="top",
                fontsize=2.1 * upscale, color="black", fontstyle="italic")
        ax.text(x_center, 13 * upscale, code, ha="center", va="top",
                fontsize=2.5 * upscale, color="black", fontweight="bold")
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
    PILImage.fromarray(final).save(out, dpi=(300, 300))
    print(f"Saved: {out}  ({final.shape[1]}x{final.shape[0]})")
    print(f"Print size: {final.shape[1]/300:.1f} x {final.shape[0]/300:.1f} inches at 300dpi")


if __name__ == "__main__":
    main()
