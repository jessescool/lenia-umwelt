"""Visualize a Lenia creature alongside its kernel, showing relative size and weights."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from substrate.animals import load_animals
from substrate.lenia import Config, build_kernel


def render_kernel_vs_creature(
    code: str,
    catalog: str | Path | None = None,
    scale: int = 1,
    output: str | Path | None = None,
):
    catalog = catalog or ROOT / "animals.json"
    animals = load_animals(catalog, codes=[code])
    if not animals:
        raise ValueError(f"Animal '{code}' not found in {catalog}")
    animal = animals[0]
    params = animal.params

    # Decode creature cells
    cells = animal.cells.numpy()
    creature_h, creature_w = cells.shape

    # Build kernel at base scale
    R = int(float(params.get("R", 13)))
    kn = int(float(params.get("kn", 1)))
    kernel_size = 2 * R + 1
    kernel = build_kernel(
        (kernel_size, kernel_size), radius=R, kn=kn, params=params,
        device=torch.device("cpu"), dtype=torch.float32,
    ).numpy()

    # Normalize kernel for display (show raw weights, colorbar gives scale)
    kernel_norm = kernel / kernel.max() if kernel.max() > 0 else kernel

    fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")

    # Use GridSpec: creature (left), kernel 2D (center), kernel radial profile (right)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35,
                          left=0.06, right=0.94, top=0.88, bottom=0.12)

    ax_creature = fig.add_subplot(gs[0])
    im_c = ax_creature.imshow(cells, cmap="inferno", vmin=0, vmax=1,
                               interpolation="nearest", origin="upper")
    # Overlay a circle showing kernel radius centered on the creature
    cy, cx = creature_h / 2, creature_w / 2
    circle = patches.Circle((cx - 0.5, cy - 0.5), R, linewidth=2,
                             edgecolor="#00ffcc", facecolor="none",
                             linestyle="--", label=f"Kernel R={R}")
    ax_creature.add_patch(circle)
    ax_creature.set_title(f"{code}  ({creature_h}×{creature_w} cells)",
                          color="white", fontsize=13, fontweight="bold")
    ax_creature.set_xlabel("cells", color="white", fontsize=10)
    ax_creature.set_ylabel("cells", color="white", fontsize=10)
    ax_creature.tick_params(colors="white", labelsize=8)
    ax_creature.legend(loc="upper right", fontsize=9, facecolor="#2a2a4a",
                       edgecolor="#00ffcc", labelcolor="white")

    ax_kernel = fig.add_subplot(gs[1])
    im_k = ax_kernel.imshow(kernel_norm, cmap="magma", vmin=0, vmax=1,
                             interpolation="nearest", origin="upper",
                             extent=[-R, R, R, -R])
    ax_kernel.set_title(f"Kernel  (kn={kn}, {kernel_size}×{kernel_size})",
                        color="white", fontsize=13, fontweight="bold")
    ax_kernel.set_xlabel("offset (cells)", color="white", fontsize=10)
    ax_kernel.set_ylabel("offset (cells)", color="white", fontsize=10)
    ax_kernel.tick_params(colors="white", labelsize=8)
    cb = fig.colorbar(im_k, ax=ax_kernel, fraction=0.046, pad=0.04)
    cb.set_label("weight (normalized)", color="white", fontsize=9)
    cb.ax.tick_params(colors="white", labelsize=8)

    ax_profile = fig.add_subplot(gs[2])
    # Compute radial distance from kernel center
    center = R
    yy, xx = np.mgrid[:kernel_size, :kernel_size]
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    # Bin by integer distance
    max_d = int(np.ceil(dist.max()))
    radii = np.arange(0, max_d + 1)
    profile_mean = np.zeros(len(radii))
    for i, r in enumerate(radii):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.any():
            profile_mean[i] = kernel[mask].mean()
    ax_profile.fill_between(radii, 0, profile_mean, color="#ff6b6b", alpha=0.3)
    ax_profile.plot(radii, profile_mean, color="#ff6b6b", linewidth=2)
    ax_profile.axvline(R, color="#00ffcc", linestyle="--", linewidth=1.5,
                       label=f"R = {R}")
    ax_profile.set_title("Radial weight profile", color="white", fontsize=13,
                         fontweight="bold")
    ax_profile.set_xlabel("distance from center (cells)", color="white", fontsize=10)
    ax_profile.set_ylabel("kernel weight", color="white", fontsize=10)
    ax_profile.tick_params(colors="white", labelsize=8)
    ax_profile.set_facecolor("#1a1a2e")
    ax_profile.spines["bottom"].set_color("white")
    ax_profile.spines["left"].set_color("white")
    ax_profile.spines["top"].set_visible(False)
    ax_profile.spines["right"].set_visible(False)
    ax_profile.legend(loc="upper right", fontsize=9, facecolor="#2a2a4a",
                      edgecolor="#00ffcc", labelcolor="white")

    # Annotation: size comparison
    fig.text(0.5, 0.96,
             f"{code}:  creature {creature_h}×{creature_w}  |  kernel diameter {kernel_size}  "
             f"|  R = {R}  |  dt = 1/{int(float(params.get('T', 10)))}",
             ha="center", va="top", color="#aaaacc", fontsize=11,
             fontfamily="monospace")

    out = output or ROOT / "results" / "new" / f"{code}_kernel_vs_creature.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "O2u"
    render_kernel_vs_creature(code)
