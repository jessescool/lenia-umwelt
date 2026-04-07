"""Generate preview PNGs from precomputed environment tensors."""

import argparse
import math
from pathlib import Path

import torch

from environments.environments import ENVIRONMENTS


def main():
    parser = argparse.ArgumentParser(description="Generate environment preview PNGs from .pt files")
    parser.add_argument("--input-dir", "-i", default="environments/", help="Directory with .pt files")
    parser.add_argument("--output-dir", "-o", default="environments/previews/", help="PNG output directory")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = {}
    for name in ENVIRONMENTS:
        pt = in_dir / f"{name}.pt"
        if not pt.exists():
            print(f"  WARNING: {pt} not found, skipping")
            continue
        masks[name] = torch.load(pt, weights_only=False)

    if not masks:
        print(f"No .pt files found in {in_dir}")
        return

    # individual PNGs
    for name, mask in masks.items():
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.imshow(1.0 - mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}.png")

    # overview grid
    n = len(masks)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 1.7 * nrows))
    axes = axes.flatten()

    for i, (name, mask) in enumerate(masks.items()):
        axes[i].imshow(1.0 - mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(name, fontsize=8, fontweight="bold", pad=3)
        axes[i].set_aspect("equal")
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(wspace=0.02, hspace=0.2)
    overview_path = in_dir / "overview.png"
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  overview.png")

    print(f"\nDone! {n} previews -> {out_dir.resolve()}")
    print(f"Overview -> {overview_path.resolve()}")


if __name__ == "__main__":
    main()
