"""20 pegs variants with center clearance, different seeds.

Usage:
    PYTHONPATH=. python scripts/pegs_gallery.py
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt

from environments import make_pegs

H, W = 128, 256
D, T = torch.device("cpu"), torch.float32

SEEDS = list(range(1, 21))


def main():
    masks = {f"seed_{s}": make_pegs((H, W), D, T, seed=s) for s in SEEDS}

    ncols = 5
    nrows = -(-len(masks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    for i, (name, mask) in enumerate(masks.items()):
        axes[i].imshow(1.0 - mask.numpy(), cmap='gray', vmin=0, vmax=1, aspect='equal')
        axes[i].set_title(name, fontsize=9, fontweight='bold')
        axes[i].axis('off')

    fig.tight_layout()
    path = Path("results/new/pegs_gallery.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # individual PNGs
    out_dir = Path("results/new/pegs_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        ax2.imshow(1.0 - mask.numpy(), cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax2.set_title(name, fontsize=9); ax2.axis('off')
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)

    print(f"Gallery: {path.resolve()}")
    print(f"Individual: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
