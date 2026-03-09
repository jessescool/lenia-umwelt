"""
Prepare settled "situation" tensors for sweep consumption.

A situation is a creature that has been placed at a specific orientation,
settled for 30*T steps, and recentered — ready to be perturbed by sweep.py.

Usage:
    python initializations/situations.py -c O2u -s 2 --grid 128 --num-orientations 3
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from substrate import Animal, Config, Simulation, load_animals
from substrate.lenia import _auto_device
from substrate.scaling import recenter_field
from utils.core import rotate_tensor


DEFAULT_GRID = 128


def prepare_situations(
    creature: Animal,
    scale: int,
    grid_size: int = DEFAULT_GRID,
    num_orientations: int = 3,
    settle_mult: float = 30.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Settle a creature at N evenly-spaced orientations and return each as
    a self-contained "situation" dict ready for sweep consumption.

    For each orientation:
      1. Deep-copy creature, rotate raw cells
      2. Create Simulation -> place_animal -> settle for settle_mult*T steps -> recenter
      3. Store settled board tensor to CPU

    Returns list of situation dicts, one per orientation.
    """
    device = _auto_device()
    T = creature.params.get("T", 10)
    settle_steps = int(settle_mult * T)
    angles = [i * 360.0 / num_orientations for i in range(num_orientations)]

    cfg = Config.from_animal(creature, grid_size, scale=scale)
    H, W = cfg.grid_shape

    situations = []
    for sit_idx, angle in enumerate(angles):
        t0 = time.time()

        creature_rotated = copy.deepcopy(creature)
        cells = torch.as_tensor(creature.cells, device=device, dtype=torch.float32)
        cells_rotated = rotate_tensor(cells, angle, device)
        creature_rotated.cells = cells_rotated.cpu().numpy()

        # Settle
        sim = Simulation(cfg)
        sim.place_animal(creature_rotated, center=True)
        for _ in range(settle_steps):
            sim.lenia.step()

        # Recenter after settling
        sim.board.replace_tensor(recenter_field(sim.board.tensor))
        settled_tensor = sim.board.tensor.detach().cpu()

        sit = {
            'tensor': settled_tensor,        # (H, W)
            'code': creature.code,
            'name': creature.name,
            'scale': scale,
            'base_grid': grid_size,
            'angle': angle,
            'sit_idx': sit_idx,
            'T': float(T),
        }
        situations.append(sit)

        elapsed = time.time() - t0
        if verbose:
            mass = settled_tensor.sum().item()
            print(
                f"  orientation {sit_idx} ({angle:5.1f}°): "
                f"mass={mass:.1f}, {elapsed:.1f}s"
            )

    return situations


def main():
    parser = argparse.ArgumentParser(
        description="Prepare settled situation tensors for sweep"
    )
    parser.add_argument("--code", "-c", type=str, required=True,
                        help="Creature code (e.g., O2u)")
    parser.add_argument("--scale", "-s", type=int, default=1,
                        help="Scale factor (default: 1)")
    parser.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID,
                        help=f"Base grid size (default: {DEFAULT_GRID})")
    parser.add_argument("--num-orientations", "-n", type=int, default=3,
                        help="Number of orientation angles (default: 3)")
    parser.add_argument("--settle-mult", type=float, default=30.0,
                        help="Settle multiplier on T (default: 30.0)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: results/initializations/<code>/s<scale>)")
    parser.add_argument("--catalog", type=str, default="animals.json",
                        help="Creature catalog path")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    catalog_path = Path(__file__).parent.parent / args.catalog
    creatures = load_animals(catalog_path, codes=[args.code])
    if not creatures:
        print(f"Error: creature '{args.code}' not found")
        sys.exit(1)
    creature = creatures[0]

    verbose = not args.quiet
    if verbose:
        T = creature.params.get("T", 10)
        print(f"Preparing situations: {args.code} scale={args.scale} grid={args.grid}")
        print(f"  {args.num_orientations} orientations, settle={int(args.settle_mult * T)} steps ({args.settle_mult}*T)")

    situations = prepare_situations(
        creature,
        scale=args.scale,
        grid_size=args.grid,
        num_orientations=args.num_orientations,
        settle_mult=args.settle_mult,
        verbose=verbose,
    )

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(f"initializations/{args.code}/s{args.scale}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sit in situations:
        stem = f"{args.code}_s{args.scale}_o{sit['sit_idx']}"
        out_path = out_dir / f"{stem}.pt"
        torch.save(sit, out_path)

        # Save a quick PNG preview
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(sit['tensor'].numpy(), cmap='inferno', interpolation='nearest')
        ax.set_title(f"{sit['name']}  ori {sit['sit_idx']} ({sit['angle']:.0f}°)", fontsize=10)
        ax.axis('off')
        png_path = out_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

        if verbose:
            print(f"  Saved {out_path} + {png_path.name}")

    if verbose:
        print(f"\nAll {len(situations)} situations saved to {out_dir}/")


if __name__ == "__main__":
    main()
