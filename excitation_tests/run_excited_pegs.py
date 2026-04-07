"""Run a creature through a pegs environment with amplified salience at peg locations.

Inverse of blind-pegs: the creature "sees harder" at peg locations instead of
being blinded there. Background salience is 1.0 (normal), pegs get the specified
amplitude. Loops over multiple amplitudes to compare effects.

Usage (via dispatch):
    ./dispatch --preempt "python excitation_tests/run_excited_pegs.py"
    ./dispatch --preempt "python excitation_tests/run_excited_pegs.py --code S1s --orientation 3"
    ./dispatch --preempt "python excitation_tests/run_excited_pegs.py --amplitudes 1.5 2.0"
"""

from __future__ import annotations

from pathlib import Path

import torch

from substrate import Config, Simulation, load_animals
from environments.environments import make_pegs, compute_scale_factor
from viz.gif import write_gif


def extract_pattern(tensor: torch.Tensor, threshold: float = 0.01, padding: int = 2) -> torch.Tensor:
    """Crop settled board to bounding box of non-zero region."""
    active = tensor > threshold
    rows, cols = torch.where(active)
    if len(rows) == 0:
        return tensor
    r0 = max(0, rows.min().item() - padding)
    r1 = min(tensor.shape[0], rows.max().item() + padding + 1)
    c0 = max(0, cols.min().item() - padding)
    c1 = min(tensor.shape[1], cols.max().item() + padding + 1)
    return tensor[r0:r1, c0:c1]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="O2u", help="Animal code")
    parser.add_argument("--orientation", type=int, default=3)
    parser.add_argument("--amplitudes", type=float, nargs="+",
                        default=[1.1, 1.25, 1.5, 1.75, 2.0])
    args = parser.parse_args()

    code = args.code
    scale = 4
    base_grid = (128, 256)
    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale  # 512 x 1024
    shape = (grid_h, grid_w)
    steps = 600
    orientation = args.orientation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Load animal from catalog
    animals = load_animals(Path("animals.json"), codes=[code])
    animal = animals[0]

    # Build config
    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)

    # Build pegs mask once (reused across amplitudes)
    sf = compute_scale_factor(shape)
    peg_mask = make_pegs(shape, device, dtype, seed=19, scale_factor=sf)

    # Load pre-settled orientation once
    init_path = Path(f"initializations/{code}/s{scale}/{code}_s{scale}_o{orientation}.pt")
    sit = torch.load(init_path, weights_only=False)
    pattern = extract_pattern(sit["tensor"])
    ph, pw = pattern.shape
    pos = (grid_h // 2 - ph // 2, grid_w // 2 - pw // 2)

    out_dir = Path("excitation_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{code} o{orientation} | grid {grid_h}x{grid_w} | {steps} steps")
    print(f"Amplitudes: {args.amplitudes}\n")

    for amp in args.amplitudes:
        salience = 1.0 + peg_mask * (amp - 1.0)

        sim = Simulation(cfg)
        sim.set_salience(salience)
        sim.add_animal(pattern.numpy(), position=pos, wrap=False)

        print(f"  W={amp:.2f} ... ", end="", flush=True)
        rollout = sim.run(steps)

        gif_path = out_dir / f"{code}_excited_pegs_w{amp}_o{orientation}.gif"
        write_gif(rollout, gif_path, fps=30, upscale=1, barrier_mask=peg_mask)
        print(f"{gif_path}")

    print(f"\nDone! Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
