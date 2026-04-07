"""Run a creature through barrier-geometry environments with amplified salience.

Takes any environment geometry from environments.py, converts the binary mask
to a salience field (W=amplitude at masked locations, 1.0 elsewhere), and runs
the creature through it.

Usage (via dispatch):
    ./dispatch --preempt "python excitation_tests/run_excited_envs.py --code K4s --envs pegs chips shuriken"
    ./dispatch --preempt "python excitation_tests/run_excited_envs.py --code K4s --envs all --amplitude 2.0"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from substrate import Config, Simulation, load_animals
from environments.environments import ENVIRONMENTS, compute_scale_factor, make_env
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="O2u", help="Animal code")
    parser.add_argument("--orientation", type=int, default=3)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--envs", nargs="+", default=["all"],
                        help="Environment names, or 'all' for every geometry")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    # Resolve env list
    if args.envs == ["all"]:
        env_names = sorted(ENVIRONMENTS.keys())
    else:
        env_names = args.envs
        unknown = [e for e in env_names if e not in ENVIRONMENTS]
        if unknown:
            print(f"ERROR: Unknown environments: {unknown}")
            print(f"Available: {sorted(ENVIRONMENTS.keys())}")
            return

    code = args.code
    scale = 4
    base_grid = (128, 256)
    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale
    shape = (grid_h, grid_w)
    orientation = args.orientation
    amp = args.amplitude

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Load animal
    animals = load_animals(Path("animals.json"), codes=[code])
    animal = animals[0]
    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)

    # Load pre-settled orientation
    init_path = Path(f"initializations/{code}/s{scale}/{code}_s{scale}_o{orientation}.pt")
    sit = torch.load(init_path, weights_only=False)
    pattern = extract_pattern(sit["tensor"])
    ph, pw = pattern.shape
    pos = (grid_h // 2 - ph // 2, grid_w // 2 - pw // 2)

    out_dir = Path(f"excitation_tests/{code}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{code} o{orientation} | grid {grid_h}x{grid_w} | W={amp} | {args.steps} steps")
    print(f"Envs: {env_names}\n")

    for env_name in env_names:
        mask = make_env(env_name, shape, device, dtype, scaled=True)

        # Convert barrier mask to excited salience: 1.0 background, amp at masked locations
        salience = 1.0 + mask * (amp - 1.0)

        sim = Simulation(cfg)
        sim.set_salience(salience)
        sim.add_animal(pattern.numpy(), position=pos, wrap=False)

        print(f"  {env_name} ... ", end="", flush=True)
        rollout = sim.run(args.steps)

        gif_path = out_dir / f"{code}_excited_{env_name}_w{amp}_o{orientation}.gif"
        write_gif(rollout, gif_path, fps=30, upscale=1, barrier_mask=mask)
        print(f"{gif_path}")

    print(f"\nDone! Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
