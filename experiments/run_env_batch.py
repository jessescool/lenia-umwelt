# experiments/run_env_batch.py — Jesse Cool (jessescool)
"""Batch-run a single creature through all environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from substrate import Simulation, load_animals, Config
from substrate.scaling import interpolate_pattern
from environments import ENVIRONMENTS, load_env, make_env
from viz.gif import write_gif


def _spawn_left_of_wall(grid_h, grid_w, ph, pw):
    """Left of vertical wall at W/3."""
    return (grid_h // 2 - ph // 2, grid_w // 6 - pw // 2)


def _spawn_center(grid_h, grid_w, ph, pw):
    return (grid_h // 2 - ph // 2, grid_w // 2 - pw // 2)


def _spawn_far_left(grid_h, grid_w, ph, pw):
    return (grid_h // 2 - ph // 2, pw)


def _spawn_left_in_mouth(grid_h, grid_w, ph, pw):
    """Left side, in wide mouth of funnel."""
    return (grid_h // 2 - ph // 2, pw + 10)


# Map environment name -> spawn function
SPAWN_POSITIONS = {
    "pegs": _spawn_center,
    "funnel": _spawn_left_in_mouth,
    "membrane_1px": _spawn_left_of_wall,
    "membrane_3px": _spawn_left_of_wall,
    "membrane_5px": _spawn_left_of_wall,
    "maze": _spawn_far_left,
    "corridor": _spawn_far_left,
    "box": _spawn_center,
    "circular_enclosure": _spawn_center,
}


def get_spawn_position(env_name, grid_h, grid_w, ph, pw):
    fn = SPAWN_POSITIONS.get(env_name, _spawn_center)
    return fn(grid_h, grid_w, ph, pw)


def extract_pattern(tensor, threshold=0.01, padding=2):
    """Crop settled board to bounding box of non-zero region."""
    active = tensor > threshold
    rows, cols = torch.where(active)
    if len(rows) == 0:
        return tensor  # fallback: whole board
    r0 = rows.min().item() - padding
    r1 = rows.max().item() + padding + 1
    c0 = cols.min().item() - padding
    c1 = cols.max().item() + padding + 1
    r0, c0 = max(0, r0), max(0, c0)
    r1 = min(tensor.shape[0], r1)
    c1 = min(tensor.shape[1], c1)
    return tensor[r0:r1, c0:c1]


def run_single_env_from_init(env_name, code, base_grid, scale, steps,
                             output_dir, animal, situations, speed=1.0,
                             scaled=False):
    """Run one environment using pre-settled situation dicts (from .pt files).

    situations: list of dicts loaded from .pt files, each containing
                'tensor', 'sit_idx', etc.
    """
    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale
    shape = (grid_h, grid_w)

    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)
    if speed != 1.0:
        cfg.dt *= speed

    for sit in situations:
        sit_idx = sit['sit_idx']

        pattern = extract_pattern(sit['tensor'])
        ph, pw = pattern.shape

        sim = Simulation(cfg)

        mask = load_env(env_name, cfg.device, cfg.dtype)
        sim.set_barrier(mask)

        pos = get_spawn_position(env_name, grid_h, grid_w, ph, pw)
        pos = (max(0, min(pos[0], grid_h - ph)),
               max(0, min(pos[1], grid_w - pw)))

        sim.add_animal(pattern.numpy(), position=pos, wrap=False)

        frames = sim.run(steps)

        gif_path = output_dir / f"{code}_{env_name}__o{sit_idx}.gif"
        write_gif(frames, gif_path, fps=30, upscale=2, barrier_mask=mask)
        print(f"  {gif_path.name}")


def run_single_env(env_name, code, base_grid, scale, steps, output_dir, animal,
                   rotations=None, speed=1.0, scaled=False):
    """Run one environment with specified rotations, saving one GIF each.

    rotations: list of k values (0-3) where rotation = k*90 degrees.
               Defaults to [0, 1, 2, 3] (all four rotations).
    speed: dt multiplier (2.0 = creature evolves 2x faster per step).
    scaled: if True, environment features scale proportionally to grid size.
    """
    if rotations is None:
        rotations = [0, 1, 2, 3]

    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale
    shape = (grid_h, grid_w)

    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)
    if speed != 1.0:
        cfg.dt *= speed

    scaled_pattern = interpolate_pattern(animal.cells, scale)
    ph, pw = scaled_pattern.shape
    scaled_t = torch.from_numpy(scaled_pattern).float()

    for k in rotations:
        rot_deg = k * 90
        # Rotate the creature pattern
        if k == 0:
            pattern = scaled_t
        else:
            pattern = torch.rot90(scaled_t, k)

        rph, rpw = pattern.shape

        sim = Simulation(cfg)

        mask = load_env(env_name, cfg.device, cfg.dtype)
        sim.set_barrier(mask)

        # Compute spawn position (using rotated pattern size)
        pos = get_spawn_position(env_name, grid_h, grid_w, rph, rpw)

        # Ensure creature doesn't overlap barrier (clamp to valid region)
        pos = (max(0, min(pos[0], grid_h - rph)),
               max(0, min(pos[1], grid_w - rpw)))

        sim.add_animal(pattern.numpy(), position=pos, wrap=False)

        frames = sim.run(steps)

        gif_path = output_dir / f"{code}_{env_name}__rot{rot_deg}.gif"
        write_gif(frames, gif_path, fps=30, upscale=2, barrier_mask=mask)
        print(f"  {gif_path.name}")


def load_manifest(path: str | Path) -> list[dict]:
    """Load animals_to_run.json and return the animals list."""
    with open(path) as f:
        data = json.load(f)
    return data["animals"]


def parse_grid(grid_str: str) -> tuple[int, int]:
    """Parse 'HxW' string into (H, W) tuple."""
    parts = grid_str.lower().split("x")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="Run creature through barrier environments")
    parser.add_argument("--envs", nargs="+", required=True,
                        help="Environment names to run")

    # Init-dir mode (pre-settled orientations from .pt files)
    parser.add_argument("--init-dir", default=None,
                        help="Directory of .pt situation files (e.g. initializations/K4s/s2)")

    # Single-animal mode
    parser.add_argument("--code", default=None, help="Animal code")
    parser.add_argument("--grid", default=None,
                        help="Base grid as HxW, e.g. 128x256")

    # Manifest mode
    parser.add_argument("--manifest", default=None,
                        help="Path to animals_to_run.json (manifest mode)")

    # Shared
    parser.add_argument("--scale", type=int, default=None,
                        help="Grid scale factor (default: inferred from .pt or 2)")
    parser.add_argument("--steps", type=int, default=400, help="Simulation steps per run")
    parser.add_argument("--output", default="results/environment_gifs/",
                        help="Output directory for GIFs")
    parser.add_argument("--rotation", type=int, default=None,
                        choices=[0, 90, 180, 270],
                        help="Single rotation angle in degrees (legacy mode only)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="dt multiplier — creature evolves faster (default: 1.0)")
    parser.add_argument("--scaled", action="store_true",
                        help="Scale environment features proportionally to grid size")
    parser.add_argument("--catalog", default="animals.json", help="Animal catalog path")
    args = parser.parse_args()

    # Validate environment names
    unknown = [e for e in args.envs if e not in ENVIRONMENTS]
    if unknown:
        print(f"ERROR: Unknown environments: {unknown}")
        print(f"Available: {sorted(ENVIRONMENTS.keys())}")
        return

    # Init-dir mode: load pre-settled .pt files
    if args.init_dir:
        init_dir = Path(args.init_dir)
        pt_files = sorted(init_dir.glob("*.pt"))
        if not pt_files:
            print(f"ERROR: No .pt files found in {init_dir}")
            return

        # Load all situations
        situations = [torch.load(f, weights_only=False) for f in pt_files]
        print(f"Loaded {len(situations)} orientations from {init_dir}")

        # Infer code from .pt metadata if not given
        code = args.code or situations[0]['code']

        # Infer scale from .pt metadata if not given
        scale = args.scale or situations[0]['scale']

        # Base grid for the environment (not the creature's settling grid)
        base_grid = parse_grid(args.grid) if args.grid else (128, 256)

        # Load animal from catalog for physics params (kernels, T, etc.)
        animals = load_animals(Path(args.catalog), codes=[code])
        if not animals:
            print(f"ERROR: Animal '{code}' not found in catalog")
            return
        animal = animals[0]

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        actual = (base_grid[0] * scale, base_grid[1] * scale)
        print(f"Animal: {code}, base_grid={base_grid}, actual={actual}, "
              f"scale={scale}, orientations={len(situations)}, scaled_env={args.scaled}")

        for env_name in args.envs:
            print(f"  [{env_name}]")
            run_single_env_from_init(
                env_name=env_name,
                code=code,
                base_grid=base_grid,
                scale=scale,
                steps=args.steps,
                output_dir=output_dir,
                animal=animal,
                situations=situations,
                speed=args.speed,
                scaled=args.scaled,
            )

        total_gifs = len(args.envs) * len(situations)
        print(f"\nDone! {total_gifs} GIFs saved to: {output_dir.resolve()}")
        return

    # Legacy modes: single-animal (with rotations) or manifest
    scale = args.scale or 2

    # Convert --rotation degree to k value list
    if args.rotation is not None:
        rotations = [args.rotation // 90]
    else:
        rotations = None  # default: all four

    # Build work list: [(code, base_grid, animal), ...]
    work = []

    if args.manifest and not args.code:
        # Manifest mode: iterate all animals
        manifest = load_manifest(args.manifest)
        all_codes = [entry["code"] for entry in manifest]
        animals_by_code = {a.code: a for a in load_animals(Path(args.catalog), codes=all_codes)}

        for entry in manifest:
            code = entry["code"]
            env_grid = entry.get("env_grid")
            if env_grid is None:
                print(f"WARNING: {code} has no env_grid, skipping")
                continue
            base_grid = tuple(env_grid)
            work.append((code, base_grid, animals_by_code[code]))
    else:
        # Single-animal mode
        code = args.code or "O2u"
        base_grid = parse_grid(args.grid) if args.grid else (128, 256)
        animals = load_animals(Path(args.catalog), codes=[code])
        work.append((code, base_grid, animals[0]))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rots = len(rotations) if rotations else 4
    total_gifs = 0

    for code, base_grid, animal in work:
        actual = (base_grid[0] * scale, base_grid[1] * scale)
        print(f"Animal: {code}, base_grid={base_grid}, actual={actual}, "
              f"scale={scale}, scaled_env={args.scaled}")

        # Per-animal output subdirectory when running manifest
        if len(work) > 1:
            animal_dir = output_dir / code
            animal_dir.mkdir(parents=True, exist_ok=True)
        else:
            animal_dir = output_dir

        for env_name in args.envs:
            print(f"  [{env_name}]")
            run_single_env(
                env_name=env_name,
                code=code,
                base_grid=base_grid,
                scale=scale,
                steps=args.steps,
                output_dir=animal_dir,
                animal=animal,
                rotations=rotations,
                speed=args.speed,
                scaled=args.scaled,
            )

        total_gifs += len(args.envs) * n_rots
        print()

    print(f"Done! {total_gifs} GIFs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
