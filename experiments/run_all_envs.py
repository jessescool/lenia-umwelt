"""Run creatures through all barrier environments, producing GIFs."""

import argparse
from pathlib import Path

import torch

from substrate import load_animals
from environments import ENVIRONMENTS
from experiments.run_env_batch import (
    extract_pattern,
    run_single_env_from_init,
    load_manifest,
    parse_grid,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run all creatures × all environments from pre-settled inits"
    )
    parser.add_argument("--code", default=None,
                        help="Single animal code (default: all from manifest)")
    parser.add_argument("--scale", type=int, default=2,
                        help="Scale factor (default: 2)")
    parser.add_argument("--steps", type=int, default=600,
                        help="Simulation steps per run (default: 600)")
    parser.add_argument("--grid", default="128x256",
                        help="Base grid as HxW (default: 128x256)")
    parser.add_argument("--output", default="results/env_sweeps",
                        help="Output root directory (default: results/env_sweeps)")
    parser.add_argument("--scaled", action="store_true", default=True,
                        help="Scale env features to grid size (default: True)")
    parser.add_argument("--no-scaled", dest="scaled", action="store_false",
                        help="Disable environment feature scaling")
    parser.add_argument("--manifest", default="animals_to_run.json",
                        help="Path to animals_to_run.json (default: animals_to_run.json)")
    parser.add_argument("--catalog", default="animals.json",
                        help="Animal catalog path (default: animals.json)")
    args = parser.parse_args()

    base_grid = parse_grid(args.grid)
    env_names = sorted(ENVIRONMENTS.keys())

    if args.code:
        codes = [args.code]
    else:
        manifest = load_manifest(args.manifest)
        codes = [entry["code"] for entry in manifest]

    # Load animal objects for physics params
    animals_by_code = {a.code: a for a in load_animals(Path(args.catalog), codes=codes)}

    print(f"Grid: {args.grid} (base) × scale {args.scale} "
          f"= {base_grid[0]*args.scale}×{base_grid[1]*args.scale}")
    print(f"Environments ({len(env_names)}): {', '.join(env_names)}")
    print(f"Animals: {', '.join(codes)}")
    print(f"Steps: {args.steps}, scaled_env: {args.scaled}")
    print()

    total_gifs = 0

    for code in codes:
        # Load pre-settled initializations
        init_dir = Path(f"initializations/{code}/s{args.scale}")
        pt_files = sorted(init_dir.glob("*.pt"))
        if not pt_files:
            print(f"SKIP {code}: no .pt files in {init_dir}")
            continue

        initializations = [torch.load(f, weights_only=False) for f in pt_files]

        animal = animals_by_code.get(code)
        if animal is None:
            print(f"SKIP {code}: not found in catalog")
            continue

        output_dir = Path(args.output) / code
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {code} ({len(initializations)} orientations) ===")

        for env_name in env_names:
            print(f"  [{env_name}]")
            run_single_env_from_init(
                env_name=env_name,
                code=code,
                base_grid=base_grid,
                scale=args.scale,
                steps=args.steps,
                output_dir=output_dir,
                animal=animal,
                initializations=initializations,
                scaled=args.scaled,
            )

        n = len(env_names) * len(initializations)
        total_gifs += n
        print(f"  → {n} GIFs in {output_dir}")
        print()

    print(f"Done! {total_gifs} GIFs total → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
