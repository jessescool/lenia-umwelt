"""
Measure each creature's natural heading direction at 0° rotation.

Runs an unperturbed simulation, tracks centroids, and computes the
circular-mean heading angle. Output is used by generate_initializations.py
to align o0 with the +x axis (rightward travel).

Usage:
    python initializations/calibrate_headings.py --scale 2 --grid 128
    python initializations/calibrate_headings.py --scale 2 --grid 128 --code O2u
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from substrate import Animal, Config, Simulation, load_animals
from substrate.lenia import _auto_device
from substrate.scaling import recenter_field
from metrics_and_machinery.trajectory_metrics import centroid
from viz._helpers import heading_from_centroids, rolling_circular_mean


DEFAULT_GRID = 128
DEFAULT_OBSERVE_STEPS = 100


def measure_heading(
    creature: Animal,
    scale: int,
    grid_size: int = DEFAULT_GRID,
    settle_mult: float = 30.0,
    observe_steps: int = DEFAULT_OBSERVE_STEPS,
    smooth_window: int = 10,
    verbose: bool = True,
) -> float:
    """
    Measure a creature's natural heading at 0° rotation.

    1. Build config, place creature centered, settle for settle_mult*T steps
    2. Recenter, then run observe_steps more steps collecting centroids
    3. Compute heading angles via heading_from_centroids, circular mean

    Returns:
        Heading angle in radians (arctan2(dr, dc), 0 = rightward).
    """
    device = _auto_device()
    T = creature.params.get("T", 10)
    settle_steps = int(settle_mult * T)

    cfg = Config.from_animal(creature, grid_size, scale=scale)
    H, W = cfg.grid_shape

    # Settle
    sim = Simulation(cfg)
    sim.place_animal(creature, center=True)
    for _ in range(settle_steps):
        sim.lenia.step()
    sim.board.replace_tensor(recenter_field(sim.board.tensor))

    # Collect centroids over observe_steps
    centroids_list = []
    for _ in range(observe_steps):
        sim.lenia.step()
        r, c = centroid(sim.board.tensor)
        centroids_list.append([r, c])

    # Shape into (1, T, 2) for heading_from_centroids
    centroids_arr = np.array(centroids_list)[np.newaxis, :, :]  # (1, N, 2)
    headings = heading_from_centroids(centroids_arr, (H, W))  # (1, N-1)

    # Smooth then take circular mean over all timesteps
    if headings.shape[1] >= smooth_window:
        headings = rolling_circular_mean(headings, smooth_window)

    # Circular mean of all heading samples
    sin_mean = np.sin(headings).mean()
    cos_mean = np.cos(headings).mean()
    mean_heading = float(np.arctan2(sin_mean, cos_mean))

    if verbose:
        deg = np.degrees(mean_heading)
        print(f"  {creature.code}: heading = {mean_heading:.4f} rad ({deg:.1f}°)")

    return mean_heading


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate natural heading direction for each creature"
    )
    parser.add_argument("--code", "-c", type=str, default=None,
                        help="Single creature code (default: all in animals_to_run.json)")
    parser.add_argument("--scale", "-s", type=int, default=2,
                        help="Scale factor (default: 2)")
    parser.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID,
                        help=f"Base grid size (default: {DEFAULT_GRID})")
    parser.add_argument("--settle-mult", type=float, default=30.0,
                        help="Settle multiplier on T (default: 30.0)")
    parser.add_argument("--observe-steps", type=int, default=DEFAULT_OBSERVE_STEPS,
                        help=f"Steps to observe after settling (default: {DEFAULT_OBSERVE_STEPS})")
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="Rolling circular mean window (default: 10)")
    parser.add_argument("--catalog", type=str, default="animals.json",
                        help="Creature catalog path")
    parser.add_argument("--animals-config", type=str, default="animals_to_run.json",
                        help="Animals config for creature list")
    parser.add_argument("--output", "-o", type=str, default="initializations/heading_offsets.json",
                        help="Output JSON path")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    root = Path(__file__).parent.parent
    catalog_path = root / args.catalog
    verbose = not args.quiet

    # Determine which creatures to calibrate
    if args.code:
        codes = [args.code]
    else:
        config_path = root / args.animals_config
        with open(config_path) as f:
            codes = [a["code"] for a in json.load(f)["animals"]]

    if verbose:
        print(f"Calibrating headings: scale={args.scale} grid={args.grid} observe={args.observe_steps}")
        print(f"Creatures: {codes}")

    # Load existing offsets to merge (don't clobber other scales' data)
    out_path = root / args.output
    if out_path.exists():
        with open(out_path) as f:
            all_offsets = json.load(f)
    else:
        all_offsets = {}

    for code in codes:
        creatures = load_animals(catalog_path, codes=[code])
        if not creatures:
            print(f"Warning: creature '{code}' not found in catalog, skipping")
            continue

        t0 = time.time()
        heading_rad = measure_heading(
            creatures[0],
            scale=args.scale,
            grid_size=args.grid,
            settle_mult=args.settle_mult,
            observe_steps=args.observe_steps,
            smooth_window=args.smooth_window,
            verbose=verbose,
        )
        elapsed = time.time() - t0

        # Scale-aware nested format: {code: {scale_str: {heading_rad, heading_deg, grid}}}
        if code not in all_offsets:
            all_offsets[code] = {}
        # Migrate old flat format if present
        if "heading_deg" in all_offsets.get(code, {}):
            old = all_offsets[code]
            old_scale = str(old.get("scale", 2))
            all_offsets[code] = {
                old_scale: {
                    "heading_rad": old["heading_rad"],
                    "heading_deg": old["heading_deg"],
                    "grid": old.get("grid", 128),
                }
            }
        all_offsets[code][str(args.scale)] = {
            "heading_rad": round(heading_rad, 6),
            "heading_deg": round(np.degrees(heading_rad), 2),
            "grid": args.grid,
        }

        if verbose:
            print(f"    ({elapsed:.1f}s)")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_offsets, f, indent=2)

    if verbose:
        print(f"\nHeading offsets saved to {out_path}")


if __name__ == "__main__":
    main()
