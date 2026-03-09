"""Run full orbit pipeline (raw → profile → distances → orbit) for a creature at multiple scales."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orbits import (
    build_distances,
    build_orbit,
    build_orbit_summary,
    build_profiles,
)
from substrate import load_animals

import json
import torch
from torch import Tensor


def run_pipeline(code: str, scales: list[int], grid: int = 64,
                 skip_raw: bool = False, catalog: str = "animals.json"):
    """Run full orbit pipeline for one creature across multiple scales."""
    catalog_path = Path(__file__).parent.parent / catalog
    creatures = load_animals(catalog_path, codes=[code])
    if not creatures:
        print(f"Error: creature '{code}' not found")
        sys.exit(1)
    creature = creatures[0]

    for scale in scales:
        out_dir = Path(f"orbits/{code}/s{scale}")
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / f"{code}_s{scale}_raw.pt"
        profile_path = out_dir / f"{code}_s{scale}_profile.pt"
        distances_path = out_dir / f"{code}_s{scale}_distances.pt"
        orbit_path = out_dir / f"{code}_s{scale}_orbit.pt"
        json_path = out_dir / f"{code}_s{scale}_orbit.json"

        # Stage 1: raw frames
        if skip_raw:
            if not raw_path.exists():
                print(f"ERROR: --skip-raw but {raw_path} missing, skipping scale {scale}")
                continue
            print(f"\n=== {code} s{scale}: skipping raw (already exists) ===")
        else:
            print(f"\n=== {code} s{scale}: building raw frames ===")
            raw_data = build_orbit(creature, scale=scale, grid_size=grid)
            torch.save(raw_data, raw_path)
            print(f"  Saved {raw_path}")

        # Stage 2: profiles
        print(f"=== {code} s{scale}: building profiles ===")
        raw_data = torch.load(raw_path, weights_only=False)
        profile_data = build_profiles(raw_data)
        torch.save(profile_data, profile_path)
        print(f"  Saved {profile_path}")

        # Stage 3: distances
        print(f"=== {code} s{scale}: building distances ===")
        dist_data = build_distances(profile_data)
        torch.save(dist_data, distances_path)
        print(f"  Saved {distances_path}")

        # Stage 4: orbit summary
        print(f"=== {code} s{scale}: building orbit summary ===")
        orbit_data = build_orbit_summary(profile_data)
        torch.save(orbit_data, orbit_path)

        json_data = {k: v for k, v in orbit_data.items() if not isinstance(v, Tensor)}
        json_path.write_text(json.dumps(json_data, indent=2) + "\n")
        print(f"  Saved {orbit_path}")
        print(f"  Saved {json_path}")

    print(f"\nDone: {code} scales {scales}")
    print(f"Output: orbits/{code}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full orbit pipeline for a creature")
    parser.add_argument("--code", "-c", required=True, help="Creature code")
    parser.add_argument("--scales", "-s", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Scales to run (default: 1 2 3 4)")
    parser.add_argument("--grid", "-g", type=int, default=64, help="Base grid size")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip raw stage (use existing raw files)")
    parser.add_argument("--catalog", default="animals.json")
    args = parser.parse_args()
    run_pipeline(args.code, args.scales, args.grid, args.skip_raw, args.catalog)
