"""Rebuild ALL neighborhoods from scratch at 128x128 base grid.

Full pipeline: raw frames → sorted activation profiles → distances → neighborhood summary.
The raw stage runs simulation (GPU), so this is the expensive part.

Usage:
    python scripts/rebuild_all_neighborhoods.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from substrate import load_animals
from neighborhoods.neighborhoods import (
    build_neighborhood, build_profiles, build_distances, build_neighborhood_summary,
    DEFAULT_GRID, DEFAULT_ROTATIONS, DEFAULT_FRAMES, DEFAULT_WARMUP_MULT,
)

CREATURES = ["3R4s", "H3s", "K4s", "K6s", "O2u", "O2v", "P4al", "S1s"]
SCALES = [4]
EXTRA = []

ORBITS_DIR = Path("neighborhoods")


def run_full_pipeline(creature, code: str, scale: int, creature_base_grid: int):
    """Run raw → profile → distances → neighborhood for one creature/scale."""
    out_dir = ORBITS_DIR / code / f"s{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = f"{code}_s{scale}"

    t0 = time.time()

    # Stage 1: raw frames (simulation)
    print(f"  [1/4] Raw frames...")
    raw_data = build_neighborhood(
        creature, scale=scale,
        grid_size=DEFAULT_GRID,
        num_rotations=DEFAULT_ROTATIONS,
        num_frames=DEFAULT_FRAMES,
        warmup_mult=DEFAULT_WARMUP_MULT,
        verbose=True,
        creature_base_grid=creature_base_grid,
    )
    raw_path = out_dir / f"{base_stem}_raw.pt"
    torch.save(raw_data, raw_path)
    frames = raw_data["frames"]
    size_mb = frames.element_size() * frames.numel() / 1e6
    print(f"  Saved {raw_path.name} ({frames.shape}, {size_mb:.1f} MB)")

    # Stage 2: histogram profiles
    print(f"  [2/4] Profiles...")
    profile_data = build_profiles(raw_data, verbose=True)
    profile_path = out_dir / f"{base_stem}_profile.pt"
    torch.save(profile_data, profile_path)
    print(f"  Saved {profile_path.name}")

    # Stage 3: pairwise distances
    print(f"  [3/4] Distances...")
    dist_data = build_distances(profile_data, verbose=True)
    dist_path = out_dir / f"{base_stem}_distances.pt"
    torch.save(dist_data, dist_path)
    print(f"  Saved {dist_path.name}")

    # Stage 4: neighborhood summary
    print(f"  [4/4] Neighborhood summary...")
    neighborhood_data = build_neighborhood_summary(profile_data)
    neighborhood_path = out_dir / f"{base_stem}_neighborhood.pt"
    torch.save(neighborhood_data, neighborhood_path)

    json_path = neighborhood_path.with_suffix(".json")
    json_data = {k: v for k, v in neighborhood_data.items() if not isinstance(v, torch.Tensor)}
    json_path.write_text(json.dumps(json_data, indent=2) + "\n")

    c_hat = neighborhood_data['c_hat']
    d_max = neighborhood_data['d_max']
    m = neighborhood_data['m']
    elapsed = time.time() - t0
    print(f"  m={m}, ĉ={c_hat:.6f}, d_max={d_max:.6f}")
    print(f"  Total: {elapsed:.1f}s")


def main():
    # Build job list
    jobs = [(code, scale) for code in CREATURES for scale in SCALES] + EXTRA
    print(f"Rebuilding {len(jobs)} neighborhoods at base grid {DEFAULT_GRID}x{DEFAULT_GRID}")
    print(f"Creatures: {CREATURES}")
    print(f"Scales: {SCALES} + extras {EXTRA}")
    print()

    # Load all creatures once
    catalog = load_animals(Path("animals.json"))
    by_code = {c.code: c for c in catalog}

    # Load creature bounding-box sizes from manifest
    manifest = json.loads(Path("animals_to_run.json").read_text())
    bg_lookup = {a["code"]: a["crop"] for a in manifest["animals"]}

    t0 = time.time()
    for i, (code, scale) in enumerate(jobs):
        creature = by_code.get(code)
        if creature is None:
            print(f"WARNING: {code} not found in animals.json, skipping")
            continue
        creature_bg = bg_lookup.get(code)
        if creature_bg is None:
            print(f"WARNING: {code} not in animals_to_run.json (no crop), skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(jobs)}] {creature.name} ({code}) scale={scale}")
        print(f"{'='*60}")

        run_full_pipeline(creature, code, scale, creature_base_grid=creature_bg)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: rebuilt {len(jobs)} neighborhoods in {total/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
