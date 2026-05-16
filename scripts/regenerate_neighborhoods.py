"""Regenerate all neighborhood profile/distances/neighborhood files from existing raw frames.

Runs the profile pipeline on every *_raw.pt found under
neighborhoods/. Overwrites existing profile, distances, and neighborhood files.

Usage:
    python scripts/regenerate_neighborhoods.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from neighborhoods.neighborhoods import build_profiles, build_distances, build_neighborhood_summary

ORBITS_DIR = Path("neighborhoods")


def regenerate_one(raw_path: Path):
    """Run profile → distances → neighborhood for a single raw .pt file."""
    stem = raw_path.stem  # e.g. "O2u_s2_raw"
    base_stem = stem.replace("_raw", "")
    out_dir = raw_path.parent

    print(f"\n{'─'*60}")
    print(f"  {base_stem}")
    print(f"{'─'*60}")

    # Load raw frames
    t0 = time.time()
    data = torch.load(raw_path, weights_only=False)
    print(f"  Loaded {raw_path.name} ({data['frames'].shape})")

    # Stage 1: profiles (sorted activation, m=μ+2σ)
    profile_data = build_profiles(data, verbose=True)
    profile_path = out_dir / f"{base_stem}_profile.pt"
    torch.save(profile_data, profile_path)
    print(f"  Saved {profile_path.name}")

    # Stage 2: pairwise distances
    dist_data = build_distances(profile_data, verbose=True)
    dist_path = out_dir / f"{base_stem}_distances.pt"
    torch.save(dist_data, dist_path)
    print(f"  Saved {dist_path.name}")

    # Stage 3: neighborhood summary (c_bar, c_hat, d_max)
    neighborhood_data = build_neighborhood_summary(profile_data)
    neighborhood_path = out_dir / f"{base_stem}_neighborhood.pt"
    torch.save(neighborhood_data, neighborhood_path)

    # JSON sidecar
    import json
    json_path = neighborhood_path.with_suffix(".json")
    json_data = {k: v for k, v in neighborhood_data.items() if not isinstance(v, torch.Tensor)}
    json_path.write_text(json.dumps(json_data, indent=2) + "\n")

    c_hat = neighborhood_data['c_hat']
    d_max = neighborhood_data['d_max']
    m = neighborhood_data['m']
    print(f"  Saved {neighborhood_path.name} + {json_path.name}")
    print(f"  m={m}, ĉ={c_hat:.6f}, d_max={d_max:.6f}")
    print(f"  [{time.time() - t0:.1f}s]")


def main():
    raw_files = sorted(ORBITS_DIR.rglob("*_raw.pt"))
    if not raw_files:
        print(f"No *_raw.pt files found under {ORBITS_DIR}")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw neighborhood files to regenerate:")
    for f in raw_files:
        print(f"  {f.relative_to(ORBITS_DIR)}")

    t0 = time.time()
    for raw_path in raw_files:
        regenerate_one(raw_path)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: regenerated {len(raw_files)} neighborhoods in {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
